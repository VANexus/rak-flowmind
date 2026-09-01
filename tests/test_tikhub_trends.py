"""TikHub client + TikTokTikHubTrendAdapter 单测（全部打桩，不打真实网络）。"""
from __future__ import annotations

import pytest

from flowmind.skills._tikhub_client import (
    TikHubClient,
    TikHubError,
    normalize_time_range,
)
from flowmind.skills._tikhub_trends import TikTokTikHubTrendAdapter
from flowmind.skills._trend_adapters import TrendError, resolve_adapter


# ── 纯函数：时间窗归一 ──────────────────────────────────────────────


@pytest.mark.parametrize(("given", "expected"), [
    (7, 7), (30, 30), (90, 90),
    (1, 7), (14, 7), (20, 30), (1000, 90),
    (None, 7), ("x", 7),
])
def test_normalize_time_range(given, expected) -> None:
    assert normalize_time_range(given) == expected


# ── 打桩 httpx.Client ───────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        if self._payload is _BAD_JSON:
            raise ValueError("bad json")
        return self._payload


_BAD_JSON = object()


class _FakeHttp:
    """记录请求参数，按队列/固定规则返回响应。"""

    def __init__(self, responses=None, exc=None):
        self.responses = list(responses or [])
        self.exc = exc
        self.calls: list[dict] = []

    def request(self, method, url, *, json=None, params=None, headers=None):
        self.calls.append({
            "method": method, "url": url, "json": json, "params": params, "headers": headers,
        })
        if self.exc is not None:
            raise self.exc
        if not self.responses:
            raise AssertionError("fake http 响应已耗尽")
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def _envelope(items, *, has_more=False, page=1, limit=20, total=None, code=200):
    return {
        "code": code,
        "message": "ok",
        "data": {
            "BaseResp": {"StatusCode": 0},
            "items": items,
            "pagination": {"hasMore": has_more, "limit": limit, "page": page,
                           "totalCount": total if total is not None else len(items)},
        },
    }


def _page(items, **kwargs) -> dict:
    """模拟 TikHubClient.trending_hashtags 的返回：信封解包后的 data。"""
    return _envelope(items, **kwargs)["data"]


def _item(name: str, rank: int, vv: int = 1000, first=100.0, last=120.0) -> dict:
    return {
        "hashtagID": rank,
        "hashtagName": name,
        "industryIDs": [],
        "popularityCurve": [
            {"timestamp": "1", "value": first},
            {"timestamp": "2", "value": last},
        ],
        "publishCnt": 10,
        "rankIndex": rank,
        "vv": vv,
    }


# ── TikHubClient：信封与错误分类 ────────────────────────────────────


def test_client_trending_hashtags_unwraps_data() -> None:
    http = _FakeHttp([_FakeResponse(200, _envelope([_item("a", 1)]))])
    client = TikHubClient(api_base="https://api.tikhub.dev", api_key="k", client=http)
    data = client.trending_hashtags(time_range=7, country_code="US", page=1, limit=20)
    assert data["pagination"]["totalCount"] == 1
    call = http.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/api/v1/tiktok/ads/get_trends_hashtag_list")
    assert call["headers"]["Authorization"] == "Bearer k"
    assert call["json"] == {"time_range": 7, "country_code": "US", "page": 1, "limit": 20}


def test_client_industry_and_cookie_passed_through() -> None:
    http = _FakeHttp([_FakeResponse(200, _envelope([]))])
    client = TikHubClient(api_base="https://api.tikhub.dev", api_key="k", client=http)
    client.trending_hashtags(industry_id=22000000000, cookie="sessionid=x")
    body = http.calls[0]["json"]
    assert body["industry_id"] == 22000000000
    assert body["cookie"] == "sessionid=x"


def test_client_missing_key_raises_environment() -> None:
    client = TikHubClient(api_base="https://api.tikhub.dev", api_key="")
    with pytest.raises(TikHubError) as ei:
        client.trending_hashtags()
    assert ei.value.category == "environment"
    assert ei.value.retriable is False


@pytest.mark.parametrize(("status", "category", "retriable"), [
    (401, "environment", False),
    (403, "environment", False),
    (402, "environment", False),
    (429, "transient", True),
    (500, "transient", True),
    (503, "transient", True),
    (400, "unknown", False),
    (404, "unknown", False),
    (422, "unknown", False),
])
def test_client_http_status_mapping(status, category, retriable) -> None:
    http = _FakeHttp([_FakeResponse(status, {"detail": "x"})])
    client = TikHubClient(api_base="https://api.tikhub.dev", api_key="k", client=http)
    with pytest.raises(TikHubError) as ei:
        client.trending_hashtags()
    assert ei.value.category == category
    assert ei.value.retriable is retriable


def test_client_envelope_business_code_error() -> None:
    http = _FakeHttp([_FakeResponse(200, {"code": 500, "message": "boom", "data": None})])
    client = TikHubClient(api_base="https://api.tikhub.dev", api_key="k", client=http)
    with pytest.raises(TikHubError):
        client.trending_hashtags()


# ── Adapter：解析 / 分页 / 截断 / 错误映射 ──────────────────────────


class _FakeTikHubClient:
    """adapter 层打桩：直接实现 trending_hashtags。"""

    def __init__(self, pages=None, exc=None, api_key: str = "k"):
        self.pages = list(pages or [])
        self.exc = exc
        self.api_key = api_key
        self.calls: list[dict] = []

    def trending_hashtags(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.pages.pop(0)


def test_adapter_single_page_rows() -> None:
    data = _page([_item("dollyparton", 1, vv=1937783877, first=100, last=160),
                  _item("dolly", 2, vv=364989556)], has_more=False)
    adapter = TikTokTikHubTrendAdapter(client=_FakeTikHubClient([data]))
    rows = adapter.fetch("tiktok", limit=20)
    assert [r["word"] for r in rows] == ["dollyparton", "dolly"]
    assert rows[0]["heat"] == 1937783877
    assert rows[0]["delta"] == 60
    assert rows[0]["rank"] == 1
    assert all(r["source"] == "tikhub" for r in rows)


def test_adapter_paginates_until_limit() -> None:
    page1 = _page([_item(f"w{i}", i) for i in range(1, 21)], has_more=True, page=1)
    page2 = _page([_item(f"w{i}", i) for i in range(21, 26)], has_more=False, page=2)
    fake = _FakeTikHubClient([page1, page2])
    adapter = TikTokTikHubTrendAdapter(client=fake)
    rows = adapter.fetch("tiktok", limit=25)
    assert len(rows) == 25
    assert [c["page"] for c in fake.calls] == [1, 2]
    assert rows[-1]["word"] == "w25"


def test_adapter_respects_limit_smaller_than_page() -> None:
    data = _page([_item(f"w{i}", i) for i in range(1, 6)], has_more=False)
    fake = _FakeTikHubClient([data])
    adapter = TikTokTikHubTrendAdapter(client=fake)
    rows = adapter.fetch("tiktok", limit=3)
    assert len(rows) == 3
    # 只打一页（拿到即止）
    assert len(fake.calls) == 1


def test_adapter_passes_industry_and_period() -> None:
    data = _page([_item("a", 1)], has_more=False)
    fake = _FakeTikHubClient([data])
    adapter = TikTokTikHubTrendAdapter(period_days=30, country="GB", client=fake)
    adapter.fetch("tiktok", industry_id=22000000000, limit=5)
    kwargs = fake.calls[0]
    assert kwargs["industry_id"] == 22000000000
    assert kwargs["time_range"] == 30
    assert kwargs["country_code"] == "GB"


def test_adapter_empty_result_raises() -> None:
    adapter = TikTokTikHubTrendAdapter(client=_FakeTikHubClient([_page([], has_more=False)]))
    with pytest.raises(TrendError) as ei:
        adapter.fetch("tiktok")
    assert ei.value.category == "unknown"


def test_adapter_without_api_key_raises_environment() -> None:
    adapter = TikTokTikHubTrendAdapter(client=_FakeTikHubClient(api_key=""))
    with pytest.raises(TrendError) as ei:
        adapter.fetch("tiktok")
    assert ei.value.category == "environment"
    assert ei.value.retriable is False


def test_adapter_maps_tikhub_error() -> None:
    err = TikHubError("余额不足", category="environment", retriable=False)
    adapter = TikTokTikHubTrendAdapter(client=_FakeTikHubClient(exc=err))
    with pytest.raises(TrendError) as ei:
        adapter.fetch("tiktok")
    assert ei.value.category == "environment"
    assert ei.value.retriable is False


def test_adapter_rejects_other_platform() -> None:
    adapter = TikTokTikHubTrendAdapter(client=_FakeTikHubClient())
    with pytest.raises(TrendError):
        adapter.fetch("instagram")


# ── resolve_adapter 路由 ────────────────────────────────────────────


class _Cfg:
    default_country = "US"
    tiktok_trend_source = "tikhub"
    tikhub_timeout_s = 30.0
    tikhub_max_pages = 5
    cc_scrape_page_url = "https://ads.tiktok.com/x"
    cc_scrape_period_days = 7
    cc_scrape_timeout_s = 90.0
    cc_scrape_headless = True
    cc_scrape_proxy = ""
    trend_timeout_s = 15.0


def test_resolve_tiktok_defaults_to_tikhub() -> None:
    adapter = resolve_adapter("tiktok", _Cfg())
    assert isinstance(adapter, TikTokTikHubTrendAdapter)
    assert adapter.name == "tikhub"


def test_resolve_tiktok_tikhub_receives_session_cookie() -> None:
    adapter = resolve_adapter("tiktok", _Cfg(), session_cookie="sessionid=abc")
    assert adapter.session_cookie == "sessionid=abc"


def test_resolve_tiktok_fallback_to_cc_scraper() -> None:
    from flowmind.skills._cc_scraper import TikTokCreativeScraperAdapter

    cfg = _Cfg()
    cfg.tiktok_trend_source = "cc_scraper"
    adapter = resolve_adapter("tiktok", cfg)
    assert isinstance(adapter, TikTokCreativeScraperAdapter)
    assert adapter.name == "cc_scraper"
