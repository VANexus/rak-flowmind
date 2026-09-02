"""TikHub 情报三 skill + client 新端点 + 解析器单测。

原则：传输层全部打桩（不打真实网络）；解析器对 tests/fixtures/tikhub 下真机录制的
真实响应做断言（录制响应不是 mock，是线上真实结构快照）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flowmind.skills import _tikhub_intel_parse as P
from flowmind.skills._tikhub_client import TikHubClient, TikHubError
from flowmind.skills.tiktok_ad_intel import AdIntelInput, tiktok_ad_intel
from flowmind.skills.tiktok_content_intel import ContentIntelInput, tiktok_content_intel
from flowmind.skills.tiktok_shop_intel import ShopIntelInput, tiktok_shop_intel

FIX = Path(__file__).parent / "fixtures" / "tikhub"


def fx(name: str):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


# ── 打桩 httpx（与 test_tikhub_trends 一致）─────────────────────────

class _FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


class _FakeHttp:
    def __init__(self, payload=None, exc=None):
        self.payload = payload if payload is not None else {"code": 200, "data": {}}
        self.exc = exc
        self.calls: list[dict] = []

    def request(self, method, url, *, json=None, params=None, headers=None):
        self.calls.append({"method": method, "url": url, "json": json, "params": params})
        if self.exc is not None:
            raise self.exc
        return _FakeResponse(self.payload)


def _client(http) -> TikHubClient:
    return TikHubClient(api_base="https://api.tikhub.dev", api_key="k", client=http)


# ════════════════════ client：路径 / 参数 / 字符串 body ════════════════

def test_client_locations_uses_string_body() -> None:
    # 双层信封：统一信封 data 内再包 {code,msg,data:{country}}
    inner = fx("ads_location_list.json")
    http = _FakeHttp({"code": 200, "data": inner})
    out = _client(http).ads_locations()
    assert http.calls[0]["method"] == "POST"
    assert http.calls[0]["url"].endswith("/get_location_list")
    assert http.calls[0]["json"] == ""  # 无 cookie 必须传空字符串而非 {}
    assert "country" in out


def test_client_ads_search_query() -> None:
    inner = fx("ads_search_ads_skincare.json")
    http = _FakeHttp({"code": 200, "data": inner})
    out = _client(http).ads_search(keyword="skincare", objective=3, industry="291")
    body = http.calls[0]["json"]
    assert body["keyword"] == "skincare"
    assert body["objective"] == 3
    assert body["industry"] == "291"
    assert "materials" in out  # 已解平双层信封


@pytest.mark.parametrize(("method", "path_suffix"), [
    ("web_trending_searchwords", "/fetch_trending_searchwords"),
    ("app_music_chart", "/fetch_music_chart_list"),
])
def test_client_get_endpoints(method, path_suffix) -> None:
    http = _FakeHttp({"code": 200, "data": {}})
    getattr(_client(http), method)()
    assert http.calls[0]["method"] == "GET"
    assert http.calls[0]["url"].endswith(path_suffix)


def test_client_shop_search_uses_search_word() -> None:
    inner = fx("shop_search_products.json")
    http = _FakeHttp({"code": 200, "data": inner})
    _client(http).shop_search_products(keyword="dress", region="US", offset=10)
    p = http.calls[0]["params"]
    assert p["search_word"] == "dress"  # 不是 keyword
    assert p["offset"] == 10


def test_client_reviews_use_v2() -> None:
    inner = fx("shop_product_reviews_v2.json")
    http = _FakeHttp({"code": 200, "data": inner})
    _client(http).shop_product_reviews(product_id="p1", page_start=2)
    assert "/fetch_product_reviews_v2" in http.calls[0]["url"]
    assert http.calls[0]["params"]["page_start"] == 2


def test_client_shop_suggest_returns_strlist() -> None:
    http = _FakeHttp({"code": 200, "data": {"code": 0, "data": ["dress a", "dress b"]}})
    assert _client(http).shop_search_suggest(keyword="dress") == ["dress a", "dress b"]


def test_client_missing_key_raises() -> None:
    c = TikHubClient(api_base="x", api_key="")
    with pytest.raises(TikHubError):
        c.ads_search(keyword="x")


# ════════════════════ 解析器：真机录制 fixture 驱动 ════════════════════

def test_parse_ad_materials_real() -> None:
    rows = P.parse_ad_materials(fx("ads_search_ads_skincare.json")["data"])
    assert len(rows) == 5
    r0 = rows[0]
    assert r0["id"]
    assert isinstance(r0["ctr"], float)
    assert r0["video_url"]  # 720p/default 至少一个
    pg = P.parse_ad_pagination(fx("ads_search_ads_skincare.json")["data"])
    assert pg["total"] == 313 and pg["has_more"] is True


def test_parse_filters_real() -> None:
    f = P.parse_ad_filters(fx("ads_top_ads_filters.json")["data"])
    assert len(f["industry"]) == 258
    assert len(f["objective"]) == 7
    assert f["industry"][0]["id"]


def test_parse_hashtag_detail_real() -> None:
    d = P.parse_hashtag_detail(fx("ads_trends_hashtag_detail.json"))
    assert d["name"] == "dollyparton"
    assert d["vv"] > 0
    assert len(d["curve"]) == 30
    assert d["age_profile"] and d["country_profile"] and d["videos"]


def test_parse_trending_words_real() -> None:
    rows = P.parse_trending_searchwords(fx("web_trending_searchwords.json"))
    assert len(rows) == 98 and rows[0]["word"]


def test_parse_video_search_real() -> None:
    rows = P.parse_video_search(fx("appv3_video_search.json"))
    assert len(rows) == 5
    r = rows[0]
    assert r["aweme_id"] and r["play"] == 41024 and r["likes"] == 453
    assert r["author"] and r["video_url"]  # 无水印地址


def test_parse_music_chart_real() -> None:
    rows = P.parse_music_chart(fx("appv3_music_chart.json"))
    assert rows[0]["rank"] == 1 and rows[0]["title"] == "oh yeah?"
    assert rows[0]["author"] == "Steve Lacy"


def test_parse_creator_insights_real() -> None:
    rows = P.parse_creator_insights(fx("appv3_creator_search_insights.json"))
    assert rows[0]["query"] and len(rows[0]["trend_seq"]) == 7


def test_parse_user_profile_real() -> None:
    u = P.parse_user_profile(fx("appv3_user_profile.json"))
    assert u["unique_id"] == "newsnews.69"
    assert u["followers"] == 13527 and u["aweme_count"] == 222


def test_parse_shop_products_real() -> None:
    rows = P.parse_shop_products(fx("shop_search_products.json")["data"])
    assert len(rows) == 30
    r = rows[0]
    assert r["product_id"] == "1732316229382607396"
    assert r["price"] == "65.99" and r["currency"] == "$"
    assert r["sold_count"] == 4211 and r["rating"] == 4.8
    assert r["seller_name"] and r["image_url"] and r["url"]


def test_parse_shop_categories_real() -> None:
    tree = P.parse_shop_categories(fx("shop_category_list.json"))
    assert len(tree) == 28 and tree[0]["children"]


def test_parse_product_detail_real() -> None:
    d = P.parse_product_detail(fx("shop_product_detail_v3.json")["product_data"])
    assert d["product_id"] == "1732316229382607396"
    assert len(d["images"]) == 9 and d["sku_count"] == 20 and len(d["specs"]) == 26
    assert d["shop"]["shop_name"] == "CurvySweet-US"


def test_parse_reviews_real() -> None:
    out = P.parse_product_reviews(fx("shop_product_reviews_v2.json")["data"])
    assert len(out["reviews"]) == 20
    assert out["summary"]["total"] == "232"
    assert out["summary"]["avg"] == 4.8
    assert out["summary"]["distribution"]["5"] == "205"
    assert out["reviews"][0]["text"]


def test_parse_ig_posts_real() -> None:
    out = P.parse_ig_hashtag_posts(fx("ig_hashtag_posts.json")["data"])
    assert len(out["posts"]) == 24
    p = out["posts"][0]
    assert p["username"] and p["likes"] == 5151 and p["thumbnail"]


def test_parsers_tolerate_empty() -> None:
    """空/异常结构不抛错，返回空集合（绝不编造）。"""
    assert P.parse_ad_materials({}) == []
    assert P.parse_video_search({}) == []
    assert P.parse_shop_products({}) == []
    assert P.parse_product_reviews({})["reviews"] == []
    assert P.parse_ig_hashtag_posts({})["posts"] == []
    assert P.parse_product_detail({}) == {}


# ════════════════════ skill：成功 + degraded 契约 ════════════════════

class _FakeIntelClient:
    """按方法名返回固定值的情报 client；值为 Exception 时抛出。"""

    def __init__(self, mapping):
        self._m = mapping

    def __getattr__(self, name):
        if name not in self._m:
            raise AttributeError(name)
        value = self._m[name]

        def _call(**kwargs):
            if isinstance(value, Exception):
                raise value
            return value
        return _call


def _patch_client(monkeypatch, fake):
    monkeypatch.setattr("flowmind.skills.tiktok_ad_intel.intel_client", lambda: fake)
    monkeypatch.setattr("flowmind.skills.tiktok_shop_intel.intel_client", lambda: fake)
    monkeypatch.setattr("flowmind.skills.tiktok_content_intel.intel_client", lambda: fake)


def test_skill_ad_search_ok(monkeypatch) -> None:
    fake = _FakeIntelClient({"ads_search": fx("ads_search_ads_skincare.json")["data"]})
    _patch_client(monkeypatch, fake)
    out = tiktok_ad_intel(AdIntelInput(action="search_ads", keyword="skincare", limit=5))
    assert not out.data.degraded
    assert len(out.data.materials) == 5
    assert out.data.pagination["total"] == 313


def test_skill_ad_search_limit_clamped_to_20(monkeypatch) -> None:
    # TikHub 单页硬上限 20：传 30 必须在 skill 内钳到 20，否则上游直接返空
    captured: dict = {}

    class _CapClient:
        def ads_search(self, **kwargs):
            captured.update(kwargs)
            return fx("ads_search_ads_skincare.json")["data"]

    monkeypatch.setattr("flowmind.skills.tiktok_ad_intel.intel_client", lambda: _CapClient())
    tiktok_ad_intel(AdIntelInput(action="search_ads", keyword="skincare", limit=30))
    assert captured["limit"] == 20


def test_skill_ad_search_missing_keyword_degraded(monkeypatch) -> None:
    _patch_client(monkeypatch, _FakeIntelClient({}))
    out = tiktok_ad_intel(AdIntelInput(action="search_ads"))
    assert out.data.degraded
    assert out.data.failure_category == "invalid_argument"
    assert out.data.materials == []


def test_skill_shop_search_ok(monkeypatch) -> None:
    fake = _FakeIntelClient({"shop_search_products": fx("shop_search_products.json")["data"]})
    _patch_client(monkeypatch, fake)
    out = tiktok_shop_intel(ShopIntelInput(action="search", keyword="dress", limit=10))
    assert not out.data.degraded and len(out.data.products) == 10
    assert out.data.products[0].price == "65.99"


def test_skill_shop_reviews_ok(monkeypatch) -> None:
    fake = _FakeIntelClient({"shop_product_reviews": fx("shop_product_reviews_v2.json")["data"]})
    _patch_client(monkeypatch, fake)
    out = tiktok_shop_intel(ShopIntelInput(action="reviews", product_id="p", limit=5))
    assert not out.data.degraded and len(out.data.reviews) == 5
    assert out.data.review_summary["avg"] == 4.8


def test_skill_content_videos_ok(monkeypatch) -> None:
    fake = _FakeIntelClient({"app_video_search": fx("appv3_video_search.json")})
    _patch_client(monkeypatch, fake)
    out = tiktok_content_intel(ContentIntelInput(action="video_search", keyword="x", limit=5))
    assert not out.data.degraded and len(out.data.videos) == 5


def test_skill_content_ig_posts_ok(monkeypatch) -> None:
    fake = _FakeIntelClient({"instagram_hashtag_posts": fx("ig_hashtag_posts.json")["data"]})
    _patch_client(monkeypatch, fake)
    out = tiktok_content_intel(ContentIntelInput(action="ig_hashtag_posts", keyword="fashion", limit=5))
    assert not out.data.degraded and len(out.data.ig_posts) == 5


def test_skill_transient_error_degraded(monkeypatch) -> None:
    err = TikHubError("限流", category="transient", retriable=True)
    fake = _FakeIntelClient({"web_trending_searchwords": err})
    _patch_client(monkeypatch, fake)
    out = tiktok_content_intel(ContentIntelInput(action="trending_words"))
    assert out.data.degraded
    assert out.data.failure_category == "transient" and out.data.retriable is True
    assert out.data.trending_words == []
