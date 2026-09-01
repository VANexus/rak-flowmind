"""InstagramTikHubTrendAdapter 单测（全部打桩，不打真实网络）。

真机验证过的线上结构（2026-09）：``GET /api/v1/instagram/v2/search_hashtags?keyword=...``
→ 信封 ``{code, data: {data: {count, items}}}``（data 内再包一层 data），
items 元素 ``{id, name, media_count, profile_pic_url, allow_following}``。
"""
from __future__ import annotations

import pytest

from flowmind.skills._tikhub_client import TikHubError
from flowmind.skills._tikhub_instagram import InstagramTikHubTrendAdapter


class _FakeTikHubClient:
    """打桩 TikHubClient：记录入参，返回 instagram_search_hashtags 的解包后 data。"""

    def __init__(self, *, data=None, exc=None, api_key="k"):
        self.data = data
        self.exc = exc
        self.api_key = api_key
        self.called = None

    def instagram_search_hashtags(self, *, keyword):
        self.called = {"keyword": keyword}
        if self.exc:
            raise self.exc
        return self.data


def test_parse_sorts_by_media_count_and_ranks():
    data = {"count": 3, "items": [
        {"id": "1", "name": "fashionhijab", "media_count": 9435623},
        {"id": "2", "name": "fashionstyle", "media_count": 134478084},
        {"id": "3", "name": "fashiongram", "media_count": 40835119},
    ]}
    adapter = InstagramTikHubTrendAdapter(client=_FakeTikHubClient(data=data))
    rows = adapter.fetch("instagram", keyword="#fashion", limit=20)

    assert [r["word"] for r in rows] == ["fashionstyle", "fashiongram", "fashionhijab"]
    assert rows[0]["heat"] == 134478084
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert all(r["delta"] is None for r in rows)
    assert all(r["source"] == "tikhub-instagram" for r in rows)
    # 关键词剥掉 # 前缀
    assert adapter._client.called == {"keyword": "fashion"}


def test_limit_truncates():
    data = {"count": 3, "items": [
        {"id": str(i), "name": f"tag{i}", "media_count": 100 - i} for i in range(3)
    ]}
    adapter = InstagramTikHubTrendAdapter(client=_FakeTikHubClient(data=data))
    rows = adapter.fetch("instagram", keyword="fashion", limit=2)
    assert len(rows) == 2
    assert [r["rank"] for r in rows] == [1, 2]


def test_missing_keyword_raises_unknown():
    adapter = InstagramTikHubTrendAdapter(client=_FakeTikHubClient())
    with pytest.raises(Exception) as ei:  # noqa: PT011
        adapter.fetch("instagram")
    assert getattr(ei.value, "category", "") == "unknown"
    assert "关键词" in str(ei.value)


def test_missing_api_key_raises_environment():
    adapter = InstagramTikHubTrendAdapter(client=_FakeTikHubClient(api_key=""))
    with pytest.raises(Exception) as ei:  # noqa: PT011
        adapter.fetch("instagram", keyword="fashion")
    assert getattr(ei.value, "category", "") == "environment"
    assert "TIKHUB_API_KEY" in str(ei.value)


def test_tikhub_error_maps_category_and_retriable():
    exc = TikHubError("余额不足 HTTP 402", category="environment", retriable=False)
    adapter = InstagramTikHubTrendAdapter(client=_FakeTikHubClient(exc=exc))
    with pytest.raises(Exception) as ei:  # noqa: PT011
        adapter.fetch("instagram", keyword="fashion")
    assert getattr(ei.value, "category", "") == "environment"
    assert getattr(ei.value, "retriable", True) is False


def test_transient_error_is_retriable():
    exc = TikHubError("限流 HTTP 429", category="transient", retriable=True)
    adapter = InstagramTikHubTrendAdapter(client=_FakeTikHubClient(exc=exc))
    with pytest.raises(Exception) as ei:  # noqa: PT011
        adapter.fetch("instagram", keyword="fashion")
    assert getattr(ei.value, "category", "") == "transient"
    assert getattr(ei.value, "retriable", False) is True


def test_missing_items_raises_unknown():
    adapter = InstagramTikHubTrendAdapter(client=_FakeTikHubClient(data={"count": 0}))
    with pytest.raises(Exception) as ei:  # noqa: PT011
        adapter.fetch("instagram", keyword="fashion")
    assert getattr(ei.value, "category", "") == "unknown"


def test_empty_items_raises_unknown():
    adapter = InstagramTikHubTrendAdapter(client=_FakeTikHubClient(data={"count": 0, "items": []}))
    with pytest.raises(Exception) as ei:  # noqa: PT011
        adapter.fetch("instagram", keyword="fashion")
    assert getattr(ei.value, "category", "") == "unknown"
    assert "返回为空" in str(ei.value)


def test_unsupported_platform_raises():
    adapter = InstagramTikHubTrendAdapter(client=_FakeTikHubClient())
    with pytest.raises(Exception) as ei:  # noqa: PT011
        adapter.fetch("tiktok")
    assert "不支持平台" in str(ei.value)
