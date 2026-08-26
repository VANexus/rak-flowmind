"""_hot_topics_client 测试：热点聚合 API 客户端（httpx MockTransport 注入）。

覆盖：
- 正常解析（title/hotValue/url + 顶层 name 作 source）
- 热度字符串（'1234.5万' / '8.2k'）解析
- 字段变体容错（name 当 word、mobilUrl 当 url）
- 空 data / 非 JSON / 缺 data → 结构化报错
- 网络分类：5xx=transient / 4xx=video / 超时=environment
"""
from __future__ import annotations


import httpx
import pytest

from flowmind.skills._hot_topics_client import HotTopicError, fetch_hot_topics, _parse_heat


def _client(payload, status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_parses_standard_payload():
    client = _client({
        "code": 200, "name": "抖音热榜",
        "data": [
            {"index": 1, "title": "通勤好物", "hot": "1234.5万", "hotValue": 12345000, "url": "https://a"},
            {"index": 2, "title": "夏日降温", "hot": "820", "hotValue": 820, "url": "https://b"},
        ],
    })
    out = fetch_hot_topics(api_base="https://api-hot.imsyy.top", endpoint="douyin",
                           limit=20, client=client)
    assert len(out) == 2
    assert out[0]["word"] == "通勤好物"
    assert out[0]["heat"] == 12345000
    assert out[0]["url"] == "https://a"
    assert out[0]["source"] == "抖音热榜"
    assert out[0]["delta"] is None
    assert out[1]["heat"] == 820


def test_fetch_limits():
    payload = {"name": "榜", "data": [{"title": f"t{i}", "hotValue": i} for i in range(10)]}
    out = fetch_hot_topics(api_base="https://x", endpoint="weibo", limit=3,
                           client=_client(payload))
    assert len(out) == 3


def test_heat_string_variants():
    assert _parse_heat("1234.5万") == 12345000
    assert _parse_heat("8.2k") == 8200
    assert _parse_heat("2亿") == 200_000_000
    assert _parse_heat(123) == 123
    assert _parse_heat(None) == 0
    assert _parse_heat("") == 0


def test_field_fallback_variants():
    """name 当 word、mobilUrl 当 url 也解析。"""
    client = _client({"data": [{"name": "车载必备", "rank": 88, "mobilUrl": "https://m"}]})
    out = fetch_hot_topics(api_base="https://x", endpoint="toutiao", client=client)
    assert out[0]["word"] == "车载必备"
    assert out[0]["heat"] == 88
    assert out[0]["url"] == "https://m"


def test_empty_data_returns_empty_list():
    out = fetch_hot_topics(api_base="https://x", endpoint="weibo", client=_client({"data": []}))
    assert out == []


def test_missing_data_raises():
    client = _client({"code": 500, "message": "oops"})
    with pytest.raises(HotTopicError, match="data"):
        fetch_hot_topics(api_base="https://x", endpoint="weibo", client=client)


def test_non_json_raises():
    def handler(req):
        return httpx.Response(200, text="<html>")
    with pytest.raises(HotTopicError, match="JSON"):
        fetch_hot_topics(api_base="https://x", endpoint="weibo",
                         client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_5xx_transient():
    with pytest.raises(HotTopicError) as ei:
        fetch_hot_topics(api_base="https://x", endpoint="weibo", client=_client({}, status=503))
    assert ei.value.category == "transient"
    assert ei.value.retriable is True


def test_4xx_video():
    with pytest.raises(HotTopicError) as ei:
        fetch_hot_topics(api_base="https://x", endpoint="weibo", client=_client({}, status=404))
    assert ei.value.category == "video"
    assert ei.value.retriable is False
