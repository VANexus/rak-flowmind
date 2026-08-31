"""crawler_sentiment 技能测试：通过 invoke() 走信封层。

覆盖：正常抓取 / 部分平台失败 degraded / 多平台 / 空关键词。
HTTP 依赖类：成功 ok=True，降级 ok=True + degraded=True。
"""
from __future__ import annotations

from unittest.mock import patch

import flowmind.skills.crawler_sentiment as mod
from flowmind.skill import invoke
from flowmind.skills._crawler_client import CrawlerError, FetchedPage


def _mock_page(keyword: str, links_count: int = 5):
    return FetchedPage(
        url=f"https://s.weibo.com/weibo?q={keyword}",
        status_code=200,
        title=f"{keyword} - 微博搜索",
        text=f"关于 {keyword} 的讨论内容" * 20,
        links=[f"https://weibo.com/{keyword}/{i}" for i in range(links_count)],
    )


def test_sentiment_success():
    page = _mock_page("保温杯", 5)
    with patch.object(mod, "fetch_url", return_value=page):
        r = invoke("crawler_sentiment", {
            "keyword": "保温杯",
            "platforms": ["weibo"],
            "limit": 10,
        })
    assert r.ok is True
    d = r.data
    assert d.keyword == "保温杯"
    assert d.total_mentions == 5
    assert len(d.items) == 5
    assert all(item.platform == "weibo" for item in d.items)
    assert r.metrics.degraded is False


def test_sentiment_multi_platform():
    page = _mock_page("测试", 3)
    with patch.object(mod, "fetch_url", return_value=page):
        r = invoke("crawler_sentiment", {
            "keyword": "测试",
            "platforms": ["weibo", "toutiao"],
            "limit": 5,
        })
    assert r.ok is True
    d = r.data
    # 2 platforms × 3 links = 6 items
    assert d.total_mentions == 6
    assert "weibo" in d.platforms_queried
    assert "toutiao" in d.platforms_queried


def test_sentiment_partial_failure_degrades():
    """单平台 CrawlerError → 该平台返回 error item，整体 degraded 但不断阻断。"""

    def side_effect(url, **kw):
        if "weibo" in url:
            return _mock_page("保温杯", 2)
        raise CrawlerError("头条反爬限制", category="environment", retriable=False)

    with patch.object(mod, "fetch_url", side_effect=side_effect):
        r = invoke("crawler_sentiment", {
            "keyword": "保温杯",
            "platforms": ["weibo", "toutiao"],
        })
    assert r.ok is True
    d = r.data
    # 有失败也有成功 → degraded=True
    assert r.metrics.degraded is True
    # toutiao 那条应带 error
    toutiao_items = [i for i in d.items if i.platform == "toutiao"]
    assert len(toutiao_items) == 1
    assert toutiao_items[0].error is not None


def test_sentiment_all_failure_degrades():
    """全部平台失败 → degraded + failure_category。"""
    err = CrawlerError("连接超时", category="transient", retriable=True)
    with patch.object(mod, "fetch_url", side_effect=err):
        r = invoke("crawler_sentiment", {
            "keyword": "保温杯",
            "platforms": ["weibo"],
        })
    assert r.ok is True
    d = r.data
    assert d.failure_category == "transient"
    assert d.retriable is True
    assert d.total_mentions == 0
    assert r.metrics.degraded is True


def test_sentiment_empty_keyword_rejected():
    r = invoke("crawler_sentiment", {"keyword": "", "platforms": ["weibo"]})
    assert r.ok is False
    assert r.error.code == "VALIDATION"


def test_sentiment_respects_limit():
    page = _mock_page("k", 20)
    with patch.object(mod, "fetch_url", return_value=page):
        r = invoke("crawler_sentiment", {
            "keyword": "k",
            "platforms": ["weibo"],
            "limit": 3,
        })
    assert r.ok is True
    # 单平台 limit=3 → 最多 3 条
    assert len(r.data.items) <= 3


def test_sentiment_unsupported_platform_returns_error_item():
    """不支持的平台不触发 HTTP 调用，返回带 error 的 item。"""
    r = invoke("crawler_sentiment", {
        "keyword": "测试",
        "platforms": ["unknown_platform"],
        "limit": 5,
    })
    assert r.ok is True
    d = r.data
    assert len(d.items) == 1
    assert d.items[0].error is not None
    assert "不支持的平台" in d.items[0].error
