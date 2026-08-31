"""content_web_fetch 技能测试：通过 invoke() 走信封层。

覆盖：正常抓取 / 降级（无效 URL）/ 参数边界。
HTTP 依赖类：成功时 ok=True，降级时 ok=True + degraded=True。
"""
from __future__ import annotations

from unittest.mock import patch

import flowmind.skills.content_web_fetch as mod
from flowmind.skill import invoke
from flowmind.skills._crawler_client import CrawlerError, FetchedPage


def _mock_page(url="https://example.com", title="示例标题", text="正文内容" * 50):
    return FetchedPage(
        url=url, status_code=200, title=title, text=text,
        links=["https://example.com/a", "https://example.com/b"],
    )


def test_fetch_success():
    page = _mock_page()
    with patch.object(mod, "fetch_url", return_value=page):
        r = invoke("content_web_fetch", {"url": "https://example.com"})
    assert r.ok is True
    d = r.data
    assert d.status_code == 200
    assert d.title == "示例标题"
    assert "正文内容" in d.text
    assert len(d.links) == 2
    assert r.metrics.degraded is False


def test_fetch_degrades_on_crawler_error():
    err = CrawlerError("连接超时", category="transient", retriable=True)
    with patch.object(mod, "fetch_url", side_effect=err):
        r = invoke("content_web_fetch", {"url": "https://unreachable.example.com"})
    assert r.ok is True  # HTTP 依赖类：ok=True + degraded
    d = r.data
    assert d.status_code == 0
    assert d.failure_category == "transient"
    assert d.retriable is True
    assert r.metrics.degraded is True


def test_fetch_environment_error_not_retriable():
    err = CrawlerError("DNS 解析失败", category="environment", retriable=False)
    with patch.object(mod, "fetch_url", side_effect=err):
        r = invoke("content_web_fetch", {"url": "https://no-such-host.invalid"})
    assert r.ok is True
    d = r.data
    assert d.failure_category == "environment"
    assert d.retriable is False


def test_fetch_respects_max_text_length():
    page = _mock_page(text="x" * 20000)
    with patch.object(mod, "fetch_url", return_value=page):
        r = invoke("content_web_fetch", {"url": "https://example.com", "max_text_length": 5000})
    assert r.ok is True
    assert len(r.data.text) <= 5000


def test_fetch_invalid_url_rejected():
    r = invoke("content_web_fetch", {"url": ""})
    assert r.ok is False
    assert r.error.code == "VALIDATION"
