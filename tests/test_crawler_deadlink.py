"""crawler_deadlink 技能测试：通过 invoke() 走信封层。

覆盖：混合存活 / 降级（并发异常）/ 空 URL 快速返回 / 限流。
HTTP 依赖类：成功 ok=True，降级 ok=True + degraded=True。
"""
from __future__ import annotations

from unittest.mock import patch

import flowmind.skills.crawler_deadlink as mod
from flowmind.skill import invoke
from flowmind.skills._crawler_client import CrawlerError, LinkStatus


def _make_link(url: str, alive: bool, status: int = 200):
    return LinkStatus(url=url, alive=alive, status_code=status)


def test_deadlink_mixed_results():
    links = [
        _make_link("https://a.com", True, 200),
        _make_link("https://b.com", False, 404),
        _make_link("https://c.com", True, 200),
    ]
    with patch.object(mod, "check_links", return_value=links):
        r = invoke("crawler_deadlink", {
            "urls": ["https://a.com", "https://b.com", "https://c.com"],
        })
    assert r.ok is True
    d = r.data
    assert d.total == 3
    assert d.alive == 2
    assert d.dead == 1
    assert len(d.links) == 3


def test_deadlink_all_alive():
    links = [_make_link(f"https://ok{i}.com", True) for i in range(3)]
    with patch.object(mod, "check_links", return_value=links):
        r = invoke("crawler_deadlink", {"urls": [f"https://ok{i}.com" for i in range(3)]})
    assert r.ok is True
    assert r.data.dead == 0
    assert r.data.alive == 3


def test_deadlink_degrades_on_concurrent_error():
    """并发异常（CrawlerError）→ degraded。"""
    err = CrawlerError("连接池耗尽", category="transient", retriable=True)
    with patch.object(mod, "check_links", side_effect=err):
        r = invoke("crawler_deadlink", {"urls": ["https://a.com"]})
    assert r.ok is True
    d = r.data
    assert d.failure_category == "transient"
    assert d.retriable is True
    assert r.metrics.degraded is True


def test_deadlink_empty_urls_fast_return():
    """空 URL 列表 → 快速返回（不调用 check_links）。"""
    with patch.object(mod, "check_links") as mock_check:
        r = invoke("crawler_deadlink", {"urls": []})
    assert r.ok is True
    assert r.data.total == 0
    assert r.data.alive == 0
    assert r.data.dead == 0
    mock_check.assert_not_called()


def test_deadlink_respects_max_links():
    """超过 max_links_per_check 应截断后再传给 check_links。"""
    urls = [f"https://x.com/{i}" for i in range(150)]
    links = [_make_link(u, True) for u in urls[:100]]
    with patch.object(mod, "check_links", return_value=links) as mock_check:
        r = invoke("crawler_deadlink", {"urls": urls})
    assert r.ok is True
    called_urls = mock_check.call_args[1].get("urls") or mock_check.call_args[0][0]
    assert len(called_urls) <= 100
