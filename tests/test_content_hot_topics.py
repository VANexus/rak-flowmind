"""content_hot_topics 技能测试：通过 invoke() 走信封层，monkeypatch mock 聚合 API。

覆盖：成功抓取 / 聚合 API 失败 → degraded 种子兜底（failure_category/retriable/warning）/
非法平台 VALIDATION / limit 透传。
"""
from __future__ import annotations

import flowmind.skills.content_hot_topics as mod
from flowmind.skill import invoke
from flowmind.skills._hot_topics_client import HotTopicError


def _stub(monkeypatch, fn):
    monkeypatch.setattr(mod, "fetch_hot_topics", fn)


def test_hot_topics_happy_path(monkeypatch):
    _stub(monkeypatch, lambda **kw: [
        {"word": "通勤好物", "heat": 12345000, "delta": None, "url": "https://a", "source": "抖音热榜"},
        {"word": "夏日降温", "heat": 820, "delta": None, "url": "", "source": "抖音热榜"},
    ])
    r = invoke("content_hot_topics", {"platform": "douyin", "limit": 2})
    assert r.ok is True
    assert r.metrics.degraded is False
    d = r.data
    assert d.degraded is False
    assert len(d.topics) == 2
    assert d.topics[0].word == "通勤好物"
    assert d.endpoint == "douyin"


def test_hot_topics_degraded_fallback(monkeypatch):
    def boom(**kw):
        raise HotTopicError("热点抓取超时", category="environment", retriable=False)

    _stub(monkeypatch, boom)
    r = invoke("content_hot_topics", {"platform": "xhs"})
    assert r.ok is True  # degraded 契约：ok 仍为 True
    assert r.metrics.degraded is True
    assert r.metrics.degradation_reason == "environment"
    d = r.data
    assert d.degraded is True
    assert d.failure_category == "environment"
    assert d.retriable is False
    assert d.topics  # 种子兜底非空
    assert "种子" in (d.warning or "")


def test_hot_topics_transient_retriable(monkeypatch):
    def boom(**kw):
        raise HotTopicError("5xx", category="transient", retriable=True)

    _stub(monkeypatch, boom)
    r = invoke("content_hot_topics", {"platform": "wechat"})
    assert r.metrics.degraded is True
    assert r.data.retriable is True


def test_hot_topics_invalid_platform(monkeypatch):
    _stub(monkeypatch, lambda **kw: [])
    r = invoke("content_hot_topics", {"platform": "ins"})
    assert r.ok is False
    assert r.error.code == "VALIDATION"


def test_hot_topics_limit_passthrough(monkeypatch):
    seen = {}

    def capture(**kw):
        seen["limit"] = kw.get("limit")
        return []

    _stub(monkeypatch, capture)
    invoke("content_hot_topics", {"platform": "douyin", "limit": 10})
    assert seen["limit"] == 10
