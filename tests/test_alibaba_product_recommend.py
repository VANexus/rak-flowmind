"""alibaba_product_recommend 技能测试：走 invoke() 信封层，mock LLM。"""
from __future__ import annotations

import flowmind.skills.alibaba_product_recommend as mod
from flowmind.skill import invoke


def _stub(monkeypatch, reply: dict, key: str | None = "test-key"):
    monkeypatch.setattr(mod, "get_api_key", lambda _env: key)
    monkeypatch.setattr(mod, "llm_json", lambda **kw: reply)


def test_recommend_happy(monkeypatch):
    _stub(monkeypatch, {"recommendations": [
        {"product_id": "1", "subject": "a", "score": 95, "reasons": ["skincare 关键词热度 Top1"]},
    ]})
    r = invoke("alibaba_product_recommend", {
        "preference": "alibaba",
        "products": [{"product_id": "1", "subject": "a", "keywords": ["skincare"]}],
        "trend_keywords": [{"word": "skincare", "heat": 100, "delta": 5}],
        "longtail_keywords": [],
    })
    assert r.ok is True
    assert len(r.data.recommendations) == 1
    assert "热度 Top1" in r.data.recommendations[0].reasons[0]


def test_recommend_empty_products_raises(monkeypatch):
    _stub(monkeypatch, {"recommendations": []})
    r = invoke("alibaba_product_recommend", {"preference": "mix", "products": []})
    assert r.ok is False
    assert r.error.code == "INTERNAL"


def test_recommend_no_key_raises(monkeypatch):
    _stub(monkeypatch, {"recommendations": []}, key=None)
    r = invoke("alibaba_product_recommend", {
        "preference": "mix", "products": [{"product_id": "1"}],
    })
    assert r.ok is False
    assert "API_KEY" in r.error.message


def test_recommend_invalid_preference(monkeypatch):
    _stub(monkeypatch, {"recommendations": []})
    r = invoke("alibaba_product_recommend", {"preference": "other", "products": [{"a": 1}]})
    assert r.ok is False
    assert r.error.code == "VALIDATION"