"""b2b_longtail_keywords 技能测试：走 invoke() 信封层，mock LLM。"""
from __future__ import annotations

import flowmind.skills.b2b_longtail_keywords as mod
from flowmind.skill import invoke


def _stub(monkeypatch, reply: dict, key: str | None = "test-key"):
    monkeypatch.setattr(mod, "get_api_key", lambda _env: key)
    monkeypatch.setattr(mod, "llm_json", lambda **kw: reply)


def test_longtail_happy(monkeypatch):
    _stub(monkeypatch, {"keywords": [
        {"word": "ceramide face cream", "category": "成分", "search_intent": "commercial"},
        {"word": "glass dropper bottle", "category": "包装", "search_intent": "transactional"},
    ]})
    r = invoke("b2b_longtail_keywords", {"industry": "美妆个护", "seed_keywords": ["skincare"]})
    assert r.ok is True
    assert len(r.data.keywords) == 2
    assert r.data.keywords[0].category == "成分"


def test_longtail_capped(monkeypatch):
    _stub(monkeypatch, {"keywords": [{"word": f"kw{i}", "category": "c", "search_intent": ""} for i in range(100)]})
    r = invoke("b2b_longtail_keywords", {"industry": "家居", "limit": 3})
    assert len(r.data.keywords) == 3


def test_longtail_no_key_raises(monkeypatch):
    _stub(monkeypatch, {"keywords": [{"word": "x", "category": "c", "search_intent": ""}]}, key=None)
    r = invoke("b2b_longtail_keywords", {"industry": "家居"})
    assert r.ok is False
    assert r.error.code == "INTERNAL"
    assert "API_KEY" in r.error.message


def test_longtail_empty_result_raises(monkeypatch):
    _stub(monkeypatch, {"keywords": []})
    r = invoke("b2b_longtail_keywords", {"industry": "家居"})
    assert r.ok is False
    assert r.error.code == "INTERNAL"


def test_longtail_industry_required(monkeypatch):
    _stub(monkeypatch, {"keywords": []})
    r = invoke("b2b_longtail_keywords", {"industry": ""})
    assert r.ok is False
    assert r.error.code == "VALIDATION"