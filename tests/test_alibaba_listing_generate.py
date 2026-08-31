"""alibaba_listing_generate 技能测试：走 invoke() 信封层，mock LLM。"""
from __future__ import annotations

import flowmind.skills.alibaba_listing_generate as mod
from flowmind.skill import invoke


def _stub(monkeypatch, reply: dict, key: str | None = "test-key"):
    monkeypatch.setattr(mod, "get_api_key", lambda _env: key)
    monkeypatch.setattr(mod, "llm_json", lambda **kw: reply)


def test_listing_happy(monkeypatch):
    _stub(monkeypatch, {
        "title": "Private Label Vitamin C Serum & Face Cream",
        "description": "High quality skincare OEM factory...",
        "keywords": ["serum", "skincare"],
        "image_prompt": "white background product photography",
    })
    r = invoke("alibaba_listing_generate", {"product_id": "1", "subject": "serum"})
    assert r.ok is True
    assert r.data.title
    assert r.data.keywords == ["serum", "skincare"]


def test_listing_title_cleaned_and_capped(monkeypatch):
    _stub(monkeypatch, {
        "title": "A&B | Best# Serum*" + "x" * 200,
        "description": "d",
        "keywords": [],
        "image_prompt": "img",
    })
    r = invoke("alibaba_listing_generate", {"product_id": "1"})
    assert r.ok is True
    assert "&" not in r.data.title
    assert "|" not in r.data.title
    assert len(r.data.title) <= 128


def test_listing_no_key_raises(monkeypatch):
    _stub(monkeypatch, {"title": "t", "description": "d"}, key=None)
    r = invoke("alibaba_listing_generate", {"product_id": "1"})
    assert r.ok is False
    assert "API_KEY" in r.error.message


def test_listing_missing_fields_raises(monkeypatch):
    _stub(monkeypatch, {"keywords": []})
    r = invoke("alibaba_listing_generate", {"product_id": "1"})
    assert r.ok is False
    assert r.error.code == "INTERNAL"