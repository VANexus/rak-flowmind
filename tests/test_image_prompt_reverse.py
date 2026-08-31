"""image_prompt_reverse 技能测试：走 invoke() 信封层，mock 视觉反推。"""
from __future__ import annotations

import flowmind.skills.image_prompt_reverse as mod
from flowmind.skill import invoke


def _stub(monkeypatch, reply: dict, key: str | None = "test-key"):
    monkeypatch.setattr(mod, "get_api_key", lambda _env: key)
    monkeypatch.setattr(mod, "reverse_prompt", lambda **kw: reply)


def test_reverse_happy(monkeypatch):
    _stub(monkeypatch, {
        "prompt": "minimal skincare product photography, soft light",
        "style_tags": ["minimal", "soft"],
        "negative_prompt": "no text",
    })
    r = invoke("image_prompt_reverse", {"image_url": "http://img/cover.jpg"})
    assert r.ok is True
    assert r.data.prompt
    assert r.data.style_tags == ["minimal", "soft"]
    assert r.data.negative_prompt == "no text"


def test_reverse_default_negative(monkeypatch):
    _stub(monkeypatch, {"prompt": "a", "style_tags": []})
    r = invoke("image_prompt_reverse", {"image_url": "http://img/x.jpg"})
    assert r.ok is True
    assert "no text" in r.data.negative_prompt


def test_reverse_no_key_raises(monkeypatch):
    _stub(monkeypatch, {"prompt": "a"}, key=None)
    r = invoke("image_prompt_reverse", {"image_url": "http://img/x.jpg"})
    assert r.ok is False
    assert "API_KEY" in r.error.message


def test_reverse_missing_prompt_raises(monkeypatch):
    _stub(monkeypatch, {"style_tags": []})
    r = invoke("image_prompt_reverse", {"image_url": "http://img/x.jpg"})
    assert r.ok is False
    assert r.error.code == "INTERNAL"


def test_reverse_missing_url_validation(monkeypatch):
    _stub(monkeypatch, {"prompt": "a"})
    r = invoke("image_prompt_reverse", {"image_url": ""})
    assert r.ok is False
    assert r.error.code == "VALIDATION"