"""content_image_gen 技能测试：通过 invoke() 走信封层。

覆盖：mock 后端返回按平台尺寸的图 / count 张数 / auto 无 key 显式报错 / 非法平台 VALIDATION。
"""
from __future__ import annotations

import flowmind.skills.content_image_gen as mod
from flowmind.skill import invoke


def test_image_gen_mock_happy_path(monkeypatch):
    monkeypatch.setattr(mod, "get_api_key", lambda _env: None)
    r = invoke("content_image_gen", {
        "platform": "xhs", "prompt": "通勤场景保温杯", "count": 2, "backend": "mock",
    })
    assert r.ok is True
    d = r.data
    assert d.platform == "xhs"
    assert d.width == 1080 and d.height == 1440  # xhs 3:4
    assert d.backend_used == "mock"
    assert len(d.images) == 2
    assert all(img.url for img in d.images)
    assert r.reasoning and r.reasoning[0].conclusion


def test_image_gen_platform_dimensions(monkeypatch):
    monkeypatch.setattr(mod, "get_api_key", lambda _env: None)
    for platform, wh in (("wechat", (1920, 1080)), ("douyin", (1080, 1920))):
        r = invoke("content_image_gen", {
            "platform": platform, "prompt": "p", "count": 1, "backend": "mock",
        })
        assert r.ok is True, platform
        assert (r.data.width, r.data.height) == wh, platform


def test_image_gen_auto_no_key_raises(monkeypatch):
    monkeypatch.setattr(mod, "get_api_key", lambda _env: None)
    r = invoke("content_image_gen", {"platform": "xhs", "prompt": "p", "count": 1})
    assert r.ok is False
    assert r.error.code == "INTERNAL"
    assert "API_KEY" in r.error.message


def test_image_gen_unknown_backend_raises(monkeypatch):
    monkeypatch.setattr(mod, "get_api_key", lambda _env: None)
    r = invoke("content_image_gen", {
        "platform": "xhs", "prompt": "p", "count": 1, "backend": "sdxl",
    })
    assert r.ok is False
    assert r.error.code == "INTERNAL"


def test_image_gen_invalid_platform(monkeypatch):
    monkeypatch.setattr(mod, "get_api_key", lambda _env: None)
    r = invoke("content_image_gen", {"platform": "ins", "prompt": "p", "count": 1})
    assert r.ok is False
    assert r.error.code == "VALIDATION"
