"""content_idea_design 技能测试：通过 invoke() 走信封层，monkeypatch mock LLM。

覆盖：成功解析 / count 裁剪 / subject 校验（VALIDATION）/ 无 key 显式报错 / LLM 无 ideas 报错。
"""
from __future__ import annotations


import flowmind.skills.content_idea_design as mod
from flowmind.skill import invoke


def _stub(monkeypatch, llm_reply: dict, key: str | None = "test-key"):
    monkeypatch.setattr(mod, "get_api_key", lambda _env: key)
    monkeypatch.setattr(mod, "llm_json", lambda **kw: llm_reply)


def test_idea_design_happy_path(monkeypatch):
    _stub(monkeypatch, {"ideas": [
        {"angle": "痛点 + 场景", "title": "通勤党救星", "reason": "贴合通勤场景"},
        {"angle": "科普 · 攻略", "title": "316 不锈钢怎么挑", "reason": "科普有干货"},
    ]})
    r = invoke("content_idea_design", {"platform": "xhs", "subject": "保温杯", "count": 2})
    assert r.ok is True
    assert r.error is None
    d = r.data
    assert d.platform == "xhs"
    assert len(d.ideas) == 2
    assert d.ideas[0].title == "通勤党救星"
    assert r.reasoning and r.reasoning[0].conclusion


def test_idea_design_count_capped_by_config_max(monkeypatch):
    _stub(monkeypatch, {"ideas": [{"angle": "a", "title": f"选题{i}"} for i in range(20)]})
    r = invoke("content_idea_design", {"platform": "wechat", "subject": "品牌成长", "count": 6})
    assert r.ok is True
    # 入参 count=6 但 LLM 给了 20 条，保留最多 cfg.max_ideas(6) 条
    assert len(r.data.ideas) == 6


def test_idea_design_subject_required(monkeypatch):
    _stub(monkeypatch, {})
    r = invoke("content_idea_design", {"platform": "xhs", "subject": ""})
    assert r.ok is False
    assert r.error.code == "VALIDATION"


def test_idea_design_invalid_platform(monkeypatch):
    _stub(monkeypatch, {})
    r = invoke("content_idea_design", {"platform": "tiktok", "subject": "保温杯"})
    assert r.ok is False
    assert r.error.code == "VALIDATION"


def test_idea_design_no_key_raises_structured(monkeypatch):
    _stub(monkeypatch, {}, key=None)
    r = invoke("content_idea_design", {"platform": "xhs", "subject": "保温杯"})
    assert r.ok is False
    assert r.error.code == "INTERNAL"
    assert "API_KEY" in r.error.message


def test_idea_design_no_ideas_raises(monkeypatch):
    _stub(monkeypatch, {})
    r = invoke("content_idea_design", {"platform": "xhs", "subject": "保温杯"})
    assert r.ok is False
    assert r.error.code == "INTERNAL"
