"""content_copywrite 技能测试：通过 invoke() 走信封层，monkeypatch mock LLM。

覆盖：成功解析 / tags 数量与长度裁剪 / body 截断 / subject 校验 / 无 key 报错。
"""
from __future__ import annotations

import flowmind.skills.content_copywrite as mod
from flowmind.skill import invoke


def _stub(monkeypatch, llm_reply: dict, key: str | None = "test-key"):
    monkeypatch.setattr(mod, "get_api_key", lambda _env: key)
    monkeypatch.setattr(mod, "llm_json", lambda **kw: llm_reply)


def test_copywrite_happy_path(monkeypatch):
    _stub(monkeypatch, {
        "title": "通勤路上也能优雅喝热水",
        "body": "316 不锈钢内胆，一键开盖防漏。",
        "tags": ["通勤好物", "316不锈钢"],
    })
    r = invoke("content_copywrite", {"platform": "xhs", "subject": "保温杯"})
    assert r.ok is True
    d = r.data
    assert d.title == "通勤路上也能优雅喝热水"
    assert d.platform == "xhs"
    assert d.tags == ["通勤好物", "316不锈钢"]
    assert r.reasoning and r.reasoning[0].conclusion


def test_copywrite_tags_capped(monkeypatch):
    _stub(monkeypatch, {
        "title": "标题", "body": "正文",
        "tags": [f"标签{i}" for i in range(20)],
    })
    r = invoke("content_copywrite", {"platform": "douyin", "subject": "车载杯"})
    assert r.ok is True
    assert len(r.data.tags) == 6  # cfg.max_tags


def test_copywrite_body_truncated(monkeypatch):
    _stub(monkeypatch, {
        "title": "标题", "body": "长" * 5000, "tags": [],
    })
    r = invoke("content_copywrite", {"platform": "wechat", "subject": "品牌"})
    assert r.ok is True
    assert len(r.data.body) <= 2000  # cfg.max_copy_length


def test_copywrite_subject_required(monkeypatch):
    _stub(monkeypatch, {"title": "标题", "body": "正文", "tags": []})
    r = invoke("content_copywrite", {"platform": "xhs", "subject": ""})
    assert r.ok is False
    assert r.error.code == "VALIDATION"


def test_copywrite_no_key_raises(monkeypatch):
    _stub(monkeypatch, {"title": "标题", "body": "正文", "tags": []}, key=None)
    r = invoke("content_copywrite", {"platform": "xhs", "subject": "保温杯"})
    assert r.ok is False
    assert r.error.code == "INTERNAL"
    assert "API_KEY" in r.error.message


def test_copywrite_missing_title_body_raises(monkeypatch):
    _stub(monkeypatch, {"tags": []})
    r = invoke("content_copywrite", {"platform": "xhs", "subject": "保温杯"})
    assert r.ok is False
    assert r.error.code == "INTERNAL"
