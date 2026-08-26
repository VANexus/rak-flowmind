"""content_audit 技能测试：通过 invoke() 走信封层。

覆盖：规则扫描命中违禁词 / 干净文案通过 / LLM 复核合并 findings / LLM 复核失败不回滚规则结果。
"""
from __future__ import annotations

import flowmind.skills.content_audit as mod
from flowmind.skill import invoke
from flowmind.skills._llm_client import LLMClientError


def _stub(monkeypatch, key: str | None = None):
    monkeypatch.setattr(mod, "get_api_key", lambda _env: key)


def test_audit_rules_catch_absolute_word(monkeypatch):
    _stub(monkeypatch)  # 无 LLM key → 纯规则扫描
    r = invoke("content_audit", {
        "platform": "xhs", "title": "全网最低价的保温杯", "body": "正文", "tags": [],
    })
    assert r.ok is True
    d = r.data
    assert d.passed is False
    assert d.llm_reviewed is False
    assert any(f.category == "absolute" and f.severity == "error" for f in d.findings)
    assert d.rule_finding_count >= 1


def test_audit_clean_copy_passes(monkeypatch):
    _stub(monkeypatch)
    r = invoke("content_audit", {
        "platform": "xhs",
        "title": "316 不锈钢内胆，一键开盖防漏",
        "body": "夏天保冷、冬天保暖，450ml 容量。",
        "tags": ["通勤好物"],
    })
    assert r.ok is True
    assert r.data.passed is True
    assert r.data.findings == []


def test_audit_llm_merges_findings(monkeypatch):
    _stub(monkeypatch, key="test-key")
    monkeypatch.setattr(mod, "llm_json", lambda **kw: {
        "findings": [
            {"category": "platform", "severity": "warning", "message": "建议补充拍摄场景", "suggestion": "增加使用场景"},
        ],
    })
    r = invoke("content_audit", {
        "platform": "xhs", "title": "保温杯", "body": "干净正文", "tags": [],
    })
    assert r.ok is True
    d = r.data
    assert d.llm_reviewed is True
    assert d.llm_finding_count == 1
    assert any(f.rule_id == "llm" for f in d.findings)


def test_audit_llm_failure_keeps_rules(monkeypatch):
    _stub(monkeypatch, key="test-key")

    def boom(**kw):
        raise LLMClientError("LLM HTTP 503", category="transient", retriable=True)

    monkeypatch.setattr(mod, "llm_json", boom)
    r = invoke("content_audit", {
        "platform": "xhs", "title": "全网最低价", "body": "正文", "tags": [],
    })
    # LLM 失败不导致整个审计失败：规则结果仍在，ok=True
    assert r.ok is True
    d = r.data
    assert d.llm_reviewed is False
    assert d.passed is False
    assert any(f.rule_id != "llm" for f in d.findings)


def test_audit_invalid_platform(monkeypatch):
    _stub(monkeypatch)
    r = invoke("content_audit", {"platform": "pinterest", "title": "t", "body": "b", "tags": []})
    assert r.ok is False
    assert r.error.code == "VALIDATION"
