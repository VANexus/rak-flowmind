"""content_xhs_e2e 技能测试：通过 invoke() 走信封层。

覆盖：选题失败中断 / 草稿步骤失败 / 空主题校验。
HTTP 依赖类：成功 ok=True，降级 ok=True + degraded=True。
"""
from __future__ import annotations

from unittest.mock import patch

import flowmind.skills.content_xhs_e2e as mod
from flowmind.contracts import ReliabilityMetrics, SkillOutput, SkillResult, new_trace
from flowmind.skill import invoke


def _wrap(skill_id: str, output: SkillOutput) -> SkillResult:
    """将 SkillOutput 包装为 invoke() 返回的 SkillResult 信封。"""
    return SkillResult(
        ok=True,
        skill=skill_id,
        version="0.1.0",
        trace=new_trace(),
        data=output.data,
        reasoning=output.reasoning,
        metrics=ReliabilityMetrics(
            latency_ms=100.0,
            confidence=output.confidence,
            sample_size=output.sample_size,
            degraded=output.degraded,
            degradation_reason=output.degradation_reason,
        ),
    )


def test_xhs_e2e_idea_step_failure():
    """选题步骤失败 → 中断并 degraded。"""

    def mock_invoke(skill_id, args):
        if skill_id == "content_idea_design":
            raise Exception("LLM 不可达")
        raise ValueError(f"不应调用 {skill_id}")

    with patch.object(mod, "invoke", side_effect=mock_invoke):
        r = invoke("content_xhs_e2e", {"subject": "保温杯"})
    assert r.ok is True
    d = r.data
    assert r.metrics.degraded is True
    assert any(s.step == "idea_design" and not s.ok for s in d.steps)


def test_xhs_e2e_draft_step_failure():
    """草稿步骤失败 → 中断。"""

    def mock_invoke(skill_id, args):
        from flowmind.skills.content_idea_design import ContentIdeaPlan, IdeaAngle
        from flowmind.skills.content_copywrite import ContentCopyPlan
        from flowmind.skills.content_image_gen import ContentImagePlan
        from flowmind.skills.content_audit import ContentAuditPlan
        from flowmind.skills.content_hot_topics import ContentHotPlan, HotTopic

        if skill_id == "content_xhs_draft":
            raise Exception("草稿生成异常")
        if skill_id == "content_idea_design":
            data = ContentIdeaPlan(
                platform="xhs", subject="保温杯",
                ideas=[IdeaAngle(title="测试选题", angle="测试", reason="r")],
                prompt_source="mock", fallback=False,
            )
        elif skill_id == "content_copywrite":
            data = ContentCopyPlan(
                platform="xhs", subject="保温杯", angle="测试", tone=None,
                title="t", body="正文内容" * 10, tags=["tag1"],
            )
        elif skill_id == "content_image_gen":
            data = ContentImagePlan(
                platform="xhs", width=1080, height=1440,
                backend_used="mock", images=[],
            )
        elif skill_id == "content_audit":
            data = ContentAuditPlan(
                platform="xhs", passed=True, findings=[],
                llm_reviewed=False, rule_finding_count=0, llm_finding_count=0,
            )
        elif skill_id == "content_hot_topics":
            data = ContentHotPlan(
                platform="xhs", source="mock", endpoint="mock",
                degraded=False, topics=[HotTopic(word="热点", heat=100, source="mock")],
            )
        else:
            raise ValueError(f"未知 skill: {skill_id}")
        return _wrap(skill_id, SkillOutput(data=data, reasoning=[], confidence=0.9, sample_size=1))

    with patch.object(mod, "invoke", side_effect=mock_invoke):
        r = invoke("content_xhs_e2e", {"subject": "保温杯", "use_hot_topics": True})
    assert r.ok is True
    d = r.data
    assert r.metrics.degraded is True
    assert any(s.step == "draft" and not s.ok for s in d.steps)


def test_xhs_e2e_empty_subject_rejected():
    """空主题 → VALIDATION。"""
    r = invoke("content_xhs_e2e", {"subject": ""})
    assert r.ok is False
    assert r.error.code == "VALIDATION"
