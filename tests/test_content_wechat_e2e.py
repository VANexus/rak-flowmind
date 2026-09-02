"""content_wechat_e2e 技能测试：通过 invoke() 走信封层。

覆盖：凭证缺失降级 / 子步骤失败中断。
HTTP 依赖类：成功 ok=True，降级 ok=True + degraded=True。
"""
from __future__ import annotations

from unittest.mock import patch

import flowmind.skills.content_wechat_publish as wechat_publish_mod
import flowmind.skills.content_wechat_e2e as mod
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


def test_wechat_e2e_degrades_without_credentials():
    """无微信公众号凭证 → 发布步骤 degraded。"""

    def mock_invoke(skill_id, args):
        from flowmind.skills.content_idea_design import ContentIdeaPlan, IdeaAngle
        from flowmind.skills.content_copywrite import ContentCopyPlan
        from flowmind.skills.content_image_gen import ContentImagePlan, ContentImageResult
        from flowmind.skills.content_publish_check import PublishCheckResult
        from flowmind.skills.content_typeset import TypesetResult

        if skill_id == "content_idea_design":
            data = ContentIdeaPlan(
                platform="wechat", subject="保温杯",
                ideas=[IdeaAngle(title="通勤好物推荐", angle="通勤场景", reason="r")],
                prompt_source="mock", fallback=False,
            )
        elif skill_id == "content_copywrite":
            data = ContentCopyPlan(
                platform="wechat", subject="保温杯", angle="通勤场景", tone=None,
                title="通勤好物推荐", body="正文内容" * 20, tags=["通勤", "好物"],
            )
        elif skill_id == "content_image_gen":
            data = ContentImagePlan(
                platform="wechat", width=1920, height=1080,
                backend_used="mock",
                images=[ContentImageResult(index=0, url="https://img.mock/cover.jpg")],
            )
        elif skill_id == "content_publish_check":
            data = PublishCheckResult(
                platform="wechat", can_publish=True, title_length=6,
                body_length=200, image_count=1, limit_warnings=[], rule_findings=[],
            )
        elif skill_id == "content_typeset":
            data = TypesetResult(
                html="<section><p>正文内容</p></section>", theme="default",
                theme_label="经典", stats={"chars": 80},
            )
        elif skill_id == "content_wechat_publish":
            # 模拟无凭证：get_api_key 已 patch 为 None，这里会真实调用并失败
            # 但为了可控，直接 raise
            raise RuntimeError("未配置微信公众号 API 凭证")
        else:
            raise ValueError(f"未知 skill: {skill_id}")
        return _wrap(skill_id, SkillOutput(data=data, reasoning=[], confidence=0.9, sample_size=1))

    with patch.object(mod, "invoke", side_effect=mock_invoke), \
         patch.object(wechat_publish_mod, "get_api_key", lambda _env: None):
        r = invoke("content_wechat_e2e", {
            "subject": "保温杯",
            "angle": "通勤场景",
            "auto_publish": True,
        })
    assert r.ok is True  # HTTP 依赖类
    d = r.data
    assert d.status == "failed"
    assert r.metrics.degraded is True
    assert any(s.step == "publish" and not s.ok for s in d.steps)


def test_wechat_e2e_idea_step_failure():
    """选题步骤失败 → 中断并 degraded。"""

    def mock_invoke(skill_id, args):
        if skill_id == "content_idea_design":
            raise Exception("LLM 不可达")
        raise ValueError(f"不应调用 {skill_id}")

    with patch.object(mod, "invoke", side_effect=mock_invoke):
        r = invoke("content_wechat_e2e", {"subject": "保温杯"})
    assert r.ok is True
    d = r.data
    assert d.status == "failed"
    assert r.metrics.degraded is True
    assert any(s.step == "idea_design" and not s.ok for s in d.steps)


def test_wechat_e2e_empty_subject_rejected():
    """空主题 → VALIDATION。"""
    r = invoke("content_wechat_e2e", {"subject": ""})
    assert r.ok is False
    assert r.error.code == "VALIDATION"
