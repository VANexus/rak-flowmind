"""edit_pipeline 技能：剪辑项目流水线执行器（任务引擎的 worker 入口）。

**内部技能**：由 edit_project / edit_review 经 TaskManager.submit 驱动，
Agent 一般不直连（语义同 localize_video 之于 localize_submit）。入参只有
project_id——从 ProjectStore 载入项目继续执行（首次跑或审批后续跑）。

信封语义：
- 审批等待（awaiting_review）/ 全部完成（succeeded）→ ok=True 无
  failure_category → 任务落 succeeded（项目级状态在 store 里另行表达）；
- 阶段失败 → degraded 信封带 failure_category（environment/video/transient/
  unknown）→ 任务落 failed；
- 取消 → failure_category="cancelled" 的 degraded 信封 → 任务落 cancelled。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import Evidence, ReasoningChain, SkillOutput
from flowmind.editing.pipeline import (
    EditingStageError,
    project_dir,
    run_pipeline,
)
from flowmind.editing.store import ProjectStoreError
from flowmind.skill import skill
from flowmind.tasks import CancelledError

_VERSION = "0.1.0"


class PipelineInput(BaseModel):
    """Internal pipeline task input, including approval continuation data."""
    project_id: str = Field(min_length=8, description="剪辑项目 id")
    review_action: Literal["approve", "revise", "reject"] | None = None
    review_stage: Literal["copy", "script", "storyboard", "pre_render"] | None = None
    feedback: str | None = None


class PipelineReport(BaseModel):
    project_id: str
    state: str                      # running 已收敛：succeeded/awaiting_review/failed/cancelled
    stage: str
    message: str = ""
    artifact_dir: str = ""


@skill(id="edit_pipeline", name="剪辑项目流水线执行器（内部）",
       version=_VERSION)
def edit_pipeline(inp: PipelineInput) -> SkillOutput[PipelineReport]:
    """驱动剪辑项目执行/续跑：阶段推进、审批闸、产物落盘。"""
    pid = inp.project_id
    try:
        backend = load_config().editing.backend
        if backend == "cybercut":
            from flowmind.editing.cybercut_backend import (
                apply_review,
            )
            from flowmind.editing.cybercut_backend import (
                run_pipeline as run_cybercut,
            )
            if inp.review_action:
                apply_review(pid, inp.review_stage or "", inp.review_action,
                             inp.feedback)
            project = run_cybercut(pid)
        elif backend == "flowmind":
            project = run_pipeline(pid)
        else:
            raise ProjectStoreError(f"未知 editing.backend：{backend!r}")
    except CancelledError as exc:
        return SkillOutput(
            data=PipelineReport(project_id=pid, state="cancelled", stage=str(exc)),
            reasoning=[ReasoningChain(
                conclusion="项目已取消",
                causal_analysis="调用方取消 → 阶段边界协作退出 → 项目落 cancelled",
                risk_note="已产出的中间产物保留，可在有效期内查询",
                confidence=1.0)],
            confidence=1.0,
            degraded=True, degradation_reason="cancelled",
        )
    except EditingStageError as exc:
        return SkillOutput(
            data=PipelineReport(
                project_id=pid, state="failed",
                stage=exc.stage or "unknown", message=str(exc)),
            reasoning=[ReasoningChain(
                conclusion=f"阶段 {exc.stage or '?'} 执行失败",
                causal_analysis=str(exc)[:500],
                risk_note="可用 edit_retry 断点续跑（done 阶段与中间产物自动复用）",
                confidence=1.0)],
            confidence=1.0,
            degraded=True, degradation_reason=exc.category,
        )
    except ProjectStoreError as exc:
        return SkillOutput(
            data=PipelineReport(project_id=pid, state="failed", stage="store",
                                message=str(exc)),
            reasoning=[ReasoningChain(
                conclusion="项目存储不可用", causal_analysis=str(exc)[:500],
                risk_note="检查 PG 配置或文件回落开关", confidence=1.0)],
            confidence=1.0, degraded=True, degradation_reason="environment",
        )

    state_message = {
        "succeeded": "流水线完成，成片已产出",
        "awaiting_review": f"进入审批等待（闸门：{project.stage}）",
    }.get(project.state, project.state)
    done_stages = [s for s, v in project.stages.items() if v == "done"]
    return SkillOutput(
        data=PipelineReport(
            project_id=pid, state=project.state, stage=project.stage,
            message=state_message,
            artifact_dir=getattr(project, "artifact_dir", str(project_dir(pid)))),
        reasoning=[ReasoningChain(
            conclusion=state_message,
            triggered_rules=[], evidence=[Evidence(
                metric="done_stages", value=str(len(done_stages)),
                comparison="==", window="本次执行")],
            causal_analysis=f"已完成阶段：{' → '.join(done_stages) or '无'}；"
                            f"审批闸配置：{project.review_stages or '全自动'}",
            risk_note=("等待人工审批期间任务结束不占 GPU；"
                       "approve/revise 经 edit_review 驱动续跑"
                       if project.state == "awaiting_review"
                       else "产物目录含全部中间 JSON 与成片"),
            confidence=1.0)],
        confidence=1.0,
        sample_size=len(done_stages),
    )
