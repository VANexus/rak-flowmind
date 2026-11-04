"""edit_review 技能：剪辑项目审批动作（人工复查回路）。

动作语义（设计文档 §3.3 / §4）：
- approve  放行：该闸记入 approved → 提交续跑任务进入下一阶段；
- revise   重跑：feedback 必填 → 打回该阶段（storyboard 闸）或带注续跑
            （pre_render 闸：渲染尚未开跑，反馈记档供后续参考）→ 提交续跑；
- reject   终止：项目落 failed（error=review_rejected），不再续跑。

仅 awaiting_review 状态且 stage 与当前闸一致的项目可动作（否则显式
报错）；跨租户与不存在同形。续跑 = 新的 edit_pipeline 任务（引擎语义
不变：进度/取消/事件全复用）。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from flowmind import auth
from flowmind.config import load_config
from flowmind.contracts import Evidence, ReasoningChain, SkillOutput
from flowmind.editing.models import (
    PSTATE_AWAITING_REVIEW,
    PSTATE_RUNNING,
    STAGE_RENDER,
    resolve_backend,
)
from flowmind.editing.store import get_project_store
from flowmind.skill import skill
from flowmind.tasks.manager import get_task_manager

_VERSION = "0.1.0"


class ReviewInput(BaseModel):
    project_id: str = Field(min_length=8, description="项目 id")
    stage: str = Field(description="待审闸门：copy/script/storyboard/pre_render")
    action: Literal["approve", "revise", "reject"] = Field(
        description="approve=放行；revise=带反馈重跑；reject=终止项目")
    feedback: str | None = Field(
        default=None,
        description="revise 必填（进该阶段 LLM 提示词）；reject 建议填（记档）")


class ReviewReport(BaseModel):
    project_id: str
    action: str
    state: str
    task_id: str | None = None       # approve/revise 续跑任务；reject 无
    message: str = ""


@skill(id="edit_review", name="剪辑项目审批（approve/revise/reject）",
       version=_VERSION)
def edit_review(inp: ReviewInput) -> SkillOutput[ReviewReport]:
    """审批闸动作：放行 / 带新提示词重跑本阶段 / 终止项目。"""
    cfg = load_config().editing
    if inp.stage not in cfg.reviewable_stages:
        raise ValueError(
            f"stage 必须是 {cfg.reviewable_stages} 之一，收到：{inp.stage}")
    if inp.action == "revise" and not (inp.feedback or "").strip():
        raise ValueError("revise 必须带 feedback（重跑提示词）")
    if inp.action == "reject" and not (inp.feedback or "").strip():
        raise ValueError("reject 需要 feedback 说明拒绝原因（记档）")

    store = get_project_store()
    project = store.get_project(inp.project_id)
    required, tenant = auth.tenant_scope()
    if required and (project is None or tenant is None
                     or project.tenant_id != tenant):
        raise ValueError(f"项目不存在：{inp.project_id}")   # 跨租户同形
    if project is None:
        raise ValueError(f"项目不存在：{inp.project_id}")
    if project.state != PSTATE_AWAITING_REVIEW:
        raise ValueError(
            f"项目当前状态 {project.state} 不在审批等待，无法对闸门 "
            f"{inp.stage} 执行动作（当前闸门：{project.stage}）")
    if project.stage != inp.stage:
        raise ValueError(
            f"闸门不匹配：当前等待 {project.stage}，收到 {inp.stage}")

    feedback = (inp.feedback or "").strip()

    openfilm_backend = (
        project.backend == "openfilm" or "openfilm" in project.input)
    if openfilm_backend:
        return _review_openfilm(inp, project, feedback)

    cybercut_backend = resolve_backend(project) == "cybercut"

    if inp.action == "reject":
        if cybercut_backend:
            from flowmind.editing.cybercut_backend import apply_review
            apply_review(inp.project_id, inp.stage, inp.action, feedback)
        project.state = "failed"
        project.error = f"review_rejected[{inp.stage}]: {feedback}"
        store.save_project(project)
        return SkillOutput(
            data=ReviewReport(
                project_id=inp.project_id, action="reject",
                state=project.state, message="项目已按人工审批终止"),
            reasoning=[ReasoningChain(
                conclusion=f"闸门 {inp.stage} 被拒绝，项目终止",
                causal_analysis=f"人工审批反馈：{feedback[:200]}",
                risk_note="可基于原素材重新发起新项目", confidence=1.0)],
            confidence=1.0)

    # ── approve / revise：驱动续跑 ──
    previous = project.model_copy(deep=True)
    if inp.action == "approve":
        if inp.stage not in project.approved:
            project.approved.append(inp.stage)
    else:   # revise
        project.review_feedback.setdefault(inp.stage, []).append(feedback)
        if inp.stage != "pre_render":
            # storyboard 闸：打回分镜重跑；copy/script 闸同理（阶段名=闸名）
            if inp.stage in project.stages:
                project.stages[inp.stage] = "pending"
    project.state = PSTATE_RUNNING
    project.stage = STAGE_RENDER if inp.action == "revise" and \
        inp.stage == "pre_render" else project.stage
    store.save_project(project)

    manager = get_task_manager()
    try:
        task_args = {"project_id": inp.project_id}
        if cybercut_backend:
            task_args.update({"review_action": inp.action,
                              "review_stage": inp.stage,
                              "feedback": feedback})
        task_id = manager.submit(
            "edit_pipeline", task_args)
    except Exception:
        store.save_project(previous)
        raise
    # 回存前重载：续跑任务可能已推进项目状态，不能用本副本覆盖
    fresh = store.get_project(inp.project_id) or project
    fresh.task_ids.append(task_id)
    store.save_project(fresh)

    message = {
        "approve": f"闸门 {inp.stage} 已放行，续跑下一阶段",
        "revise": (f"闸门 {inp.stage} 已带反馈重跑"
                   if inp.stage != "pre_render"
                   else f"闸门 {inp.stage} 已带反馈放行（渲染续跑，反馈记档）"),
    }[inp.action]
    return SkillOutput(
        data=ReviewReport(
            project_id=inp.project_id, action=inp.action,
            state=project.state, task_id=task_id, message=message),
        reasoning=[ReasoningChain(
            conclusion=message,
            evidence=[Evidence(metric="feedback_len",
                               value=str(len(feedback)), comparison=">=",
                               window="本次动作")],
            causal_analysis=(f"人工审批 {inp.action} → 项目状态更新 → "
                             "提交 edit_pipeline 续跑任务"),
            risk_note="续跑任务复用引擎进度事件；再次到闸仍会暂停等待审批",
            confidence=1.0)],
        confidence=1.0, sample_size=1,
    )


def _review_openfilm(inp: ReviewInput, project, feedback: str):
    """OpenFilm 审批：approve 触发渲染；revise 退回外部 Agent；reject 终止。"""
    from flowmind.editing.openfilm_backend import (
        OpenFilmError,
        mark_render_submit_failed,
        prepare_render,
        record_render_task,
    )
    from flowmind.editing.openfilm_backend import (
        apply_review as apply_openfilm_review,
    )

    try:
        outcome = apply_openfilm_review(
            inp.project_id, inp.stage, inp.action, feedback)
        prepared = (prepare_render(inp.project_id)
                    if outcome.get("submit_render") else None)
    except OpenFilmError as exc:
        # 并发请求可能已先领取/完成：不要把本请求早先的快照写回去覆盖
        # 对方状态（claim 冲突时尤其如此）。
        raise ValueError(str(exc)) from exc

    if not outcome.get("submit_render"):
        return SkillOutput(
            data=ReviewReport(
                project_id=inp.project_id, action=inp.action,
                state=outcome.get("state", project.state), task_id=None,
                message=outcome.get("message", "")),
            reasoning=[ReasoningChain(
                conclusion=outcome.get("message", "OpenFilm 审批已记录"),
                evidence=[Evidence(metric="feedback_len",
                                   value=str(len(feedback)), comparison=">=",
                                   window="本次动作")],
                causal_analysis=(
                    f"edit_review {inp.action} → OpenFilm 项目状态 "
                    f"{outcome.get('state')} → 反馈写入工作区 "
                    "FLOWMIND_REVIEW.md"),
                risk_note=("revise 不渲染：外部 Agent 修改 film.html 后需"
                           "重新调用 edit_openfilm_render"
                           if inp.action == "revise" else
                           "reject 为终态，可基于原素材重新建项目"),
                confidence=1.0)],
            confidence=1.0, sample_size=1)

    # approve：复用任务引擎执行渲染。提交失败则释放领取并落 failed。
    if not prepared or not prepared.get("submit"):
        raise ValueError("OpenFilm 渲染领取失败，未提交任务")
    claim_token = prepared.get("claim_token")
    manager = get_task_manager()
    try:
        task_id = manager.submit("edit_pipeline", {"project_id": inp.project_id})
    except Exception as exc:
        mark_render_submit_failed(
            inp.project_id, f"提交渲染任务失败：{exc}", claim_token)
        raise
    if not record_render_task(inp.project_id, task_id, claim_token):
        manager.cancel(task_id)
        mark_render_submit_failed(
            inp.project_id, "渲染领取已失效，任务已取消", claim_token)
        raise ValueError("OpenFilm 渲染领取已失效，任务已取消")
    latest = get_project_store().get_project(inp.project_id) or project
    message = outcome.get("message", "pre_render 已放行，开始渲染")
    return SkillOutput(
        data=ReviewReport(
            project_id=inp.project_id, action="approve",
            state=latest.state, task_id=task_id,
            message=message),
        reasoning=[ReasoningChain(
            conclusion=message,
            evidence=[Evidence(metric="feedback_len",
                               value=str(len(feedback)), comparison=">=",
                               window="本次动作")],
            causal_analysis=(
                "edit_review approve → pre_render 记入 approved → 提交 "
                "edit_pipeline → OpenFilm 渲染任务执行"),
            risk_note="渲染进度/取消复用任务引擎；成片固定落 output.mp4",
            confidence=1.0)],
        confidence=1.0, sample_size=1,
    )
