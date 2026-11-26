"""edit_openfilm_render：按 project_id 受理 FlowMind 管理的 OpenFilm 渲染。

调用方只传 project_id。流程：

1. 校验项目由 OpenFilm 后端创建、工作区含 film.html；
2. 若配置了 ``pre_render`` 审批闸且尚未放行 → 项目进入 ``awaiting_review``，
   由既有 ``edit_review`` 做 approve / revise / reject（复用审批语义）；
3. 无闸（或已放行）→ 依赖预检（Node/OpenFilm/FFmpeg）→ 提交
   ``edit_pipeline`` 任务，由任务引擎执行 OpenFilm CLI 渲染、上报进度、
   支持取消，并把失败/成片写回项目状态。

调用方永远不能用本技能传命令、参数或输出路径。
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from flowmind.contracts import Evidence, ReasoningChain, SkillOutput
from flowmind.editing.openfilm_backend import (
    OpenFilmError,
    current_state,
    mark_render_submit_failed,
    prepare_render,
    record_render_task,
)
from flowmind.editing.store import get_project_store
from flowmind.skill import skill
from flowmind.tasks import TaskQueueFull
from flowmind.tasks.manager import get_task_manager

_VERSION = "0.1.0"
RENDER_TASK_SKILL = "edit_pipeline"


class RenderInput(BaseModel):
    """渲染入参：只有 FlowMind 项目 id，不接受命令/路径/输出位置。"""

    project_id: str = Field(
        min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$",
        description="FlowMind 剪辑项目 id（edit_project 返回）")


class RenderReport(BaseModel):
    project_id: str
    state: str = ""
    stage: str = ""
    task_id: str | None = None
    render_status: str = ""
    output_path: str | None = None
    message: str = ""


@skill(id="edit_openfilm_render", name="OpenFilm 项目渲染",
       version=_VERSION, category="智能剪辑",
       tags=["edit", "openfilm", "render"])
def edit_openfilm_render(
        inp: RenderInput) -> SkillOutput[RenderReport]:
    """受理 OpenFilm 渲染：审批闸优先；否则提交任务并返回 task_id。"""
    project_id = inp.project_id
    try:
        prepared = prepare_render(project_id)
    except OpenFilmError as exc:
        return SkillOutput(
            data=RenderReport(
                project_id=project_id,
                state=current_state(project_id) or "failed",
                stage="render", task_id=None,
                render_status="blocked", message=str(exc)[:1000]),
            reasoning=[ReasoningChain(
                conclusion="OpenFilm 渲染未受理",
                causal_analysis=str(exc)[:500],
                risk_note=(
                    "先修复依赖（Node/OpenFilm/FFmpeg）或项目工作区后重试；"
                    "项目状态已记录失败原因，见 edit_status"
                    if exc.category == "environment" else
                    "确认项目由 OpenFilm 后端创建、film.html 存在，"
                    "且没有正在进行的渲染"),
                confidence=1.0)],
            confidence=1.0, degraded=True,
            degradation_reason=exc.category)

    if not prepared.get("submit"):
        return SkillOutput(
            data=RenderReport(
                project_id=project_id,
                state=prepared.get("state", "awaiting_review"),
                stage=prepared.get("stage", "pre_render"),
                task_id=None,
                render_status=prepared.get("render_status", "awaiting_review"),
                message=str(prepared.get("message", ""))),
            reasoning=[ReasoningChain(
                conclusion="OpenFilm 项目进入 pre_render 审批等待",
                evidence=[Evidence(metric="review_gates",
                                   value="pre_render", comparison="==")],
                causal_analysis=(
                    "外部 Agent 完成工作区 → edit_openfilm_render → 审批闸 "
                    "未放行 → 项目落 awaiting_review（不占任务槽位）"),
                risk_note=("经 edit_review approve 开始渲染；revise 带反馈"
                           "退回外部 Agent 修改；reject 终止项目"),
                confidence=1.0)],
            confidence=1.0, sample_size=1)

    manager = get_task_manager()
    claim_token = prepared.get("claim_token")
    try:
        task_id = manager.submit(RENDER_TASK_SKILL, {"project_id": project_id})
    except TaskQueueFull:
        mark_render_submit_failed(project_id, "队列满未受理", claim_token)
        raise   # invoke 兜底 429 语义
    except Exception as exc:
        mark_render_submit_failed(
            project_id, f"提交渲染任务失败：{exc}", claim_token)
        raise

    if not record_render_task(project_id, task_id, claim_token):
        manager.cancel(task_id)
        mark_render_submit_failed(
            project_id, "渲染领取已失效（可能被并发请求取代）", claim_token)
        return SkillOutput(
            data=RenderReport(
                project_id=project_id,
                state=current_state(project_id) or "failed",
                stage="render", task_id=task_id,
                render_status="failed",
                message="渲染领取已失效，任务已取消；请重试"),
            reasoning=[ReasoningChain(
                conclusion="渲染领取已失效，未重复渲染",
                causal_analysis=(
                    "并发请求/超时导致 claim token 不匹配；已取消刚提交的"
                    "任务，保证同一项目只有一个活跃渲染"),
                risk_note="重新调用 edit_openfilm_render 会重新领取",
                confidence=1.0)],
            confidence=1.0, degraded=True,
            degradation_reason="transient")

    # 从最新项目记录读取真实状态：inline/极快任务可能已在 submit 返回前
    # 完成，此时不能再硬编码 queued / output_path=null。
    project = get_project_store().get_project(project_id)
    render = {}
    if project is not None:
        meta = project.input.get("openfilm")
        if isinstance(meta, dict) and isinstance(meta.get("render"), dict):
            render = meta["render"]
    final_state = (project.state if project is not None
                   else (current_state(project_id) or "running"))
    render_status = str(render.get("status") or "queued")
    output_path: str | None = None
    if final_state == "succeeded":
        candidate = render.get("output")
        if isinstance(candidate, str) and Path(candidate).is_file():
            output_path = candidate
    message = {
        "succeeded": "渲染已完成，成片见 output_path",
        "failed": f"渲染失败：{render.get('error') or '见 edit_status'}",
        "cancelled": "渲染已取消",
    }.get(final_state, "渲染任务已受理；轮询 edit_status 查看进度与 output.mp4")
    return SkillOutput(
        data=RenderReport(
            project_id=project_id, state=final_state, stage="render",
            task_id=task_id, render_status=render_status,
            output_path=output_path, message=message),
        reasoning=[ReasoningChain(
            conclusion=f"OpenFilm 渲染任务已受理：{task_id}",
            evidence=[Evidence(metric="render_fps", value="配置值",
                               comparison="==", window="本次任务")],
            causal_analysis=(
                "edit_openfilm_render(project_id) → OpenFilm 工作区校验 + "
                "依赖预检 → edit_pipeline 任务提交 → 任务引擎执行 openfilm "
                "render（进度/取消/终态复用任务引擎）"),
            risk_note=("渲染输出固定为 FlowMind 项目目录下的 output.mp4；"
                       "失败/取消不会留下半截成片"),
            confidence=1.0)],
        confidence=1.0, sample_size=1,
    )
