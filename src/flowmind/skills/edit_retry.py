"""edit_retry 技能：失败/取消的剪辑项目断点续跑（恢复回路）。

run_pipeline 本身支持断点（done 阶段跳过、产物落盘复用），但此前没有
任何入口能把 failed/cancelled 项目重新驱动起来——edit_pipeline 的失败
信封长期"承诺可重跑"而无实现（v1.2 补齐）。语义对齐 localize_retry：

- failed / cancelled（项目终态）→ 定位第一个未 done 阶段置 pending、
  清 error、提交新 edit_pipeline 任务续跑（done 阶段与中间产物全复用）；
- succeeded → 拒绝（degraded）：项目已成功无需重跑；
- running / awaiting_review（非终态）→ 拒绝：运行中项目不允许重跑
  （awaiting_review 应走 edit_review 审批动作）；
- 项目不存在 → 拒绝（跨租户与不存在同形，防枚举）。

review_rejected[stage] 的项目（edit_review reject 落 failed）同样可重跑：
被拒闸的阶段产物仍是 done，续跑会推进到下一道未放行闸再次等待审批。

队列背压：提交遇 TaskQueueFull 回滚项目状态后向上抛 → invoke() 兜底
ok=False（429 语义，与 edit_review/localize_submit 一致）。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind import auth
from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.editing.models import (
    PSTATE_CANCELLED,
    PSTATE_FAILED,
    PSTATE_RUNNING,
    PSTATE_SUCCEEDED,
    STAGE_DONE,
    STAGE_ORDER,
    STAGE_PENDING,
)
from flowmind.editing.store import get_project_store
from flowmind.skill import skill
from flowmind.tasks import TaskQueueFull
from flowmind.tasks.manager import get_task_manager

_VERSION = "0.1.0"

# 允许续跑的项目终态（succeeded 显式排除；running/awaiting_review 非终态）
_RETRYABLE_STATES = frozenset({PSTATE_FAILED, PSTATE_CANCELLED})


class RetryInput(BaseModel):
    """edit_retry 入参。"""
    project_id: str = Field(min_length=8, description="要续跑的剪辑项目 id")


class RetryReport(BaseModel):
    """edit_retry 业务载荷。"""
    project_id: str
    state: str                       # 续跑提交后为 running
    resume_stage: str                # 断点续跑的起始阶段
    task_id: str | None = None       # 新驱动的 edit_pipeline 任务
    message: str = ""                # 拒绝/成功说明


@skill(id="edit_retry", name="剪辑项目重跑（失败/取消恢复）", version=_VERSION)
def edit_retry(inp: RetryInput) -> SkillOutput[RetryReport]:
    """把 failed/cancelled 项目从第一个未完成阶段断点续跑。

    数据流：project_id → ProjectStore 读项目（终态校验 + 租户同形）→
    断点阶段置 pending → 提交 edit_pipeline 任务 → RetryReport +
    ReasoningChain → SkillResult 信封。
    """
    store = get_project_store()
    project = store.get_project(inp.project_id)
    required, tenant = auth.tenant_scope()
    if required and (project is None or tenant is None
                     or project.tenant_id != tenant):
        project = None   # 跨租户与不存在同形，防项目 id 枚举
    if project is None:
        return _reject(inp.project_id, "", "项目不存在，无法重跑")
    if project.state == PSTATE_SUCCEEDED:
        return _reject(inp.project_id, project.state,
                       "项目已成功，无需重跑")
    if project.state not in _RETRYABLE_STATES:
        return _reject(
            inp.project_id, project.state,
            f"项目非终态（{project.state}），不允许重跑；"
            f"运行中任务请先取消，审批等待请走 edit_review")
    # 断点续跑语义建立在 flowmind 内置流水线上（stages 字典 + 产物复用）；
    # CyberCut 后端的节点重试由其执行器内部处理（node_attempts），不归本技能
    from flowmind.config import load_config
    backend = load_config().editing.backend
    if backend != "flowmind":
        return _reject(
            inp.project_id, project.state,
            f"editing.backend={backend} 暂不支持 edit_retry"
            f"（CyberCut 的重试由其执行器内部 node_attempts 处理）")

    # 断点：第一个未 done 的阶段（全部 done 视为收尾阶段，续跑直接收敛）
    resume = next((s for s in STAGE_ORDER
                   if project.stages.get(s) != STAGE_DONE), STAGE_ORDER[-1])
    prev_state, prev_error = project.state, project.error
    project.stages[resume] = STAGE_PENDING
    project.error = None
    project.state = PSTATE_RUNNING
    project.stage = resume
    store.save_project(project)

    manager = get_task_manager()
    try:
        task_id = manager.submit("edit_pipeline", {"project_id": inp.project_id})
    except TaskQueueFull:
        project.state, project.error = prev_state, prev_error   # 回滚可重试
        project.stage = resume
        store.save_project(project)
        raise
    # 回存前重载：内联 worker 可能已推进项目，不能用本副本覆盖
    fresh = store.get_project(inp.project_id) or project
    fresh.task_ids.append(task_id)
    store.save_project(fresh)

    message = (f"项目已从阶段 {resume} 断点续跑"
               f"（此前状态：{prev_state}{'：' + prev_error[:120] if prev_error else ''}）")
    return SkillOutput(
        data=RetryReport(
            project_id=inp.project_id, state=PSTATE_RUNNING,
            resume_stage=resume, task_id=task_id, message=message),
        reasoning=[ReasoningChain(
            conclusion=message,
            causal_analysis=(
                f"failed/cancelled 项目 → 断点定位 {resume}（done 阶段与"
                f"中间产物全复用）→ 提交 edit_pipeline 任务 {task_id}"),
            risk_note="原失败原因若仍存在（配额/素材损坏）会再次失败，"
                      "先看项目 error 字段定位根因",
            confidence=1.0)],
        confidence=1.0, sample_size=1,
    )


def _reject(project_id: str, state: str, message: str) -> SkillOutput[RetryReport]:
    """统一拒绝返回：degraded SkillOutput，原因在 message 字段里。"""
    return SkillOutput(
        data=RetryReport(
            project_id=project_id, state=state or "not_found",
            resume_stage="", message=message),
        reasoning=[ReasoningChain(
            conclusion=f"重跑项目 {project_id} 被拒绝",
            causal_analysis=message,
            risk_note="仅 failed/cancelled 项目可断点续跑；"
                      "awaiting_review 应走 edit_review 审批动作",
            confidence=1.0)],
        confidence=0.0, sample_size=1,
        degraded=True, degradation_reason="retry_rejected",
    )
