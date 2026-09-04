"""localize_cancel 技能：取消一个本地化任务（本地任务引擎）。

薄包装 TaskManager.cancel(task_id) 的协作式取消语义：
- queued：worker 未启动，直接落 cancelled 终态
- running：置取消 flag，流水线在下一阶段边界检查后停止（协作式取消）
- 终态任务：幂等返回（cancelled=False + 说明文案，不是错误）
- 任务不存在：degraded + video（资源缺失）

TaskManager.cancel 对「查询后竞态进入终态」也幂等处理（返回 False），
本技能按同样的幂等语义透出，不报错。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.errors import is_retriable
from flowmind.skill import skill
from flowmind.tasks import STATUS_QUEUED, TERMINAL_STATUSES
from flowmind.tasks.manager import get_task_manager

_VERSION = "0.2.0"


# ── 入参 ──

class CancelInput(BaseModel):
    """cancel 技能入参。"""
    task_id: str = Field(..., min_length=1, description="要取消的 task_id")


# ── 出参 ──

class CancelReport(BaseModel):
    """cancel 技能业务载荷。"""
    task_id: str
    cancelled: bool              # True=取消信号已受理/已落终态；False=无需取消
    previous_status: str | None  # 取消时的任务状态
    message: str
    failure_category: str | None = None  # 仅任务不存在时为 "video"
    retriable: bool = False
    warning: str | None = None


# ── 入口 ──

@skill(id="localize_cancel", name="取消本地化任务", version=_VERSION)
def localize_cancel(inp: CancelInput) -> SkillOutput[CancelReport]:
    """取消任务：queued 直接落终态；running 阶段边界生效；终态幂等返回。

    数据流：task_id → TaskStore 读状态 → TaskManager.cancel（协作式）
    → CancelReport + ReasoningChain → 框架套 SkillResult 信封。
    """
    manager = get_task_manager()
    rec = manager.get_task(inp.task_id)

    if rec is None:
        return _not_found_output(inp.task_id)

    status = rec.get("status")
    if status in TERMINAL_STATUSES:
        # 幂等：终态任务无需取消，正常返回（不是错误）
        report = CancelReport(
            task_id=inp.task_id,
            cancelled=False,
            previous_status=status,
            message=f"任务已是终态（{status}），无需取消（幂等返回）",
        )
        chain = ReasoningChain(
            conclusion=f"任务 {inp.task_id} 已是终态（{status}），取消请求幂等忽略",
            triggered_rules=[], evidence=[],
            causal_analysis=f"TaskStore 读到终态 status={status}，未发取消信号",
            risk_note="终态任务不可取消；failed/interrupted 任务可 localize_retry 重提。",
        )
        return SkillOutput(
            data=report, reasoning=[chain], confidence=1.0, sample_size=1,
        )

    accepted = manager.cancel(inp.task_id)
    if not accepted:
        # 查询后竞态进入终态：与终态分支同样的幂等语义
        report = CancelReport(
            task_id=inp.task_id,
            cancelled=False,
            previous_status=status,
            message="任务在取消前已进入终态（幂等返回）",
        )
        chain = ReasoningChain(
            conclusion=f"任务 {inp.task_id} 在取消前已进入终态，幂等忽略",
            triggered_rules=[], evidence=[],
            causal_analysis="cancel 返回 False（任务不存在或已是终态）",
            risk_note="如需结果可再查 localize_status。",
        )
        return SkillOutput(
            data=report, reasoning=[chain], confidence=1.0, sample_size=1,
        )

    if status == STATUS_QUEUED:
        message = "排队中任务已直接取消（终态 cancelled）"
        risk_note = "worker 尚未启动，任务已落 cancelled 终态。"
    else:
        message = "取消信号已发出：running 任务在下一阶段边界生效（协作式取消）"
        risk_note = "流水线在阶段边界检查取消 flag；已进入的 GPU 阶段会跑完当前阶段。"

    report = CancelReport(
        task_id=inp.task_id,
        cancelled=True,
        previous_status=status,
        message=message,
    )
    chain = ReasoningChain(
        conclusion=f"已请求取消任务 {inp.task_id}（原状态 {status}）",
        triggered_rules=[], evidence=[],
        causal_analysis=message,
        risk_note=risk_note,
    )
    return SkillOutput(
        data=report, reasoning=[chain], confidence=1.0, sample_size=1,
    )


def _not_found_output(task_id: str) -> SkillOutput[CancelReport]:
    """任务不存在的统一返回：degraded + video（资源缺失，与原 404 语义对齐）。"""
    report = CancelReport(
        task_id=task_id,
        cancelled=False,
        previous_status=None,
        message="任务不存在，无法取消",
        failure_category="video",
        retriable=is_retriable("video"),
        warning="任务不存在（可能从未创建或已被 TTL 回收）",
    )
    chain = ReasoningChain(
        conclusion=f"取消任务 {task_id} 失败（video）",
        triggered_rules=[], evidence=[],
        causal_analysis="TaskStore 无该任务行",
        risk_note="确认 task_id 是否正确；任务行不会因终态被删（GC 只清工作目录）。",
    )
    return SkillOutput(
        data=report, reasoning=[chain], confidence=0.0, sample_size=1,
        degraded=True, degradation_reason="video",
    )
