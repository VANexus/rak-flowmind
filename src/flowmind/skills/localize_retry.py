"""localize_retry 技能：重提一个终态的本地化任务（本地任务引擎）。

内部两步：TaskStore 读原任务行（args_json 即原始入参）→ TaskManager.submit
复制原参数重新创建任务，返回 original_task_id + new_task_id。对 Agent 来说
一次调用就行，不用自己取 args 再调 localize_submit。

重提准入语义：
- failed / cancelled / interrupted（终态）→ 复制原 args 重新 submit
- succeeded → 拒绝（degraded）：任务已成功无需重提
- queued / running（非终态）→ 拒绝（degraded）：排队/运行中任务不允许重提，
  先 localize_cancel 再重提
- 任务不存在 → degraded + video（与原 VL 404 语义对齐）

队列背压：重提遇 TaskQueueFull 向上抛 → invoke() 兜底 ok=False（429 语义，
决策记录同 localize_submit）。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.errors import is_retriable
from flowmind.skill import skill
from flowmind.tasks.manager import get_task_manager

_VERSION = "0.2.0"

# 允许重提的终态（succeeded 显式排除：成功任务无重提价值）
_RETRYABLE_STATUSES = frozenset({"failed", "cancelled", "interrupted"})


# ── 入参 ──

class RetryInput(BaseModel):
    """retry 技能入参。"""
    task_id: str = Field(..., min_length=1, description="要重提的原 task_id")


# ── 出参 ──

class RetryReport(BaseModel):
    """retry 技能业务载荷。"""
    original_task_id: str
    new_task_id: str          # 重提创建的新任务 id（拒绝时为空串）
    original_status: str | None
    skill_id: str             # 原任务的技能（重提沿用）
    source_video: str
    target_lang: str | None
    failure_category: str | None = None
    retriable: bool = False
    message: str | None = None    # 拒绝/失败时的人类可读原因


# ── 入口 ──

@skill(id="localize_retry", name="重提本地化任务", version=_VERSION)
def localize_retry(inp: RetryInput) -> SkillOutput[RetryReport]:
    """复制原任务 args_json 重新 submit，返回新 task_id。

    数据流：task_id → TaskStore 读原行（终态校验）→ TaskManager.submit
    （同 skill_id + 原 args）→ RetryReport + ReasoningChain → SkillResult 信封。
    """
    manager = get_task_manager()
    rec = manager.get_task(inp.task_id)

    if rec is None:
        return _reject(inp.task_id, None, "", "任务不存在，无法重提", "video")

    status = rec.get("status")
    if status == "succeeded":
        return _reject(inp.task_id, status, "", "任务已成功，无需重提", "video")
    if status not in _RETRYABLE_STATUSES:
        return _reject(
            inp.task_id, status, "",
            f"任务未到终态（{status}），排队/运行中任务拒绝重提；"
            f"如需终止可先 localize_cancel",
            "video",
        )

    original_args = dict(rec.get("args") or {})
    source_video = str(original_args.get("video_path") or "")
    if not source_video:
        return _reject(
            inp.task_id, status, "",
            "原任务 args 缺 video_path，无法重提（存储行异常）", "video",
        )

    skill_id = str(rec.get("skill_id") or "localize_video")
    new_task_id = manager.submit(skill_id, original_args)

    report = RetryReport(
        original_task_id=inp.task_id,
        new_task_id=new_task_id,
        original_status=status,
        skill_id=skill_id,
        source_video=source_video,
        target_lang=original_args.get("target_lang"),
    )
    chain = ReasoningChain(
        conclusion=(
            f"已重提任务 {inp.task_id} → 新任务 {new_task_id}"
            f"（status={status} → queued）"
        ),
        triggered_rules=[],
        evidence=[],
        causal_analysis=(
            f"复制原 args 重新 submit：skill_id={skill_id} / "
            f"video_path={source_video} / target_lang={original_args.get('target_lang')}"
        ),
        risk_note=(
            "新任务独立调度；原任务失败原因若仍存在会再次失败，"
            "先看原任务 error 字段定位。"
        ),
    )
    return SkillOutput(
        data=report, reasoning=[chain], confidence=1.0, sample_size=1,
    )


def _reject(task_id: str, status: str | None, new_task_id: str,
            message: str, category: str) -> SkillOutput[RetryReport]:
    """统一的拒绝返回：degraded SkillOutput，原因在 message 字段里。"""
    report = RetryReport(
        original_task_id=task_id,
        new_task_id=new_task_id,
        original_status=status,
        skill_id="",
        source_video="",
        target_lang=None,
        failure_category=category,
        retriable=is_retriable(category),
        message=message,
    )
    chain = ReasoningChain(
        conclusion=f"重提任务 {task_id} 被拒绝（{category}）",
        triggered_rules=[],
        evidence=[],
        causal_analysis=message,
        risk_note=(
            f"{'可重试' if is_retriable(category) else '需确认任务状态'}；"
            f"queued/running 任务须先 localize_cancel 才能重提。"
        ),
    )
    return SkillOutput(
        data=report, reasoning=[chain], confidence=0.0, sample_size=1,
        degraded=True, degradation_reason=f"retry_rejected_{category}",
    )
