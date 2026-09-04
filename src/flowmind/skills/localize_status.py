"""localize_status 技能：批量查询本地化任务状态（本地任务引擎）。

读 TaskStore（经 TaskManager）而非外部 VL 服务：每个 task_id 取 PG 里的
任务行（status/stage/progress/args/output_paths/时间戳），判定终态 / 卡住 /
不存在；输出汇总 + 四段式推理链。

- 不存在的 task_id → 该任务标 not_found（partial success，不影响其他任务）
- store 读失败（PG 不可达）→ 异常上抛，invoke() 兜底为 ok=False INTERNAL
  （错误永不静默）
- 卡住判定：running 且 started_at 距今超过 stall_threshold_seconds
  （阈值走 config；协作取消/阶段边界语义见 tasks/manager.py）
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import Evidence, ReasoningChain, SkillOutput
from flowmind.rules import Rule, evaluate_rules
from flowmind.skill import skill
from flowmind.tasks import TERMINAL_STATUSES
from flowmind.tasks.manager import get_task_manager

_VERSION = "0.2.0"

# not_found 是查询侧合成状态（store 无此行），视为终态便于 Agent 收敛
_TERMINAL = frozenset({*TERMINAL_STATUSES, "not_found"})


# ── 入参 ──

class StatusInput(BaseModel):
    """状态查询入参。"""
    task_ids: list[str] = Field(..., min_length=1, description="要查询的 task_id 列表")
    stall_threshold_seconds: int | None = Field(
        default=None, description="运行中任务的卡住阈值（秒）；None = 用 config 默认"
    )


# ── 出参 ──

class TaskStatusReport(BaseModel):
    """单个任务状态报告。"""
    task_id: str
    status: str            # queued/running/succeeded/failed/cancelled/interrupted/not_found
    stage: str | None      # 流水线阶段（extract_audio/asr/.../vectorize）
    progress: float        # 0-100
    source_video: str | None
    target_lang: str | None
    output_paths: list[str]
    error: str | None
    created_at: str | None
    started_at: str | None
    finished_at: str | None
    duration_seconds: float | None  # started→finished 或 started→now（未完）
    is_terminal: bool               # succeeded/failed/cancelled/interrupted/not_found
    is_stalled: bool                # running 且持续 > threshold


class StatusReport(BaseModel):
    """批量状态汇总。"""
    tasks: list[TaskStatusReport]
    succeeded: int
    failed: int
    cancelled: int
    interrupted: int
    running: int
    queued: int
    stalled: int
    not_found: int
    all_terminal: bool
    failure_category: str | None = None  # store 不可达等错误时填充（预留）
    retriable: bool = False
    warning: str | None = None


# ── 规则 ──

def _rules(cfg) -> list[Rule]:
    return [
        Rule(
            id="STAL-01",
            name="运行卡住",
            expression=f"running 持续 > {cfg.stall_threshold_seconds}s",
            predicate=lambda m: m["stalled"] > 0,
            evidence=lambda m: [Evidence(
                metric="卡住任务数",
                value=m["stalled"],
                threshold=0,
                comparison=">",
            )],
        ),
        Rule(
            id="STAL-02",
            name="存在失败",
            expression="failed > 0",
            predicate=lambda m: m["failed"] > 0,
            evidence=lambda m: [Evidence(
                metric="失败任务数",
                value=m["failed"],
                threshold=0,
                comparison=">",
            )],
        ),
        Rule(
            id="STAL-03",
            name="全部终态",
            expression="all_terminal=True",
            predicate=lambda m: m["all_terminal"],
            evidence=lambda m: [Evidence(
                metric="未完成任务数",
                value=m["pending"],
                threshold=0,
                comparison="==",
            )],
        ),
    ]


# ── 工具 ──

def _parse_iso(s: str | None) -> datetime | None:
    """解析 ISO8601 字符串；naive datetime 一律当 UTC 处理。"""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _duration_seconds(started: str | None, finished: str | None) -> float | None:
    s = _parse_iso(started)
    if s is None:
        return None
    e = _parse_iso(finished) or datetime.now(timezone.utc)
    return max(0.0, (e - s).total_seconds())


# ── 单任务读取 ──

def _fetch_one(manager, task_id: str) -> TaskStatusReport:
    """读单个任务行。

    - 任务不存在 → 标 not_found（partial success，不影响其他任务）
    - store 异常不在此吞——上抛由 invoke() 兜底为结构化错误
    """
    rec = manager.get_task(task_id)
    if rec is None:
        return TaskStatusReport(
            task_id=task_id, status="not_found",
            stage=None, progress=0.0,
            source_video=None, target_lang=None, output_paths=[],
            error="任务不存在", created_at=None, started_at=None,
            finished_at=None, duration_seconds=None,
            is_terminal=True, is_stalled=False,
        )
    args = rec.get("args") or {}
    started = rec.get("started_at")
    finished = rec.get("finished_at")
    return TaskStatusReport(
        task_id=rec.get("task_id") or task_id,
        status=rec.get("status", "unknown"),
        stage=rec.get("stage"),
        progress=float(rec.get("progress") or 0.0),
        source_video=args.get("video_path"),
        target_lang=args.get("target_lang"),
        output_paths=list(rec.get("output_paths") or []),
        error=rec.get("error"),
        created_at=rec.get("created_at"),
        started_at=started,
        finished_at=finished,
        duration_seconds=_duration_seconds(started, finished),
        is_terminal=rec.get("status") in TERMINAL_STATUSES,
        is_stalled=False,  # 由 _aggregate 设
    )


# ── 汇总 / 推理链 ──

def _aggregate(
    tasks: list[TaskStatusReport], cfg, stall_threshold: int
) -> tuple[StatusReport, list, list[Evidence]]:
    """统计 + 标记 stalled；返回 (汇总报告, 规则命中, 证据)."""
    succeeded = failed = cancelled = interrupted = 0
    running = queued = stalled = not_found = 0
    now = datetime.now(timezone.utc)

    for t in tasks:
        if t.status == "succeeded":
            succeeded += 1
        elif t.status == "failed":
            failed += 1
        elif t.status == "cancelled":
            cancelled += 1
        elif t.status == "interrupted":
            interrupted += 1
        elif t.status == "not_found":
            not_found += 1
        elif t.status == "queued":
            queued += 1
        elif t.status == "running":
            running += 1
            started = _parse_iso(t.started_at)
            if started is not None:
                elapsed = (now - started).total_seconds()
                if elapsed > stall_threshold:
                    t.is_stalled = True
                    stalled += 1

    pending = running + queued
    all_terminal = pending == 0
    metrics = {
        "stalled": stalled,
        "failed": failed,
        "interrupted": interrupted,
        "all_terminal": all_terminal,
        "pending": pending,
    }
    hits, evidence = evaluate_rules(_rules(cfg), metrics)

    report = StatusReport(
        tasks=tasks,
        succeeded=succeeded,
        failed=failed,
        cancelled=cancelled,
        interrupted=interrupted,
        running=running,
        queued=queued,
        stalled=stalled,
        not_found=not_found,
        all_terminal=all_terminal,
    )
    return report, hits, evidence


def _build_chain(
    report: StatusReport, hits: list, evidence: list[Evidence], cfg
) -> ReasoningChain:
    rule_names = "、".join(h.name for h in hits) if hits else "（无）"
    conclusion = (
        f"查询 {len(report.tasks)} 个任务：成功 {report.succeeded}、"
        f"失败 {report.failed}、取消 {report.cancelled}、中断 {report.interrupted}、"
        f"运行中 {report.running}、排队 {report.queued}、"
        f"卡住 {report.stalled}、未找到 {report.not_found}。"
    )
    if report.all_terminal:
        risk_note = (
            f"全部进入终态。命中规则：{rule_names}。"
            if hits else "全部进入终态，无异常。"
        )
    elif report.stalled > 0:
        risk_note = (
            f"有 {report.stalled} 个任务卡住超过 {cfg.stall_threshold_seconds}s，"
            f"建议查 worker 日志，或 cancel 后 localize_retry 重提。"
        )
    elif report.failed > 0:
        risk_note = f"有 {report.failed} 个任务失败，建议查看 error 字段后 localize_retry 重提。"
    else:
        risk_note = "任务正常推进中，继续轮询。"
    causal_analysis = (
        f"基于各任务 status + started_at 与当前时间差，"
        f"按 stall_threshold_seconds={cfg.stall_threshold_seconds} 阈值求值。"
    )
    return ReasoningChain(
        conclusion=conclusion,
        triggered_rules=hits,
        evidence=evidence,
        causal_analysis=causal_analysis,
        risk_note=risk_note,
    )


# ── 入口 ──

@skill(id="localize_status", name="本地化任务状态查询", version=_VERSION)
def localize_status(inp: StatusInput) -> SkillOutput[StatusReport]:
    """批量查询任务状态，返回每任务详情 + 汇总 + 四段式推理链。

    task_ids 数 > 1 时用 ThreadPoolExecutor 并发读（max_workers =
    min(N, poll_max_concurrency)），单 task 串行避免线程开销。
    汇总与推理链仍单线程（顺序无关）。
    """
    cfg = load_config().localizer
    stall_threshold = inp.stall_threshold_seconds or cfg.stall_threshold_seconds

    manager = get_task_manager()
    task_ids = inp.task_ids
    if len(task_ids) <= 1:
        task_reports = [_fetch_one(manager, tid) for tid in task_ids]
    else:
        max_workers = max(1, min(len(task_ids), cfg.poll_max_concurrency))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            task_reports = list(pool.map(lambda tid: _fetch_one(manager, tid), task_ids))

    report, hits, evidence = _aggregate(task_reports, cfg, stall_threshold)
    chain = _build_chain(report, hits, evidence, cfg)

    return SkillOutput(
        data=report,
        reasoning=[chain],
        confidence=1.0,
        sample_size=len(task_ids),
    )
