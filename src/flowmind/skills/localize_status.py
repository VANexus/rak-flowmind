"""localize_status 技能：批量查询任务状态。

对一组 task_id 并发查 video-localizer GET /api/v1/tasks/{id}，
判定每个任务：终态 / 卡住 / 健康；输出汇总 + 四段式推理链。
HTTP 层统一走 VLClient（vl_client.py），本文件不直接发请求。

阈值（stall_threshold_seconds / poll_max_concurrency）走 config，
个性化由终端用户对话初始化覆盖。
"""
from __future__ import annotations

import requests  # noqa: F401  保留模块级引用：测试 fixture 经 <mod>.requests 打桩拦截 VLClient
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from flowmind.config import LocalizerConfig, load_config
from flowmind.contracts import Evidence, ReasoningChain, SkillOutput
from flowmind.rules import Rule, evaluate_rules
from flowmind.skill import skill
from flowmind.vl_client import VLAPIError, VLClient

_VERSION = "0.1.0"

_TERMINAL = {"completed", "failed", "cancelled", "not_found"}


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
    status: str            # queued/running/retrying/completed/failed/cancelled/not_found
    source_video: str | None
    target_language: str | None
    output_dir: str | None
    outputs: dict[str, str]
    error: str | None
    created_at: str | None
    started_at: str | None
    finished_at: str | None
    duration_seconds: float | None  # started→finished 或 started→now（未完）
    is_terminal: bool               # completed/failed/cancelled/not_found
    is_stalled: bool                # running/retrying 且持续 > threshold


class StatusReport(BaseModel):
    """批量状态汇总。"""
    tasks: list[TaskStatusReport]
    completed: int
    failed: int
    cancelled: int
    running: int
    queued: int
    stalled: int
    all_terminal: bool
    failure_category: str | None = None  # 网络/服务错时填充
    retriable: bool = False
    warning: str | None = None


# ── 规则 ──

def _rules(cfg: LocalizerConfig) -> list[Rule]:
    return [
        Rule(
            id="STAL-01",
            name="运行卡住",
            expression=f"running/retrying 持续 > {cfg.stall_threshold_seconds}s",
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
            name="存在重试中",
            expression="retrying > 0",
            predicate=lambda m: m["running_retrying"] > 0,
            evidence=lambda m: [Evidence(
                metric="重试中任务数",
                value=m["running_retrying"],
                threshold=0,
                comparison=">",
            )],
        ),
        Rule(
            id="STAL-04",
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


# ── HTTP 调用 ──

def _fetch_one(cfg: LocalizerConfig, task_id: str) -> TaskStatusReport:
    """单次 GET /api/v1/tasks/{task_id}（走 VLClient）。

    - 404 → 该 task 标 not_found（部分成功，不影响其他 task）
    - 其他 HTTP / 网络错 → 该 task 标 unknown + error 文本（partial success，不冒泡）
    """
    client = VLClient(cfg)
    try:
        body = client.get(f"/tasks/{task_id}")
    except VLAPIError as exc:
        if exc.code == "NOT_FOUND":
            return TaskStatusReport(
                task_id=task_id, status="not_found",
                source_video=None, target_language=None, output_dir=None,
                outputs={}, error="Task not found",
                created_at=None, started_at=None, finished_at=None,
                duration_seconds=None, is_terminal=True, is_stalled=False,
            )
        return TaskStatusReport(
            task_id=task_id, status="unknown",
            source_video=None, target_language=None, output_dir=None,
            outputs={}, error=f"[{exc.category}] {type(exc.__cause__ or exc).__name__}",  # 不放完整消息（避免泄漏）
            created_at=None, started_at=None, finished_at=None,
            duration_seconds=None, is_terminal=False, is_stalled=False,
        )
    return _body_to_report(body)


def _body_to_report(body: dict) -> TaskStatusReport:
    status = body.get("status", "unknown")
    started = body.get("started_at")
    finished = body.get("finished_at")
    duration = _duration_seconds(started, finished)
    return TaskStatusReport(
        task_id=body.get("task_id") or body.get("job_id") or "",
        status=status,
        source_video=body.get("source_video"),
        target_language=body.get("target_language"),
        output_dir=body.get("output_dir"),
        outputs=body.get("outputs") or {},
        error=body.get("error"),
        created_at=body.get("created_at"),
        started_at=started,
        finished_at=finished,
        duration_seconds=duration,
        is_terminal=status in _TERMINAL,
        is_stalled=False,  # 由 _aggregate 设
    )


# ── 汇总 / 推理链 ──

def _aggregate(
    tasks: list[TaskStatusReport], cfg: LocalizerConfig, stall_threshold: int
) -> tuple[StatusReport, list, list[Evidence]]:
    """统计 + 标记 stalled；返回 (汇总报告, 规则命中, 证据)."""
    completed = failed = cancelled = running = queued = stalled = running_retrying = 0
    now = datetime.now(timezone.utc)

    for t in tasks:
        if t.status == "completed":
            completed += 1
        elif t.status == "failed":
            failed += 1
        elif t.status == "cancelled":
            cancelled += 1
        elif t.status == "queued":
            queued += 1
        elif t.status in ("running", "retrying"):
            running += 1
            if t.status == "retrying":
                running_retrying += 1
            # 卡住判定：只对 running（不是 retrying）。retrying 表示 VL 在自动重试，
            # 是已知失败模式，不算真卡住——别让 Agent 看到 N 小时 retrying 就告警。
            if t.status == "running":
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
        "running_retrying": running_retrying,
        "all_terminal": all_terminal,
        "pending": pending,
    }
    rules = _rules(cfg)
    hits, evidence = evaluate_rules(rules, metrics)

    report = StatusReport(
        tasks=tasks,
        completed=completed,
        failed=failed,
        cancelled=cancelled,
        running=running,
        queued=queued,
        stalled=stalled,
        all_terminal=all_terminal,
    )
    return report, hits, evidence


def _build_chain(
    report: StatusReport, hits: list, evidence: list[Evidence], cfg: LocalizerConfig
) -> ReasoningChain:
    rule_names = "、".join(h.name for h in hits) if hits else "（无）"
    conclusion = (
        f"查询 {len(report.tasks)} 个任务：完成 {report.completed}、"
        f"失败 {report.failed}、运行中 {report.running}、排队 {report.queued}、"
        f"卡住 {report.stalled}。"
    )
    if report.all_terminal:
        risk_note = (
            f"全部进入终态。命中规则：{rule_names}。"
            if hits else "全部进入终态，无异常。"
        )
    elif report.stalled > 0:
        risk_note = f"有 {report.stalled} 个任务卡住超过 {cfg.stall_threshold_seconds}s，建议查 worker 日志或重提。"
    elif report.failed > 0:
        risk_note = f"有 {report.failed} 个任务失败，建议查看 error 字段后决定是否重试。"
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

@skill(id="localize_status", name="视频本地化任务状态查询", version=_VERSION)
def localize_status(inp: StatusInput) -> SkillOutput[StatusReport]:
    """批量查询 video-localizer 任务状态，返回每任务详情 + 汇总 + 四段式推理链。

    task_ids 数 > 1 时用 ThreadPoolExecutor 并发查（max_workers = min(N, poll_max_concurrency)）。
    单 task 串行避免线程开销。汇总与推理链仍单线程（顺序无关）。
    """
    from concurrent.futures import ThreadPoolExecutor

    cfg = load_config().localizer
    stall_threshold = inp.stall_threshold_seconds or cfg.stall_threshold_seconds

    task_ids = inp.task_ids
    if len(task_ids) <= 1:
        task_reports = [_fetch_one(cfg, task_ids[0])] if task_ids else []
    else:
        max_workers = max(1, min(len(task_ids), cfg.poll_max_concurrency))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            task_reports = list(pool.map(lambda tid: _fetch_one(cfg, tid), task_ids))

    report, hits, evidence = _aggregate(task_reports, cfg, stall_threshold)
    chain = _build_chain(report, hits, evidence, cfg)

    return SkillOutput(
        data=report,
        reasoning=[chain],
        confidence=1.0,
        sample_size=len(task_ids),
    )