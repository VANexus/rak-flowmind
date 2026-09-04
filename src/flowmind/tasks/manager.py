"""TaskManager：PG 持久化异步任务执行器（server_api.JobManager 的 SaaS 升级版）。

与 JobManager（内存态）的差异：
- 存储落 PostgreSQL（服务重启不丢任务历史；启动时 recover_running()
  把遗留 queued/running 标为 interrupted，pending 水位不再虚增）。
- 进度可观测：技能经 TaskContext.progress_cb 上报 → PG 落库 +
  MQTT ``mcp-base-gpu/tasks/{id}/events`` 实时推送（终态 retain）。
- 协作式取消：cancel() 置 Event；queued 任务直接落 cancelled 终态，
  running 任务由流水线在阶段边界检查（tasks.CancelledError）。
  ffmpeg/demucs 子进程不强行 kill（各自有 timeout；阶段边界为主，
  不做过度工程）。

终态语义（JobManager "failed 仅 runner 崩溃"语义在此收紧为业务语义）：
- succeeded：ok=True 且非技能级失败（degraded 但无 failure_category，
  如无人声空结果，也算 succeeded）。
- failed：ok=False（VALIDATION/INTERNAL/NOT_FOUND），或 ok=True 但
  data.failure_category ∈ {video, environment, transient, unknown}
  （技能级失败，error 列记 warning 文本）。
- cancelled：CancelledError 路径 / 排队时取消。

TTL GC：后台 daemon 线程周期清扫——
- 终态任务 finished_at 超 task_ttl_seconds → 删 data_dir/tasks/<task_id>/
  工作目录（DB 行保留：审计与状态查询需要；**决策：不删行**）；
- data_dir/tasks/ 下无 DB 行且 mtime 超 7 天的孤儿目录（直连 invoke
  产生的随机 workdir）一并清理（远大于任何任务运行时长，安全）。

单例：get_task_manager() 惰性创建（服务进程生命周期一个实例；
创建即连接 PG 做 recover——PG 不可达时快速失败，属部署错误）。
"""
from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from flowmind.config import load_config
from flowmind.contracts import new_trace
from flowmind.skill import invoke
from flowmind.tasks import (
    NON_TERMINAL_STATUSES,
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    TERMINAL_STATUSES,
    TaskContext,
    TaskQueueFull,
    reset_task_context,
    set_task_context,
)
from flowmind.tasks.events import TaskEventPublisher
from flowmind.tasks.store import TaskStore

logger = logging.getLogger(__name__)

_ORPHAN_DIR_TTL_SECONDS = 7 * 86400  # 孤儿 workdir（无 DB 行）mtime 超过 7 天清理


class TaskManager:
    """异步任务执行器：PG store + MQTT events + 单 GPU worker + TTL GC。"""

    def __init__(
        self,
        *,
        store: TaskStore | None = None,
        events: TaskEventPublisher | None = None,
        max_pending: int | None = None,
        ttl_seconds: int | None = None,
        workers: int = 1,
        data_dir: str | None = None,
    ) -> None:
        cfg = load_config().localizer
        self._max_pending = max(1, max_pending if max_pending is not None
                                else cfg.max_pending_tasks)
        self._ttl = max(1, ttl_seconds if ttl_seconds is not None
                        else cfg.task_ttl_seconds)
        self._data_dir = Path(data_dir or cfg.data_dir).expanduser().resolve()
        self.store = store or TaskStore()
        self.events = events or TaskEventPublisher()
        self.workers = max(1, workers)  # GPU 单卡：绝不放宽（防 OOM）
        self._pool = ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix="flowmind-task")
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()       # 保护 _cancel_events
        self._submit_lock = threading.Lock()  # pending 检查 + 建行的原子性
        self._gc_stop = threading.Event()

        # 启动恢复：服务重启后遗留 queued/running → interrupted（PG 快速失败）
        self.store.recover_running()

        self._gc_thread = threading.Thread(
            target=self._gc_loop, name="flowmind-task-gc", daemon=True)
        self._gc_thread.start()

    # ── 提交 / 取消 ──

    def submit(self, skill_id: str, args: dict) -> str:
        """提交任务，返回 task_id；pending 超限抛 TaskQueueFull（调用方回 429）。"""
        task_id = uuid.uuid4().hex
        with self._submit_lock:
            pending = self.store.count_pending()
            if pending >= self._max_pending:
                raise TaskQueueFull(
                    f"待处理任务已达上限 {self._max_pending}（当前 {pending}），稍后重试")
            self.store.create_task(task_id, skill_id, args)
        with self._lock:
            self._cancel_events[task_id] = threading.Event()
        try:
            self._pool.submit(self._worker, task_id, skill_id, args)
        except RuntimeError as exc:
            # 执行器已关闭等极端情况：错误永不静默，落 failed 终态
            with self._lock:
                self._cancel_events.pop(task_id, None)
            self._finish(task_id, STATUS_FAILED, error=f"executor rejected: {exc}")
            raise
        logger.info("任务已提交：%s skill=%s pending=%s", task_id, skill_id, pending + 1)
        return task_id

    def cancel(self, task_id: str) -> bool:
        """协作式取消。queued 直接落终态；running 置 flag 由阶段边界生效。

        返回 False：任务不存在或已是终态（幂等，不报错）。
        """
        rec = self.store.get_task(task_id)
        if rec is None or rec["status"] in TERMINAL_STATUSES:
            return False
        with self._lock:
            ev = self._cancel_events.get(task_id)
        if ev is not None:
            ev.set()
        if rec["status"] == STATUS_QUEUED:
            # worker 尚未启动：直接落终态（worker 启动时看到 flag 会跳过）
            self._finish(task_id, STATUS_CANCELLED, message="任务在排队时被取消")
        logger.info("任务取消信号已发出：%s（%s）", task_id, rec["status"])
        return True

    def get_task(self, task_id: str) -> dict | None:
        return self.store.get_task(task_id)

    def list_tasks(self, status: str | None = None, limit: int = 100) -> list[dict]:
        return self.store.list_tasks(status=status, limit=limit)

    def shutdown(self) -> None:
        """停 GC 与线程池（进程退出；running 任务由下次启动 recover）。"""
        self._gc_stop.set()
        self._pool.shutdown(wait=False, cancel_futures=True)
        self.events.close()

    # ── 执行 ──

    def _worker(self, task_id: str, skill_id: str, args: dict) -> None:
        with self._lock:
            ev = self._cancel_events.get(task_id)
        try:
            rec = self.store.get_task(task_id)
            if rec is None or rec["status"] != STATUS_QUEUED:
                return  # 排队时已被取消（终态已落），直接退出
            if ev is not None and ev.is_set():
                self._finish(task_id, STATUS_CANCELLED, message="任务在启动前被取消")
                return
            self.store.set_status(task_id, STATUS_RUNNING)
            self._publish(task_id, status=STATUS_RUNNING, pct=0.0, message="任务开始执行")

            ctx = TaskContext(
                task_id=task_id,
                workdir=self._data_dir / "tasks" / task_id / "work",
                progress_cb=lambda stage, pct, message="": self._on_progress(
                    task_id, stage, pct, message),
                cancel_check=(lambda: True) if ev is None else ev.is_set,
            )
            token = set_task_context(ctx)
            try:
                result = invoke(skill_id, args,
                                new_trace(source="task-manager", trace_id=task_id))
            except Exception as exc:  # noqa: BLE001  invoke 理论不抛；防御兜底
                self._finish(task_id, STATUS_FAILED,
                             error=f"{type(exc).__name__}: {exc}")
                return
            finally:
                reset_task_context(token)

            self._classify_and_finish(task_id, result)
        finally:
            with self._lock:
                self._cancel_events.pop(task_id, None)

    def _classify_and_finish(self, task_id: str, result) -> None:
        """SkillResult 信封 → 任务终态（见模块 docstring 终态语义）。"""
        data = result.data
        failure_category = getattr(data, "failure_category", None) if data is not None else None
        if result.ok and not failure_category:
            out = getattr(data, "output_path", None) if data is not None else None
            output_paths = [out] if isinstance(out, str) and out else None
            self._finish(task_id, STATUS_SUCCEEDED, output_paths=output_paths,
                         message="任务完成")
            return
        if result.ok and failure_category == STATUS_CANCELLED:
            self._finish(task_id, STATUS_CANCELLED, message="任务已取消")
            return
        if result.ok:  # 技能级失败（degraded 信封）
            warning = getattr(data, "warning", None) if data is not None else None
            self._finish(task_id, STATUS_FAILED,
                         error=warning or f"技能级失败（{failure_category}）",
                         message=f"任务失败（{failure_category}）")
            return
        error = result.error.message if result.error is not None else "未知错误"
        self._finish(task_id, STATUS_FAILED,
                     error=f"[{result.error.code}] {error}" if result.error else error,
                     message="任务失败")

    def _on_progress(self, task_id: str, stage: str, pct: float, message: str) -> None:
        """进度回调：落库 + 发 MQTT（各自降级，互不影响）。"""
        try:
            self.store.update_progress(task_id, stage, pct)
        except Exception as exc:  # noqa: BLE001  进度落库失败不阻断任务
            logger.warning("进度落库失败 task=%s: %s", task_id, exc)
        self._publish(task_id, status=STATUS_RUNNING, stage=stage,
                      pct=pct, message=message)

    def _finish(self, task_id: str, status: str, *,
                error: str | None = None,
                output_paths: list[str] | None = None,
                message: str = "") -> None:
        """落终态 + retain 事件（终态幂等：重复调用仅多一次事件）。"""
        self.store.set_status(task_id, status, error=error, output_paths=output_paths)
        self._publish(task_id, status=status,
                      pct=100.0 if status == STATUS_SUCCEEDED else 0.0,
                      message=message or error or status)

    def _publish(self, task_id: str, **kw) -> None:
        """事件发布（TaskEventPublisher 自身绝不抛；此处兜底防御）。"""
        try:
            self.events.publish(task_id, **kw)
        except Exception as exc:  # noqa: BLE001
            logger.debug("事件发布异常（已忽略）: %s", exc)

    # ── TTL GC ──

    def _gc_loop(self) -> None:
        interval = max(30, min(300, self._ttl // 2))
        while not self._gc_stop.wait(interval):
            try:
                self._gc_once()
            except Exception as exc:  # noqa: BLE001  错误永不静默
                logger.warning("任务 GC 失败: %s", exc)

    def _gc_once(self) -> None:
        """终态任务 workdir 清理（DB 行保留）+ 孤儿目录清扫。"""
        cutoff = datetime.now(timezone.utc).timestamp() - self._ttl
        known: set[str] = set()
        for status in (*NON_TERMINAL_STATUSES, *TERMINAL_STATUSES):
            for rec in self.store.list_tasks(status=status, limit=1000):
                task_id = rec["task_id"]
                known.add(task_id)
                if status in TERMINAL_STATUSES and rec["finished_at"]:
                    finished = datetime.fromisoformat(rec["finished_at"]).timestamp()
                    if finished < cutoff:
                        shutil.rmtree(
                            self._data_dir / "tasks" / task_id, ignore_errors=True)
        tasks_root = self._data_dir / "tasks"
        if not tasks_root.is_dir():
            return
        orphan_cutoff = time.time() - _ORPHAN_DIR_TTL_SECONDS
        for child in tasks_root.iterdir():
            if not child.is_dir() or child.name in known:
                continue
            try:
                if child.stat().st_mtime < orphan_cutoff:
                    shutil.rmtree(child, ignore_errors=True)
                    logger.info("孤儿 workdir 已清理：%s", child)
            except OSError as exc:
                logger.debug("孤儿目录检查失败 %s: %s", child, exc)


# ── 惰性单例 ──

_manager: TaskManager | None = None
_manager_lock = threading.Lock()


def get_task_manager() -> TaskManager:
    """进程级单例（首次调用即建 store 连接 + recover + GC 线程）。"""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = TaskManager()
        return _manager
