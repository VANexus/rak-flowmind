"""TaskManager：PG 持久化异步任务执行器（早期内存态 JobManager 的 SaaS 升级版）。

与内存态方案（已随独立 REST 后端移除）的差异：
- 存储落 PostgreSQL（服务重启不丢任务历史；启动时 recover_running()
  把遗留 queued/running 标为 interrupted，pending 水位不再虚增）。
- 进度可观测：技能经 TaskContext.progress_cb 上报 → PG 落库 +
  MQTT ``mcp-base-gpu/tasks/{id}/events`` 实时推送（终态 retain）。
- 取消语义：cancel() 置协作取消 Event（流水线在阶段边界抛
  tasks.CancelledError 提前退出），同时带 CAS 守卫落 cancelled 终态
  （仅 queued/running 可写）；worker 落终态同样带 CAS——终态
  first-writer-wins，cancel 与完成并发时后者幂等让位，绝不互相覆盖
  （防 retain 污染与 cancelled 被 retry 重复烧 GPU）。
  ffmpeg/demucs 子进程不强行 kill（各自有 timeout；阶段边界为主，
  不做过度工程）。

终态语义（早期 "failed 仅 runner 崩溃"语义在此收紧为业务语义）：
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
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flowmind import auth
from flowmind.config import load_config
from flowmind.contracts import new_trace
from flowmind.pipeline_steps import resolve_selection
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
from flowmind.tasks.events import TaskEventPublisher, get_event_publisher
from flowmind.tasks.store import TaskStore

logger = logging.getLogger(__name__)

_ORPHAN_DIR_TTL_SECONDS = 7 * 86400  # 孤儿 workdir（无 DB 行）mtime 超过 7 天清理
# 进度写合并最小间隔（秒）：stage 未变且距上次 PG 落库不足此值时跳过 UPDATE
# （MQTT 照发；终态不受影响）。2s 对 2s 轮询的前端进度条无感知差异。
_PROGRESS_MIN_INTERVAL = 2.0


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
        # SaaS 租户公平性（2026-10-31）：单租户 pending 上限，防一个租户的
        # 批量提交塞满全队列把其他租户全部挤成 429
        self._max_pending_per_tenant = max(
            1, getattr(cfg, "max_pending_per_tenant", 20))
        self._ttl = max(1, ttl_seconds if ttl_seconds is not None
                        else cfg.task_ttl_seconds)
        self._data_dir = Path(data_dir or cfg.data_dir).expanduser().resolve()
        self.store = store or TaskStore()
        self.events = events or get_event_publisher()
        self.workers = max(1, workers)  # GPU 单卡：绝不放宽（防 OOM）
        self._pool = ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix="flowmind-task")
        self._cancel_events: dict[str, threading.Event] = {}
        # 终态已写标志：终态落库（无论谁写）后置位，_on_progress 据此停发，
        # 防 cancel 后流水线残余进度把 MQTT retain 的终态事件洗回 running
        self._terminal_flags: dict[str, threading.Event] = {}
        # 进度写合并账本（task_id → (stage, ts)）：stage 未变且距上次落库
        # 不足 _PROGRESS_MIN_INTERVAL 秒时跳过 PG UPDATE（进度条体验不变，
        # worker 内 PG 往返 -80%）；MQTT 发布不受合并影响
        self._last_progress: dict[str, tuple[str, float]] = {}
        # 最新步骤快照（2026-10-31）：终态收尾时 PG 可能还没有快照（取消太快）
        self._last_steps: dict[str, list[dict]] = {}
        self._lock = threading.Lock()       # 保护 _cancel_events/_terminal_flags/_last_progress
        self._gc_stop = threading.Event()

        # 启动恢复（队列持久化，2026-10-31）：running → interrupted（流水线
        # 不可断点续跑）；queued **保留**并由下方重新入队——202 受理的任务
        # 不再因重启静默丢失。
        self.store.recover_running()
        self._requeue_orphans()

        self._gc_thread = threading.Thread(
            target=self._gc_loop, name="flowmind-task-gc", daemon=True)
        self._gc_thread.start()

    # ── 提交 / 取消 ──

    def submit(self, skill_id: str, args: dict,
               tenant_id: str | None = None) -> str:
        """提交任务，返回 task_id；pending 超限抛 TaskQueueFull（调用方回 429）。

        原子化准入（2026-10-31）：单条 SQL 完成全局水位 + 租户水位检查 +
        插入——取代旧的全局互斥锁 + 两次 PG 往返（提交全序列化的瓶颈）。
        水位为软上限（READ COMMITTED 并发下 ±N 超越，见 store 模块 docstring）。

        租户归属（ECO-ADR-0011）：``tenant_id`` 不传时从当前调用方上下文
        （请求中间件绑定的凭证）取——这样 MCP 通道的技能（localize_submit /
        localize_retry 调本方法时并不持租户参数）也自动归因，REST 端点
        显式传值则优先。已启用鉴权而上下文无凭证 → **拒写**（fail-closed），
        绝不落下无主任务。
        """
        required, scope_tenant = auth.tenant_scope()
        if tenant_id is None:
            if required and scope_tenant is None:
                raise PermissionError(
                    "无调用方凭证，拒绝创建无租户归属的任务（fail-closed）")
            tenant_id = scope_tenant
        task_id = uuid.uuid4().hex
        admitted = self.store.create_task_admitted(
            task_id, skill_id, args, tenant_id,
            global_max=self._max_pending,
            tenant_max=self._max_pending_per_tenant,
        )
        if not admitted:
            raise TaskQueueFull(
                f"待处理任务已达上限（全局 {self._max_pending}"
                f"/租户 {self._max_pending_per_tenant}），稍后重试")
        with self._lock:
            self._cancel_events[task_id] = threading.Event()
            self._terminal_flags[task_id] = threading.Event()
        try:
            self._pool.submit(self._worker, task_id, skill_id, args)
        except RuntimeError as exc:
            # 执行器已关闭等极端情况：错误永不静默，落 failed 终态
            with self._lock:
                self._cancel_events.pop(task_id, None)
                self._terminal_flags.pop(task_id, None)
            self._finish(task_id, STATUS_FAILED, error=f"executor rejected: {exc}",
                         from_statuses=NON_TERMINAL_STATUSES)
            raise
        logger.info("任务已提交：%s skill=%s tenant=%s", task_id, skill_id, tenant_id)
        return task_id

    def _requeue_orphans(self) -> None:
        """重启后把遗留 queued 任务重新入队（队列持久化的后半段）。

        行已在 PG（不用重新 INSERT），仅向线程池补投 worker；worker 的
        CAS queued→running 天然幂等。上限按全局 pending 水位截断（再多
        的留待 GC/人工，正常场景不会有——能 queued 必然当初过了水位）。
        """
        try:
            orphans = self.store.list_queued(limit=self._max_pending)
        except Exception as exc:  # noqa: BLE001  重入队失败不阻断启动
            logger.warning("启动重入队失败（任务留在 queued，下轮重启再试）: %s", exc)
            return
        for rec in orphans:
            task_id = rec["task_id"]
            with self._lock:
                self._cancel_events[task_id] = threading.Event()
                self._terminal_flags[task_id] = threading.Event()
            try:
                self._pool.submit(self._worker, task_id, rec["skill_id"],
                                  rec.get("args") or {})
            except RuntimeError as exc:  # noqa: PERF203  执行器关闭
                logger.warning("重入队投递失败 task=%s: %s", task_id, exc)
                break
        if orphans:
            logger.warning("启动重入队：%s 个遗留 queued 任务已恢复执行", len(orphans))

    def cancel(self, task_id: str) -> bool:
        """取消任务：置协作取消 flag + CAS 落 cancelled 终态。

        终态写入带 ``from_statuses=NON_TERMINAL`` 原子守卫（单条 UPDATE）：
        与 worker 落终态并发时 first-writer-wins——谁先把终态写进 DB 谁赢，
        后写方 CAS False 幂等返回，杜绝 cancelled 被 succeeded 覆盖、
        retain 被污染、cancelled 任务被 retry 重复烧 GPU。
        running 任务置 flag 后流水线在阶段边界退出（协作取消语义不变），
        但终态立即对用户可见（cancelled）。

        返回 False：任务不存在、已是终态、**不属于当前租户**（含并发下已被
        人先写终态），幂等不报错——与读路径一致，不给跨租户取消留口子。
        """
        rec = self._visible(self.store.get_task(task_id))
        if rec is None or rec["status"] in TERMINAL_STATUSES:
            return False
        with self._lock:
            ev = self._cancel_events.get(task_id)
        if ev is not None:
            ev.set()
        ok = self._finish(task_id, STATUS_CANCELLED, message="任务已取消",
                          from_statuses=NON_TERMINAL_STATUSES)
        if ok:
            logger.info("任务已取消：%s（%s）", task_id, rec["status"])
        else:
            logger.info("任务取消未生效（并发已入终态）：%s（%s）",
                        task_id, rec["status"])
        return ok

    def get_task(self, task_id: str, *, include_args: bool = True) -> dict | None:
        """按调用方租户范围读任务（跨租户与不存在**同形**返回 None）。

        故意不给 403而是回 404：“此 task_id 存在但不属于你”本身就是泄密
        （枚举探针可用它探测他人任务是否存在）。本方法是所有读/操作
        技能的唯一入口（localize_status / cancel / retry / download）。
        后台路径（worker / GC / recover）直接用 store.get_task，不受影响。
        include_args=False 供轮询热路径跳过 args 解析（见 store.get_task）。
        """
        return self._visible(
            self.store.get_task(task_id, include_args=include_args))

    def list_tasks(self, status: str | None = None, limit: int = 100) -> list[dict]:
        required, tenant = auth.tenant_scope()
        if required and tenant is None:
            return []  # 启用鉴权但无凭证：fail-closed，不返回任何行
        rows = self.store.list_tasks(status=status, limit=limit)
        if not required or tenant is None:
            return rows
        return [r for r in rows if r.get("tenant_id") == tenant]

    @staticmethod
    def _visible(rec: dict | None) -> dict | None:
        """按当前调用方租户过滤单条记录（无权限一律当作不存在）。"""
        required, tenant = auth.tenant_scope()
        if not required:
            return rec
        if rec is None or tenant is None or rec.get("tenant_id") != tenant:
            return None
        return rec

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
            try:
                rec = self.store.get_task(task_id)
                if rec is None or rec["status"] != STATUS_QUEUED:
                    return  # 排队时已被取消（终态已落），直接退出
                if ev is not None and ev.is_set():
                    self._finish(task_id, STATUS_CANCELLED,
                                 message="任务在启动前被取消",
                                 from_statuses=NON_TERMINAL_STATUSES)
                    return
                # queued → running 也走 CAS：cancel 在取走与置 running 之间
                # 落了 cancelled 时，这里必须让位（防止取消后任务复活成 running）
                if not self.store.set_status(task_id, STATUS_RUNNING,
                                             from_statuses=(STATUS_QUEUED,)):
                    return
                self._publish(task_id, status=STATUS_RUNNING, pct=0.0,
                              message="任务开始执行")

                # 步骤选择（2026-10-31）：调用方在 args 里带 steps/skip_steps 时
                # 由注册表解析成启用集合（提交端点已做同样校验，此处是兜底：
                # 解析失败 → 显式失败，绝不静默跑半截）
                enabled_steps: tuple[str, ...] = ()
                try:
                    sel, _sk = resolve_selection(
                        skill_id,
                        steps=args.get("steps"),
                        skip_steps=args.get("skip_steps"),
                    )
                    enabled_steps = tuple(s.id for s in sel)
                except ValueError as exc:
                    self._finish(task_id, STATUS_FAILED,
                                 error=f"[VALIDATION] 步骤选择非法: {exc}",
                                 from_statuses=NON_TERMINAL_STATUSES)
                    return

                ctx = TaskContext(
                    task_id=task_id,
                    workdir=self._data_dir / "tasks" / task_id / "work",
                    data_dir=self._data_dir,
                    progress_cb=lambda stage, pct, message="": self._on_progress(
                        task_id, stage, pct, message),
                    step_cb=lambda payload: self._on_step(task_id, payload),
                    enabled_steps=enabled_steps,
                    cancel_check=(lambda: False) if ev is None else ev.is_set,
                )
                token = set_task_context(ctx)
                try:
                    result = invoke(skill_id, args,
                                    new_trace(source="task-manager", trace_id=task_id))
                except Exception as exc:  # noqa: BLE001  invoke 理论不抛；防御兜底
                    self._finish(task_id, STATUS_FAILED,
                                 error=f"{type(exc).__name__}: {exc}",
                                 from_statuses=NON_TERMINAL_STATUSES)
                    return
                finally:
                    reset_task_context(token)

                self._classify_and_finish(task_id, result)
            except Exception:  # noqa: BLE001  最外层兜底：store 异常不得被 Future 静默吞掉
                logger.exception("任务 worker 异常 task=%s skill=%s", task_id, skill_id)
                try:
                    self._finish(task_id, STATUS_FAILED,
                                 error=f"worker crashed: {sys.exc_info()[0].__name__}: {sys.exc_info()[1]}",
                                 from_statuses=NON_TERMINAL_STATUSES)
                except Exception:  # noqa: BLE001  兜底自身失败：留待下次启动 recover_running
                    logger.critical(
                        "worker 崩溃兑底落终态失败 task=%s（状态留在非终态，"
                        "下次启动 recover_running 会标为 interrupted）", task_id,
                        exc_info=True)
        finally:
            with self._lock:
                self._cancel_events.pop(task_id, None)
                self._terminal_flags.pop(task_id, None)
                self._last_progress.pop(task_id, None)
                self._last_steps.pop(task_id, None)

    def _finalize_steps(self, task_id: str, task_status: str,
                        error: str | None) -> None:
        """终态时把步骤快照收尾（幂等；无步骤快照的任务直接跳过）。"""
        with self._lock:
            steps = list(self._last_steps.get(task_id) or [])
        if not steps:
            rec = self.store.get_task(task_id, include_args=False)
            steps = (rec or {}).get("steps") or []
        if not steps:
            return
        now = datetime.now(timezone.utc).isoformat()
        changed = False
        for st in steps:
            cur = st.get("status")
            if cur in ("done", "skipped", "failed"):
                continue
            if task_status == STATUS_FAILED and cur == "running":
                st["status"] = "failed"
                st["skip_reason"] = None
                st["note"] = (f"任务失败：{(error or '')[:200]}")
                # 失败也记耗时（从 started_at 推算，前端可展示"跑了多久才失败"）
                st["duration_ms"] = _duration_since(st.get("started_at"))
            else:
                st["status"] = "skipped"
                st["skipped"] = True
                st["skip_reason"] = (
                    "任务已取消，该步未执行" if task_status == STATUS_CANCELLED
                    else "任务已结束，该步未执行")
            st["ended_at"] = now
            changed = True
        if changed:
            self.store.update_steps(task_id, steps)

    def _classify_and_finish(self, task_id: str, result) -> None:
        """SkillResult 信封 → 任务终态（见模块 docstring 终态语义）。"""
        data = result.data
        failure_category = getattr(data, "failure_category", None) if data is not None else None
        if result.ok and not failure_category:
            out = getattr(data, "output_path", None) if data is not None else None
            output_paths = [out] if isinstance(out, str) and out else None
            self._finish(task_id, STATUS_SUCCEEDED, output_paths=output_paths,
                         message="任务完成", from_statuses=NON_TERMINAL_STATUSES)
            return
        if result.ok and failure_category == STATUS_CANCELLED:
            self._finish(task_id, STATUS_CANCELLED, message="任务已取消",
                         from_statuses=NON_TERMINAL_STATUSES)
            return
        if result.ok:  # 技能级失败（degraded 信封）
            warning = getattr(data, "warning", None) if data is not None else None
            self._finish(task_id, STATUS_FAILED,
                         error=warning or f"技能级失败（{failure_category}）",
                         message=f"任务失败（{failure_category}）",
                         from_statuses=NON_TERMINAL_STATUSES)
            return
        error = result.error.message if result.error is not None else "未知错误"
        self._finish(task_id, STATUS_FAILED,
                     error=f"[{result.error.code}] {error}" if result.error else error,
                     message="任务失败",
                     from_statuses=NON_TERMINAL_STATUSES)

    def _on_progress(self, task_id: str, stage: str, pct: float, message: str) -> None:
        """进度回调：落库（按间隔合并）+ 发 MQTT（各自降级，互不影响）。

        终态守卫：终态已落库（本 worker 落完或 cancel 线程抢先落）后停发——
        防止 cancel 后流水线在阶段边界间的残余进度把 retain 的终态事件
        洗回 running（实现选型：worker 侧 flag 停发，比逐次 update 走
        状态守卫少一次 DB 往返且覆盖 MQTT 发布路径）。

        写合并（2026-10-31）：stage 未变且距上次 PG 落库不足
        ``_PROGRESS_MIN_INTERVAL`` 秒时跳过 UPDATE（MQTT 照发，前端实时性
        不依赖 DB 频率）——localize_video 单任务 ~25 次回调，合并后 worker
        内 PG 往返 -80%。终态由 _finish/set_status 无条件落库，不受合并影响。
        """
        with self._lock:
            flag = self._terminal_flags.get(task_id)
            if flag is not None and flag.is_set():
                return
            last = self._last_progress.get(task_id)
            now = time.monotonic()
            if last is not None and last[0] == stage and (now - last[1]) < _PROGRESS_MIN_INTERVAL:
                skip_write = True
            else:
                self._last_progress[task_id] = (stage, now)
                skip_write = False
        if not skip_write:
            try:
                self.store.update_progress(task_id, stage, pct)
            except Exception as exc:  # noqa: BLE001  进度落库失败不阻断任务
                logger.warning("进度落库失败 task=%s: %s", task_id, exc)
        self._publish(task_id, status=STATUS_RUNNING, stage=stage,
                      pct=pct, message=message)

    def _on_step(self, task_id: str, payload: dict) -> None:
        """步骤状态回调：落库（按步合并）+ 发 MQTT（终态守卫同 _on_progress）。

        payload = StepTracker.build() 的全量快照 {skill_id,total,completed,steps[]}。
        落库走 ``update_steps``（单行 JSON 更新，仅当快照有变化才写——
        StepTracker 只在状态迁移/backend/note 变化时回调，天然稀疏）；
        MQTT 载荷带全量 steps，前端一次拿到完整面板。
        """
        with self._lock:
            flag = self._terminal_flags.get(task_id)
        if flag is not None and flag.is_set():
            return
        steps = payload.get("steps") or []
        with self._lock:
            self._last_steps[task_id] = steps   # 终态收尾用（PG 可能尚未写入）
        try:
            self.store.update_steps(task_id, steps)
        except Exception as exc:  # noqa: BLE001  步骤落库失败不阻断任务
            logger.warning("步骤落库失败 task=%s: %s", task_id, exc)
        self._publish(task_id, status=STATUS_RUNNING,
                      stage=(next((s["id"] for s in steps
                                   if s.get("status") == "running"), "")) or None,
                      pct=None, message="", steps=steps)

    def _finish(self, task_id: str, status: str, *,
                error: str | None = None,
                output_paths: list[str] | None = None,
                message: str = "",
                from_statuses: tuple[str, ...] | None = None) -> bool:
        """落终态（CAS 可选）+ retain 事件；返回本次写入是否生效。

        - CAS 生效（或未带守卫且写入成功）→ 置终态 flag（停发残余进度）
          并发布事件；
        - CAS False（已被人先写终态）→ 不置事件不发布，幂等返回 False。
          终态重复调用不再产生重复 retain 事件（终态真正只写一次）。
        """
        ok = self.store.set_status(task_id, status, error=error,
                                   output_paths=output_paths,
                                   from_statuses=from_statuses)
        if not ok:
            return False
        # 步骤收尾（2026-10-31）：任务终态时不应留下 running/pending 的步骤——
        # 取消 → 中断步骤记 skipped；失败 → 当前 running 步骤记 failed，其余 pending
        # 记 skipped（未执行）；成功 → 未执行的 pending 记 skipped。
        try:
            self._finalize_steps(task_id, status, error)
        except Exception as exc:  # noqa: BLE001  收尾失败不影响终态
            logger.warning("步骤收尾失败 task=%s: %s", task_id, exc)
        with self._lock:
            flag = self._terminal_flags.get(task_id)
        if flag is not None:
            flag.set()
        self._publish(task_id, status=status,
                      pct=100.0 if status == STATUS_SUCCEEDED else 0.0,
                      message=message or error or status)
        return True

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
        """终态任务 workdir/outputs 清理（DB 行保留）+ 孤儿目录清扫。

        清扫范围覆盖 data_dir/tasks/<id>/（工作目录）与 data_dir/outputs/<id>/
        （URL 输入任务的产物目录），同一 TTL 策略。过期任务由
        store.list_expired 按 SQL（``status IN 终态 AND finished_at < cutoff``）
        直接圈定，替代逐状态 limit=1000 扫描；孤儿目录（无 DB 行且 mtime
        超 7 天，直连 invoke 产生的随机目录）逐目录查 DB 存在性后清理
        （孤儿本就稀少，且 7 天内目录不做 DB 查询）。
        """
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self._ttl)
        for task_id in self.store.list_expired(finished_before=cutoff, limit=500):
            shutil.rmtree(
                self._data_dir / "tasks" / task_id, ignore_errors=True)
            shutil.rmtree(
                self._data_dir / "outputs" / task_id, ignore_errors=True)
        orphan_cutoff = time.time() - _ORPHAN_DIR_TTL_SECONDS
        for root_name in ("tasks", "outputs"):
            root = self._data_dir / root_name
            if not root.is_dir():
                continue
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                try:
                    if (child.stat().st_mtime < orphan_cutoff
                            and not self.store.task_exists(child.name)):
                        shutil.rmtree(child, ignore_errors=True)
                        logger.info("孤儿目录已清理：%s", child)
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


def _duration_since(started_at: str | None) -> int | None:
    """从 ISO8601 起始时间推算已运行毫秒（失败/取消步骤补耗时用）。"""
    if not started_at:
        return None
    try:
        t0 = datetime.fromisoformat(started_at)
        return max(0, int((datetime.now(timezone.utc) - t0).total_seconds() * 1000))
    except Exception:  # noqa: BLE001
        return None
