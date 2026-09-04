"""localize_submit 技能演示 —— 批量提交本地化异步任务（本地任务引擎）。

运行：PYTHONPATH=<repo>/src conda run -n flowmind python examples/localize_submit_demo.py

展示：
1. happy path：3 个视频 URL → 逐视频一个 task，立即返回 task_ids
2. 扩展名预检：合法 + 非法混提 → 受理/拒绝分桶
3. degraded partial：队列中途满 → 已受理 task_ids 保留 + transient 可重试
4. error：队列满且一个都没受理 → ok=False（429 背压语义）

mock 方式：patch 技能模块级 get_task_manager 符号为内存 fake
（FakeManager/FakeStore，本文件内实现），不依赖 PG / MQTT / 外部 VL 服务。
"""

from __future__ import annotations

from datetime import datetime, timezone

import flowmind.skills  # noqa: F401  触发 @skill 注册
import flowmind.skills.localize_submit as lsub
from flowmind.discover import field_names
from flowmind.skill import invoke
from flowmind.tasks import TERMINAL_STATUSES, TaskQueueFull


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── 内存 fake：TaskStore / TaskManager 最小语义替身 ─────────────────

class _FakeStore:
    """dict 存储的任务行（字段结构与 TaskStore._row_to_dict 对齐）。"""

    def __init__(self) -> None:
        self.tasks: dict[str, dict] = {}
        self._seq = 0

    def create_task(self, skill_id: str, args: dict) -> str:
        self._seq += 1
        tid = f"task-{self._seq:03d}"
        self.tasks[tid] = {
            "task_id": tid, "skill_id": skill_id, "args": dict(args),
            "status": "queued", "stage": None, "progress": 0.0,
            "error": None, "created_at": _now(), "started_at": None,
            "finished_at": None, "tenant_id": None, "output_paths": None,
        }
        return tid

    def get_task(self, task_id: str) -> dict | None:
        return self.tasks.get(task_id)

    def count_pending(self) -> int:
        return sum(1 for t in self.tasks.values()
                   if t["status"] in ("queued", "running"))

    def set_status(self, task_id: str, status: str, *,
                   error: str | None = None,
                   output_paths: list[str] | None = None) -> None:
        rec = self.tasks[task_id]
        rec["status"] = status
        if error is not None:
            rec["error"] = error
        if output_paths is not None:
            rec["output_paths"] = output_paths
        if status in TERMINAL_STATUSES:
            rec["finished_at"] = _now()


class _FakeManager:
    """submit 背压（TaskQueueFull）+ get_task/cancel 最小语义。"""

    def __init__(self, store: _FakeStore, max_pending: int = 100):
        self.store = store
        self._max_pending = max_pending

    def submit(self, skill_id: str, args: dict) -> str:
        pending = self.store.count_pending()
        if pending >= self._max_pending:
            raise TaskQueueFull(
                f"待处理任务已达上限 {self._max_pending}（当前 {pending}），稍后重试")
        return self.store.create_task(skill_id, args)

    def get_task(self, task_id: str) -> dict | None:
        return self.store.get_task(task_id)


def _install(store: _FakeStore, max_pending: int = 100) -> _FakeManager:
    """patch localize_submit 模块级 get_task_manager 符号。"""
    manager = _FakeManager(store, max_pending)
    lsub.get_task_manager = lambda: manager
    return manager


def main() -> None:
    section("0) discover('localize_submit') —— Agent 自查字段")
    for p, names in field_names("localize_submit").items():
        print(f"  {p}: {names}")

    section("1) Happy path：3 个视频 URL → 3 个独立任务")
    store = _FakeStore()
    _install(store)
    r = invoke("localize_submit", {
        "videos": [
            "https://cdn.example.com/promo-v1.mp4",
            "https://cdn.example.com/promo-v2.mp4",
            "https://cdn.example.com/promo-v3.mp4",
        ],
        "target_lang": "th",
    })
    print(f"  ok          : {r.ok}")
    print(f"  task_ids    : {r.data.task_ids}")
    print(f"  accepted    : {r.data.accepted}")
    print(f"  skill_id    : {r.data.skill_id}")
    print(f"  reasoning   : {r.reasoning[0].conclusion}")
    check(r.ok, "happy path 应 ok")
    check(len(r.data.task_ids) == 3 and r.data.accepted == 3, "应受理 3 个任务")
    check(all(store.get_task(t)["status"] == "queued" for t in r.data.task_ids),
          "任务应落 queued")
    check(store.get_task(r.data.task_ids[0])["args"]["video_path"]
          == "https://cdn.example.com/promo-v1.mp4", "args 应含 video_path")
    check(store.get_task(r.data.task_ids[0])["args"]["target_lang"] == "th",
          "显式 target_lang 应透传进任务 args")

    section("2) 扩展名预检：合法 + 非法混提 → 分桶")
    r = invoke("localize_submit", {
        "videos": ["https://cdn.example.com/a.mp4", "/data/b.mp4", "/data/c.txt"],
    })
    print(f"  accepted      : {r.data.accepted}")
    print(f"  rejected      : {r.data.rejected_count} {r.data.rejected_paths}")
    print(f"  命中规则      : {[h.name for h in r.reasoning[0].triggered_rules]}")
    check(r.ok and r.data.accepted == 2, "URL + .mp4 应受理 2 个")
    check(r.data.rejected_paths == ["/data/c.txt"], ".txt 应被拒")

    section("3) degraded partial：队列中途满（max_pending=2）")
    store = _FakeStore()
    _install(store, max_pending=2)
    r = invoke("localize_submit", {
        "videos": ["/data/v1.mp4", "/data/v2.mp4", "/data/v3.mp4", "/data/v4.mp4"],
    })
    print(f"  ok              : {r.ok}（信封 ok=True + degraded=True）")
    print(f"  task_ids        : {r.data.task_ids}")
    print(f"  degraded        : {r.metrics.degraded}")
    print(f"  failure_category: {r.data.failure_category}（transient → 可重试）")
    print(f"  retriable       : {r.data.retriable}")
    print(f"  warning         : {r.data.warning}")
    check(r.ok and r.metrics.degraded, "partial success 应 degraded")
    check(len(r.data.task_ids) == 2 and r.data.accepted == 2, "应已受理 2 个")
    check(r.data.failure_category == "transient" and r.data.retriable,
          "队列满属 transient 可重试")

    section("4) error：队列满且一个都没受理 → ok=False（429 背压语义）")
    store = _FakeStore()
    manager = _install(store, max_pending=2)
    store.create_task("localize_video", {"video_path": "/data/pre1.mp4"})  # 预填 2 个 pending
    store.create_task("localize_video", {"video_path": "/data/pre2.mp4"})
    check(manager.store.count_pending() == 2, "预填 pending 应为 2")
    r = invoke("localize_submit", {"videos": ["/data/v.mp4"]})
    print(f"  ok          : {r.ok}")
    print(f"  error.code  : {r.error.code if r.error else None}")
    print(f"  error.msg   : {r.error.message if r.error else None}")
    check(r.ok is False, "队列全满应 ok=False")
    check(r.error is not None and r.error.code == "INTERNAL",
          "TaskQueueFull 经 invoke 兜底为 INTERNAL（429 语义由 HTTP 层映射）")
    check("稍后重试" in (r.error.message or ""), "message 应带背压提示")

    print("\n✅ localize_submit_demo 全部通过")


if __name__ == "__main__":
    main()
