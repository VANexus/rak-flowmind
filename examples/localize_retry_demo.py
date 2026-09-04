"""localize_retry 技能演示 —— 重提终态任务（复制原 args 重新 submit）。

运行：PYTHONPATH=<repo>/src conda run -n flowmind python examples/localize_retry_demo.py

展示：
1. discover() 自动字段发现
2. happy path：failed 任务 → 复制原 args 重提，返回 original + new task_id
3. degraded：running 拒绝重提（须先 localize_cancel）/ succeeded 拒绝 / 任务不存在
4. error：队列满（TaskQueueFull）→ ok=False（429 背压语义）

mock 方式：patch 技能模块级 get_task_manager 符号为内存 fake
（FakeManager/FakeStore，本文件内实现），不依赖 PG / MQTT / 外部 VL 服务。
"""

from __future__ import annotations

import flowmind.skills  # noqa: F401  触发 @skill 注册
import flowmind.skills.localize_retry as lr
from flowmind.discover import field_names
from flowmind.skill import invoke
from flowmind.tasks import TaskQueueFull

_ORIGINAL_ARGS = {
    "video_path": "/data/promo.mp4",
    "target_lang": "th",
    "source_lang": "zh",
    "keep_background_audio": True,
    "tts_backend": "auto",
}


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


class _FakeStore:
    """dict 存储的任务行（字段结构与 TaskStore._row_to_dict 对齐）。"""

    def __init__(self, max_pending: int = 100) -> None:
        self.tasks: dict[str, dict] = {}
        self._seq = 0
        self.max_pending = max_pending

    def add(self, task_id: str, status: str, args: dict,
            skill_id: str = "localize_video") -> dict:
        rec = {
            "task_id": task_id, "skill_id": skill_id, "args": dict(args),
            "status": status, "stage": None, "progress": 0.0, "error": None,
            "created_at": "2026-09-04T08:00:00+00:00",
            "started_at": "2026-09-04T08:01:00+00:00" if status != "queued" else None,
            "finished_at": "2026-09-04T08:05:00+00:00" if status in
            ("succeeded", "failed", "cancelled", "interrupted") else None,
            "tenant_id": None, "output_paths": None,
        }
        self.tasks[task_id] = rec
        return rec

    def create_task(self, skill_id: str, args: dict) -> str:
        self._seq += 1
        tid = f"task-new-{self._seq:03d}"
        self.tasks[tid] = {
            "task_id": tid, "skill_id": skill_id, "args": dict(args),
            "status": "queued", "stage": None, "progress": 0.0,
            "error": None, "created_at": "2026-09-04T09:00:00+00:00",
            "started_at": None, "finished_at": None,
            "tenant_id": None, "output_paths": None,
        }
        return tid

    def get_task(self, task_id: str) -> dict | None:
        return self.tasks.get(task_id)

    def count_pending(self) -> int:
        return sum(1 for t in self.tasks.values()
                   if t["status"] in ("queued", "running"))


class _FakeManager:
    def __init__(self, store: _FakeStore):
        self.store = store

    def submit(self, skill_id: str, args: dict) -> str:
        pending = self.store.count_pending()
        if pending >= self.store.max_pending:
            raise TaskQueueFull(
                f"待处理任务已达上限 {self.store.max_pending}"
                f"（当前 {pending}），稍后重试")
        return self.store.create_task(skill_id, args)

    def get_task(self, task_id: str) -> dict | None:
        return self.store.get_task(task_id)


def _install(store: _FakeStore) -> _FakeManager:
    """patch localize_retry 模块级 get_task_manager 符号。"""
    manager = _FakeManager(store)
    lr.get_task_manager = lambda: manager
    return manager


def main() -> None:
    section("0) discover('localize_retry') —— Agent 自查字段")
    for p, names in field_names("localize_retry").items():
        print(f"  {p}: {names}")

    section("1) Happy path：重提 failed 任务（复制原 args → 新任务 queued）")
    store = _FakeStore()
    _install(store)
    store.add("t-old-failed", status="failed", args=_ORIGINAL_ARGS)
    r = invoke("localize_retry", {"task_id": "t-old-failed"})
    print(f"  ok              : {r.ok}")
    print(f"  original_task_id: {r.data.original_task_id}")
    print(f"  new_task_id     : {r.data.new_task_id}（新独立任务）")
    print(f"  original_status : {r.data.original_status}")
    print(f"  skill_id 沿用   : {r.data.skill_id}")
    print(f"  source_video    : {r.data.source_video}")
    print(f"  target_lang     : {r.data.target_lang}")
    print(f"  推理            : {r.reasoning[0].conclusion}")

    new_id = r.data.new_task_id
    check(r.ok, "重提 failed 应 ok")
    check(new_id and new_id != "t-old-failed", "应返回新 task_id")
    rec = store.get_task(new_id)
    check(rec is not None and rec["status"] == "queued", "新任务应落 queued")
    check(rec["skill_id"] == "localize_video", "skill_id 应沿用原任务")
    check(rec["args"] == _ORIGINAL_ARGS, "原 args 应完整复制")
    check(r.data.original_status == "failed" and r.data.source_video
          == "/data/promo.mp4" and r.data.target_lang == "th", "报告字段应正确")

    section("2) degraded：running 拒绝重提（须先 localize_cancel）")
    store.add("t-old-running", status="running", args=_ORIGINAL_ARGS)
    n_before = len(store.tasks)
    r = invoke("localize_retry", {"task_id": "t-old-running"})
    print(f"  degraded        : {r.metrics.degraded}")
    print(f"  failure_category: {r.data.failure_category}")
    print(f"  new_task_id     : {r.data.new_task_id!r}（拒绝时空串）")
    print(f"  message         : {r.data.message}")
    check(r.ok and r.metrics.degraded, "running 拒绝应 degraded 不报错")
    check("拒绝重提" in (r.data.message or "") and "localize_cancel"
          in (r.data.message or ""), "message 应引导先取消")
    check(len(store.tasks) == n_before, "拒绝重提不应创建新任务")

    section("3) degraded：succeeded 拒绝 / 任务不存在")
    store.add("t-old-done", status="succeeded", args=_ORIGINAL_ARGS)
    r = invoke("localize_retry", {"task_id": "t-old-done"})
    print(f"  succeeded 拒绝  : degraded={r.metrics.degraded}"
          f" message={r.data.message}")
    check(r.ok and r.metrics.degraded and "已成功" in (r.data.message or ""),
          "succeeded 应拒绝（无需重提）")

    r = invoke("localize_retry", {"task_id": "t-ghost"})
    print(f"  任务不存在      : degraded={r.metrics.degraded}"
          f" category={r.data.failure_category}")
    check(r.ok and r.metrics.degraded and r.data.failure_category == "video",
          "ghost 应 degraded + video")

    section("4) error：队列满（TaskQueueFull）→ ok=False（429 背压语义）")
    store = _FakeStore(max_pending=1)
    _install(store)
    store.add("t-old-failed", status="failed", args=_ORIGINAL_ARGS)
    store.create_task("localize_video", {"video_path": "/data/pre.mp4"})  # 占满队列
    r = invoke("localize_retry", {"task_id": "t-old-failed"})
    print(f"  ok          : {r.ok}")
    print(f"  error.code  : {r.error.code if r.error else None}")
    print(f"  error.msg   : {r.error.message if r.error else None}")
    check(r.ok is False, "队列全满重提应 ok=False")
    check(r.error is not None and r.error.code == "INTERNAL",
          "TaskQueueFull 经 invoke 兜底为 INTERNAL（429 语义由 HTTP 层映射）")
    check("稍后重试" in (r.error.message or ""), "message 应带背压提示")

    print("\n✅ localize_retry_demo 全部通过")


if __name__ == "__main__":
    main()
