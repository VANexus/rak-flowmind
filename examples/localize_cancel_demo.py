"""localize_cancel 技能演示 —— 取消本地化任务（协作式取消语义）。

运行：PYTHONPATH=<repo>/src conda run -n flowmind python examples/localize_cancel_demo.py

展示：
1. discover() 自动字段发现
2. happy path：queued → 直接落 cancelled 终态；running → 协作取消（阶段边界生效）
3. 幂等：终态任务（succeeded）取消请求正常返回 cancelled=False，不是错误
4. degraded：任务不存在 → degraded + video

mock 方式：patch 技能模块级 get_task_manager 符号为内存 fake
（FakeManager/FakeStore，本文件内实现），不依赖 PG / MQTT / 外部 VL 服务。
"""

from __future__ import annotations

import flowmind.skills  # noqa: F401  触发 @skill 注册
import flowmind.skills.localize_cancel as lc
from flowmind.discover import field_names
from flowmind.skill import invoke
from flowmind.tasks import TERMINAL_STATUSES


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


class _FakeStore:
    """dict 存储的任务行（字段结构与 TaskStore._row_to_dict 对齐）。"""

    def __init__(self) -> None:
        self.tasks: dict[str, dict] = {}

    def add(self, task_id: str, status: str) -> dict:
        rec = {
            "task_id": task_id, "skill_id": "localize_video",
            "args": {"video_path": f"/data/{task_id}.mp4", "target_lang": "th"},
            "status": status, "stage": "asr" if status == "running" else None,
            "progress": 40.0 if status == "running" else 0.0, "error": None,
            "created_at": "2026-09-04T08:00:00+00:00",
            "started_at": "2026-09-04T08:01:00+00:00" if status == "running" else None,
            "finished_at": "2026-09-04T08:05:00+00:00"
            if status in TERMINAL_STATUSES else None,
            "tenant_id": None, "output_paths": None,
        }
        self.tasks[task_id] = rec
        return rec

    def get_task(self, task_id: str) -> dict | None:
        return self.tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        """TaskManager.cancel 最小语义：非终态 → 落 cancelled 终态；终态/不存在 → False。"""
        rec = self.tasks.get(task_id)
        if rec is None or rec["status"] in TERMINAL_STATUSES:
            return False
        rec["status"] = "cancelled"
        rec["stage"] = None
        rec["finished_at"] = "2026-09-04T08:10:00+00:00"
        return True


class _FakeManager:
    def __init__(self, store: _FakeStore):
        self.store = store

    def cancel(self, task_id: str) -> bool:
        return self.store.cancel(task_id)

    def get_task(self, task_id: str) -> dict | None:
        return self.store.get_task(task_id)


def _install(store: _FakeStore) -> _FakeManager:
    """patch localize_cancel 模块级 get_task_manager 符号。"""
    manager = _FakeManager(store)
    lc.get_task_manager = lambda: manager
    return manager


def _print(r) -> None:
    print(f"  ok              : {r.ok}")
    print(f"  task_id         : {r.data.task_id}")
    print(f"  cancelled       : {r.data.cancelled}")
    print(f"  previous_status : {r.data.previous_status}")
    print(f"  message         : {r.data.message}")
    print(f"  推理            : {r.reasoning[0].conclusion}")


def main() -> None:
    section("0) discover('localize_cancel') —— Agent 自查字段")
    for p, names in field_names("localize_cancel").items():
        print(f"  {p}: {names}")

    section("1) Happy path：queued 任务 → 直接取消（落 cancelled 终态）")
    store = _FakeStore()
    _install(store)
    store.add("t-queued-001", "queued")
    r = invoke("localize_cancel", {"task_id": "t-queued-001"})
    _print(r)
    check(r.ok and r.data.cancelled, "queued 取消应 ok + cancelled=True")
    check(r.data.previous_status == "queued", "应记录原状态 queued")
    check(store.get_task("t-queued-001")["status"] == "cancelled",
          "queued 任务应已落 cancelled 终态")
    check(store.get_task("t-queued-001")["finished_at"] is not None,
          "终态应带 finished_at")

    section("2) Happy path：running 任务 → 协作取消（阶段边界生效）")
    store.add("t-running-001", "running")
    r = invoke("localize_cancel", {"task_id": "t-running-001"})
    _print(r)
    check(r.ok and r.data.cancelled, "running 取消请求应受理")
    check("阶段边界" in r.data.message, "message 应说明协作式取消语义")
    check(store.get_task("t-running-001")["status"] == "cancelled",
          "fake 已模拟阶段边界检查后落终态")

    section("3) 幂等：终态任务（succeeded）→ ok=True cancelled=False（非错误）")
    store.add("t-done-001", "succeeded")
    r = invoke("localize_cancel", {"task_id": "t-done-001"})
    _print(r)
    check(r.ok and r.data.cancelled is False, "终态任务应幂等返回 cancelled=False")
    check("终态" in r.data.message and "幂等" in r.data.message,
          "message 应说明幂等语义")
    check(store.get_task("t-done-001")["status"] == "succeeded",
          "终态任务状态不应被改动")

    section("4) degraded：任务不存在 → degraded + video")
    r = invoke("localize_cancel", {"task_id": "t-ghost"})
    _print(r)
    print(f"  degraded        : {r.metrics.degraded}")
    print(f"  failure_category: {r.data.failure_category}")
    print(f"  warning         : {r.data.warning}")
    check(r.ok and r.metrics.degraded, "ghost 应 degraded 不报错")
    check(r.data.failure_category == "video" and r.data.cancelled is False,
          "ghost 应 category=video 且未取消")

    print("\n✅ localize_cancel_demo 全部通过")


if __name__ == "__main__":
    main()
