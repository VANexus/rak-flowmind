"""localize_download 技能演示 —— 读任务产物清单（output_paths → 下载 URL）。

运行：PYTHONPATH=<repo>/src conda run -n flowmind python examples/localize_download_demo.py

展示：
1. discover() 自动字段发现
2. happy path：succeeded 任务 2 个产物 → filename/local_path/url（指向
   GET /api/v1/tasks/{task_id}/download?file=<name>，端点阶段 4 实现）
3. degraded：succeeded 但无产物（空结果任务）/ 未完成任务 / 任务不存在
mock 方式：patch 技能模块级 get_task_manager 符号为内存 fake
（FakeManager/FakeStore，本文件内实现），不依赖 PG / MQTT / 外部 VL 服务。
"""

from __future__ import annotations

import flowmind.skills  # noqa: F401  触发 @skill 注册
import flowmind.skills.localize_download as ld
from flowmind.discover import field_names
from flowmind.skill import invoke

_OK_OUTPUTS = [
    "/data/work/t-ok/output_sub.mp4",
    "/data/work/t-ok/trans.srt",
]


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


class _FakeStore:
    """dict 存储的任务行（字段结构与 TaskStore._row_to_dict 对齐）。"""

    def __init__(self) -> None:
        self.tasks: dict[str, dict] = {}

    def add(self, task_id: str, status: str,
            output_paths: list[str] | None) -> dict:
        terminal = status in ("succeeded", "failed", "cancelled", "interrupted")
        rec = {
            "task_id": task_id, "skill_id": "localize_video",
            "args": {"video_path": f"/data/{task_id}.mp4", "target_lang": "th"},
            "status": status, "stage": None,
            "progress": 100.0 if status == "succeeded" else 0.0, "error": None,
            "created_at": "2026-09-04T08:00:00+00:00",
            "started_at": "2026-09-04T08:01:00+00:00" if status != "queued" else None,
            "finished_at": "2026-09-04T08:05:00+00:00" if terminal else None,
            "tenant_id": None, "output_paths": output_paths,
        }
        self.tasks[task_id] = rec
        return rec

    def get_task(self, task_id: str) -> dict | None:
        return self.tasks.get(task_id)


class _FakeManager:
    def __init__(self, store: _FakeStore):
        self.store = store

    def get_task(self, task_id: str) -> dict | None:
        return self.store.get_task(task_id)


def _install(store: _FakeStore) -> _FakeManager:
    """patch localize_download 模块级 get_task_manager 符号。"""
    manager = _FakeManager(store)
    ld.get_task_manager = lambda: manager
    return manager


def main() -> None:
    section("0) discover('localize_download') —— Agent 自查字段")
    for p, names in field_names("localize_download").items():
        print(f"  {p}: {names}")

    section("1) Happy path：succeeded 任务 2 个产物 → 下载 URL")
    store = _FakeStore()
    _install(store)
    store.add("t-ok", "succeeded", output_paths=_OK_OUTPUTS)
    r = invoke("localize_download", {"task_id": "t-ok"})
    print(f"  ok          : {r.ok}")
    print(f"  task_id     : {r.data.task_id}")
    print(f"  status      : {r.data.status}")
    print(f"  files ({len(r.data.files)})：")
    for f in r.data.files:
        print(f"    • {f.filename:20s} local={f.local_path}")
        print(f"      {'':20s} url={f.url}")
    print(f"  推理        : {r.reasoning[0].conclusion}")

    check(r.ok, "succeeded 有产物应 ok")
    check(len(r.data.files) == 2, "应列出 2 个产物")
    f0 = r.data.files[0]
    check(f0.filename == "output_sub.mp4"
          and f0.local_path == "/data/work/t-ok/output_sub.mp4",
          "filename/local_path 应取自 output_paths")
    check(f0.url == "/api/v1/tasks/t-ok/download?file=output_sub.mp4",
          "URL 应指向阶段 4 的下载端点")
    check(r.data.files[1].filename == "trans.srt", "第二产物应为 trans.srt")
    check(r.data.degraded is False and r.data.failure_category is None,
          "happy path 不 degraded")

    section("2) degraded：succeeded 但无产物（空结果任务，如无人声视频）")
    store.add("t-empty", "succeeded", output_paths=None)
    r = invoke("localize_download", {"task_id": "t-empty"})
    print(f"  ok          : {r.ok}（信封 ok=True + degraded=True）")
    print(f"  degraded    : {r.data.degraded}")
    print(f"  files       : {len(r.data.files)}")
    print(f"  warning     : {r.data.warning}")
    check(r.ok and r.data.degraded, "空结果任务应 degraded 不报错")
    check(r.data.files == [] and "空结果任务" in (r.data.warning or ""),
          "空产物应 files=[] + warning 说明")

    section("3) degraded：未完成任务（running）→ 无产物可下载")
    store.add("t-run", "running", output_paths=None)
    r = invoke("localize_download", {"task_id": "t-run"})
    print(f"  degraded         : {r.metrics.degraded}")
    print(f"  failure_category : {r.data.failure_category}（video）")
    print(f"  retriable        : {r.data.retriable}")
    print(f"  warning          : {r.data.warning}")
    check(r.ok and r.metrics.degraded, "未完成任务应 degraded 不报错")
    check(r.data.failure_category == "video" and "running" in (r.data.warning or ""),
          "未完成应 category=video + 状态说明")

    section("4) degraded：任务不存在（ghost）→ video（资源缺失）")
    r = invoke("localize_download", {"task_id": "t-ghost"})
    print(f"  degraded         : {r.metrics.degraded}")
    print(f"  failure_category : {r.data.failure_category}")
    print(f"  retriable        : {r.data.retriable}")
    check(r.ok and r.metrics.degraded and r.data.failure_category == "video",
          "ghost 应 degraded + video")

    print("\n✅ localize_download_demo 全部通过")


if __name__ == "__main__":
    main()
