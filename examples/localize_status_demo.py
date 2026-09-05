"""localize_status 技能演示 —— 批量查询本地任务引擎的任务状态 + 推理链。

运行：PYTHONPATH=<repo>/src conda run -n flowmind python examples/localize_status_demo.py

展示：
1. discover() 自动字段发现
2. happy path：4 个任务（succeeded / stalled running / failed / queued）+ 并发查询
3. per-task 不存在 → 标 not_found（查询侧合成终态，partial success 语义）
4. error：store 读失败 → ok=False INTERNAL（错误永不静默）

mock 方式：patch 技能模块级 get_task_manager 符号为内存 fake
（FakeManager/FakeStore，本文件内实现），不依赖 PG / MQTT / 外部 VL 服务。
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import flowmind.skills  # noqa: F401  触发 @skill 注册
import flowmind.skills.localize_status as ls
from flowmind.discover import field_names
from flowmind.skill import invoke


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _FakeStore:
    """dict 存储的任务行（字段结构与 TaskStore._row_to_dict 对齐）。"""

    def __init__(self) -> None:
        self.tasks: dict[str, dict] = {}
        self.fail_reads = False  # 段4：模拟 PG 不可达

    def add(self, task_id: str, **fields) -> dict:
        rec = {
            "task_id": task_id, "skill_id": "localize_video",
            "args": fields.pop("args", {}), "status": fields.pop("status"),
            "stage": fields.pop("stage", None),
            "progress": fields.pop("progress", 0.0),
            "error": fields.pop("error", None),
            "created_at": fields.pop("created_at", _now()),
            "started_at": fields.pop("started_at", None),
            "finished_at": fields.pop("finished_at", None),
            "tenant_id": None, "output_paths": fields.pop("output_paths", None),
        }
        rec.update(fields)
        self.tasks[task_id] = rec
        return rec

    def get_task(self, task_id: str) -> dict | None:
        if self.fail_reads:
            raise RuntimeError("connection refused")  # 模拟 PG 不可达
        time.sleep(0.02)  # 模拟 I/O 延迟（并发收益可见）
        return self.tasks.get(task_id)


class _FakeManager:
    def __init__(self, store: _FakeStore):
        self.store = store

    def get_task(self, task_id: str) -> dict | None:
        return self.store.get_task(task_id)


def _install(store: _FakeStore) -> _FakeManager:
    """patch localize_status 模块级 get_task_manager 符号。"""
    manager = _FakeManager(store)
    ls.get_task_manager = lambda: manager
    return manager


def _print_report(r) -> None:
    print(f"  ok         : {r.ok}")
    d = r.data
    print(f"  汇总       : 成功 {d.succeeded} / 失败 {d.failed} / 运行 {d.running}"
          f" / 排队 {d.queued} / 卡住 {d.stalled} / 未找到 {d.not_found}")
    print(f"  all_terminal = {d.all_terminal}")
    for t in d.tasks:
        flags = []
        if t.is_stalled:
            flags.append("stalled")
        if t.is_terminal:
            flags.append("terminal")
        print(f"    • {t.task_id}: {t.status:10s} stage={t.stage!r:10s}"
              f" progress={t.progress:5.1f}"
              + (f"  [{','.join(flags)}]" if flags else ""))
    print(f"  推理：{r.reasoning[0].conclusion}")


def main() -> None:
    section("0) discover('localize_status') —— Agent 自查字段")
    for p, names in field_names("localize_status").items():
        print(f"  {p}: {names}")

    section("1) Happy path：4 个任务（含 1 个卡住超 600s），并发查询")
    now = datetime.now(timezone.utc)
    store = _FakeStore()
    _install(store)
    store.add(
        "t-done", status="succeeded", stage=None, progress=100.0,
        started_at=(now - timedelta(minutes=15)).isoformat(),
        finished_at=(now - timedelta(minutes=12)).isoformat(),
        args={"video_path": "/data/promo-a.mp4", "target_lang": "th"},
        output_paths=["/data/work/t-done/output_sub.mp4", "/data/work/t-done/trans.srt"],
    )
    store.add(
        "t-stall", status="running", stage="asr", progress=45.0,
        started_at=(now - timedelta(minutes=15)).isoformat(),  # 900s > 600s → stalled
        args={"video_path": "/data/promo-b.mp4", "target_lang": "en"},
    )
    store.add(
        "t-fail", status="failed", stage="tts", progress=70.0,
        error="TTS 合成失败：qwen-audio-3.0-tts-flash 超时",
        started_at=(now - timedelta(minutes=20)).isoformat(),
        finished_at=(now - timedelta(minutes=19)).isoformat(),
        args={"video_path": "/data/promo-c.mp4", "target_lang": "ja"},
    )
    store.add(
        "t-queue", status="queued", progress=0.0,
        args={"video_path": "/data/promo-d.mp4", "target_lang": "ko"},
    )

    t0 = time.time()
    r = invoke("localize_status", {"task_ids": ["t-done", "t-stall", "t-fail", "t-queue"]})
    elapsed_ms = (time.time() - t0) * 1000
    print(f"  用时       : {elapsed_ms:.0f}ms（4 个并发 × 20ms 模拟 I/O）")
    _print_report(r)

    check(r.ok, "happy path 应 ok")
    d = r.data
    check((d.succeeded, d.failed, d.running, d.queued) == (1, 1, 1, 1), "计数应 1/1/1/1")
    check(d.stalled == 1, "15min running 应标 stalled")
    stall = next(t for t in d.tasks if t.task_id == "t-stall")
    check(stall.is_stalled and not stall.is_terminal and stall.stage == "asr",
          "t-stall 应 stalled + 非终态 + stage=asr")
    done = next(t for t in d.tasks if t.task_id == "t-done")
    check(done.is_terminal and len(done.output_paths) == 2
          and done.source_video == "/data/promo-a.mp4" and done.target_lang == "th",
          "t-done 应终态 + 产物 + args 透出")
    check(d.all_terminal is False, "running+queued 未完，all_terminal 应 False")
    check(len(d.tasks) == 4, "4 个任务报告")

    section("2) per-task 不存在 → 标 not_found（partial success，不影响其他任务）")
    r = invoke("localize_status", {"task_ids": ["t-done", "t-ghost", "t-fail"]})
    print(f"  ok               : {r.ok}")
    d = r.data
    print(f"  tasks[1].status  : {d.tasks[1].status}（not_found）")
    print(f"  tasks[1].error   : {d.tasks[1].error}")
    print(f"  tasks[1].is_terminal : {d.tasks[1].is_terminal}（查询侧合成终态）")
    print(f"  not_found 计数   : {d.not_found}")
    check(r.ok, "not_found 走 partial success，信封仍 ok")
    check(d.tasks[1].status == "not_found" and d.tasks[1].is_terminal,
          "ghost 应标 not_found 终态")
    check(d.not_found == 1 and d.succeeded == 1 and d.failed == 1, "计数应 1/1/1")

    section("3) error：store 读失败（PG 不可达）→ ok=False INTERNAL")
    store.fail_reads = True
    r = invoke("localize_status", {"task_ids": ["t-done"]})
    print(f"  ok          : {r.ok}")
    print(f"  error.code  : {r.error.code if r.error else None}")
    print(f"  error.msg   : {r.error.message if r.error else None}")
    check(r.ok is False, "store 读失败应 ok=False")
    check(r.error is not None and r.error.code == "INTERNAL",
          "store 异常经 invoke 兜底为 INTERNAL（错误永不静默）")
    store.fail_reads = False

    print("\n✅ localize_status_demo 全部通过")


if __name__ == "__main__":
    main()
