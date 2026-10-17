"""dsp_bgm_match 技能演示 —— 背景音乐（**占位降级**，契约 §3.3 明确规定）。

运行：PYTHONPATH=<repo>/src conda run -n flowmind python examples/dsp_bgm_match_demo.py

展示：
1. discover() 自动字段发现
2. 契约现状：选曲能力 / 接口均未给 → **一律 degraded + 「BGM 选曲能力未接通」**，成片空轨继续
3. 无锁定时间线时同样如实降级（本节点属 degrade，不阻塞成片）

无需任何 mock：本技能当前不做外呼（拿到选曲接口后在 ``_pick_track`` 一处接通即可）。
"""

from __future__ import annotations

import flowmind.skills  # noqa: F401  触发 @skill 注册
from flowmind.discover import field_names
from flowmind.skill import invoke


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _ctx(deps: dict | None = None) -> dict:
    return {
        "mode": "content", "platform": "douyin",
        "cachePrefix": "dsp/scratch/t1", "outputPrefix": "dsp/output/t1",
        "runId": "run-demo-3", "stepId": "bgm", "idempotencyKey": "run-demo-3:bgm",
        "shots": [], "materials": [], "mood": "轻快",
        "deps": deps if deps is not None else {"timeline": {"edl": {"version": 1, "totalMs": 9000}}},
    }


def main() -> None:
    section("0) discover('bgm_match') —— Agent 自查字段")
    for path, names in field_names("bgm_match").items():
        print(f"  {path}: {names}")

    section("1) 契约现状：选曲接口未给 → degraded + 原因（成片空轨继续）")
    r = invoke("bgm_match", _ctx())
    d = r.data
    print(f"  ok                 : {r.ok}")
    print(f"  metrics.degraded   : {r.metrics.degraded}")
    print(f"  degradation_reason : {r.metrics.degradation_reason}")
    print(f"  data.bgmKey        : {d.bgmKey!r}（空 = 成片空轨）")
    print(f"  data.durationMs    : {d.durationMs}（取自锁定时间线）")
    print(f"  data.ducking       : {d.ducking.model_dump()}")
    check(r.metrics.degraded is True, "契约 §3.3：未接通期间一律 degraded")
    check(r.metrics.degradation_reason == "BGM 选曲能力未接通", "降级原因须逐字为契约口径")
    check(d.bgmKey == "", "未接通即无 bgmKey（不伪造曲目）")
    check(d.durationMs == 9000, "参考时长应取自 deps.timeline.edl.totalMs")

    section("2) 无锁定时间线：仍如实降级（degrade 节点不阻塞成片）")
    r = invoke("bgm_match", _ctx(deps={}))
    print(f"  metrics.degraded : {r.metrics.degraded}")
    print(f"  data.durationMs  : {r.data.durationMs}（无时间线 → 0，如实）")
    check(r.metrics.degraded is True and r.data.durationMs == 0, "缺时间线时应如实降级且时长为 0")

    print("\n✅ dsp_bgm_match_demo 全部通过")


if __name__ == "__main__":
    main()
