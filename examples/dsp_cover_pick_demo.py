"""dsp_cover_pick 技能演示 —— 封面取帧（图片源：封面 = 指定的那张图 + 标题位）。

运行：PYTHONPATH=<repo>/src conda run -n flowmind python examples/dsp_cover_pick_demo.py

展示：
1. discover() 自动字段发现
2. happy path：默认取第一镜的图；**标题未确认 → 如实 degraded（不阻塞成片）**
3. 给定标题 → 不再 degraded
4. error：时间线与素材池均无图 → degraded + 原因

无需 mock：本技能只出参数（选图 + 标题位），不写对象存储。
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


def _ctx(**over) -> dict:
    base = {
        "mode": "content", "platform": "douyin",
        "cachePrefix": "dsp/scratch/t1", "outputPrefix": "dsp/output/t1",
        "runId": "run-demo-5", "stepId": "cover", "idempotencyKey": "run-demo-5:cover",
        "shots": [{"id": "s1", "durationMs": 3000, "copy": "开门的瞬间"}],
        # 素材池条目（2026-10-15 起**没有** category 字段——类目维度已退役，身份改判为来源）
        "materials": [{"kind": "image", "key": "dsp/scratch/t1/门店/门头1.jpg"}],
        "deps": {"timeline": {"edl": {"version": 1, "totalMs": 9000, "shots": [
            {"id": "s1", "atMs": 0, "durationMs": 3000,
             "imageSourceKey": "dsp/scratch/t1/门店/门头1.jpg"},
        ]}}},
    }
    base.update(over)
    return base


def main() -> None:
    section("0) discover('cover_pick') —— Agent 自查字段")
    for path, names in field_names("cover_pick").items():
        print(f"  {path}: {names}")

    section("1) Happy path：默认取第一镜的图；标题未确认 → 如实 degraded")
    r = invoke("cover_pick", _ctx())
    d = r.data
    print(f"  coverKey           : {d.coverKey}")
    print(f"  atMs / aspect      : {d.atMs} / {d.aspect}")
    print(f"  titleSlot          : {d.titleSlot.model_dump()}")
    print(f"  metrics.degraded   : {r.metrics.degraded}（标题属 L1 人工确认）")
    print(f"  degradation_reason : {r.metrics.degradation_reason}")
    check(d.coverKey == "dsp/scratch/t1/门店/门头1.jpg", "默认封面 = 第一镜的图")
    check(d.aspect == "9:16" and d.titleSlot.needConfirm is True, "9:16 + 标题位待确认")
    check(r.metrics.degraded is True and r.metrics.degradation_reason == "cover-title-unconfirmed",
          "标题未确认 → degraded，但不阻塞成片")

    section("2) 给定标题 → 不再降级")
    r = invoke("cover_pick", _ctx(title="保温杯 · 12 小时"))
    print(f"  metrics.degraded : {r.metrics.degraded}")
    print(f"  titleSlot.text   : {r.data.titleSlot.text}")
    check(r.metrics.degraded is False, "标题已确认即不降级")

    section("3) error：时间线与素材池均无图 → degraded + 原因")
    r = invoke("cover_pick", _ctx(shots=[], materials=[], deps={}))
    print(f"  metrics.degraded   : {r.metrics.degraded}")
    print(f"  degradation_reason : {r.metrics.degradation_reason}")
    check(r.metrics.degraded is True and r.data.coverKey == "", "无图即不产封面（不臆造）")

    print("\n✅ dsp_cover_pick_demo 全部通过")


if __name__ == "__main__":
    main()
