"""dsp_cart_sticker 技能演示 —— 小黄车贴片（**条件节点**，仅带货形态生成）。

运行：PYTHONPATH=<repo>/src conda run -n flowmind python examples/dsp_cart_sticker_demo.py

展示：
1. discover() 自动字段发现
2. happy path：带货形态 + 商品锚点 → 贴片位（L1 人工确认）
3. 纯内容形态 → 如实降级（该形态编排里本就没有这个节点）
4. error：带货但未给商品锚点 → degraded；贴片区间越界 → degraded

无需 mock：本技能只出参数（贴片位 + 锚点），不写对象存储。
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
        "mode": "commerce", "platform": "douyin",
        "cachePrefix": "dsp/scratch/t1", "outputPrefix": "dsp/output/t1",
        "runId": "run-demo-6", "stepId": "cart", "idempotencyKey": "run-demo-6:cart",
        "shots": [], "materials": [],
        "productRef": "https://item.example.com/12345",
        "deps": {"timeline": {"edl": {"version": 1, "totalMs": 15000}}},
    }
    base.update(over)
    return base


def main() -> None:
    section("0) discover('cart_sticker') —— Agent 自查字段")
    for path, names in field_names("cart_sticker").items():
        print(f"  {path}: {names}")

    section("1) Happy path：带货形态 + 商品锚点 → 贴片位")
    r = invoke("cart_sticker", _ctx())
    check(r.ok, "happy path 应 ok")
    d = r.data
    print(f"  贴片数 : {len(d.stickers)}")
    for s in d.stickers:
        print(f"    • {s.atMs}–{s.atMs + s.durationMs}ms {s.position} ref={s.productRef.model_dump()}")
    check(len(d.stickers) == 1, "一枚贴片")
    check(d.stickers[0].productRef.kind == "manual", "手工档锚点")
    check(d.stickers[0].durationMs == 9000, "默认时长 = min(9s, 余量)")
    print(f"  推理   : {r.reasoning[0].risk_note}")

    section("2) 纯内容形态 → 该形态编排里没有本节点（如实降级，非失败）")
    r = invoke("cart_sticker", _ctx(mode="content"))
    print(f"  metrics.degraded   : {r.metrics.degraded}")
    print(f"  degradation_reason : {r.metrics.degradation_reason}")
    check(r.metrics.degradation_reason == "not-commerce-mode", "纯内容形态应如实标注")

    section("3) error：带货但未给商品锚点 → degraded")
    r = invoke("cart_sticker", _ctx(productRef="  "))
    print(f"  degradation_reason : {r.metrics.degradation_reason}")
    check(r.metrics.degradation_reason == "missing-product-ref", "缺锚点应如实降级")

    section("4) error：贴片区间越界（起始 ≥ 时间线末尾）→ degraded")
    r = invoke("cart_sticker", _ctx(atMs=20000))
    print(f"  degradation_reason : {r.metrics.degradation_reason}")
    print(f"  warnings           : {r.data.warnings}")
    check(r.metrics.degradation_reason == "sticker-out-of-timeline", "越界应如实降级")

    print("\n✅ dsp_cart_sticker_demo 全部通过")


if __name__ == "__main__":
    main()
