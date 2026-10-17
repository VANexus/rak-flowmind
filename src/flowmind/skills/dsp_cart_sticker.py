"""cart_sticker 技能：小黄车贴片（**条件节点**）——只出贴片参数，由 render 烧制。

契约 §3.6：**仅 ``commerce``（带货）形态生成**（纯内容形态编排里没有这个节点，
不是降级也不是失败）；依赖 ``timeline``；降级 **degrade**；风险 **L1**（含商品链接
→ 人工确认）。

商品锚点 ``productRef`` 为**手工档**（商品链接 / 商品 ID）；抖音小店自动档属后续。
商品链接是**平台组件、不烧进画面**（契约 §7.2）——本步只给贴片位与锚点参数。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills import _dsp_common as common
from flowmind.skills._dsp_common import DspBaseInput

_VERSION = "0.1.0"

_DEFAULT_DURATION_MS = 9000
_DEFAULT_POSITION = "bottom-right"


class CartStickerInput(DspBaseInput):
    """小黄车贴片入参（公共上下文 + 商品锚点）。"""
    productRef: str = Field(default="", description="商品链接 / 商品 ID（手工档）")
    atMs: int = Field(default=0, description="贴片起始（ms，须落锁定时间线内）")
    durationMs: int | None = Field(default=None, description="贴片时长（ms）；缺省 = min(9s, 余量)")
    position: str = Field(default=_DEFAULT_POSITION)


class StickerProductRef(BaseModel):
    """商品锚点（手工档）。"""
    kind: str = "manual"
    value: str


class CartSticker(BaseModel):
    """单枚贴片。"""
    atMs: int
    durationMs: int
    position: str
    productRef: StickerProductRef


class CartStickerData(BaseModel):
    """出参：贴片列表。"""
    stickers: list[CartSticker] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@skill(id="cart_sticker", name="小黄车贴片", version=_VERSION,
       category="抖音成片", tags=["dsp", "cart"])
def cart_sticker(inp: CartStickerInput) -> SkillOutput[CartStickerData]:
    """带货形态的贴片参数（含商品锚点 → L1 人工确认）；纯内容形态不含本节点。"""
    if inp.mode != "commerce":
        return SkillOutput(
            data=CartStickerData(),
            reasoning=[common.chain(
                f"小黄车贴片不适用：本单为 {inp.mode} 形态（贴片仅带货形态生成）",
                risk_note="纯内容形态的编排里本就没有该节点；此为调用侧误用，已如实降级。",
            )],
            confidence=1.0, degraded=True, degradation_reason="not-commerce-mode",
        )

    if not inp.productRef.strip():
        return SkillOutput(
            data=CartStickerData(),
            reasoning=[common.chain(
                "小黄车贴片未生成：带货形态但未给商品锚点（productRef）",
                risk_note="请提供商品链接 / 商品 ID（手工档）；抖音小店自动档属后续。",
            )],
            confidence=0.0, degraded=True, degradation_reason="missing-product-ref",
        )

    edl = common.dep_edl(inp.deps)
    total_ms = edl.totalMs if edl else 0
    warnings: list[str] = []
    if edl is None:
        warnings.append("缺少锁定时间线（deps.timeline.edl），贴片时长未按时间线裁剪")

    at_ms = max(0, inp.atMs)
    span = (total_ms - at_ms) if total_ms > 0 else _DEFAULT_DURATION_MS
    duration = inp.durationMs if inp.durationMs is not None else min(_DEFAULT_DURATION_MS, max(span, 0))
    if duration <= 0:
        warnings.append(f"贴片起始 {at_ms}ms 已到/超过时间线末尾（{total_ms}ms），未生成贴片")
        return SkillOutput(
            data=CartStickerData(warnings=warnings),
            reasoning=[common.chain(
                "小黄车贴片未生成：贴片区间落在锁定时间线之外",
                risk_note="贴片须落 [0, totalMs]；请调整 atMs 后重试。",
            )],
            confidence=0.0, degraded=True, degradation_reason="sticker-out-of-timeline",
        )

    if edl is not None:
        warnings.extend(common.check_in_timeline(
            total_ms, [(at_ms, "贴片起始"), (at_ms + duration, "贴片结束")]))

    sticker = CartSticker(
        atMs=at_ms, durationMs=duration, position=inp.position,
        productRef=StickerProductRef(kind="manual", value=inp.productRef.strip()),
    )
    return SkillOutput(
        data=CartStickerData(stickers=[sticker], warnings=warnings),
        reasoning=[common.chain(
            f"小黄车贴片 {at_ms}–{at_ms + duration}ms（{inp.position}）",
            causal_analysis=f"带货形态；商品锚点手工档（{inp.productRef.strip()}）；贴片位须落锁定时间线内。",
            risk_note="商品链接属 L1 人工确认项；链接是平台组件，不烧进画面（只出贴片位与锚点）。",
        )],
        confidence=1.0, sample_size=1,
    )
