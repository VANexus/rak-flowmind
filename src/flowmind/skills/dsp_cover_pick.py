"""cover_pick 技能：封面取帧——只出参数（封面临界位），由 render 烧制。

契约 §3.5：依赖 ``timeline``；降级 **degrade**；风险 **L1**（封面标题需人工确认），
未确认即如实标 ``degraded``（用默认封面图），**不阻塞成片**。

图片源简化（契约 §3.5 · 2026-10-12）：素材是图片，「封面」不必再「取锁定时间线首帧」
——**封面 = 指定的那张图**（默认第一镜的图，或用户指定），本步退化为**选图 + 标题位**。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills import _dsp_common as common
from flowmind.skills._dsp_common import DspBaseInput

_VERSION = "0.1.0"

_ASPECT = "9:16"


class CoverPickInput(DspBaseInput):
    """封面入参（公共上下文 + 可选用户指定）。"""
    coverKey: str | None = Field(default=None, description="用户指定的封面图 key；缺省 = 第一镜的图")
    title: str | None = Field(default=None, description="封面标题；未给即视为未确认（L1）")


class TitleSlot(BaseModel):
    """封面标题位（L1 人工确认）。"""
    text: str = ""
    needConfirm: bool = True


class CoverPickData(BaseModel):
    """出参：封面图 key + 取位 + 标题位。"""
    coverKey: str = ""
    atMs: int = 0
    aspect: str = _ASPECT
    titleSlot: TitleSlot = Field(default_factory=TitleSlot)
    warnings: list[str] = Field(default_factory=list)


def _resolve_cover(inp: CoverPickInput, edl: common.DspEdl | None) -> tuple[str, list[str]]:
    """按「用户指定 → 第一镜的图（EDL）→ 第一镜指名 → 素材池首图」解析封面图。"""
    notes: list[str] = []
    if inp.coverKey:
        return inp.coverKey, notes
    if edl is not None:
        for shot in edl.shots:
            if shot.imageSourceKey:
                return shot.imageSourceKey, notes
        notes.append("锁定时间线里各镜均未选到图，封面回落到素材池首图/第一镜指名")
    for shot in inp.shots:
        if shot.asset_key:  # ADR-077：指名图取 image.assetKey（旧 clipKeys 由 lifter 提升）
            return shot.asset_key, notes
    if inp.materials:
        return inp.materials[0].key, notes
    return "", notes


@skill(id="cover_pick", name="封面取帧", version=_VERSION,
       category="抖音成片", tags=["dsp", "cover"])
def cover_pick(inp: CoverPickInput) -> SkillOutput[CoverPickData]:
    """定封面图（默认第一镜）+ 标题位；标题未确认即如实降级，不阻塞成片。"""
    edl = common.dep_edl(inp.deps)
    cover_key, warnings = _resolve_cover(inp, edl)
    title_text = inp.title or (inp.shots[0].copy_text if inp.shots else "")
    slot = TitleSlot(text=title_text, needConfirm=True)

    if not cover_key:
        return SkillOutput(
            data=CoverPickData(coverKey="", atMs=0, aspect=_ASPECT, titleSlot=slot, warnings=warnings),
            reasoning=[common.chain(
                "封面未定：时间线与素材池均无可用的图（已如实降级）",
                risk_note="素材池为空或各镜未选到图；请补齐素材后重跑，或显式传入 coverKey。",
            )],
            confidence=0.0, degraded=True,
            degradation_reason="no-cover-image",
        )

    unconfirmed = inp.title is None
    reason = None
    conclusion = f"封面定为 {cover_key}（9:16，取位 0ms）"
    if unconfirmed:
        reason = "cover-title-unconfirmed"
        conclusion += "；标题未人工确认（用默认标题位）"

    return SkillOutput(
        data=CoverPickData(coverKey=cover_key, atMs=0, aspect=_ASPECT, titleSlot=slot, warnings=warnings),
        reasoning=[common.chain(
            conclusion,
            causal_analysis="素材为图片，封面 = 指定的那张图（默认第一镜），不再做帧导出。",
            risk_note=("封面标题属 L1 人工确认项；未确认不阻塞成片，确认后可重跑本步。"
                       if unconfirmed else "封面标题已确认。"),
        )],
        confidence=1.0 if not unconfirmed else 0.7,
        sample_size=1,
        degraded=unconfirmed,
        degradation_reason=reason,
    )
