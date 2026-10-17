"""timeline_build 技能：素材对齐 · 切镜收口 · 时间线锁定（**唯一时间线写者**）。

契约 §3.2（**2026-10-13 改判：ADR-069 定时长真源，ADR-070 删模板改自动切镜**）：
把镜头与**图片池**对齐成轨并**锁定** EDL——此后 ``totalMs`` 与各轨入出点不得再改。

**第一段 · 时长真源 = 配音逐句实测**（ADR-069，不是模板常量）：

- ``deps.tts`` 在（配音开着）→ **实测路径**：``_dsp_align`` 把 ``deps.tts.voSegments``（厂商逐句
  时间戳）归属到各镜，每镜时长 = 该镜句子的实测时长之和 ⇒ ``totalMs`` 即音频总长，**画面与配音逐帧对齐**。
  ``durationSource = "measured"``。
- ``deps.tts`` 缺席（配音开关关掉）→ **估算路径**：按 ``shots[].durationMs``（前端按文案字数的估算）
  累加，``durationSource = "estimated"`` 并**如实告警**——没有真值就标注清楚，不假装。

**第二段 · 节奏收口 = 并镜 / 拆镜**（ADR-070）：镜头的「有几个、切在哪」由文案定，但文案没写好的
地方由规则兜底（``_dsp_align.consolidate_shots``）——实测 < ``timeline_shot_min_ms`` 的镜并入相邻镜、
> ``timeline_shot_max_ms`` 的镜在**其内部最近的合法边界**一分为二（**先句边界，句内下沉到词边界**）；
收口后镜数越界只告警不夹取。
**拆镜需要实测 token（句 / 词）边界**，故估算路径只可能并镜（如实说明）。

**第三段 · 转场**（ADR-070）：写 ``segments[i].transition``（**符号名**，语义 = 该镜末尾桥接 i→i+1）。
来源 = 下发镜头的 ``transitionAfter``；**但被拆开的镜，其内部那刀按硬切**（用户设的转场是给整镜末尾的）。
厂商资源 ID 未登记时由 ``render_compose`` 如实回落硬切并告警——本步只忠实记录符号值。

**终版镜头自带文案与运镜**：收口会并/拆镜并重排 id（``s1..sK``），故每镜的 ``copy`` /
``visualPrompt`` / ``motion`` 一并落进 ``edl.shots[]``——下游（I2V 提示词与运镜等）**一律读 EDL**，
不再回查 ``inp.shots``（否则并/拆后必然错配）。

**运镜（ADR-076）**：**不再往 segment 写 ``effect``**。运镜是**逐镜结构化方案**（前端「运镜导演」
**看图**产出，含 type/intensity/direction/target/reason/正负提示词），**原样搬运**到
``edl.shots[].motion``，由 ``image_to_video`` 经引擎适配器并入 prompt——**render 侧不再施加运镜**
（旧 MediaKit ``apply-camera-motion`` 路径已退役，留着就是同一运动施加两次）。

选图口径（契约 §3.2，**不臆造素材**）：
1. 镜头给了**指名的图**（ADR-077：`shots[].image.assetKey`）→ 用它；
2. 没给 → 该镜**留空位**并在 ``notes`` 里如实说明。

**2026-10-15：删掉「按类目从素材池选图」这一档**（cross-dashboard ADR-088 的跨仓部分）。
旧口径的第 2 档是 ``needKind == category``（两字段都来自「素材类目」那套已退役的架构），
删掉后本技能**只认指名图**——上游（智能选图 / 逐镜手工选片）本来就已逐镜给出 ``assetKey``，
这一档在报文里长期是**恒空**（两字段都空 ⇒ 旧代码会「全等命中」，等于没有筛选）。

**四块（ADR-077）**：画面提示词取 `i2v.authorPrompt`、指名图取 `image.assetKey`、运镜方案取
`i2v.motion`（旧扁平字段由 `DspShot._lift_legacy` 提升，故旧 run 的 spec 原样重跑仍能跑）。

出参：``{edl, edlKey, durationSource, notes}``；EDL json 落 ``{cachePrefix}/{runId}/timeline/edl.json``。

降级：**abort**（成片必需步）——存储写不进等真实失败即抛错，由 ``invoke()`` 兜成
``ok=False``，运行器据此中止整单（错误永不静默）。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills import _dsp_align as align
from flowmind.skills import _dsp_common as common
from flowmind.skills import _dsp_storage as storage
from flowmind.skills._dsp_common import DspBaseInput

_VERSION = "0.4.0"

DurationSource = Literal["measured", "estimated"]


class TimelineBuildInput(DspBaseInput):
    """时间线锁定入参（公共上下文已含 ``shots[]`` / ``materials[]``）。"""


class TimelineBuildData(BaseModel):
    """出参：锁定 EDL + 产物 key + **时长来源** + 如实说明。"""
    edl: common.DspEdl
    edlKey: str = Field(..., description="EDL 快照的云存储 key")
    durationSource: DurationSource = Field(
        ..., description='measured = 配音逐句实测；estimated = 前端估算（无配音时的兜底）')
    notes: list[str] = Field(default_factory=list, description="留空位 / 未入池 / 归属推断 / 并拆镜等如实说明")


def _select_image(
    shot_id: str, clip_keys: tuple[str, ...], pool_keys: set[str], avoid: str = "",
) -> tuple[str, list[str]]:
    """为单镜选图，返回 ``(sourceKey, notes)``；没指名 = 空串 + 说明。

    ``avoid`` = 上一镜已用的 key（**仅用于如实提示**：指名图与上一镜相同时画面观感等同没切）。
    """
    notes: list[str] = []
    if clip_keys:
        chosen = clip_keys[0]
        if chosen not in pool_keys:
            notes.append(f"镜头 {shot_id} 指名的图 {chosen} 不在素材池中，仍按指名使用")
        elif avoid and chosen == avoid:
            notes.append(f"镜头 {shot_id} 指名的图与上一镜相同（画面观感等同不切）")
        return chosen, notes

    notes.append(f"镜头 {shot_id} 未指名图，已留空位（选图由上游逐镜给出）")
    return "", notes


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _word_rows(raw: object) -> tuple[align.AlignWord, ...]:
    """``voSegments[].words[]`` → 词级实测 token（形状不对即丢弃该条，不臆造）。"""
    if not isinstance(raw, list):
        return ()
    out: list[align.AlignWord] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        at = _int_or_none(item.get("atMs"))
        dur = _int_or_none(item.get("durationMs"))
        if at is None or dur is None:
            continue
        out.append(align.AlignWord(start_ms=at, end_ms=at + dur, text=str(item.get("text") or "")))
    return tuple(out)


def _sentence_rows(tts: dict[str, Any] | None) -> list[align.AlignSentence]:
    """``deps.tts.voSegments`` → 对齐用的逐句实测轴（含**词级 token**；形状不对即丢弃该条，不臆造）。"""
    if not tts:
        return []
    rows = tts.get("voSegments")
    if not isinstance(rows, list):
        return []
    out: list[align.AlignSentence] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        at = _int_or_none(row.get("atMs"))
        dur = _int_or_none(row.get("durationMs"))
        if at is None or dur is None:
            continue
        out.append(align.AlignSentence(
            start_ms=at, end_ms=at + dur, text=str(row.get("text") or ""),
            words=_word_rows(row.get("words")),
        ))
    return out


def _cut_shots(inp: TimelineBuildInput) -> list[align.CutShot]:
    """下发镜头 → 收口所需的视图（比例兜底的**权重由文案字数自算**，不再有 weight 字段）。

    ADR-077：画面提示词 / 指名图 / 运镜方案一律**从四块读**（`author_prompt` / `asset_key` /
    `motion_plan` 三个便捷属性；旧扁平字段由 `DspShot._lift_legacy` 提升，故老快照照常能跑）。
    """
    return [
        align.CutShot(
            id=s.id,
            copy=s.copy_text,
            visual_prompt=s.author_prompt,
            clip_keys=(s.asset_key,) if s.asset_key else (),
            motion=s.motion_plan,  # ADR-076：运镜方案原样进收口（并镜取主导、拆镜共用）
            transition_after=(s.transitionAfter or None),
        )
        for s in inp.shots
    ]


def _resolve(
    inp: TimelineBuildInput, tts: dict[str, Any] | None, silent_ms: int,
) -> tuple[align.CutResult, int, DurationSource, list[str]]:
    """对齐 + 收口 → ``(终版镜头与落点, 总时长, 时长来源, 如实说明)``。"""
    cfg = load_config().dsp
    notes: list[str] = []
    cut_shots = _cut_shots(inp)
    sentences = _sentence_rows(tts)

    if sentences:
        result = align.align_sentences_to_shots(
            [align.AlignShot(id=s.id, copy=s.copy, weight=float(max(1, align.speakable_len(s.copy))))
             for s in cut_shots],
            sentences,
            silent_shot_ms=silent_ms,
            audio_total_ms=_int_or_none(tts.get("durationMs")) if tts else None,
        )
        notes.extend(result.warnings)
        total_ms = result.total_ms
        source: DurationSource = "measured"
    else:
        # 估算路径：无配音真值，按前端估算时长累加（**如实标注，不假装**）
        reason = "配音开关关闭（deps.tts 缺席）" if not tts else "配音产物无逐句时间戳（voSegments 为空）"
        notes.append(
            f"{reason}：镜头时长按前端估算值 ``shots[].durationMs``（按文案字数算）累加——**非实测**。"
        )
        result = align.AlignResult(
            spans=_estimated_spans(inp.shots),
            total_ms=sum(max(1, s.durationMs) for s in inp.shots),
            warnings=[],
            anchored_shots=0, inferred_shots=0, silent_shots=len(inp.shots),
            mismatched_shots=0, unused_sentences=0, trailing_ms=0, desync_ms=0,
        )
        total_ms = result.total_ms
        source = "estimated"

    cut = align.consolidate_shots(
        cut_shots,
        result.spans,
        sentences,
        min_ms=cfg.timeline_shot_min_ms,
        max_ms=cfg.timeline_shot_max_ms,
        limit_min=cfg.timeline_shot_limit_min,
        limit_max=cfg.timeline_shot_limit_max,
    )
    if not sentences and cut.merged == 0 and cut.split == 0:
        notes.append("估算路径无实测轴 ⇒ **只可能并镜、不能拆镜**（拆镜需要逐句/词实测边界）。")
    notes.extend(cut.warnings)
    return cut, total_ms, source, notes


def _estimated_spans(shots: list[common.DspShot]) -> list[align.ShotSpan]:
    """估算路径的逐镜落点（逐个累加估算时长；无口播真值 ⇒ 一律标 ``silent``/未匹配）。"""
    out: list[align.ShotSpan] = []
    cursor = 0
    for shot in shots:
        dur = max(1, shot.durationMs)
        out.append(align.ShotSpan(shotId=shot.id, atMs=cursor, durationMs=dur,
                                  sentenceCount=0, matched=False, silent=True))
        cursor += dur
    return out


@skill(id="timeline_build", name="素材对齐·切镜收口·时间线锁定", version=_VERSION,
       category="抖音成片", tags=["dsp", "timeline"])
def timeline_build(inp: TimelineBuildInput) -> SkillOutput[TimelineBuildData]:
    """按**配音逐句实测**（或估算兜底）对齐 → **收口**（并/拆镜）→ 锁定 EDL（唯一时间线写者）。

    有配音：每镜时长 = 该镜句子的实测时长之和（``_dsp_align``），总时长 = 音频总长；
    收口后终版镜头的 id 重排为 ``s1..sK``，**文案随镜落进 EDL**（下游不再回查 ``inp.shots``）。
    无配音：按前端估算累加并如实标注 ``durationSource="estimated"``（不能拆镜，只可能并镜）。
    选不到图的镜头留空位并如实写入 notes。
    """
    if not inp.shots:
        raise ValueError("timeline_build 需要 shots[]：无镜头无法锁时间线。")

    cfg = load_config().dsp
    materials = inp.materials
    pool_keys = {m.key for m in materials}

    tts = common.dep(inp.deps, "tts")
    cut, total_ms, duration_source, notes = _resolve(inp, tts, cfg.timeline_silent_shot_ms)

    final_sum = sum(sp.durationMs for sp in cut.spans)
    if final_sum != total_ms:
        # 收口只做并/拆（不改总长），对不上说明上游数据有问题——如实报出，仍以 total_ms 为准
        notes.append(f"收口后各镜时长之和 {final_sum}ms 与总时长 {total_ms}ms 不一致（异常，已按总时长锁定）。")

    # ── 选图（相邻镜优先换图） + 转场 → 逐段 EDL ──
    image_segments: list[common.DspEdlSegment] = []
    edl_shots: list[common.DspEdlShot] = []
    used_prev = ""
    forced_hard_cut = 0
    kept_transitions = 0
    for idx, (shot, span) in enumerate(zip(cut.shots, cut.spans, strict=True)):
        source_key, shot_notes = _select_image(
            shot.id, shot.clip_keys, pool_keys, avoid=used_prev)
        notes.extend(shot_notes)
        used_prev = source_key or used_prev

        # 转场：**被拆开的镜，其内部那刀按硬切**（用户设的转场只属于整镜末尾）
        transition: str | None = None
        if idx < len(cut.shots) - 1:
            is_split_boundary = cut.boundary_split[idx] if idx < len(cut.boundary_split) else False
            if is_split_boundary:
                if shot.transition_after:
                    forced_hard_cut += 1
                transition = None
            else:
                transition = shot.transition_after
                if transition:
                    kept_transitions += 1

        image_segments.append(common.DspEdlSegment(
            atMs=span.atMs,
            durationMs=span.durationMs,
            sourceKey=source_key,
            transition=transition,
        ))
        edl_shots.append(common.DspEdlShot(
            id=shot.id,
            atMs=span.atMs,
            durationMs=span.durationMs,
            imageSourceKey=source_key,
            copy=shot.copy,
            visualPrompt=shot.visual_prompt,
            motion=shot.motion,  # ADR-076：I2V 从这里读运镜（不回查 inp.shots——并/拆会重排 id）
        ))

    if forced_hard_cut:
        notes.append(
            f"{forced_hard_cut} 处转场被改为硬切：该镜被时长规则拆成两半，"
            "拆出来的内部边界按硬切（用户设的转场保留在整镜末尾）。"
        )
    if kept_transitions:
        notes.append(
            f"时间线记录了 {kept_transitions} 处转场（符号名）；"
            "**厂商转场资源 ID 未登记时会由 render 侧如实回落硬切**（绝不编造 ID）。"
        )

    # 音轨只声明位次；sourceKey 由 render_compose 从 deps.tts.audioKey 挂上（契约 §3.2 注）。
    # totalMs 在实测路径下**即音频总长**（末尾静默已并入末镜），故音轨与画面天然对齐。
    vo_track = common.DspEdlTrack(
        kind="vo", segments=[common.DspEdlSegment(atMs=0, durationMs=total_ms, sourceKey="")])

    edl = common.DspEdl(
        version=1,
        totalMs=total_ms,
        tracks=[
            common.DspEdlTrack(kind="image", segments=image_segments),
            vo_track,
        ],
        shots=edl_shots,
    )

    if inp.materialsNote:
        notes.append(inp.materialsNote)

    key = common.edl_key(inp.cachePrefix, inp.runId)
    storage.put_text(key, edl.model_dump_json(indent=2), content_type="application/json")

    gaps = sum(1 for s in image_segments if not s.sourceKey)
    measured = duration_source == "measured"
    source_label = "配音逐句实测" if measured else "前端估算（无配音）"
    # 置信度：留空位的镜头扣分；**估算路径再扣**（时长未经验证，如实降低把握）
    confidence = (1.0 if not gaps else 0.6) * (1.0 if measured else 0.7)
    concluded = len(inp.shots)
    conclusion = (
        f"锁定时间线：{concluded} 镜 → 收口后 {len(cut.shots)} 镜 / 总时长 {total_ms}ms"
        f"（时长来源：{source_label}）"
        + (f"，{gaps} 镜未选到图（已留空位）" if gaps else "，各镜均已选到图")
    )
    return SkillOutput(
        data=TimelineBuildData(edl=edl, edlKey=key, durationSource=duration_source, notes=notes),
        reasoning=[common.chain(
            conclusion,
            causal_analysis=(
                f"时长来源 = {source_label}："
                + (
                    f"按 deps.tts.voSegments（{len(_sentence_rows(tts))} 句）归属到 {concluded} 镜，"
                    f"轨严格连续、末尾静默并入末镜；"
                    if measured else
                    "无配音真值，按 shots[].durationMs 累加（已如实标注）；"
                )
                + f"收口（并 {cut.merged} / 拆 {cut.split}，切点都在实测 token 边界上）→ {len(cut.shots)} 镜；"
                + f"选图只看镜头指名的图（clipKeys → image.assetKey），没指名即留空位；素材池 {len(materials)} 张。"
            ),
            risk_note=(
                "时间线一经锁定不再变动，字幕/渲染均读它；"
                "留空位的镜头成片对应时段为空白，请补齐素材或插图后再跑。"
                if gaps else
                "时间线已锁定，后续步骤（字幕/渲染）以本 EDL 为唯一时间基准。"
                + ("" if measured else "（**注意：本次时长未经配音实测**，只有估算值。）")
            ),
            confidence=confidence,
        )],
        confidence=confidence,
        sample_size=len(cut.shots),
    )
