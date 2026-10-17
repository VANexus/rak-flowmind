"""render_compose 技能：合成渲染（火山 AI MediaKit）——**唯一渲染、DAG 汇点**。

契约 §3.7：依赖 ``tts`` / ``timeline`` / ``bgm`` / ``subtitle`` / ``cover`` / ``cart``
（**缺席者按裁剪后的依赖表传入**）；降级 **abort**；风险 **L1**。

职责：按 EDL 铺 **主轨（图生视频片段 ``video`` / 静图 ``image``）+ 音轨**，把字幕 / 贴片 /
封面位**一次性烧制**，成片 mp4 落**成片目录**（契约 §2.4：``{outputPrefix}/{runId}/final.mp4``）。
**运镜不在这里做**（ADR-076）。

关键契约点：
- §7.3② **成片必须转存**：MediaKit 的 ``result.video_url`` 是 **24h 签名 URL**，
  **不得**直接当 ``storedKey``；须落 ``{outputPrefix}/{runId}/final.mp4`` 再回 key。
- **运镜（ADR-076 改判）**：运镜已由 ``image_to_video`` 经引擎适配器并入 prompt **随生成落地**；
  本步**不再施加**（旧的 MediaKit ``apply-camera-motion`` 事后 pan-zoom 已退役——重复施加会叠加两次运动）。
- §7.3① 素材走**本地上传**：先把每个 key 的字节直传 MediaKit，再用 ``mediakit://{file_id}``
  引用（不再签预签名 URL）——本机 / 内网的对象存储端点厂商拉不到，预签名那条在 dev 必失败
  （2026-10-17 真跑：``render_compose`` 失败于 ``DownloadFileError``）。
- §7.3④ AIGC 标识：``extra_param.upload.aigc_meta`` 由 MediaKit 客户端注入。

降级 **abort**：失败即抛错（``invoke()`` 兜成 ``ok=False``），运行器据此中止整单，
成片字段保持空——**绝不伪造成片**。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills import _dsp_common as common
from flowmind.skills import _dsp_mediakit as mediakit
from flowmind.skills import _dsp_storage as storage
from flowmind.skills._dsp_common import DspBaseInput

_VERSION = "0.1.0"


class RenderComposeInput(DspBaseInput):
    """渲染入参（公共上下文 + 可选字幕样式）。"""
    subtitleFontSize: int = Field(default=48)


class RenderComposeData(BaseModel):
    """出参：成片 key + 时长 + 分辨率 + **逐镜剪辑记录**（本仓就取 ``storedKey`` 当成品）。"""
    storedKey: str
    durationMs: int
    width: int
    height: int
    editSegments: list[common.DspEditSegment] = Field(
        default_factory=list,
        description="**逐镜剪辑记录**（ADR-080 阶段 5）：该镜在成片时间轴上的起点 / 剪辑时长 / 用哪个视频或"
                    "静图 / 覆盖的音轨 / **实际施加**的转场（null = 硬切）/ 裁切实况 / 降级模式。"
                    "**只报实况**——镜头上的 `transitionAfter` 是请求，未登记厂商资源 ID 的转场在此如实为 null。"
                    "前端落库后在成片任务的「剪辑记录」区逐镜展示")
    note: str = ""
    warnings: list[str] = Field(default_factory=list)


def _media_ref_factory(cfg, warnings: list[str]):  # noqa: ANN001, ANN202  两个返回值的闭包工厂
    """素材 key → MediaKit 可引用地址（**本地上传 + ``mediakit://``**，契约 §7.3①）。

    为什么不再用预签名 URL：``source`` 的四协议里 ``http(s)://`` 只在**对象存储公网可达**时成立，
    而本机 / 内网端点（``127.0.0.1:9000``）厂商根本拉不到——真跑里 ``render_compose`` 正是失败在
    ``DownloadFileError``。本地上传把**字节直接**送进 MediaKit，素材与公网可达性彻底解耦。

    同一 key 在**一次渲染里只传一次**（同一张静图会被多镜引用）。空 key 返回空串（调用方跳过）。
    """
    cache: dict[str, str] = {}
    uploaded = 0

    def ref(key: str) -> str:
        nonlocal uploaded
        if not key:
            return ""
        hit = cache.get(key)
        if hit is not None:
            return hit
        data = storage.get_bytes(key)
        cache[key] = mediakit.upload_media(cfg, data, storage.content_type_for(key))
        uploaded += 1
        return cache[key]

    def report() -> None:
        """把「这一轮到底传了什么」如实记进 warnings（不静默）。"""
        if uploaded:
            warnings.append(
                f"素材已上传 MediaKit：{uploaded} 个（``mediakit://`` 引用，不依赖对象存储公网可达）"
            )

    return ref, report


def _split_stickers(raw: object) -> tuple[list[dict], list[str]]:
    """拆贴片：带图片 source 的可烧制；纯平台组件（商品链接）不烧进画面（§7.2）。"""
    burnable: list[dict] = []
    notes: list[str] = []
    if isinstance(raw, list):
        for sticker in raw:
            if not isinstance(sticker, dict):
                continue
            if sticker.get("source"):
                burnable.append(sticker)
            else:
                ref = (sticker.get("productRef") or {}).get("value", "")
                notes.append(f"贴片（商品锚点 {ref or '未给'}）为平台组件，未烧进画面")
    return burnable, notes


def _edit_segments(
    edl: common.DspEdl,
    clip_by_shot: dict[str, dict],
    *,
    audio_key: str,
    bgm_key: str,
    bgm_requested: bool,
    transition_region: str = "cn-north",
) -> list[common.DspEditSegment]:
    """EDL + 逐镜片段 → **逐镜剪辑记录**（ADR-080 阶段 5）。

    **只报实况，不报请求**（ADR-077 的核心纪律）：镜头上的 ``transitionAfter`` 是**请求**，
    ``edit.transition`` 是**真加进轨的**——未登记厂商资源 ID 的转场在渲染里已回落硬切，
    故这里如实为 ``None``（硬切），并在 ``note`` 里写明原因。

    与 ``build_tracks`` **同一索引口径**（EDL image 轨 ``segments[i]`` ↔ ``edl.shots[i]``），
    故两者不会错位。无片段 ⇒ ``fallback_still``；连图都没有 ⇒ ``missing_clip``（该镜没进片子，
    如实落一条而不是从记录里消失）。
    """
    image_tracks = [t for t in edl.tracks if t.kind == "image"]
    if not image_tracks:
        return []
    segs: list[common.DspEditSegment] = []
    for idx, seg in enumerate(image_tracks[0].segments):
        shot = edl.shots[idx] if idx < len(edl.shots) else None
        shot_id = shot.id if shot is not None else f"s{idx + 1}"
        clip = clip_by_shot.get(shot_id) or {}
        video_key = str(clip.get("key") or "")
        notes: list[str] = []

        # 转场：符号名 → 厂商资源 ID。未登记 / 只支持交叠 ⇒ 渲染侧已回落硬切（见 transition_filters）
        requested = (seg.transition or "").strip()
        entry = mediakit.resolve_transition(requested, transition_region)
        applied: str | None = None
        if requested and entry and str(entry.get("mode") or "") == "non-overlap":
            applied = requested
        elif requested:
            notes.append(f"转场 {requested} 未登记厂商资源 ID，实况**硬切**")

        if video_key:
            degrade = ""
        elif seg.sourceKey:
            degrade = "fallback_still"
            notes.append("无图生视频片段，**回落静图**")
        else:
            degrade = "missing_clip"
            notes.append("既无片段也无静图（**空段**，该镜未进片子）")

        raw_trim = clip.get("trim")
        trim = common.DspEditTrim()
        if isinstance(raw_trim, dict):
            try:
                trim = common.DspEditTrim.model_validate(raw_trim)
            except Exception:  # noqa: BLE001  形状不对 ⇒ 按「未实测」如实归零（不猜）
                notes.append("裁切实况形状不合法——按未实测（全 0）记录")
        if trim.shortfallMs > 0:
            notes.append(f"源片段比镜头窗口短 {trim.shortfallMs}ms，已按**末帧定格**补齐")
        elif trim.sourceMs > seg.durationMs:
            notes.append(f"源片段 {trim.sourceMs}ms 超窗口，已裁掉 {trim.sourceMs - seg.durationMs}ms")

        if bgm_requested and not bgm_key:
            notes.append("无 BGM（**选曲能力未接通，恒降级**）")

        segs.append(common.DspEditSegment(
            shotId=shot_id, atMs=seg.atMs, durationMs=seg.durationMs,
            videoKey=video_key, imageKey=seg.sourceKey,
            voKey=audio_key, bgmKey=bgm_key,
            transition=applied, trim=trim, degradeMode=degrade,
            note="；".join(notes),
        ))
    return segs


# **运镜施加已退役**（ADR-076）：旧路径 `apply-camera-motion` 是对**已生成片段**做事后 pan-zoom，
# 而运镜现已由「运镜导演」产出方案、经适配器并入 I2V prompt **随生成一起落地**——本侧再施加就是
# **同一运动施加两次**（ADR-073 当初正是为此才禁止把运镜写进 prompt）。故整段删除，不留死代码。
# MediaKit 的 `apply_camera_motion` / `MOTION_STYLES` / `EFFECT_MOTION_STYLE` / `motion_style_for`
# 一并退役（见 `_dsp_mediakit.py`）；将来若要用引擎**参数级**运镜，落点是 `_dsp_video_gen.MOTION_ENGINE`。


@skill(id="render_compose", name="合成渲染", version=_VERSION,
       category="抖音成片", tags=["dsp", "render", "video"])
def render_compose(inp: RenderComposeInput) -> SkillOutput[RenderComposeData]:
    """按 EDL 铺轨并一次性烧制，成片落 ``{outputPrefix}/{runId}/final.mp4`` 并回 key。"""
    cfg = load_config().dsp
    edl = common.dep_edl(inp.deps)
    if edl is None:
        raise ValueError("render_compose 需要 deps.timeline.edl（渲染以锁定 EDL 为准）。")
    if not mediakit.mediakit_configured(cfg):
        raise mediakit.DspMediakitError(
            f"未配置渲染密钥 {cfg.mediakit_key_env}（dsp 成片必需）；"
            "契约 §7.1：MediaKit 密钥由后端经集群 Secret 持有。",
            category="environment",
        )

    warnings: list[str] = []
    # 素材引用：本地上传 MediaKit 后得到 ``mediakit://``（契约 §7.3①；见该工厂的注释）
    ref, report_uploads = _media_ref_factory(cfg, warnings)

    # ── 逐镜素材：``video`` 节点（图生视频）的片段优先，缺片段回落静图 ──
    video_dep = common.dep(inp.deps, "video") or {}
    raw_clips = video_dep.get("clips")
    clips = [c for c in raw_clips if isinstance(c, dict)] if isinstance(raw_clips, list) else []
    clip_by_shot = {
        str(c.get("shotId")): c for c in clips if c.get("shotId") and c.get("key")
    }

    # 主轨：有 I2V 片段的镜头用片段、其余回落静图。**运镜不在这里施加**（ADR-076：已随生成落地）。
    clip_urls: dict[str, str] = {}
    for shot_id, clip in clip_by_shot.items():
        clip_urls[shot_id] = ref(str(clip["key"]))
    if clip_urls:
        warnings.append(f"主轨用图生视频片段：{len(clip_urls)} 镜")

    # 静图轨：仅无片段（或片段缺失）的镜头需要其图
    still_keys = {
        seg.sourceKey
        for track in edl.tracks if track.kind == "image"
        for seg in track.segments if seg.sourceKey
    }
    image_urls = {key: ref(key) for key in still_keys}

    tts = common.dep(inp.deps, "tts") or {}
    bgm = common.dep(inp.deps, "bgm") or {}
    subtitle = common.dep(inp.deps, "subtitle") or {}
    cover = common.dep(inp.deps, "cover") or {}
    cart = common.dep(inp.deps, "cart") or {}

    audio_key = str(tts.get("audioKey") or "")
    if not audio_key:
        warnings.append("无音轨（deps.tts.audioKey 缺失）：成片无口播（若已关「智能配音」属预期）")
    else:
        # 新不变量（ADR-069）：锁定时间线的 totalMs **就是音频实测总长**（timeline 按逐句实测锁）。
        # 两者不符即说明 EDL 与配音不同源，如实报出（不静默重算）。
        try:
            audio_ms = int(tts.get("durationMs"))
        except (TypeError, ValueError):
            audio_ms = 0
        if audio_ms and abs(audio_ms - edl.totalMs) > 50:
            warnings.append(
                f"配音实测总长 {audio_ms}ms 与锁定时间线 {edl.totalMs}ms 不一致"
                "（按 ADR-069，时间线应即音频总长）——请核对，渲染仍以 EDL 为准。"
            )

    stickers, sticker_notes = _split_stickers(cart.get("stickers"))
    warnings.extend(sticker_notes)

    # 转场（ADR-070/080）：EDL 记的是**符号名**，未登记厂商资源 ID 的会被如实回落硬切——在此告警。
    # 传入有片段的镜 id：厂商文档的转场适用范围是「同轨道相邻两 **Video**」，相邻含**静图**时会追加
    # 「静图邻接是否生效未经实测」的告警（**文档支持范围 ≠ 我们实际用到的形态**）。
    warnings.extend(mediakit.transition_warnings(
        edl, set(clip_by_shot), region=cfg.mediakit_transition_region))

    tracks = mediakit.build_tracks(
        edl,
        image_urls=image_urls,
        clip_urls=clip_urls,
        audio_url=ref(audio_key),
        bgm_url=ref(str(bgm.get("bgmKey") or "")),
        srt_url=ref(str(subtitle.get("srtKey") or "")),
        subtitle_font_size=inp.subtitleFontSize,
        canvas_width=cfg.mediakit_canvas_width,
        canvas_height=cfg.mediakit_canvas_height,
        transition_region=cfg.mediakit_transition_region,
    )
    # 「这一轮到底传了哪些素材」如实进 warnings（在 build_tracks 之后：此时所有引用都已生成）
    report_uploads()
    if not tracks:
        raise mediakit.DspMediakitError(
            "渲染轨道为空：EDL 各镜均未选到图（timeline 已留空位）——无可渲染内容。",
            category="video",
        )

    cover_at = cover.get("atMs")
    body = mediakit.build_body(
        cfg, tracks=tracks,
        client_token=inp.idempotencyKey or f"{inp.runId}:{inp.stepId}",
        cover_at_ms=int(cover_at) if cover_at is not None else None,
    )
    result = mediakit.render(cfg, body)

    video_url = str(result.get("video_url") or "")
    duration_s = float(result.get("duration") or 0)
    duration_ms = int(duration_s * 1000) or edl.totalMs
    final = common.final_key(inp.outputPrefix, inp.runId, ext=cfg.mediakit_output_format)

    if video_url.startswith(("http://", "https://")):
        # §7.3② 转存：签名 URL（24h 失效）→ 自有对象存储的不可变 key
        with tempfile.TemporaryDirectory(prefix="dsp-render-") as tmp:
            local = storage.fetch_to_file(video_url, str(Path(tmp) / f"final.{cfg.mediakit_output_format}"))
            storage.put_file(local, final, content_type="video/mp4")
    else:
        raise mediakit.DspMediakitError(
            f"MediaKit 返回的成片地址不可直接取用（{video_url[:80]}）。"
            "本仓只支持 http(s) 结果的下载-转存；若启用了 media_output_destination（tos://…），"
            "需另配火山 TOS 客户端才能搬回自有对象存储。请清空对应 output_destination 配置"
            "或补 TOS 接线。",
            category="environment",
        )

    if result.get("resolution"):
        warnings.append(f"MediaKit 实际分辨率：{result['resolution']}")

    # ── 逐镜**剪辑记录**（ADR-080）：只报实况（未被施加的转场在这里就是硬切） ──
    edit_segments = _edit_segments(
        edl, clip_by_shot,
        audio_key=audio_key,
        bgm_key=str(bgm.get("bgmKey") or ""),
        bgm_requested=common.dep(inp.deps, "bgm") is not None,
        transition_region=cfg.mediakit_transition_region,
    )
    still_used = sum(1 for s in edit_segments if s.degradeMode == "fallback_still")
    missing = sum(1 for s in edit_segments if s.degradeMode == "missing_clip")
    if edit_segments:
        warnings.append(
            f"剪辑记录 {len(edit_segments)} 段：视频 {len(edit_segments) - still_used - missing} 段 / "
            f"静图回落 {still_used} 段" + (f" / 空段 {missing} 段" if missing else "")
            + "（逐镜实况见 editSegments）"
        )

    note = (f"成片 {duration_ms}ms → {final}"
            + (f"；剪辑记录 {len(edit_segments)} 段（视频 "
               f"{len(edit_segments) - still_used - missing} / 静图 {still_used}"
               + (f" / 空段 {missing}" if missing else "") + "）" if edit_segments else ""))
    return SkillOutput(
        data=RenderComposeData(
            storedKey=final, durationMs=duration_ms,
            width=cfg.mediakit_canvas_width, height=cfg.mediakit_canvas_height,
            editSegments=edit_segments,
            note=note, warnings=warnings,
        ),
        reasoning=[common.chain(
            f"合成渲染完成：{final}（{duration_ms}ms，{cfg.mediakit_canvas_width}×{cfg.mediakit_canvas_height}）",
            causal_analysis=(
                f"按锁定 EDL（totalMs={edl.totalMs}ms）铺 {len(tracks)} 轨；"
                f"主轨 {len(clip_urls)} 段视频片段 + {len(still_keys)} 张静图；"
                "字幕/贴片/封面位一次性烧制；AIGC 标识元数据已注入；"
                f"逐镜剪辑记录 {len(edit_segments)} 段（含裁切实况与**实际施加**的转场）。"
            ),
            risk_note=(
                "渲染为一次性烧制（不可再编辑）；成片已转存为不可变 key，重跑 = 新 runId = 新 key。"
                + (" 有镜既无片段也无图 ⇒ 该段为空（成片里是一段黑/空位），已在剪辑记录标 missing_clip。"
                   if missing else "")
            ),
        )],
        confidence=1.0,
        sample_size=1,
    )
