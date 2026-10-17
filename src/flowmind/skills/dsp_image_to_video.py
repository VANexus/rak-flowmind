"""image_to_video 技能：图生视频（逐镜产出视频底 · 契约 §7.5 第 1 步）。

契约 §7.5：运镜（``apply-camera-motion``）**只接受视频**，故「让静态图动起来」需
**先有视频底**。本技能就是那个第 1 步——**逐镜**把素材图 + **画面提示词**喂给图生视频模型，产出**真视频片段**，
落 ``{cachePrefix}/{runId}/clips/{shotId}.mp4``。

**prompt 取画面提示词、不是台词**（ADR-073）：把口播文案当画面提示词会让模型把**字幕烧进画面**，
故优先用 ``edl.shots[].visualPrompt``（``copy`` 只在缺画面提示词时兜底，并**如实告警**）。

**首尾帧桥**（ADR-074）：逐镜把**下一镜的图**当**尾帧**下发（``images`` 数组第 2 位）——
相邻镜在画面里首尾咬合，取代「厂商转场资源 ID 未登记 ⇒ 一律硬切」。开关
``FLOWMIND_DSP_VIDEOGEN_BRIDGE_NEXT_IMAGE=0`` 可关（聚合商未透出指南的 ``content.role``，
本落法语义**待真跑验证**）。

**输入降采样**（ADR-074）：素材图（补生 ``1152x2048``，为**静图主轨**口径）在**下发前**等比缩到
不超过 I2V 输出分辨率（``videogen_size``）——指南 FAQ：输入超过输出分辨率易出**指纹状纹理**。
只缩不放；降采样失败**回落原图并如实告警**（画质风险，不是成片失败）。
开关 ``FLOWMIND_DSP_VIDEOGEN_DOWNSCALE_INPUT=0`` 可关。

链路位置：``timeline``（锁时长）→ **本步** → ``render`` 侧把片段铺上主轨拼成成片。

**裁到镜头窗口**（ADR-080 阶段 5）：出片后本地 ffmpeg 把每镜裁到**锁定的镜头窗口**——超长裁掉、
不足则**末帧定格**补齐并如实报 ``shortfallMs``（``trim`` 随 ``ShotClip`` 带出，render 侧透进剪辑记录）。
两个行为各配 env 开关（``FLOWMIND_DSP_VIDEOGEN_TRIM_WINDOW`` / ``..._FREEZE_PAD``，默认都开）；
裁切失败（ffmpeg 缺失 / 编码器缺失）**回落原片段 + 如实告警**，不丢镜。

粒度与时长：**一镜一段**；镜头建议时长（``shots[].durationMs``）按秒下发给模型
（文档：``seconds`` 取值视模型而定，本仓**原样下发、不夹取**），出片后再由本步裁到窗口。

降级 **degrade**（本步为可关开关 ``videoGen`` 控制的增强步）：
- 某镜生成失败 → **该镜不出片段**（render 侧回落静图）+ ``warnings`` 如实说明；
- 全部失败 / 未配密钥 / 无时间线 → ``degraded=True`` + 明确 ``degradation_reason``。
**绝不伪造片段**（不拿静图当视频、不编 URL）。
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, Field

from flowmind.config import DspConfig, load_config
from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills import _dsp_common as common
from flowmind.skills import _dsp_qc as qc
from flowmind.skills import _dsp_storage as storage
from flowmind.skills import _dsp_video_gen as videogen
from flowmind.skills import _media as media
from flowmind.skills._dsp_common import DspBaseInput

_VERSION = "0.1.0"

# 负向约束（ADR-073；依据厂商《Seedance 2.5 提示词指南》「字幕约束不生效」一节）：
# 台词被写进 prompt 时模型容易把字幕**烧进画面**——除画面提示词本身不写台词外，这里再兜一层明文禁止。
# 需要更强的负向约束时经 env ``FLOWMIND_DSP_VIDEOGEN_PROMPT_SUFFIX``（``DspConfig.videogen_prompt_suffix``）追加。
_PROMPT_NEGATIVE = "。画面中不要出现任何文字、字幕、水印、logo 或品牌标识。"

# 降采样派生件的**文件夹**（在 run 的工作缓存目录内）与**可选后缀**（认不出即按 JPEG）。
_I2V_INPUT_DIR = "i2v_input"
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")

# 提交的**有界重试**次数（ADR-079）：只对厂商标 `retriable` 的瞬时错误重试，最多这么多次。
# 为什么是 2：一次重试足以越过常见的瞬时抖动，又不至于把「配额已尽」这类问题拖成长时间等待
# （后者分类为 environment、`retriable=False`，**根本不会**进重试）。
_SUBMIT_ATTEMPTS = 2


class ImageToVideoInput(DspBaseInput):
    """图生视频入参（公共上下文已含 ``shots[]``——逐镜的**画面提示词**取自其 ``visualPrompt``，
    缺则回落 ``copy`` 口播文案并如实告警，见 ADR-073）。"""

    maxShots: int = Field(
        default=0, ge=0,
        description="最多生成几镜（0 = 不限）；图生视频按镜计费，用于控成本")
    regenerateShotIds: list[str] = Field(
        default_factory=list,
        description="**只重生成这些镜**（ADR-079 的按镜重试 / 换图重生成）：未列出的镜头若在确定性 key "
                    "上已有产物则**复用**（不重复出片、不重复计费）；**留空 = 全部重新生成**（正常首跑）")


class ShotClip(BaseModel):
    """单镜视频片段（供 render 侧作 ``video`` 元素）+ **质检的测量产物**（ADR-079）+ **裁切实况**（ADR-080）。"""
    shotId: str
    key: str = Field(..., description="片段的云存储 key")
    atMs: int
    durationMs: int
    frames: list[str] = Field(
        default_factory=list,
        description="抽帧 key（ADR-079；qc 开关关 / 抽帧失败时为空——**不伪造帧**）。"
                    "前端质检层读这些帧做视觉判定（画面 / 主体一致性 / 运动质量 / 镜头意图）")
    frameStats: dict[str, Any] = Field(
        default_factory=dict,
        description="**客观量**（PIL 本地算，无需 numpy）：分辨率 / 逐帧平均亮度 / 相邻帧差异 / "
                    "黑帧与疑似冻结帧计数。**是测量不是判决**（阈值见 `_dsp_qc`）")
    trim: common.DspEditTrim = Field(
        default_factory=common.DspEditTrim,
        description="**裁切实况**（ADR-080）：已把片段裁到镜头窗口（`sourceMs` = 源实测时长、"
                    "`outMs` = 裁/补后时长、`shortfallMs` > 0 = 源比窗口短、已按末帧定格补齐）。"
                    "**全 0 = 未裁切/未实测**（开关关、裁切失败、或复用上次产物）——render 侧原样透出")


# 逐镜状态的**稳定原因码**（前端 / 运行器据此区分「前置条件不满足」与「厂商调用失败」；
# 中文 detail 只给人看，状态判定不解析文本）。
ShotVideoStatusId = Literal["generated", "reused", "skipped", "failed"]


class ShotVideoStatus(BaseModel):
    """一镜的图生视频**真实结局**（ADR-123）。

    为什么单列：``clips[]`` 只装成功片段，``warnings[]`` 是散装文本——「这镜到底出了视频、
    复用了旧片、还是回落静图，为什么」事后无法按结构查。运行器把本列表汇总进
    step note 并以 ``summary`` 决定 ``video`` 步是否降级（**不因静图回落成功就标完成**）。
    """
    shotId: str
    status: ShotVideoStatusId
    reason: str = Field(
        default="",
        description="稳定原因码：source_url_unreachable / no_image / no_prompt / max_shots / "
                    "reused_existing / vendor_submit_failed / vendor_generation_failed / transfer_failed")
    detail: str = Field(default="", description="给人看的原因（**不含预签名 URL 查询串**）")
    key: str = Field(default="", description="片段对象 key（generated / reused 时非空）")
    runId: str = Field(default="", description="产出该片段的运行 id（与 key 命名空间一致）")
    atMs: int = 0
    durationMs: int = 0
    imageKey: str = Field(default="", description="本镜绑定的素材 key（回落静图时保留绑定关系）")


class VideoStatusSummary(BaseModel):
    """逐镜状态汇总（运行器 / 阶段进度只看这四个计数，不解析 warning 文本）。"""
    total: int = 0
    generated: int = 0
    reused: int = 0
    skipped: int = 0
    failed: int = 0


class ImageToVideoData(BaseModel):
    """出参：逐镜片段 + 逐镜状态 + 如实说明。"""
    clips: list[ShotClip] = Field(default_factory=list)
    shots: list[ShotVideoStatus] = Field(
        default_factory=list,
        description="逐镜真实结局（含跳过 / 失败原因；**成功生成时带片段 key 与 runId**）")
    summary: VideoStatusSummary = Field(default_factory=VideoStatusSummary)
    warnings: list[str] = Field(default_factory=list)


class _Job(NamedTuple):
    """一镜的图生视频作业（尾帧 = 下一镜的图；空串 = 无桥）。"""
    shot_id: str
    at_ms: int
    duration_ms: int
    prompt: str
    image_key: str
    tail_key: str
    seconds: str
    """运镜的**引擎参数**（ADR-076）。本聚合端点无运动字段 ⇒ 恒空；留着是给将来引擎透出参数的接口位。"""
    motion_params: dict[str, object]


def _presign_public(key: str) -> str:
    """素材 key → **公网可读**预签名 URL（空 key 返回空串）。

    与 ``storage.presign_get``（内部端点）分开：厂商要求公网可访问，而 SigV4 把 Host 签进
    URL——必须由**公网端点**签发（ADR-123）。未配置公网入口时抛 ``DspStorageError``。
    """
    return storage.presign_get_public(key) if key else ""


def _next_image_key(shots: list[common.DspEdlShot], idx: int) -> str:
    """本镜之后**最近一张有图的镜**的素材 key（首尾帧桥的尾帧；没有则空串）。

    取「最近的有图镜」而非「紧邻镜」：中间镜无图时桥不该断（尾帧仍是下一张会出画的图）。
    """
    for nxt in shots[idx + 1:]:
        if nxt.imageSourceKey:
            return nxt.imageSourceKey
    return ""


def _i2v_image_url(
    inp: ImageToVideoInput, cfg: DspConfig, key: str, cap: tuple[int, int],
    cache: dict[str, tuple[str, str]], warnings: list[str], stats: dict[str, int],
) -> tuple[str, str]:
    """素材 key → ``(公网预签名 URL, 不可用原因码)``；原因码空串 = 可用。

    三步前置检查（**先检查、再提交**，不浪费厂商按镜计费）：
      ① 公网端点签发（内部端点签的地址厂商拉不到；签完改 host 会 403）；
      ② 字面主机检查（本机 / 私网 / 集群服务名 ⇒ ``source_url_unreachable``）；
      ③ HTTP 预读 1 字节（签名 / 反代 / 路径不匹配 ⇒ 同上；开关可关）。
    同一素材 key **只准备一次**（首尾帧桥下相邻镜共用同一张图）；降采样失败**回落原图**
    并如实告警——画质风险，不是成片失败，不因此丢掉整镜。
    """
    if key in cache:
        return cache[key]
    try:
        src_url = _presign_public(key)
    except storage.DspStorageError as exc:
        code = getattr(exc, "code", "") or "presign_failed"
        warnings.append(
            f"素材 {Path(key).name} 无法签发公网预签名地址（{code}：{exc}）——"
            "该镜**跳过图生视频**（回落静图，不提交厂商）。")
        cache[key] = ("", "source_url_unreachable")
        return cache[key]
    # **厂商硬约束**（2026-10-17 真调，ADR-111）：参考素材必须是公网 http(s) 或已入库的 asset://，
    # base64 / mediakit:// / 本机与内网地址一律拒绝。公网端点若被配成本机/内网，这里提前拦下。
    public, host = storage.public_url_host(src_url)
    if not public:
        warnings.append(
            f"素材 {Path(key).name} 的预签名地址是本机/内网（{host or '未知主机'}）——"
            "厂商要求**公网可访问的 http(s) URL** 或已入库的 asset://，本机对象存储厂商拉不到；"
            "该镜**跳过图生视频**（回落静图）。"
            "修复：配置公网可达的对象存储端点（`FLOWMIND_DSP_S3_PUBLIC_ENDPOINT`），"
            "或先在厂商素材库入库。")
        cache[key] = ("", "source_url_unreachable")
        return cache[key]
    if cfg.videogen_public_probe:
        readable, why = storage.probe_public_read(src_url)
        if not readable:
            warnings.append(
                f"素材 {Path(key).name} 公网预检失败（{why}）——**不提交厂商**（不浪费按镜计费）；"
                "该镜回落静图。请检查公网端点 / 反向代理是否透传 Host、签名是否与公网域名匹配。")
            cache[key] = ("", "source_url_unreachable")
            return cache[key]
    else:
        warnings.append(
            f"素材 {Path(key).name} 仅做字面公网判定（公网 HTTP 预检开关已关）——"
            "签名 / 反代是否真正匹配未验证，厂商若拒绝该镜将如实记为失败。")
    if not (cap[0] and cap[1]):
        cache[key] = (src_url, "")
        return cache[key]

    suffix = Path(key).suffix.lower()
    out_suffix = suffix if suffix in _IMAGE_SUFFIXES else ".jpg"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    derived_key = common.step_key(
        inp.cachePrefix, inp.runId, _I2V_INPUT_DIR,
        f"{Path(key).stem[:24]}-{digest}{out_suffix}")
    try:
        with tempfile.TemporaryDirectory(prefix="dsp-i2v-in-") as tmp:
            src = storage.fetch_to_file(src_url, str(Path(tmp) / f"orig{suffix or out_suffix}"))
            w, h = media.probe_image_size(src)
            if w <= cap[0] and h <= cap[1]:
                cache[key] = (src_url, "")           # 已不超 ⇒ 原图直用，不落派生件
                return cache[key]
            fitted = media.scale_image_to_fit(src, str(Path(tmp) / f"fit{out_suffix}"), *cap)
            storage.put_file(fitted, derived_key,
                             content_type=storage.content_type_for(derived_key))
    except (media.MediaError, storage.DspStorageError) as exc:
        warnings.append(
            f"素材 {Path(key).name} 降采样失败（{exc}）——**回落原图下发**"
            f"（指南：输入**超过输出分辨率** {cap[0]}x{cap[1]} 易出指纹状纹理；本镜未能核实尺寸，"
            "画质可能受影响）")
        cache[key] = (src_url, "")
        return cache[key]

    stats["downscaled"] += 1
    try:
        url = _presign_public(derived_key)
    except storage.DspStorageError as exc:
        code = getattr(exc, "code", "") or "presign_failed"
        warnings.append(
            f"降采样派生件 {Path(derived_key).name} 无法签发公网预签名地址（{code}：{exc}）——"
            "该镜**跳过图生视频**（回落静图）。")
        cache[key] = ("", "source_url_unreachable")
        return cache[key]
    cache[key] = (url, "")
    return cache[key]


def _degraded(reason: str, note: str, risk: str) -> SkillOutput[ImageToVideoData]:
    """统一的「未生成」出口（不伪造片段）。"""
    return SkillOutput(
        data=ImageToVideoData(warnings=[note]),
        reasoning=[common.chain(f"图生视频未生成：{note}", risk_note=risk)],
        confidence=0.0,
        degraded=True,
        degradation_reason=reason,
    )


@skill(id="image_to_video", name="图生视频", version=_VERSION,
       category="抖音成片", tags=["dsp", "video"])
def image_to_video(inp: ImageToVideoInput) -> SkillOutput[ImageToVideoData]:
    """逐镜把素材图 + 文案喂图生视频模型，产出片段 key（失败即该镜回落静图并如实说明）。"""
    cfg = load_config().dsp
    edl = common.dep_edl(inp.deps)
    if edl is None:
        return _degraded(
            "missing-timeline", "缺少锁定时间线（deps.timeline.edl）",
            "图生视频按锁定时间线的镜头与时长出片；请先跑通 timeline。")
    if not inp.shots:
        return _degraded("no-shots", "入参无 shots[]（缺画面提示词来源）",
                         "图生视频的 prompt 取自镜头的画面提示词（缺则回落文案）；无 shots 即无可生成的镜头。")
    if not videogen.videogen_configured(cfg):
        return _degraded(
            "videogen-not-configured",
            f"未配置图生视频密钥 {cfg.videogen_key_env}——本步跳过（成片回落静图）",
            "开启「图生视频」开关即需配置该密钥；未配置时如实降级，绝不伪造片段。")

    warnings: list[str] = []
    # 逐镜真实结局（含跳过 / 失败；成功时带片段 key 与 runId）。运行器只认这份结构化结果，
    # **不解析 warning 文本**——否则「静图回落成功」会被误当成图生视频阶段完成（ADR-123）。
    statuses: dict[str, ShotVideoStatus] = {}

    def _mark(shot_id: str, status: ShotVideoStatusId, *, reason: str = "", detail: str = "",
              key: str = "", at_ms: int = 0, duration_ms: int = 0, image_key: str = "") -> None:
        statuses[shot_id] = ShotVideoStatus(
            shotId=shot_id, status=status, reason=reason, detail=detail,
            key=key, runId=inp.runId if key else "",
            atMs=at_ms, durationMs=duration_ms, imageKey=image_key)

    def _mark_job(job: _Job, status: ShotVideoStatusId, *, reason: str = "", detail: str = "",
                  key: str = "") -> None:
        _mark(job.shot_id, status, reason=reason, detail=detail, key=key,
              at_ms=job.at_ms, duration_ms=job.duration_ms, image_key=job.image_key)

    def _summary() -> VideoStatusSummary:
        values = list(statuses.values())
        return VideoStatusSummary(
            total=len(values),
            generated=sum(1 for s in values if s.status == "generated"),
            reused=sum(1 for s in values if s.status == "reused"),
            skipped=sum(1 for s in values if s.status == "skipped"),
            failed=sum(1 for s in values if s.status == "failed"),
        )

    def _reason_counts(status: str) -> list[str]:
        """某状态下的原因分布（「source_url_unreachable 3」这种），便于一眼定位。"""
        counts: dict[str, int] = {}
        for item in statuses.values():
            if item.status == status and item.reason:
                counts[item.reason] = counts.get(item.reason, 0) + 1
        return [f"{reason} {count}" for reason, count in
                sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    # 文案 / 画面提示词**从 EDL 读**（ADR-070/073）：镜头在 timeline 里经收口（并/拆）后 id 会重排为
    # `s1..sK`，终版镜头的文案与画面提示词已随镜落进 `edl.shots[]`——回查 inp.shots 会**错配**。
    # 兜底：EDL 无该字段（旧快照）时按 id 回查下发的 shots（尽力而为，不臆造）。
    copy_by_shot = {s.id: s.copy_text for s in inp.shots}
    visual_by_shot = {s.id: s.author_prompt for s in inp.shots}  # ADR-077：真源在 i2v.authorPrompt

    # 输入降采样上限 = I2V **输出**分辨率（ADR-074；形状不合法即跳过降采样并如实说明）。
    cap = (0, 0)
    if cfg.videogen_downscale_input:
        cap = videogen.parse_size(cfg.videogen_size)
        if not (cap[0] and cap[1]):
            warnings.append(
                f"视频尺寸 {cfg.videogen_size!r} 形状不合法——**跳过输入降采样**（原图下发）")

    # ── 组作业：逐镜（图 + 画面提示词 + 建议时长 + 下一镜的图当尾帧）──
    shots = list(edl.shots)
    jobs: list[_Job] = []
    fell_back = 0
    motion_planned = 0   # 真下发运镜方案的镜数
    no_motion = 0        # 按固定机位处理的镜数
    motion_ignored = 0   # 有方案但因「运镜」开关关而未下发
    switches = inp.switches
    for idx, shot in enumerate(shots):
        if not shot.imageSourceKey:
            warnings.append(f"镜头 {shot.id} 无图（timeline 已留空位）——跳过图生视频")
            _mark(shot.id, "skipped", reason="no_image", detail="时间线该镜没有绑定素材图",
                  at_ms=shot.atMs, duration_ms=shot.durationMs)
            continue
        # **画面提示词优先**（ADR-073）：把**台词**当画面提示词会让模型把字幕烧进画面，
        # 故顺序是 visualPrompt（EDL → 下发 shots）→ copy 兜底，并对兜底**如实计数**。
        visual = (shot.visualPrompt or visual_by_shot.get(shot.id, "")).strip()
        fallback_copy = (shot.copy or copy_by_shot.get(shot.id, "")).strip()
        prompt = visual or fallback_copy
        if not prompt:
            warnings.append(f"镜头 {shot.id} 无画面提示词也无文案（prompt 为空）——跳过图生视频（不凭空编 prompt）")
            _mark(shot.id, "skipped", reason="no_prompt", detail="既无画面提示词也无口播文案",
                  at_ms=shot.atMs, duration_ms=shot.durationMs, image_key=shot.imageSourceKey)
            continue
        if not visual:
            fell_back += 1
        # **运镜（ADR-076）**：经引擎适配器并入 prompt。
        # 「运镜」开关**关** ⇒ 逐镜下**固定机位**条款（图照做，只是别动镜头——注意这**不是**「不做 I2V」）；
        # 开 ⇒ 用该镜的方案（缺方案 = 固定机位）。方案来自 **EDL**（并/拆后 id 会重排，回查 inp.shots 必错配）。
        effective_motion = shot.motion if switches.motion else None
        if shot.motion is not None and not switches.motion:
            motion_ignored += 1
        motion_phrase, motion_params = videogen.motion_expression(effective_motion)
        if shot.motion is not None and switches.motion and not motion_phrase:
            warnings.append(
                f"镜头 {shot.id} 的运镜符号 {shot.motion.type!r} 未登记到引擎适配表——该镜按固定机位处理")
        if motion_phrase and motion_phrase != videogen.MOTION_STATIC_CLAUSE:
            motion_planned += 1
        else:
            no_motion += 1
        tail_key = _next_image_key(shots, idx) if cfg.videogen_bridge_next_image else ""
        jobs.append(_Job(
            shot_id=shot.id, at_ms=shot.atMs, duration_ms=shot.durationMs,
            prompt=prompt + _PROMPT_NEGATIVE + motion_phrase + cfg.videogen_prompt_suffix,
            image_key=shot.imageSourceKey, tail_key=tail_key,
            seconds=videogen.seconds_str(shot.durationMs),
            motion_params=motion_params,
        ))
        # 先记「待处理」；后续提交 / 收片会把它改写成 generated / failed，maxShots 截断则改 skipped。
        _mark_job(jobs[-1], "skipped", reason="not_attempted", detail="待生成")
    if fell_back:
        warnings.append(
            f"{fell_back} 个镜头没有画面提示词（visualPrompt），已**回落用口播文案**出 prompt——"
            "台词容易被模型画成画面字幕，建议在镜头剪辑页为这些镜补画面提示词（ADR-073）。")
    if motion_ignored:
        # 如实告警（不静默）：方案在 EDL 上、但「运镜」开关关着 ⇒ 本次按固定机位生成。
        warnings.append(
            f"{motion_ignored} 个镜头带运镜方案，但「运镜」开关已关——本次**未下发**这些方案"
            "（按固定机位生成）；要启用请在镜头剪辑页打开「运镜」开关。")
    if inp.maxShots and len(jobs) > inp.maxShots:
        warnings.append(f"超过 maxShots={inp.maxShots}：仅生成前 {inp.maxShots} 镜（其余回落静图）")
        for job in jobs[inp.maxShots:]:
            _mark_job(job, "skipped", reason="max_shots",
                      detail=f"超过本次成本上限 maxShots={inp.maxShots}，未提交厂商")
        jobs = jobs[:inp.maxShots]
    if not jobs:
        summary = _summary()
        return SkillOutput(
            data=ImageToVideoData(shots=list(statuses.values()), summary=summary, warnings=warnings),
            reasoning=[common.chain("图生视频未生成：无可生成的镜头",
                                    risk_note="各镜缺图或缺文案；请补齐素材/文案后重跑。")],
            confidence=0.0, degraded=True, degradation_reason="no-generatable-shots")

    # ── 按镜重生成（ADR-079）：`regenerateShotIds` 之外的镜头，若确定性 key 上已有产物就**复用** ──
    # 用法：前端质检/调度层发现某镜没过，换图后只把该镜交给本步重跑；其余镜的产物原样带出，
    # **不重复出片、不重复计费**。`regenerateShotIds` 为空 = 正常首跑（全部生成）。
    regen = set(inp.regenerateShotIds)
    reused_clips: list[ShotClip] = []
    if regen:
        to_generate: list[_Job] = []
        for job in jobs:
            key = common.step_key(inp.cachePrefix, inp.runId, "clips", f"{job.shot_id}.mp4")
            if job.shot_id not in regen and storage.exists(key):
                reused_clips.append(ShotClip(
                    shotId=job.shot_id, key=key, atMs=job.at_ms, durationMs=job.duration_ms))
                _mark_job(job, "reused", reason="reused_existing",
                          detail="确定性 key 上已有片段，按 ADR-079 复用（未重新出片、未重复计费）",
                          key=key)
            else:
                to_generate.append(job)
        if reused_clips:
            warnings.append(
                f"按镜重生成：{len(reused_clips)} 镜**复用**已有片段（未重新出片、未重复计费）；"
                f"本次只生成 {len(to_generate)} 镜")
        jobs = to_generate
    if not jobs and not reused_clips:
        summary = _summary()
        return SkillOutput(
            data=ImageToVideoData(shots=list(statuses.values()), summary=summary, warnings=warnings),
            reasoning=[common.chain("没有可生成的镜头（缺图 / 缺文案）",
                                    risk_note="各镜缺图或缺文案；请补齐素材/文案后重跑。")],
            confidence=0.0, degraded=True, degradation_reason="no-generatable-shots")

    # ── 先全部提交（厂商侧并行跑），再逐个等 ──
    # 素材图在这时才逐张准备（降采样要下载+缩放+转存，只对**真要生成的镜**做）。
    url_cache: dict[str, tuple[str, str]] = {}
    stats = {"downscaled": 0}
    submitted: list[tuple[_Job, str]] = []
    for job in jobs:
        image_url, prepare_reason = _i2v_image_url(
            inp, cfg, job.image_key, cap, url_cache, warnings, stats)
        if not image_url:
            # **前置检查失败**（公网端点未配 / 端点字面内网 / 预签名失败 / 公网预检失败）：
            # 该镜 skipped（不是 failed）——厂商一次都没调，别把「没提交」说成「生成失败」。
            _mark_job(job, "skipped", reason=prepare_reason or "source_url_unreachable",
                      detail="参考素材不可公网读取，未提交厂商")
            continue
        tail_url = ""
        if job.tail_key:
            tail_url, tail_reason = _i2v_image_url(
                inp, cfg, job.tail_key, cap, url_cache, warnings, stats)
            if not tail_url:
                warnings.append(
                    f"镜头 {job.shot_id} 的尾帧素材不可用（{tail_reason or 'source_url_unreachable'}）"
                    "——本镜**按单图下发**（不因尾帧失败丢整镜）")
        # **有界重试**（ADR-079）：厂商把错误分成了 transient/其它（`DspVideoGenError.retriable`），
        # 而此前这个标志**被丢掉了**——瞬时抖动也直接判该镜回落静图，白白浪费一镜的配额。
        # 只对 `retriable` 重试，最多 `_SUBMIT_ATTEMPTS` 次；其余错误立刻如实报出（不无脑重试）。
        task_id = ""
        last: videogen.DspVideoGenError | None = None
        for attempt in range(1, _SUBMIT_ATTEMPTS + 1):
            try:
                task_id = videogen.submit(
                    cfg, prompt=job.prompt, image_url=image_url,
                    seconds=job.seconds, tail_image_url=tail_url,
                    motion_params=job.motion_params)
                break
            except videogen.DspVideoGenError as exc:
                last = exc
                if not exc.retriable or attempt >= _SUBMIT_ATTEMPTS:
                    break
                warnings.append(
                    f"镜头 {job.shot_id} 提交瞬时失败（第 {attempt} 次，{exc.category}）：{exc}——重试")
        if not task_id:
            exc = last
            detail = f"（{exc.category}）：{exc}" if exc else "未知原因"
            warnings.append(f"镜头 {job.shot_id} 提交失败（{detail}）——该镜回落静图")
            _mark_job(job, "failed", reason="vendor_submit_failed", detail=detail)
            continue
        submitted.append((job, task_id))

    # ── 收片 → 裁到镜头窗口（ADR-080） → 转存 → （qc 开时）抽帧 + 客观量 ──
    clips: list[ShotClip] = list(reused_clips)
    qc_measured = 0
    trimmed = 0        # 源比窗口长、真裁掉的镜数
    cut_ms = 0         # 共裁掉多少 ms
    padded = 0         # 源比窗口短、真按末帧定格补齐的镜数
    pad_ms = 0         # 共补齐多少 ms
    short_unpadded: list[str] = []   # 比窗口短但**未补**（`freeze_pad` 关）的镜
    trim_failed = 0
    for job, task_id in submitted:
        frames: list[str] = []
        frame_stats: dict[str, Any] = {}
        trim = common.DspEditTrim()
        key = common.step_key(inp.cachePrefix, inp.runId, "clips", f"{job.shot_id}.mp4")
        # ① **厂商侧**：轮询到终态。失败与「前置检查不通过」分开记（reason 不同）。
        try:
            raw = videogen.wait(cfg, task_id)
        except videogen.DspVideoGenError as exc:
            detail = f"（{exc.category}）：{exc}"
            warnings.append(f"镜头 {job.shot_id} 出片失败{detail}——该镜回落静图")
            _mark_job(job, "failed", reason="vendor_generation_failed", detail=detail)
            continue
        url = videogen.result_url(raw)
        if not url:
            detail = "厂商任务完成但缺少视频直链（metadata.url / url）"
            warnings.append(f"镜头 {job.shot_id} 出片失败：{detail}——该镜回落静图")
            _mark_job(job, "failed", reason="vendor_generation_failed", detail=detail)
            continue
        # ② **本仓侧**：下载 → 裁到窗口 → 转存 →（qc 开时）抽帧。任何一步失败都记 transfer_failed，
        # 与厂商失败区分开（排查方向不同：一个是配额/提示词，一个是存储/转存）。
        try:
            with tempfile.TemporaryDirectory(prefix="dsp-i2v-") as tmp:
                local = storage.fetch_to_file(url, str(Path(tmp) / f"{job.shot_id}.mp4"))
                # **裁到镜头窗口**（ADR-080）：模型给的时长与 timeline 锁定的窗口不是同一个数——
                # 超长会顶到下一镜、不足会让该镜提前卡住。失败**回落原片段 + 如实告警，不丢镜**
                # （与输入降采样同款纪律：画质/节奏风险 ≠ 成片失败）。
                upload = local
                if cfg.videogen_trim_window:
                    try:
                        fit = media.clip_to_window(
                            local, str(Path(tmp) / f"{job.shot_id}.fit.mp4"),
                            job.duration_ms, freeze_pad=cfg.videogen_freeze_pad)
                        upload = fit.path
                        trim = common.DspEditTrim(
                            sourceMs=fit.sourceMs, outMs=fit.outMs, shortfallMs=fit.shortfallMs)
                        if fit.sourceMs > job.duration_ms:
                            trimmed += 1
                            cut_ms += fit.sourceMs - job.duration_ms
                        elif fit.shortfallMs > 0:
                            if fit.outMs > fit.sourceMs:      # 真补齐了
                                padded += 1
                                pad_ms += fit.shortfallMs
                            else:                             # 未补（定格开关关）——照实列出
                                short_unpadded.append(
                                    f"{job.shot_id}（短 {fit.shortfallMs}ms）")
                    except media.MediaError as exc:
                        trim_failed += 1
                        warnings.append(
                            f"镜头 {job.shot_id} 裁到镜头窗口失败（{exc}）——**原片段照常入库**"
                            "（未裁切；剪辑记录里该镜的 trim 按 0 如实上报）")
                storage.put_file(upload, key, content_type="video/mp4")
                # **质检的测量侧**（ADR-079）：qc 开关开时才抽帧 + 算客观量（ffmpeg 免费但耗时，帧要上传）。
                # 判定与重试调度在**前端质检层**；本侧**只测量**。失败即**如实告警并放弃本次测量**——
                # 抽帧失败仍要保住这一镜的片段（画质风险 ≠ 丢镜），**绝不伪造帧或统计**。
                if inp.switches.qc:
                    try:
                        sample = qc.sample_frames(upload, tmp, job.duration_ms / 1000.0)
                        frame_stats = qc.frame_stats(sample.paths)
                        for i, path in enumerate(sample.paths):
                            fkey = common.step_key(
                                inp.cachePrefix, inp.runId, "qa", f"{job.shot_id}-{i}.png")
                            storage.put_file(path, fkey, content_type="image/png")
                            frames.append(fkey)
                        qc_measured += 1
                    except media.MediaError as exc:
                        warnings.append(
                            f"镜头 {job.shot_id} 抽帧/测量失败（{exc}）——本次不做客观量（如实跳过）")
        except (storage.DspStorageError, media.MediaError) as exc:
            warnings.append(f"镜头 {job.shot_id} 转存/裁切失败：{exc}——该镜回落静图")
            _mark_job(job, "failed", reason="transfer_failed", detail=str(exc))
            continue
        clips.append(ShotClip(shotId=job.shot_id, key=key, atMs=job.at_ms,
                              durationMs=job.duration_ms, frames=frames,
                              frameStats=frame_stats, trim=trim))
        _mark_job(job, "generated", key=key)

    if trimmed or padded:
        bits: list[str] = []
        if trimmed:
            bits.append(f"裁到镜头窗口 {trimmed} 镜（共裁掉 {cut_ms}ms）")
        if padded:
            bits.append(f"源片段不足、末帧定格补齐 {padded} 镜（共 {pad_ms}ms）")
        warnings.append("；".join(bits) + "——逐镜实况见各片段的 trim。")
    if short_unpadded:
        warnings.append(
            f"{len(short_unpadded)} 镜源片段比镜头窗口短且**未做定格补齐**（「末帧定格」开关关）："
            f"{'、'.join(short_unpadded[:6])}{'…' if len(short_unpadded) > 6 else ''}"
            "——这些镜会比时间线短（如实报出 shortfallMs，补不补是策略）。")
    if not cfg.videogen_trim_window:
        warnings.append("「裁到镜头窗口」开关关：片段**原样入库**（trim 实况按 0 上报）")
    if regen and reused_clips and cfg.videogen_trim_window:
        warnings.append(
            f"{len(reused_clips)} 镜复用已有片段（沿用上次产物，未重新裁切）——"
            "其 trim 实况不在本次测量范围（按 0 上报）")

    summary = _summary()
    produced = summary.generated + summary.reused
    fell_back_count = summary.skipped + summary.failed
    skip_reasons = _reason_counts("skipped")
    fail_reasons = _reason_counts("failed")
    if produced == 0:
        reason_bits = "；".join([*skip_reasons, *fail_reasons])
        return SkillOutput(
            data=ImageToVideoData(
                shots=list(statuses.values()), summary=summary, warnings=warnings),
            reasoning=[common.chain(
                "图生视频未产出片段：成片回落静图"
                + (f"（{reason_bits}）" if reason_bits else ""),
                risk_note=(
                    "本步没有真视频片段；render 侧将逐镜回落静图。"
                    "若是 source_url_unreachable：请配置公网可达的对象存储端点"
                    "（`FLOWMIND_DSP_S3_PUBLIC_ENDPOINT`），本机 / 内网 MinIO 厂商拉不到。"))],
            confidence=0.0, degraded=True,
            degradation_reason=(
                "all-shots-fell-back" if (summary.failed or summary.skipped)
                else "no-generatable-shots"))

    total_ms = sum(c.durationMs for c in clips)
    bridged = sum(1 for j in jobs if j.tail_key)
    extras = ([f"首尾帧桥 {bridged} 镜"] if bridged else []) + (
        [f"复用 {summary.reused} 镜（未重复计费）"] if summary.reused else []) + (
        [f"输入降采样 {stats['downscaled']} 张"] if stats["downscaled"] else []) + (
        [f"运镜方案 {motion_planned} 镜"] if motion_planned else []) + (
        [f"固定机位 {no_motion} 镜"] if no_motion else []) + (
        [f"{motion_ignored} 镜有运镜方案但「运镜」开关已关，未下发"] if motion_ignored else []) + (
        [f"裁到镜头窗口 {trimmed} 镜（裁掉 {cut_ms}ms）"] if trimmed else []) + (
        [f"末帧定格补齐 {padded} 镜（{pad_ms}ms）"] if padded else []) + (
        [f"裁切失败 {trim_failed} 镜（原片段照常入库）"] if trim_failed else []) + (
        [f"质检测量 {qc_measured} 镜（抽帧 + 客观量；判定在前端质检层）"] if qc_measured else []) + (
        [f"跳过 {summary.skipped} 镜（{'，'.join(skip_reasons)}）"] if summary.skipped else []) + (
        [f"失败 {summary.failed} 镜（{'，'.join(fail_reasons)}）"] if summary.failed else [])
    return SkillOutput(
        data=ImageToVideoData(
            clips=clips, shots=list(statuses.values()), summary=summary, warnings=warnings),
        reasoning=[common.chain(
            f"图生视频完成：{produced}/{summary.total} 镜出片"
            f"（生成 {summary.generated} / 复用 {summary.reused}；共 {total_ms}ms"
            f"{'；' + '，'.join(extras) if extras else ''}）",
            causal_analysis=(
                f"逐镜把素材图（**公网端点**预签名 URL，ADR-123）+ **画面提示词**喂 {cfg.videogen_model}"
                "（prompt 末尾统一追加「不要出现文字/字幕/水印」的负向约束，ADR-073）；"
                f"时长按镜头建议时长（{clips[0].durationMs}ms 等）原样下发；"
                + ("相邻镜按**首尾帧桥**下发（首帧 = 本镜图、尾帧 = 下一镜图，ADR-074），"
                   "让切点处画面首尾咬合；" if bridged else "")
                + (f"素材图超输出分辨率（{cfg.videogen_size}）的先**等比降采样**再下发"
                   f"（指南：输入超标易出指纹状纹理，ADR-074），共 {stats['downscaled']} 张；"
                   if stats["downscaled"] else "")
                + "运镜（ADR-076）经**引擎适配器**并入 prompt："
                + (f"本镜下发了运镜方案（{motion_planned} 镜）；" if motion_planned else "")
                + (f"另有 {no_motion} 镜按**固定机位**下发；" if no_motion else "")
                + (f"{motion_ignored} 镜带方案但因「运镜」开关关闭被忽略；" if motion_ignored else "")
                + "（本聚合端点**没有**运动入参，运镜只能靠提示词语言表达——是否真被执行**待真跑验证**）"
                + (f"出片后本地 ffmpeg 裁到**锁定的镜头窗口**（ADR-080）：裁掉 {trimmed} 镜 / "
                   f"末帧定格补齐 {padded} 镜" if (trimmed or padded) else "")
                + ("；" if (trimmed or padded) else "")
                + "直链（24h）已转存为不可变 key。"
            ),
            risk_note=(
                f"{fell_back_count} 镜未出片（逐镜原因见 shots[] 与 warnings），"
                "render 侧将回落静图（如实标注）。"
                if fell_back_count else
                "全部镜头已出片；运镜已随本次生成落地（render 侧**不再**施加运镜——重复施加已退役）。"
            ),
        )],
        confidence=1.0 if not fell_back_count else 0.6,
        sample_size=produced,
        degraded=bool(fell_back_count),
        degradation_reason="partial-shots-fell-back" if fell_back_count else None,
    )
