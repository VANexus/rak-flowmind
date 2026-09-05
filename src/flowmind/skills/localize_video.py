"""localize_video 技能：视频本地化流水线（单视频同步，本地优先）。

流水线：ffmpeg 提音轨 → ASR 句级时间戳（本地 faster-whisper GPU 优先，
auto 回落百炼 Paraformer）→ OCR 定位原字幕区（本地 RapidOCR 优先，
auto 回落 qwen3.5-ocr）→ LongCat 翻译 → ffmpeg 擦除原字幕区
→ 克隆/预设音色逐句配音（音色 ID 由百炼声音复刻提前生成，
配音时长自动 atempo 对齐原句）→ 混音（可选保留 -12dB 背景音）+ ASS 字幕烧录。

后端由 config 后端开关控制（local/cloud/auto）；auto 两端都不可用时
显式 degraded，绝不静默降级。
失败契约遵循 HTTP 依赖类：r.ok=True + metrics.degraded + data.failure_category。
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import requests
from pydantic import BaseModel, Field, field_validator

from flowmind.config import get_config, load_config
from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.errors import is_retriable
from flowmind.skill import skill
from flowmind.skills import (
    _bge_embed,
    _cloud_asr,
    _cloud_ocr,
    _cloud_tts,
    _inpaint,
    _llm_translate,
    _local_asr,
    _local_ocr,
    _local_tts,
    _media,
    _music_sep,
)
from flowmind.skills._secrets import get_api_key
from flowmind.tasks import CancelledError, current_task_context, vectors
from flowmind.tasks.gpu import gpu_lane

_VERSION = "0.1.0"

KEY_SPEECH = "AI_SPEECH_API_KEY"
KEY_LLM = "AI_LLM_API_KEY"

logger = logging.getLogger(__name__)


def _noop_progress(stage: str, pct: float, message: str) -> None:
    """直连 invoke（无 TaskManager）时的默认进度回调。"""


def _never_cancel() -> bool:
    """直连 invoke 时永不取消（协作取消仅经 TaskManager 路径生效）。"""
    return False


class LocalizeVideoInput(BaseModel):
    """单视频本地化入参。"""

    video_path: str = Field(..., min_length=1, description="视频路径或可公网访问 URL")
    target_lang: str | None = Field(default=None, description="目标语言；None=读 config 默认")
    source_lang: str | None = Field(default=None, description="源语言；None=zh")
    voice_id: str | None = Field(
        default=None,
        description="配音音色（预设音色名如 longanhuan_v3.6，"
                    "或百炼声音复刻生成的音色 ID）；"
                    "None=读 config.localize_voice，配置为空则不配音",
    )
    output_path: str | None = Field(
        default=None, description="输出 mp4 路径；None=输入同目录 <stem>_localized.mp4"
    )
    keep_background_audio: bool = Field(default=True, description="原声 -12dB 保留为背景（BGM/环境音不丢）；显式传 False 可输出纯配音")
    tts_backend: str | None = Field(
        default=None,
        description="配音后端：auto（本地栈可用则克隆原片人声，否则云 TTS）/ "
                    "local（强制本地 Qwen3-TTS 克隆）/ cloud（强制云端，voice_id 生效）；"
                    "None=读 config.tts_backend",
    )
    voice_ref_audio: str | None = Field(
        default=None,
        description="本地克隆参考音频路径/URL（缺省自动用原片人声）",
    )
    voice_ref_text: str | None = Field(
        default=None,
        description="参考音频转写（缺省用本次 ASR 原文/本地 ASR 补转写）",
    )
    erase_backend: str | None = Field(
        default=None,
        description="字幕擦除后端：auto（LaMa 可用则用，否则 delogo）/ "
                    "local（强制 LaMa，缺失显式报错）/ delogo（旧硬横带）；"
                    "None=读 config.erase_backend",
    )

    @field_validator("video_path")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("video_path 不能为空")
        return v.strip()


class LocalizeVideoReport(BaseModel):
    """流水线业务载荷。"""

    output_path: str
    duration_seconds: float
    asr_segment_count: int
    subtitle_region_erased: bool          # 是否定位到并擦除了原字幕区
    voice_used: str | None                # 实际使用的音色；None=未配音
    asr_backend: str = "cloud"            # 实际使用的 ASR 后端（local/cloud）
    ocr_backend: str = "cloud"            # 实际使用的 OCR 后端（local/cloud）
    erase_backend: str | None = None      # 实际使用的擦除后端（lama/delogo；增量字段）
    # 字幕向量化（增量字段）：True=已写入 Milvus；False=未写入（未配置或失败，
    # 由阶段消息与日志区分）；None=开关关闭
    vectorized: bool | None = None
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


@skill(id="localize_video", name="视频本地化流水线（本地优先）", version=_VERSION)
def localize_video(inp: LocalizeVideoInput) -> SkillOutput[LocalizeVideoReport]:
    """ASR → LLM 翻译 → 原字幕擦除 → 克隆 TTS 配音 → 合成。

    双匹配策略：OCR 负责定位原字幕区域（供擦除）；翻译文本一律以 ASR 为准。
    七阶段主体在 run_localization_pipeline（进度回调 / 协作式取消 /
    GPU 串行）；本入口负责预检、工作目录装配与信封转换
    （对外签名与 SkillResult.data 结构不变）。
    """
    cfg = load_config().localizer
    speech_key = get_api_key(KEY_SPEECH)
    llm_key = get_api_key(KEY_LLM)

    # ── 预检（确定性，不调云）──
    is_url = inp.video_path.startswith(("http://", "https://"))
    src = inp.video_path
    if not is_url and not Path(src).exists():
        return _fail(inp, f"视频文件不存在: {src}", "video")
    if not is_url:
        ext = Path(src).suffix.lower()
        if ext not in {e.lower() for e in cfg.allowed_extensions}:
            return _fail(inp, f"扩展名 {ext} 不在允许列表 {cfg.allowed_extensions}", "video")
    # 云 key 按需强制：仅当配音确定走云端时才强制 speech_key
    # （全本地方案 ASR/OCR/TTS 都不需要任何云 key）
    eff_tts_backend = (inp.tts_backend or getattr(cfg, "tts_backend", "auto")).lower()
    tts_wants_cloud = eff_tts_backend == "cloud" or (
        eff_tts_backend == "auto"
        and (bool(inp.voice_id) or not _local_tts.available())
    )
    if tts_wants_cloud and not speech_key:
        return _fail(inp, f"未设置环境变量 {KEY_SPEECH}（云端配音需要）", "environment")
    if not llm_key:
        return _fail(inp, f"未设置环境变量 {KEY_LLM}（翻译需要）", "environment")

    # ── 工作目录与任务上下文 ──
    # TaskManager 路径：workdir（data_dir/tasks/<task_id>/work，绝对路径）、
    # 进度回调、取消检查由 TaskContext 注入（contextvars 跨 invoke 传递）；
    # 直连 invoke（demo/调试）：降级默认回调 + data_dir 下随机 workdir。
    ctx = current_task_context()
    if ctx is not None and ctx.workdir is not None:
        workdir = Path(ctx.workdir)
    else:
        workdir = Path(cfg.data_dir) / "tasks" / uuid.uuid4().hex / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    progress_cb = ctx.progress_cb if ctx is not None else _noop_progress
    cancel_check = ctx.cancel_check if ctx is not None else _never_cancel

    try:
        report = run_localization_pipeline(inp, workdir, progress_cb, cancel_check)
    except CancelledError:
        # 阶段边界取消：degraded 信封 failure_category="cancelled"
        # → TaskManager 落 cancelled 终态；直连 invoke 语义不变
        return _fail(inp, "任务已取消（阶段边界停止）", "cancelled")
    except _cloud_asr.ASRError as exc:
        return _fail(inp, str(exc), exc.category, retriable=exc.retriable)
    except _local_tts.LocalTTSError as exc:
        # 本地 TTS 异常与云 TTS 同构（category/retriable 语义一致），
        # 缺此分支会让 LocalTTSError 穿透到 invoke 兜底丢 category
        return _fail(inp, str(exc), exc.category, retriable=exc.retriable)
    except _cloud_tts.TTSError as exc:
        return _fail(inp, str(exc), exc.category, retriable=exc.retriable)
    except _cloud_ocr.OCRError as exc:
        return _fail(inp, str(exc), exc.category, retriable=exc.retriable)
    except _inpaint.InpaintError as exc:
        return _fail(inp, str(exc), exc.category, retriable=exc.retriable)
    except _music_sep.MusicSepError as exc:
        return _fail(inp, f"{exc}（可设 bgm_vocal_sep=false 跳过人声分离）",
                     exc.category, retriable=exc.retriable)
    except _llm_translate.LLMTtranslateError as exc:
        return _fail(inp, str(exc), exc.category, retriable=exc.retriable)
    except _media.MediaError as exc:
        return _fail(inp, f"媒体处理失败: {type(exc).__name__}", "video")

    if report.get("empty_asr"):
        return _ok_shell(inp, workdir, duration=report["duration_seconds"],
                         msg=report["warning"], erased=False, voice=None)

    # 成功信封（data 字段与语义不变，6 个 demo 断言兼容）
    regions_erased = report["subtitle_region_erased"]
    voice_used = report["voice_used"]
    chain = ReasoningChain(
        conclusion=(
            f"本地化完成：{report['display_segment_count']} 句"
            f"（ASR {report['asr_segment_count']} 句拆分）、"
            f"{'已擦除[' + report['erase_backend'] + ']' if regions_erased else '未检出'}"
            f"原字幕区、{'克隆配音 ' + voice_used if voice_used else '未配音'}"
        ),
        triggered_rules=[], evidence=[],
        causal_analysis=(
            f"ASR[{report['asr_backend']}] {report['asr_segment_count']} 句 → "
            f"拆分为 {report['display_segment_count']} 句 → LLM 翻译 → "
            f"OCR[{report['ocr_backend']}] 定位 → 擦除[{report['erase_backend']}] → "
            f"TTS → ffmpeg 合成"
        ),
        risk_note="译文为 LLM 生成，正式投放前建议人工抽查。",
    )
    return SkillOutput(
        data=LocalizeVideoReport(
            output_path=report["output_path"],
            duration_seconds=report["duration_seconds"],
            asr_segment_count=report["asr_segment_count"],
            subtitle_region_erased=regions_erased,
            voice_used=voice_used,
            asr_backend=report["asr_backend"],
            ocr_backend=report["ocr_backend"],
            erase_backend=report["erase_backend"],
            vectorized=report["vectorized"],
        ),
        reasoning=[chain], confidence=0.9,
        sample_size=report["asr_segment_count"],
    )


# ═══ 流水线主体（阶段边界取消 / 进度回调 / GPU 串行） ═══


def _cancel_boundary(stage: str, cancel_check) -> None:
    """阶段边界协作取消检查：命中抛 tasks.CancelledError。

    TaskManager 捕获后落 cancelled 终态；直连 invoke 时 cancel_check
    恒 False（_never_cancel），行为不变。
    """
    if cancel_check():
        raise CancelledError(f"阶段 {stage} 边界检测到取消信号")


def _pipeline_task_id() -> str:
    """Milvus 向量归属 id：TaskManager 路径用任务 id；直连 invoke 用随机 id。"""
    ctx = current_task_context()
    if ctx is not None and ctx.task_id:
        return ctx.task_id
    return uuid.uuid4().hex


def _filter_tts_segments(translated: list[dict]) -> list[dict]:
    """TTS 输入分段：加 index、过滤过短句（≤1 字避免 TTS API 报错）；
    全被过滤时退回原始分段。"""
    tts_segs = [
        {**s, "index": i} for i, s in enumerate(translated)
        if len(str(s.get("text", "")).strip()) > 1
    ]
    if not tts_segs:
        tts_segs = [{**s, "index": i} for i, s in enumerate(translated)]
    return tts_segs


def _vectorize_segments(task_id: str, segments: list[dict], src: str) -> tuple[bool | None, str]:
    """成功路径尾部：ASR 分段 → BGE 嵌入 → Milvus upsert（FLOWMIND_VECTORIZE 开关）。

    开关配置源顺序：env → config.toml（infra.vectorize）→ 默认开。
    返回 (vectorized, note)，note 供阶段进度消息：
    - (True, "向量化完成")：已写入 Milvus；
    - (False, "向量化未配置，已跳过")：Milvus/嵌入服务地址均空（显式禁用
      默认形态）——info 静默跳过，不打 degraded warning 刷屏；
    - (False, "向量化失败（已降级）")：服务已配置但调用失败——warning 降级；
    - (None, "向量化已关闭")：开关显式关闭。
    向量化是增值步骤：任何未写入结局都不影响本地化主产出（ok=True 不变）。
    """
    raw = os.environ.get("FLOWMIND_VECTORIZE", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return None, "向量化已关闭"
    if not raw and not get_config().infra.vectorize:
        return None, "向量化已关闭"
    if not vectors.is_configured():
        logger.info("字幕向量化未配置（FLOWMIND_MILVUS_URI / config "
                    "infra.milvus_uri 均空）——跳过")
        return False, "向量化未配置，已跳过"
    if not _bge_embed.is_configured():
        logger.info("字幕向量化未配置（FLOWMIND_EMBEDDING_BASE_URL / config "
                    "infra.embedding_base_url 均空）——跳过")
        return False, "向量化未配置，已跳过"
    pairs = [(i, str(s.get("text", "")).strip()) for i, s in enumerate(segments)]
    pairs = [(i, t) for i, t in pairs if t]
    if not pairs:
        return False, "无有效 ASR 文本，向量化跳过"
    try:
        vecs = _bge_embed.embed_texts([t for _, t in pairs])
        video_name = Path(src).name.split("?")[0] or src
        rows = [
            {
                "task_id": task_id,
                "video_name": video_name,
                "seg_index": i,
                "start_sec": float(segments[i].get("begin", 0.0)),
                "end_sec": float(segments[i].get("end", 0.0)),
                "text": t,
                "vector": v,
            }
            for (i, t), v in zip(pairs, vecs)
        ]
        vectors.upsert_task_segments(task_id, rows)
        return True, "向量化完成"
    except Exception as exc:  # noqa: BLE001  增值步骤降级
        logger.warning("字幕向量化失败（不影响主产出）: %s", exc)
        return False, "向量化失败（已降级）"


def run_localization_pipeline(args: LocalizeVideoInput, workdir: Path,
                              progress_cb, cancel_check) -> dict:
    """七阶段本地化流水线（进度回调 + 协作取消 + GPU 串行）。

    阶段与进度区间（pct 0-100）::

        extract_audio 0→10 → asr 10→25 → ocr 25→40 → translate 40→55
        → erase 55→75 → tts 75→90 → mix 90→97 → vectorize 97→100

    - 每阶段边界调 cancel_check()；命中抛 tasks.CancelledError（协作取消）
    - GPU 推理阶段（本地 ASR / LaMa 擦除 / 本地 TTS）经 tasks.gpu.gpu_lane
      独占单卡槽；云 API 阶段（dashscope/qwen-ocr/LongCat/云 TTS）不占
    - 返回 report dict；无人声提前返回 {"empty_asr": True, ...}
    - 模块内辅助（_locate_region/_build_timed_audio 等）经模块全局查找
      调用——demo 的 monkeypatch mock 依赖此行为，调用形状不可变
    """
    cfg = load_config().localizer
    speech_key = get_api_key(KEY_SPEECH)
    llm_key = get_api_key(KEY_LLM)
    src = args.video_path
    is_url = src.startswith(("http://", "https://"))

    # ── 1) 提取音轨（extract_audio 0→10）──
    progress_cb("extract_audio", 1.0, "提取音轨")
    audio_path = _media.extract_audio(
        src, str(workdir / "audio.wav"),
        sample_rate=cfg.asr_sample_rate,
    )
    progress_cb("extract_audio", 10.0, "音轨已提取")

    # ── 2) ASR（asr 10→25；本地 faster-whisper 占 GPU 槽，auto 回落云）──
    _cancel_boundary("asr", cancel_check)
    asr_backend, asr_fn = _select_asr(cfg, speech_key)
    source_lang = args.source_lang or cfg.source_lang_default
    progress_cb("asr", 12.0, f"ASR 转写（{asr_backend}）")
    if asr_backend == "local":
        with gpu_lane():
            segments = asr_fn(audio_path, source_lang)
    else:
        segments = asr_fn(audio_path, source_lang)
    duration_s, width, height = (None, None, None)
    if not is_url:
        duration_s, width, height = _media.probe_media(src)
    if not segments:
        return {"empty_asr": True, "duration_seconds": duration_s or 0.0,
                "warning": "ASR 未识别到任何语音（可能无人声）"}
    if duration_s is None:
        duration_s = segments[-1]["end"]
    progress_cb("asr", 25.0, f"ASR 完成：{len(segments)} 句")

    # ── 3) OCR 定位原字幕区（ocr 25→40；RapidOCR 为 CPU，不占 GPU 槽）──
    _cancel_boundary("ocr", cancel_check)
    ocr_backend, ocr_fn = _select_ocr(cfg, speech_key)
    progress_cb("ocr", 27.0, f"OCR 定位字幕区（{ocr_backend}）")
    region = _locate_region(src, duration_s, workdir, ocr_fn, cfg,
                            width=width, height=height)
    progress_cb("ocr", 40.0, f"OCR 完成：{len(region or [])} 个候选区")

    # ── 4) LLM 翻译（translate 40→55；长句先拆分，翻译更稳定）──
    _cancel_boundary("translate", cancel_check)
    display_segments = _split_long_segments(segments)
    progress_cb("translate", 42.0,
                f"翻译 {len(display_segments)} 句（{len(segments)} 句 ASR 拆分）")
    # 协议/base/model 从 .env 读（AI_LLM_*），未设置回落 LongCat Anthropic 默认
    translated = _llm_translate.translate_segments(
        display_segments,
        target_lang=args.target_lang or cfg.target_lang_default,
        source_lang=args.source_lang or cfg.source_lang_default,
        api_key=llm_key,
        api_base=os.environ.get("AI_LLM_BASE_URL") or _llm_translate.DEFAULT_BASE,
        model=os.environ.get("AI_LLM_MODEL") or cfg.localize_llm_model,
        protocol=(os.environ.get("AI_LLM_PROTOCOL") or "anthropic").lower(),
    )
    progress_cb("translate", 55.0, f"翻译完成：{len(translated)} 句")

    # ── 5) 擦除原字幕区 + 烧译文字幕（erase 55→75；LaMa 推理占 GPU 槽）──
    _cancel_boundary("erase", cancel_check)
    regions = region or []
    # 字号/位置匹配原字幕：把 OCR 原始行框传给 _write_ass
    ass_path = _write_ass(translated, workdir, cfg,
                          regions=regions, width=width, height=height)
    out_path = _resolve_out_path(args, src, workdir)
    # 精准擦除：OCR 文本行框逐框外扩（不再合成大横带误伤前景主体）
    erase_regs = _prep_erase_regions(regions, width, height)
    src_local = _ensure_local(src, workdir)
    # 擦除后端：auto=LaMa 可用则用（复杂背景无拉丝）否则 delogo；
    # local=强制 LaMa（缺失/失败显式报错，不静默回落）；delogo=旧硬横带
    erase_used = "delogo"
    eff_erase = (args.erase_backend or getattr(cfg, "erase_backend", "auto")).lower()
    progress_cb("erase", 57.0, f"擦除字幕区（{eff_erase}，{len(erase_regs)} 框）")
    if erase_regs and eff_erase in ("auto", "local", "lama"):
        if _inpaint.available():
            with gpu_lane():
                _inpaint.erase_regions(
                    src_local, erase_regs,
                    str(workdir / "inpainted.mp4"), str(workdir))
            erased_video = _media.burn_subtitles(
                str(workdir / "inpainted.mp4"),
                str(workdir / "subbed.mp4"), ass_path)
            erase_used = "lama"
        elif eff_erase in ("local", "lama"):
            raise _inpaint.InpaintError(
                "erase_backend=local：simple-lama-inpainting 不可用"
                "（conda env update -f environment.yml）",
                category="environment",
            )
        # auto 且 LaMa 缺失：回落 delogo（auto 语义允许）
    if erase_used == "delogo":
        erased_video = _media.burn_subtitles(
            src_local, str(workdir / "subbed.mp4"), ass_path,
            erase_regions=erase_regs or None,
        )
    progress_cb("erase", 75.0, f"擦除完成（{erase_used}）")

    # ── 6) TTS 逐句配音（tts 75→90；本地 TTS 推理占 GPU 槽）──
    _cancel_boundary("tts", cancel_check)
    voice_used = None
    eff_voice = args.voice_id or cfg.localize_voice or ""
    eff_tts = (args.tts_backend or getattr(cfg, "tts_backend", "auto")).lower()
    # 后端选择：local=强制本地；cloud=强制云；
    # auto=未显式指定云端克隆音色且本地栈可用时优先本地（本地优先原则）
    use_local_tts = eff_tts == "local" or (
        eff_tts == "auto" and not args.voice_id and _local_tts.available()
    )
    if use_local_tts and not _local_tts.available():
        raise _local_tts.LocalTTSError(
            "tts_backend=local：qwen-tts 不可用（conda env update -f environment.yml）",
            category="environment",
        )
    # 背景音源：默认原声（人声+BGM）；开启人声分离时净化为纯伴奏
    bg_source = str(workdir / "audio.wav")
    if args.keep_background_audio and getattr(cfg, "bgm_vocal_sep", True) \
            and _music_sep.available():
        progress_cb("tts", 77.0, "人声分离（htdemucs）")
        bg_source = _music_sep.separate_vocals(bg_source, str(workdir))
    tts_segs = _filter_tts_segments(translated)
    dub_track: str | None = None
    if use_local_tts:
        # 本地 Qwen3-TTS 零样本克隆：优先显式样本；缺省克隆原片人声
        # （参考音频=已抽出的原声 wav，参考转写=本次 ASR 原文拼接）
        if args.voice_ref_audio:
            ref_audio = _ensure_local(args.voice_ref_audio, workdir)
            ref_text = args.voice_ref_text or _ref_transcript(ref_audio)
            voice_used = f"clone:{Path(ref_audio).stem}"
        else:
            ref_audio = str(workdir / "audio.wav")
            ref_text = "".join(str(s.get("text", "")) for s in segments)
            voice_used = "clone:source_speaker"
        progress_cb("tts", 80.0, f"本地 TTS 克隆配音 {len(tts_segs)} 句")
        with gpu_lane():
            dubs = _local_tts.synthesize_segments(
                tts_segs, out_dir=str(workdir / "dubs"),
                ref_audio=ref_audio, ref_text=ref_text,
                target_lang=args.target_lang or cfg.target_lang_default,
            )
        # 按原始时间戳合成配音（带静音间隔，保证音画同步）。
        # 消费 tts_segs（过滤+拆分后的同一份分段）：translated 与 dubs
        # 长度不等（≤1 字句被滤），zip(translated) 会错位前移+丢尾句
        dub_track = _build_timed_audio(tts_segs, dubs, duration_s,
                                       str(workdir / "dub.wav"))
        progress_cb("tts", 90.0, f"配音完成（{voice_used}，{len(dubs)} 段）")
    elif eff_voice:
        progress_cb("tts", 80.0, f"云端 TTS 配音 {len(tts_segs)} 句")
        try:
            dubs = _cloud_tts.synthesize_segments(
                tts_segs, voice_id=eff_voice,
                out_dir=str(workdir / "dubs"), api_key=speech_key,
                target_model=cfg.localize_tts_model,
            )
        except Exception as exc:
            logger.error("云 TTS 失败: %s: %s", type(exc).__name__, exc)
            raise
        # 同上：消费 tts_segs（与 synthesize_segments 同一份分段）防错位
        dub_track = _build_timed_audio(tts_segs, dubs, duration_s,
                                       str(workdir / "dub.wav"))
        voice_used = eff_voice
        progress_cb("tts", 90.0, f"配音完成（{voice_used}，{len(dubs)} 段）")
    else:
        progress_cb("tts", 90.0, "未配置配音，跳过 TTS")

    # ── 7) 混音合成（mix 90→97）──
    _cancel_boundary("mix", cancel_check)
    progress_cb("mix", 92.0, "混音合成")
    if dub_track is not None:
        if args.keep_background_audio:
            # 原声降 -12dB 为背景，与配音混音（保留 BGM/环境音）
            _media.mix_audio(erased_video, dub_track, out_path,
                             keep_background=True, bg_path=bg_source)
        else:
            _replace_audio(erased_video, dub_track, out_path)
    else:
        # 不配音：移除原声，仅保留擦除+字幕
        _strip_audio(erased_video, out_path)
    progress_cb("mix", 97.0, "混音完成")
    print(f"[LV] final out_path={out_path}")

    # ── 8) 字幕向量化（vectorize 97→100；增值步骤，失败降级）──
    _cancel_boundary("vectorize", cancel_check)
    progress_cb("vectorize", 98.0, "字幕向量化")
    vectorized, vectorize_note = _vectorize_segments(_pipeline_task_id(), segments, src)
    progress_cb("vectorize", 100.0, vectorize_note)

    return {
        "output_path": out_path,
        "duration_seconds": round(duration_s, 3),
        "asr_segment_count": len(segments),
        "display_segment_count": len(display_segments),
        "subtitle_region_erased": bool(regions),
        "voice_used": voice_used,
        "asr_backend": asr_backend,
        "ocr_backend": ocr_backend,
        "erase_backend": erase_used,
        "vectorized": vectorized,
    }


def _resolve_out_path(args: LocalizeVideoInput, src: str, workdir: Path) -> str:
    """产物路径推导：显式 output_path > 本地路径旁推 > URL 落任务产物区。

    - 显式 output_path：原样使用（demo/调试直连场景，行为不变）
    - 本地路径输入：源文件旁 <stem>_localized.mp4（行为不变，demo 兼容）
    - URL 输入：源路径是伪相对路径（Path("https://host/a.mp4")），
      旁推会让 ffmpeg mix 必败（SaaS 主路径：URL 总是 accepted 且
      不开放 output_path）——产物落 data_dir/outputs/<task_id>/output.mp4
      （TaskManager 注入 data_dir；下载通道经 output_paths 白名单寻址；
      TTL GC 同策略清扫；直连 invoke 无任务时用随机 id 等价落位）
    """
    if args.output_path:
        return args.output_path
    if src.startswith(("http://", "https://")):
        ctx = current_task_context()
        base = Path(ctx.data_dir) if ctx is not None and ctx.data_dir \
            else Path(load_config().localizer.data_dir)
        out_dir = base / "outputs" / _pipeline_task_id()
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / "output.mp4")
    return str(Path(src).with_stem(Path(src).stem + "_localized").with_suffix(".mp4"))


def _select_asr(cfg, speech_key: str) -> tuple[str, object]:
    """选择 ASR 后端。返回 (backend_name, fn(wav_path, language) -> segments)。

    - local  = 强制本地 faster-whisper（库缺失时 transcribe 内部显式报错）
    - cloud  = 强制 dashscope 流式
    - auto   = 本地库可导入即用本地；否则回落云；两端都不可用显式报错
    """
    chosen = (cfg.asr_backend or "auto").lower()

    def _local(p: str, lang: str | None) -> list[dict]:
        return _local_asr.transcribe_local(
            p, model=cfg.local_asr_model, device=cfg.local_asr_device, language=lang)

    def _cloud(p: str, lang: str | None) -> list[dict]:
        return _cloud_asr.transcribe_local(
            p, api_key=speech_key, sample_rate=cfg.asr_sample_rate,
            language_hints=[lang] if lang else None)

    if chosen == "local":
        return "local", _local
    if chosen == "cloud":
        return "cloud", _cloud
    if _local_asr.available():
        return "local", _local
    if speech_key:
        return "cloud", _cloud
    raise _cloud_asr.ASRError(
        "asr_backend=auto：faster-whisper 不可用且未设置 AI_SPEECH_API_KEY，无可用 ASR 后端",
        category="environment",
    )


def _select_ocr(cfg, speech_key: str) -> tuple[str, object]:
    """选择 OCR 后端。返回 (backend_name, fn(frame_paths, w, h) -> regions)。

    语义同 _select_asr：local(RapidOCR CPU) / cloud(qwen3.5-ocr) / auto。
    """
    chosen = (cfg.ocr_backend or "auto").lower()

    def _local(frames: list[str], w: int | None, h: int | None) -> list[dict]:
        return _local_ocr.locate_subtitle_region(frames, frame_width=w, frame_height=h)

    def _cloud(frames: list[str], w: int | None, h: int | None) -> list[dict]:
        return _cloud_ocr.locate_subtitle_region(
            frames, api_key=speech_key, frame_width=w, frame_height=h)

    if chosen == "local":
        return "local", _local
    if chosen == "cloud":
        return "cloud", _cloud
    if _local_ocr.available():
        return "local", _local
    if speech_key:
        return "cloud", _cloud
    raise _cloud_ocr.OCRError(
        "ocr_backend=auto：RapidOCR 不可用且未设置 AI_SPEECH_API_KEY，无可用 OCR 后端",
        category="environment",
    )


def _locate_region(src: str, duration_s: float, workdir: Path,
                   ocr_fn, cfg, *, width: int | None = None,
                   height: int | None = None) -> list[dict]:
    """离线抽样 N 帧（均匀铺开，非逐帧实时）→ OCR → 聚合字幕擦除区列表。

    可能返回多个独立区域（顶部标题 + 底部歌词）；帧数走 cfg.ocr_frame_count。
    """
    n = max(1, cfg.ocr_frame_count)
    frames = []
    for i in range(n):
        frac = (i + 0.5) / n  # 均匀取中点，避开片头片尾黑屏
        p = str(workdir / f"frame_{i:02d}.png")
        try:
            frames.append(_media.extract_frame(src, duration_s * frac, p))
        except _media.MediaError:
            continue
    if not frames:
        return []
    return ocr_fn(frames, width, height)


def _split_long_segments(segments: list[dict], max_chars: int = 20) -> list[dict]:
    """把过长的句段按自然断句拆成多行，避免"一坨字幕"覆盖全视频。

    拆分策略：在标点（。，！？；,.!?;）处切分，每段不超过 max_chars 字；
    时间按字数比例分配。输出条目统一携带 index（display 顺序），
    下游 translate（index 原样透传）与 TTS 段命名（seg_XXXX）依赖它。
    """
    import re
    result: list[dict] = []
    for seg in segments:
        text = str(seg["text"]).strip()
        duration = seg["end"] - seg["begin"]
        if len(text) <= max_chars or duration < 0.5:
            result.append(seg)
            continue
        # 按标点切分，保留分隔符
        parts = re.split(r'(?<=[。，！？；,.!?；])\s*', text)
        parts = [p for p in parts if p.strip()]
        if len(parts) <= 1:
            # 无标点可切，强制按 max_chars 硬切
            parts = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
        total_chars = sum(len(p) for p in parts)
        t = seg["begin"]
        for p in parts:
            ratio = len(p) / total_chars if total_chars else 1.0 / len(parts)
            seg_duration = duration * ratio
            result.append({
                "begin": round(t, 3),
                "end": round(t + seg_duration, 3),
                "text": p,
            })
            t += seg_duration
    return [{**s, "index": i} for i, s in enumerate(result)]


def _est_text_width(text: str, font_size: int) -> float:
    """估算 ASS 渲染宽度（PlayRes 单位）：CJK≈1.0×字号，拉丁/半角≈0.55×。"""
    return sum(
        font_size * (1.0 if ord(ch) > 0x2E7F else 0.55) for ch in text
    )


def _fit_subtitle(text: str, base_fs: int, play_res_w: int = 1280,
                  margin_lr: int = 120, min_fs: int = 14) -> tuple[int, str]:
    """按文字长度动态匹配字号，返回 (font_size, 可能折行的 text)。

    策略：先整句逐步缩号（下限 min_fs）；仍放不下则在句中找断点
    （优先标点/空格，其次几何中点）折两行，再缩号到放下。
    """
    safe_w = play_res_w - margin_lr * 2
    if _est_text_width(text, base_fs) <= safe_w:
        return base_fs, text
    # 单行缩号
    fs = base_fs
    while fs > min_fs and _est_text_width(text, fs) > safe_w:
        fs -= 1
    if _est_text_width(text, fs) <= safe_w:
        return fs, text
    # 两行折行：从几何中点向外找最近的断点字符
    mid = len(text) // 2
    split = None
    for radius in range(len(text)):
        for idx in (mid + radius, mid - radius):
            if 0 < idx < len(text) and text[idx] in " ,.!?:;、，。！？：；-/）)":
                split = idx + 1
                break
        if split is not None:
            break
    split = split or mid
    l1, l2 = text[:split].strip(), text[split:].strip()
    fs = base_fs
    while fs > min_fs and max(_est_text_width(l1, fs),
                              _est_text_width(l2, fs)) > safe_w:
        fs -= 1
    return fs, f"{l1}\\N{l2}"


_FONT_CANDIDATES = ("Noto Sans CJK SC", "Noto Serif CJK SC",
                    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "DejaVu Sans")


def _pick_font() -> str:
    """探测系统可用字体：优先 CJK（中英混排不豆腐），兜底 DejaVu（纯拉丁）。

    fonts-noto-cjk 未装的机器上 libass 会用无 CJK 字形的兜底字体渲染出
    方框乱码——所以宁可显式探测，也不假设某个字体一定存在。
    """
    import shutil
    import subprocess
    if not shutil.which("fc-match"):
        return "DejaVu Sans"
    for name in _FONT_CANDIDATES:
        try:
            out = subprocess.run(
                ["fc-match", "-f", "%{family}", name],
                capture_output=True, text=True,
                # 显式 UTF-8：防父进程 locale 被翻成 C 时解码崩溃（同 _music_sep）
                encoding="utf-8", errors="replace", timeout=10,
            ).stdout.strip()
            # fc-match 永远返回"最接近的"，要确认真的命中请求字体
            if name.split()[0].lower() in out.lower():
                return name
        except Exception:
            continue
    return "DejaVu Sans"


def _match_original_style(regions: list[dict], width: int | None,
                          height: int | None) -> tuple[int, int]:
    """从 OCR 原字幕行框推导新字幕的字号与下边距（ASS PlayRes 1280×720 空间）。

    - 字号：行高中位数 × 分辨率比 × 0.95（CJK 字形约占 em 高 0.9-1.0），
      钳制 [14, 48]——新字幕视觉上"接管"原字幕，而不是固定 22 号
    - 下边距：最低行的行底换算到 720 空间，新字幕贴原字幕位置
    - regions 空 / 尺寸缺失 → 回退默认 (22, 40)
    """
    if not regions or not width or not height:
        return 22, 40
    scale = 720.0 / float(height)
    heights = sorted(float(r["h"]) for r in regions if r.get("h"))
    if not heights:
        return 22, 40
    mid = heights[len(heights) // 2] if len(heights) % 2 else (
        (heights[len(heights) // 2 - 1] + heights[len(heights) // 2]) / 2
    )
    base_fs = max(14, min(48, int(mid * scale * 0.95)))
    bottom = max(float(r["y"]) + float(r["h"]) for r in regions)
    margin_v = max(8, min(120, int(720 - bottom * scale)))
    return base_fs, margin_v


def _write_ass(segments: list[dict], workdir: Path, cfg,
               regions: list[dict] | None = None,
               width: int | None = None, height: int | None = None) -> str:
    """把译文句段写成 ASS 字幕（底部居中，带半透明底条）。

    字号/下边距动态匹配原字幕（regions=OCR 原始行框时按行高与行底推导）；
    长句仍经 _fit_subtitle 缩号/折行防溢出（匹配字号为上限）。
    regions 缺省回退 cfg.subtitle_font_size / 固定 MarginV=40。
    """
    base_fs, margin_v = _match_original_style(regions or [], width, height)
    font_size = base_fs if regions else getattr(cfg, "subtitle_font_size", 22)
    font_name = _pick_font()
    lines = [
        "[Script Info]", "PlayResX: 1280", "PlayResY: 720",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, "
        "BorderStyle, Outline, Shadow, Alignment, MarginV, MarginL, MarginR",
        # Alignment=2: 底部居中；MarginV 由原字幕行底位置推导
        # 字体由 _pick_font 运行时探测（CJK 缺失时显式回落，避免豆腐乱码）
        f"Style: Sub,{font_name},{font_size},&H00FFFFFF,&H00000000,"
        f"1,1,0,2,{margin_v},120,120",
        "[Events]",
        "Format: Layer, Start, End, Style, Text",
    ]

    def ts(sec: float) -> str:
        h, rem = divmod(max(0.0, sec), 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):01d}:{int(m):02d}:{s:05.2f}"

    for seg in segments:
        fs, text = _fit_subtitle(str(seg["text"]).replace("\n", " "), font_size)
        styled = text if fs == font_size else f"{{\\fs{fs}}}{text}"
        lines.append(f"Dialogue: 0,{ts(seg['begin'])},{ts(seg['end'])},Sub,{styled}")
    path = workdir / "subs.ass"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _ensure_local(src: str, workdir: Path) -> str:
    """URL 输入时下载到本地（后续 ffmpeg/OCR 需要）；已本地则原样返回。"""
    if not src.startswith(("http://", "https://")):
        return src
    local = workdir / "src.mp4"
    resp = requests.get(src, stream=True, timeout=120.0)
    resp.raise_for_status()
    with open(local, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return str(local)


def _ref_transcript(ref_audio: str) -> str:
    """克隆参考样本缺转写时，本地 ASR 补转写（CUDA 预加载由 _local_asr 负责）。"""
    try:
        segs = _local_asr.transcribe_local(ref_audio)
        return "".join(str(s.get("text", "")) for s in segs).strip()
    except Exception as exc:
        raise _local_tts.LocalTTSError(
            f"参考音频转写失败（可改传 voice_ref_text）: {type(exc).__name__}: {exc}",
            category="video",
        ) from exc


def _prep_erase_regions(regions: list[dict], width: int | None,
                        height: int | None) -> list[dict]:
    """把 OCR 文本行框整理成擦除区：逐框适度外扩 + 裁边，保持紧贴文字。

    不再合成大横带——LaMa/delogo 只处理真实文字覆盖范围，避免
    "超级大横带"误伤前景主体。外扩量覆盖字形边缘/描边/阴影。
    """
    if not regions:
        return []
    out: list[dict] = []
    for r in regions:
        x0 = int(r["x"]) - 14
        y0 = int(r["y"]) - 8
        x1 = int(r["x"] + r["w"]) + 14
        y1 = int(r["y"] + r["h"]) + 8
        if width:
            x0, x1 = max(0, x0), min(int(width), x1)
        if height:
            y0, y1 = max(0, y0), min(int(height), y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue  # 退化框（噪声）跳过
        out.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
    return out


def _replace_audio(video_path: str, audio_wav: str, out_path: str) -> str:
    """用新音频替换视频原声（视频全长，音频不足部分自动补静音）。"""
    # 先取视频时长，再用 apad 补静音到该时长
    dur = _media.probe_media(video_path)[0]
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-i", audio_wav,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-af", f"apad=whole_dur={int(dur)}",  # 不足部分补静音，不截断视频
        "-t", str(dur),  # 强制输出时长 = 视频时长
        out_path,
    ]
    rc, _, err = _media.run_ffmpeg(cmd)
    if rc != 0:
        raise _media.MediaError(f"音轨替换失败: {err[-200:]}")
    return out_path


def _build_timed_audio(segments: list[dict], dubs: list[str], duration_s: float,
                       out_path: str) -> str:
    """把逐句 TTS 按原始时间戳拼回完整音轨（带静音间隔，保证音画同步）。

    segments: [{"begin": float, "end": float, "text": str, "index": int}, ...]
    dubs: 对应的 TTS mp3 路径列表（与 segments 等长同序）
    """
    # 统一转为 wav + 静音间隔，再用 amix 合成
    tmp_dir = Path(out_path).parent
    wavs: list[tuple[float, str]] = []  # (delay_ms, wav_path)
    # wav 命名按 zip 位置（而非 seg['index']）：调用方传入的 segments
    # 已是过滤后的 tts_segs（index 可能非连续/缺失），位置必唯一防覆盖
    for pos, (seg, dub) in enumerate(zip(segments, dubs)):
        # mp3 → wav；配音比原句长时用 atempo 加速压到句时长内（上限 2 倍速，
        # 再长接受轻微溢出，避免过度压缩变调）
        wav_p = str(tmp_dir / f"dub_{pos:04d}.wav")
        cmd = ["ffmpeg", "-y", "-i", dub]
        seg_dur = seg["end"] - seg["begin"]
        if seg_dur > 0.3:
            tempo = _media.probe_media(dub)[0] / seg_dur
            if tempo > 1.02:  # 2% 以内不压，避免徒增变速伪影
                cmd += ["-filter:a", f"atempo={min(tempo, 2.0):.4f}"]
        cmd += ["-ac", "1", "-ar", "16000", wav_p]
        rc, _, err = _media.run_ffmpeg(cmd)
        if rc != 0:
            raise _media.MediaError(f"dub 转 wav 失败: {err[-100:]}")
        delay_ms = int(seg["begin"] * 1000)
        wavs.append((delay_ms, wav_p))
    if not wavs:
        raise _media.MediaError("无配音片段可合成")
    # 用 adelay + amix 合成：每路延迟到对应时间点（单声道用 adelay=ms）
    # 每路独立：[0:a]adelay=ms[d0];[1:a]adelay=ms[d1];...;[d0][d1]amix=...
    delays = ";".join(
        f"[{i}:a]adelay={ms}[d{i}]" for i, (ms, _) in enumerate(wavs)
    )
    mix_inputs = "".join(f"[d{i}]" for i in range(len(wavs)))
    n = len(wavs)
    af = f"{delays};{mix_inputs}amix=inputs={n}:duration=longest:dropout_transition=0:normalize=0[aout]"
    cmd = [
        "ffmpeg", "-y",
    ]
    for _, wp in wavs:
        cmd += ["-i", wp]
    cmd += [
        "-filter_complex", af,
        "-map", "[aout]",
        "-c:a", "pcm_s16le",
        "-ar", "16000",
        out_path,
    ]
    rc, _, err = _media.run_ffmpeg(cmd)
    if rc != 0:
        raise _media.MediaError(f"配音合成失败: {err[-200:]}")
    return out_path


def _strip_audio(video_path: str, out_path: str) -> str:
    """移除视频音轨（不配音时使用）。"""
    cmd = ["ffmpeg", "-y", "-i", video_path, "-c:v", "copy", "-an", out_path]
    rc, _, err = _media.run_ffmpeg(cmd)
    if rc != 0:
        raise _media.MediaError(f"移除音轨失败: {err[-200:]}")
    return out_path


def _concat_wavs(wavs: list[str], out_path: str) -> str:
    """拼接逐句音频为整轨。全部转绝对路径：ffmpeg concat demuxer 按列表文件
    所在目录解析相对路径，跨目录时必炸。"""
    lst = Path(out_path).resolve().with_suffix(".txt")
    lst.write_text("\n".join(f"file '{Path(w).resolve()}'" for w in wavs), encoding="utf-8")
    rc, _, err = _media.run_ffmpeg([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-c", "copy", str(Path(out_path).resolve()),
    ])
    if rc != 0:
        raise _media.MediaError(f"配音拼接失败: {err[-200:]}")
    return out_path


def _ok_shell(inp, workdir, *, duration: float, msg: str,
              erased: bool, voice) -> SkillOutput[LocalizeVideoReport]:
    """空结果但非错误（如无人声）：正常返回 + warning。"""
    chain = ReasoningChain(
        conclusion=msg, triggered_rules=[], evidence=[],
        causal_analysis="ASR 返回空句段", risk_note="确认视频是否含人声。",
    )
    return SkillOutput(
        data=LocalizeVideoReport(
            output_path="", duration_seconds=round(duration, 3),
            asr_segment_count=0, subtitle_region_erased=erased,
            voice_used=voice, warning=msg,
        ),
        reasoning=[chain], confidence=0.5, sample_size=0,
        degraded=True, degradation_reason="empty_asr",
    )


def _fail(inp, warning: str, category: str, *,
          retriable: bool = False) -> SkillOutput[LocalizeVideoReport]:
    """统一 degraded 返回：脱敏（warning 不带 key/host）。"""
    report = LocalizeVideoReport(
        output_path="", duration_seconds=0.0, asr_segment_count=0,
        subtitle_region_erased=False, voice_used=None,
        failure_category=category, retriable=retriable or is_retriable(category),
        warning=f"{warning}（{category}）",
    )
    chain = ReasoningChain(
        conclusion=f"本地化失败（{category}）",
        triggered_rules=[], evidence=[],
        causal_analysis="流水线预检或云调用阶段失败，见 warning 字段",
        risk_note=(
            "按 failure_category 决策：environment 先检查网络/key；"
            "video 修输入；transient 可重试。"
        ),
    )
    return SkillOutput(
        data=report, reasoning=[chain], confidence=0.0, sample_size=0,
        degraded=True, degradation_reason=category,
    )
