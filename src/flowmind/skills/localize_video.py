"""localize_video 技能：视频本地化流水线（单视频同步，本地优先）。

流水线：ffmpeg 提音轨 → ASR 句级时间戳（本地 faster-whisper GPU 优先，
auto 回落百炼 Paraformer）→ 说话人分离（funasr CAM++，CPU，逐说话人打标签）
→ OCR 定位原字幕区（本地 RapidOCR 优先，auto 回落 qwen3.5-ocr）
→ LongCat 翻译（可选术语表 / 文化替换 / 长度预算）→ ffmpeg 擦除原字幕区
→ 克隆/预设音色逐句配音（多人对话按说话人路由参考音频；音色 ID 由百炼声音复刻
提前生成，配音时长分级 atempo 对齐原句）→ 混音（可选保留 -12dB 背景音）
+ ASS 字幕烧录 → 字幕向量化。

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
    _diarize,
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
    watermark_bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description="固定水印区域归一化坐标 (x0,y0,x1,y1)，各坐标范围 0-1",
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
    diarize_backend: str | None = Field(
        default=None,
        description="说话人分离后端：auto（本地栈可用则分离，否则跳过并单音色克隆）/ "
                    "local（强制，不可用显式报错）/ off（跳过）；"
                    "None=读 config.diarize_backend",
    )
    speaker_voices: dict[str, str] | None = Field(
        default=None,
        description="说话人 → 云端预设音色 ID（如 {\"spk0\": \"longanhuan_v3.6\"}），"
                    "仅云端 TTS 路径生效；None=整片单 voice_id",
    )

    # ── 步骤选择（2026-10-31 SaaS 化）：流水线九阶段可裁剪 ──
    steps: list[str] | None = Field(
        default=None,
        description="白名单：只执行这些步骤（如 [\"extract_audio\",\"asr\",\"translate\"]）。"
                    "None=全部执行。可用步骤见 GET /api/v1/skills/localize_video/steps 或 "
                    "manifest 的 stage_schema。依赖不满足会 422。",
    )
    skip_steps: list[str] | None = Field(
        default=None,
        description="黑名单：从全集中剔除这些步骤（如 [\"erase\",\"tts\"]）。"
                    "必需步骤（extract_audio/asr）不可跳过；依赖被剔除会 422。",
    )

    @field_validator("steps", "skip_steps")
    @classmethod
    def _validate_steps(cls, v: list[str] | None) -> list[str] | None:
        if not v:
            return v
        from flowmind.pipeline_steps import steps_for

        valid = {s.id for s in steps_for("localize_video")}
        unknown = [x for x in v if x not in valid]
        if unknown:
            raise ValueError(f"未知步骤 {unknown}；可选：{sorted(valid)}")
        return v

    @field_validator("video_path")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("video_path 不能为空")
        return v.strip()

    @field_validator("watermark_bbox")
    @classmethod
    def _valid_watermark_bbox(cls, v):
        if v is not None and (
            any(coord < 0 or coord > 1 for coord in v)
            or v[2] <= v[0]
            or v[3] <= v[1]
        ):
            raise ValueError("watermark_bbox 必须是 0-1 范围内的非空区域")
        return v


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
    watermark_removed: bool = False
    # 字幕向量化（增量字段）：True=已写入 Milvus；False=未写入（未配置或失败，
    # 由阶段消息与日志区分）；None=开关关闭
    vectorized: bool | None = None
    # 说话人分离（增量字段）：实际后端 "cam++" / "none"；quality 见 quality 块
    diarize_backend: str | None = None
    # 质量自检报告（增量字段）：speakers / diarize_backend / dub /
    # translate_over_budget / erase_coverage
    quality: dict | None = None
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


@skill(id="localize_video", name="视频本地化流水线（本地优先）", version=_VERSION)
def localize_video(inp: LocalizeVideoInput) -> SkillOutput[LocalizeVideoReport]:
    """ASR → LLM 翻译 → 原字幕擦除 → 克隆 TTS 配音 → 合成。

    双匹配策略：OCR 负责定位原字幕区域（供擦除）；翻译文本一律以 ASR 为准。
    九阶段主体在 run_localization_pipeline（进度回调 / 协作式取消 /
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
    eff_tts_backend = (inp.tts_backend or cfg.tts_backend).lower()
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
    # expanduser() 不可省：data_dir 默认 "~/flowmind-data"，不展开会在 CWD 下
    # 建出字面名为 "~" 的目录（仓库根目录被污染，与 TaskManager 展开行为不一致）。
    ctx = current_task_context()
    if ctx is not None and ctx.workdir is not None:
        workdir = Path(ctx.workdir)
    else:
        workdir = Path(cfg.data_dir).expanduser() / "tasks" / uuid.uuid4().hex / "work"
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
    except _diarize.DiarizeError as exc:
        # 说话人分离异常：auto 模式已在流水线内降级，只有 local/off 语义
        # 与环境类错误会走到这里（与 _music_sep 同写法）
        return _fail(inp, f"{exc}（可设 diarize_backend=off 跳过说话人分离）",
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
    # 质量自检（增量）：说话人数 / 配音溢出 / 译文超预算 → 进 ReasoningChain 文本
    quality = report.get("quality") or {}
    speakers = quality.get("speakers") or []
    dub_stats = quality.get("dub") or {}
    over_budget = quality.get("translate_over_budget") or []
    overflow = int(dub_stats.get("overflow") or 0)
    conclusion = (
        f"本地化完成：{report['display_segment_count']} 句"
        f"（ASR {report['asr_segment_count']} 句拆分）、"
        f"{'已擦除[' + report['erase_backend'] + ']' if regions_erased else '未检出'}"
        f"原字幕区、{'克隆配音 ' + voice_used if voice_used else '未配音'}"
        f"、说话人 {len(speakers)} 位"
    )
    if overflow:
        conclusion += f"、配音溢出 {overflow} 句"
    risk_note = "译文为 LLM 生成，正式投放前建议人工抽查。"
    hints = []
    if overflow:
        hints.append(f"配音溢出 {overflow} 句")
    if over_budget:
        hints.append(f"译文超预算 {len(over_budget)} 句")
    if hints:
        risk_note += f"检测到{'、'.join(hints)}，建议人工抽查对应片段。"
    chain = ReasoningChain(
        conclusion=conclusion,
        triggered_rules=[], evidence=[],
        causal_analysis=(
            f"ASR[{report['asr_backend']}] {report['asr_segment_count']} 句 → "
            f"说话人分离[{report['diarize_backend']}] → "
            f"拆分为 {report['display_segment_count']} 句 → LLM 翻译 → "
            f"OCR[{report['ocr_backend']}] 定位 → 擦除[{report['erase_backend']}] → "
            f"TTS → ffmpeg 合成"
        ),
        risk_note=risk_note,
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
            watermark_removed=report["watermark_removed"],
            vectorized=report["vectorized"],
            diarize_backend=report.get("diarize_backend"),
            quality=report.get("quality"),
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


# ── 步骤跟踪辅助（2026-10-31 SaaS 化）──

def _pipeline_tracker(args):
    """构造 StepTracker：读 args.steps / skip_steps + TaskContext.enabled_steps。

    返回 (tracker, enabled_set)。enabled_set 为空 = 全部执行（向后兼容）。
    步骤状态经 TaskContext.step_cb 上报（无 TaskManager 直连时静默）。
    """
    from flowmind.pipeline_steps import StepTracker
    from flowmind.tasks import current_task_context

    ctx = current_task_context()
    enabled = tuple(ctx.enabled_steps) if ctx is not None else ()
    if not enabled:
        # TaskManager 未注入（直连 invoke / demo）或该技能无注册表：
        # 从 args 解析选择；args 也没传 → enabled 保持空 = 全跑
        from flowmind.pipeline_steps import resolve_selection
        sel, _ = resolve_selection(
            "localize_video",
            steps=getattr(args, "steps", None),
            skip_steps=getattr(args, "skip_steps", None),
        )
        enabled = tuple(x.id for x in sel)
    tracker = StepTracker(
        "localize_video", enabled=enabled,
        on_change=(ctx.step_cb if ctx is not None else None),
    )
    # 起点快照：任务一启动就让前端看到完整九步（含 skipped），而不是等到首步结束
    if ctx is not None:
        try:
            ctx.step_cb(tracker.build())
        except Exception:  # noqa: BLE001  上报失败不阻断
            pass
    return tracker, set(enabled)


# 阶段起始时刻（step_id → monotonic），供 _stage_end 计算耗时
_STAGE_T0: dict[str, float] = {}


def _stage_begin(args, tracker, step_id: str) -> None:
    """阶段入口：把该步标为 running（未启用的步在 tracker 里已是 skipped）。"""
    import time as _time

    _STAGE_T0[step_id] = _time.monotonic()
    tracker._mark(step_id, status="running", started_at=_iso_now(),
                  ended_at=None, duration_ms=None, skipped=False,
                  skip_reason=None)


def _stage_end(tracker, step_id: str, *, backend: str | None = None,
               note: str | None = None, skipped: str | None = None) -> None:
    """阶段出口：标 done（或 skewed/skipped 带原因），并记录实际后端。"""
    import time as _time

    t0 = _STAGE_T0.pop(step_id, None)
    dur = int((_time.monotonic() - t0) * 1000) if t0 is not None else None
    if skipped:
        tracker._mark(step_id, status="skipped", skipped=True,
                      skip_reason=skipped, ended_at=_iso_now(),
                      duration_ms=dur, note=skipped)
        return
    fields: dict = {"status": "done", "ended_at": _iso_now(), "skipped": False,
                    "skip_reason": None, "duration_ms": dur}
    if backend:
        fields["backend"] = backend
    if note:
        fields["note"] = note
    tracker._mark(step_id, **fields)


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def run_localization_pipeline(args: LocalizeVideoInput, workdir: Path,
                              progress_cb, cancel_check) -> dict:
    """九阶段本地化流水线（进度回调 + 协作取消 + GPU 串行）。

    阶段与进度区间（pct 0-100）::

        extract_audio 0→8 → asr 8→20 → diarize 20→32 → ocr 32→44
        → translate 44→57 → erase 57→75 → tts 75→90 → mix 90→97 → vectorize 97→100

    - 每阶段边界调 cancel_check()；命中抛 tasks.CancelledError（协作取消）
    - GPU 推理阶段（本地 ASR / LaMa 擦除 / 本地 TTS）经 tasks.gpu.gpu_lane
      独占单卡槽；云 API 阶段（dashscope/qwen-ocr/LongCat/云 TTS）不占；
      说话人分离（CAM++）默认走 CPU，不占显存预算
    - 返回 report dict；无人声提前返回 {"empty_asr": True, ...}
    - 模块内辅助（_locate_region/_mix_timed_audio 等）经模块全局查找
      调用——demo 的 monkeypatch mock 依赖此行为，调用形状不可变
    """
    cfg = load_config().localizer
    speech_key = get_api_key(KEY_SPEECH)
    llm_key = get_api_key(KEY_LLM)
    src = args.video_path
    is_url = src.startswith(("http://", "https://"))

    # ── 步骤跟踪 + 裁剪（2026-10-31 SaaS 化）──
    # 调用方可通过 args.steps / args.skip_steps 选择只跑部分阶段（如"只转写
    # 不擦除""只翻译不配音"）。enabled 为空 = 全跑（向后兼容）。
    # _on(): 该步是否执行；with _step(...) 记录 running/done/skipped/failed。
    _tracker, _enabled = _pipeline_tracker(args)
    def _on(step_id: str) -> bool:
        return not _enabled or step_id in _enabled

    # ── 1) 提取音轨（extract_audio 0→8）──
    _stage_begin(args, _tracker, "extract_audio")
    progress_cb("extract_audio", 1.0, "提取音轨")
    audio_path = _media.extract_audio(
        src, str(workdir / "audio.wav"),
        sample_rate=cfg.asr_sample_rate,
    )
    progress_cb("extract_audio", 8.0, "音轨已提取")
    _stage_end(_tracker, "extract_audio")

    # ── 2) ASR（asr 8→20；本地 faster-whisper 占 GPU 槽，auto 回落云）──
    _cancel_boundary("asr", cancel_check)
    _stage_begin(args, _tracker, "asr")
    asr_backend, asr_fn = _select_asr(cfg, speech_key)
    source_lang = args.source_lang or cfg.source_lang_default
    progress_cb("asr", 9.0, f"ASR 转写（{asr_backend}）")
    if asr_backend == "local":
        with gpu_lane():
            segments = asr_fn(audio_path, source_lang)
    else:
        segments = asr_fn(audio_path, source_lang)
    duration_s, width, height = (None, None, None)
    if not is_url:
        duration_s, width, height = _media.probe_media(src)
    if not segments:
        # 无人声提前返回：ASR 步如实收尾（done 但 note 说明零识别），
        # 其余未执行步骤由 manager 终态收尾兜底——不留 running 假态
        _stage_end(_tracker, "asr", backend=asr_backend,
                   note="未识别到语音（可能无人声）")
        return {"empty_asr": True, "duration_seconds": duration_s or 0.0,
                "warning": "ASR 未识别到任何语音（可能无人声）"}
    if duration_s is None:
        duration_s = segments[-1]["end"]
    progress_cb("asr", 20.0, f"ASR 完成：{len(segments)} 句")
    _stage_end(_tracker, "asr", backend=asr_backend)

    # ── 3) 说话人分离（diarize 20→32；CAM++ 默认 CPU，不占 GPU 槽）──
    # 后端语义（对齐仓库 local/cloud/auto 惯例）：
    #   local：栈不可用 → 抛 DiarizeError(category=environment)
    #   auto ：不可用/失败 → 跳过，整片单音色克隆，并在 report 显式记录原因
    #   off  ：直接跳过
    _cancel_boundary("diarize", cancel_check)
    if _on("diarize"):
        _stage_begin(args, _tracker, "diarize")
    eff_diarize = (args.diarize_backend or cfg.diarize_backend).lower()
    want_diarize = eff_diarize != "off" and _on("diarize")
    if not _on("diarize"):
        diarize_note = "调用方未选择说话人分离"
    # 人声轨/伴奏轨一次算好：diarize 用净人声（多人混响会污染声纹），
    # 背景音直接复用伴奏轨——避免 demucs 在 diarize 与 tts 两阶段各跑一遍
    diarize_input = audio_path
    bg_source: str | None = None
    need_stems = bool(cfg.bgm_vocal_sep) and _music_sep.available() and (
        args.keep_background_audio or (want_diarize and _diarize.available()))
    if need_stems:
        progress_cb("diarize", 21.0, "人声分离（htdemucs）")
        vocals, instr = _music_sep.separate_stems(audio_path, str(workdir))
        diarize_input = vocals
        if args.keep_background_audio:
            bg_source = instr
    diarize_backend = "none"
    diarize_note: str | None = None
    refs: dict[str, dict] = {}
    if not want_diarize:
        diarize_note = "diarize_backend=off：跳过说话人分离"
    elif not _diarize.available():
        if eff_diarize == "local":
            raise _diarize.DiarizeError(
                "diarize_backend=local：funasr 不可用"
                "（conda env update -n flowmind -f environment.yml）",
                category="environment",
            )
        diarize_note = "本地说话人分离不可用，已回落整片单音色克隆"
    else:
        progress_cb("diarize", 23.0, f"说话人分离（{cfg.local_diarize_model}）")
        try:
            turns = _diarize.diarize(
                diarize_input,
                device=cfg.local_diarize_device,
                model=cfg.local_diarize_model,
                merge_thr=cfg.diarize_merge_thr,
                min_cluster_seconds=cfg.diarize_min_cluster_seconds,
            )
            if turns:
                assigned = _diarize.assign_speakers(segments, turns)
                new_refs = _diarize.build_speaker_refs(
                    assigned, diarize_input, str(workdir / "speaker_refs"),
                    min_seconds=cfg.speaker_ref_min_seconds,
                    target_seconds=cfg.speaker_ref_target_seconds,
                    max_seconds=cfg.speaker_ref_max_seconds,
                )
            else:
                assigned, new_refs = segments, {}
        except Exception as exc:  # noqa: BLE001  auto 降级；local 模式照常抛出
            if eff_diarize == "local":
                raise
            logger.warning("说话人分离失败（auto 降级为整片单音色克隆）: %s: %s",
                           type(exc).__name__, exc)
            diarize_note = (f"本地说话人分离失败（{type(exc).__name__}），"
                            "已回落整片单音色克隆")
        else:
            if turns:
                segments, refs = assigned, new_refs
                diarize_backend = "cam++"
            else:
                diarize_note = "说话人分离未检出语音，回落整片单音色克隆"
    progress_cb("diarize", 32.0,
                f"说话人分离完成（{diarize_backend}，{len(refs)} 位说话人）"
                if refs else f"说话人分离跳过：{diarize_note or '无可用轮次'}")
    _stage_end(_tracker, "diarize",
               backend=diarize_backend,
               skipped=(None if refs else (diarize_note or "无可用轮次")))

    # ── 4) OCR 定位原字幕区（ocr 32→44；RapidOCR 为 CPU，不占 GPU 槽）──
    _cancel_boundary("ocr", cancel_check)
    _stage_begin(args, _tracker, "ocr")
    if _on("ocr"):
        ocr_backend, ocr_fn = _select_ocr(cfg, speech_key)
        progress_cb("ocr", 33.0, f"OCR 定位字幕区（{ocr_backend}）")
        region = _locate_region(src, duration_s, workdir, ocr_fn, cfg,
                                width=width, height=height)
    else:
        # 未选择 OCR：无原字幕区域 → 不擦除（erase 硬依赖 ocr，已在注册表校验）
        ocr_backend, region = "none", None
        progress_cb("ocr", 33.0, "跳过 OCR（调用方未选择）")
        progress_cb("ocr", 44.0, "OCR 已跳过")
    progress_cb("ocr", 44.0, f"OCR 完成：{len(region or [])} 个候选区")
    _stage_end(_tracker, "ocr", backend=ocr_backend,
               skipped=("调用方未选择 OCR" if not _on("ocr") else None))

    # ── 5) LLM 翻译（translate 44→57；长句先拆分，翻译更稳定）──
    _cancel_boundary("translate", cancel_check)
    _stage_begin(args, _tracker, "translate")
    display_segments = _split_long_segments(segments)
    if not _on("translate"):
        # 未选择翻译：译文 = 原文（同语言场景），下游 TTS/字幕照常可用
        progress_cb("translate", 45.0, "跳过翻译（调用方未选择）")
        progress_cb("translate", 57.0, "翻译已跳过（沿用原文）")
        translated = display_segments
        target_lang = args.target_lang or cfg.target_lang_default
        _stage_end(_tracker, "translate", skipped="调用方未选择翻译")
    else:
        target_lang = args.target_lang or cfg.target_lang_default
        progress_cb("translate", 45.0,
                    f"翻译 {len(display_segments)} 句（{len(segments)} 句 ASR 拆分）")
        # 术语表 / 文化替换 / 长度预算（术语表路径为空则不启用；三者均注入系统提示，
        # 不新增 LLM 调用轮次）；load_glossary 的异常由入口 except 链统一兜底
        glossary = culture_rules = None
        if cfg.glossary_path:
            glossary, culture_rules = _llm_translate.load_glossary(cfg.glossary_path)
        length_budget = _llm_translate.estimate_length_budget(
            display_segments, target_lang,
            speed_factor=cfg.dub_speed_factors or None)
        # 协议/base/model 从 .env 读（AI_LLM_*），未设置回落 LongCat Anthropic 默认
        translated = _llm_translate.translate_segments(
            display_segments,
            target_lang=target_lang,
            source_lang=args.source_lang or cfg.source_lang_default,
            api_key=llm_key,
            api_base=os.environ.get("AI_LLM_BASE_URL") or _llm_translate.DEFAULT_BASE,
            model=os.environ.get("AI_LLM_MODEL") or cfg.localize_llm_model,
            protocol=(os.environ.get("AI_LLM_PROTOCOL") or "anthropic").lower(),
            glossary=glossary, culture_rules=culture_rules, length_budget=length_budget,
        )
        progress_cb("translate", 57.0, f"翻译完成：{len(translated)} 句")
        _stage_end(_tracker, "translate", backend="llm")

    # ── 6) 擦除原字幕区 + 烧译文字幕（erase 57→75；LaMa 推理占 GPU 槽）──
    _cancel_boundary("erase", cancel_check)
    _stage_begin(args, _tracker, "erase")
    regions = region or []
    # 字号/位置匹配原字幕：把 OCR 原始行框传给 _write_ass
    ass_path = _write_ass(translated, workdir, cfg,
                          regions=regions, width=width, height=height)
    out_path = _resolve_out_path(args, src, workdir)
    # 精准擦除：OCR 文本行框逐框外扩（不再合成大横带误伤前景主体）
    subtitle_erase_regs = _prep_erase_regions(regions, width, height)
    watermark_regions = _watermark_region(args.watermark_bbox, width, height)
    erase_regs = subtitle_erase_regs + watermark_regions
    src_local = _ensure_local(src, workdir)
    # 擦除后端：auto=LaMa 可用则用（复杂背景无拉丝）否则 delogo；
    # local=强制 LaMa（缺失/失败显式报错，不静默回落）；delogo=旧硬横带
    erase_used = "delogo"
    eff_erase = (args.erase_backend or getattr(cfg, "erase_backend", "auto")).lower()
    progress_cb(
        "erase", 57.0,
        f"擦除字幕/水印（{eff_erase}，{len(erase_regs)} 框）",
    )
    if not _on("erase"):
        # 未选择擦除：源片直通（不擦、不烧字幕），mix 直接以此为输入
        erased_video = src_local
        progress_cb("erase", 57.0, "跳过擦除（调用方未选择）")
    elif erase_regs and eff_erase in ("auto", "local", "lama"):
        if _inpaint.available():
            with gpu_lane():
                _inpaint.erase_regions(
                    src_local, subtitle_erase_regs,
                    str(workdir / "inpainted.mp4"), str(workdir),
                    solid_regions=watermark_regions,
                )
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
    if _on("erase") and erase_used == "delogo":
        erased_video = _media.burn_subtitles(
            src_local, str(workdir / "subbed.mp4"), ass_path,
            erase_regions=erase_regs or None,
        )
    progress_cb("erase", 75.0,
                "擦除已跳过（调用方未选择）" if not _on("erase")
                else f"擦除完成（{erase_used}）")
    _stage_end(_tracker, "erase", backend=erase_used,
               skipped=("调用方未选择擦除" if not _on("erase") else None))

    # ── 7) TTS 逐句配音（tts 75→90；本地 TTS 推理占 GPU 槽）──
    _cancel_boundary("tts", cancel_check)
    _stage_begin(args, _tracker, "tts")
    want_tts = _on("tts")
    progress_cb("tts", 76.0,
                "配音（TTS）" if want_tts else "跳过配音（调用方未选择）")
    voice_used = None
    eff_voice = args.voice_id or cfg.localize_voice or ""
    if not eff_voice and args.speaker_voices:
        # 只传了「说话人→音色」映射时，用首个音色兜底整片 voice_id，
        # 避免"有映射却因 voice_id 为空而整段跳过 TTS"的静默空转
        eff_voice = next(iter(args.speaker_voices.values()))
    eff_tts = (args.tts_backend or cfg.tts_backend).lower()
    # 后端选择：local=强制本地；cloud=强制云；
    # auto=未显式指定云端克隆音色且本地栈可用时优先本地（本地优先原则）
    if not want_tts:
        # 调用方显式跳过配音：直接产出"无配音"路径（mix 会做 strip/mix 兜底）
        dub_track, dubs, dub_stats = None, [], {}
        use_local_tts = False
        progress_cb("tts", 90.0, "配音已跳过（调用方未选择）")
        _stage_end(_tracker, "tts", skipped="调用方未选择配音")
    else:
        progress_cb("tts", 78.0, "配音准备")
    use_local_tts = use_local_tts if not want_tts else (
        eff_tts == "local" or (
            eff_tts == "auto" and not args.voice_id and _local_tts.available()))
    if want_tts and use_local_tts and not _local_tts.available():
        raise _local_tts.LocalTTSError(
            "tts_backend=local：qwen-tts 不可用（conda env update -f environment.yml）",
            category="environment",
        )
    if use_local_tts and cfg.tts_engine != "qwen3-tts":
        # IndexTTS2 spike 未通过（上游硬钉 torch==2.8.*，与 Pascal 6.1 的 torch 钉版
        # 冲突，见 CLAUDE.md）：配置切到未接入的引擎时显式报错，不静默回落
        raise _local_tts.LocalTTSError(
            f"tts_engine={cfg.tts_engine} 尚未接入（仅支持 qwen3-tts；"
            "IndexTTS2 因 torch 钉版冲突未通过可行性验证）",
            category="environment",
        )
    # 背景音源：diarize 阶段若已分离出伴奏轨则直接复用（不重复跑 demucs）；
    # 否则默认原声（人声+BGM），开启人声分离时按需惰性净化为纯伴奏
    if bg_source is None:
        bg_source = str(workdir / "audio.wav")
        if args.keep_background_audio and cfg.bgm_vocal_sep \
                and _music_sep.available():
            progress_cb("tts", 77.0, "人声分离（htdemucs）")
            bg_source = _music_sep.separate_vocals(bg_source, str(workdir))
    tts_segs = _filter_tts_segments(translated)
    dub_track: str | None = None
    dub_stats: dict = {}
    if use_local_tts:
        # 本地 Qwen3-TTS 零样本克隆：优先显式样本；缺省克隆原片人声
        # （参考音频=已抽出的原声 wav，参考转写=本次 ASR 原文拼接）
        if args.voice_ref_audio:
            # 显式参考样本优先级最高：不做逐说话人路由
            ref_audio = _ensure_local(args.voice_ref_audio, workdir)
            ref_text = args.voice_ref_text or _ref_transcript(ref_audio)
            voice_used = f"clone:{Path(ref_audio).stem}"
        else:
            ref_audio = str(workdir / "audio.wav")
            ref_text = "".join(str(s.get("text", "")) for s in segments)
            voice_used = "clone:source_speaker"
            if len(refs) >= 2:
                # 逐说话人音色克隆：按 seg["speaker"] 路由各自聚合的参考音频；
                # 未命中 refs 的句段不写 ref 键，回落上面的整片单参考
                for seg in tts_segs:
                    ref = refs.get(str(seg.get("speaker") or ""))
                    if ref:
                        seg["ref_audio"] = ref["ref_audio"]
                        seg["ref_text"] = ref["ref_text"]
                voice_used = f"clone:multi_speaker(n={len(refs)})"
        progress_cb("tts", 80.0, f"本地 TTS 克隆配音 {len(tts_segs)} 句")
        with gpu_lane():
            dubs = _local_tts.synthesize_segments(
                tts_segs, out_dir=str(workdir / "dubs"),
                ref_audio=ref_audio, ref_text=ref_text,
                target_lang=target_lang,
            )
        # 按原始时间戳合成配音（带静音间隔，保证音画同步）。
        # 消费 tts_segs（过滤+拆分后的同一份分段）：translated 与 dubs
        # 长度不等（≤1 字句被滤），zip(translated) 会错位前移+丢尾句
        dub_track, dub_stats = _mix_timed_audio(
            tts_segs, dubs, duration_s, str(workdir / "dub.wav"),
            max_tempo=cfg.dub_max_tempo)
        progress_cb("tts", 90.0, f"配音完成（{voice_used}，{len(dubs)} 段）")
    elif eff_voice:
        progress_cb("tts", 80.0, f"云端 TTS 配音 {len(tts_segs)} 句")
        try:
            dubs = _cloud_tts.synthesize_segments(
                tts_segs, voice_id=eff_voice,
                out_dir=str(workdir / "dubs"), api_key=speech_key,
                target_model=cfg.localize_tts_model,
                speaker_voices=args.speaker_voices,
            )
        except Exception as exc:
            logger.error("云 TTS 失败: %s: %s", type(exc).__name__, exc)
            raise
        # 同上：消费 tts_segs（与 synthesize_segments 同一份分段）防错位
        dub_track, dub_stats = _mix_timed_audio(
            tts_segs, dubs, duration_s, str(workdir / "dub.wav"),
            max_tempo=cfg.dub_max_tempo)
        voice_used = (f"cloud:multi_speaker(n={len(args.speaker_voices)})"
                      if args.speaker_voices else eff_voice)
        progress_cb("tts", 90.0, f"配音完成（{voice_used}，{len(dubs)} 段）")
        _stage_end(_tracker, "tts", backend=(voice_used or "unknown"))
    else:
        progress_cb("tts", 90.0, "未配置配音，跳过 TTS")
        _stage_end(_tracker, "tts", skipped="未配置配音（无 voice_id）")

    # ── 8) 混音合成（mix 90→97）──
    _cancel_boundary("mix", cancel_check)
    _stage_begin(args, _tracker, "mix")
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
    _stage_end(_tracker, "mix")
    print(f"[LV] final out_path={out_path}")

    # ── 9) 字幕向量化（vectorize 97→100；增值步骤，失败降级）──
    _cancel_boundary("vectorize", cancel_check)
    _stage_begin(args, _tracker, "vectorize")
    progress_cb("vectorize", 98.0, "字幕向量化")
    if _on("vectorize"):
        vectorized, vectorize_note = _vectorize_segments(
            _pipeline_task_id(), segments, src)
    else:
        vectorized, vectorize_note = False, "调用方未选择向量化"
        progress_cb("vectorize", 98.0, "跳过向量化（调用方未选择）")
    progress_cb("vectorize", 100.0, vectorize_note)
    _stage_end(_tracker, "vectorize", note=vectorize_note,
               skipped=("调用方未选择向量化" if not _on("vectorize") else None))

    # 质量自检报告（增量）：说话人参考音频 / 分离后端 / 配音溢出统计 /
    # 译文超预算句段 / 擦除覆盖框数
    quality = {
        "speakers": [
            {
                "id": sid,
                "segments": sum(
                    1 for s in segments if str(s.get("speaker") or "") == sid),
                "ref_seconds": refs[sid]["seconds"],
                "low_confidence": refs[sid]["low_confidence"],
            }
            for sid in sorted(refs)
        ],
        "diarize_backend": diarize_backend,
        "dub": dub_stats,
        "translate_over_budget": [
            s["index"] for s in tts_segs if s.get("over_budget")
        ],
        "erase_coverage": len(erase_regs),
    }
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
        "watermark_removed": bool(watermark_regions),
        "vectorized": vectorized,
        "diarize_backend": diarize_backend,
        "diarize_note": diarize_note,
        "quality": quality,
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
        base = (Path(ctx.data_dir) if ctx is not None and ctx.data_dir
                else Path(load_config().localizer.data_dir)).expanduser()
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
    chosen = (getattr(cfg, "asr_backend", None) or "auto").lower()

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
    chosen = (getattr(cfg, "ocr_backend", None) or "auto").lower()

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
    源句段的非时间/非文本键（如 ``speaker``）原样带到拆分后的每个子句段，
    保证逐说话人音色路由在拆分后仍能命中。
    """
    import re
    result: list[dict] = []
    for seg in segments:
        text = str(seg["text"]).strip()
        duration = seg["end"] - seg["begin"]
        # 非时间/非文本键（speaker 等）原样携带，拆分后不丢
        extra = {k: v for k, v in seg.items()
                 if k not in ("begin", "end", "text", "index")}
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
                **extra,
                "begin": round(t, 3),
                "end": round(t + seg_duration, 3),
                "text": p,
            })
            t += seg_duration
    return [{**s, "index": i} for i, s in enumerate(result)]


def _est_text_width(text: str, font_size: int) -> float:
    """估算 ASS 渲染宽度（像素）：CJK≈0.70×字号，拉丁/半角≈0.45×。

    系数为 libass 实测值（PlayRes=画幅、DejaVu/Noto Sans CJK），不是字形
    理论 em 值——libass 渲染出的 advance 明显小于标称字号：实测 66 字符
    英文句 0.417 em/字符、CJK 0.686 em/字符。原先假设 1.0/0.55 会把宽度
    高估 1.4~1.5 倍，导致 _fit_subtitle 过度缩号（字幕明显偏小）。
    取略高于实测下限的值（拉丁 0.45 / CJK 0.70）：低估时 libass 会自动
    折行（观感可接受），高估时只会让字幕继续变小（更糟）。
    """
    return sum(
        font_size * (0.70 if ord(ch) > 0x2E7F else 0.45) for ch in text
    )


def _fit_subtitle(text: str, base_fs: int, play_res_w: int = 1280,
                  margin_lr: int = 120) -> tuple[int, str]:
    """保持统一字号，按单词换行；无空格长词按字符折行。"""
    safe_w = play_res_w - margin_lr * 2
    lines: list[str] = []
    line = ""
    for word in text.split():
        if _est_text_width(word, base_fs) > safe_w:
            if line:
                lines.append(line)
                line = ""
            for char in word:
                if line and _est_text_width(line + char, base_fs) > safe_w:
                    lines.append(line)
                    line = ""
                line += char
            continue
        candidate = f"{line} {word}".strip()
        if line and _est_text_width(candidate, base_fs) > safe_w:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return base_fs, "\\N".join(lines)


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


# OCR 行框高 → ASS Fontsize 的实测标定系数（见 _match_original_style 说明）：
# 原字幕行框 103px ↔ 字形墨迹 74px；libass 墨迹 ≈ 0.62 × Fontsize
# → 74 / 103 / 0.62 ≈ 1.16
_FONT_PER_LINE = 1.16


def _match_original_style(regions: list[dict],
                          height: int | None) -> tuple[int, int] | None:
    """从 OCR 原字幕行框推导新字幕的字号与下边距（单位 = 视频像素）。

    PlayRes 与视频一一对应（见 _write_ass），故此处得到的就是真实像素值。

    - 锚定区：**面积最大**的候选框 = 原字幕主体。不能取"最靠下的框"：
      角标/水印框（如「@某某精选」，窄且贴画面最底部）位置比字幕更低，
      取它会把新字幕顶到画面最底边被 App UI 遮挡
    - 字号：锚定行框高 × ``_FONT_PER_LINE``（≈1.16）。该系数是实测标定：
      参考片原字幕 OCR 行框 103px 对应字形墨迹高 74px，而 libass 渲染出的
      字形墨迹高 ≈ 0.62 × Fontsize（CJK 0.61 / 拉丁大写 0.63）
      → Fontsize ≈ 行框高 × (74/103) / 0.62。钳制 [14, 画面高 9%]；
      超宽长句仍由 _fit_subtitle 缩号/折行收口
    - 下边距：画面高 − 锚定行底 → 新字幕接管原字幕的垂直位置

    无法推导（无框 / 行高为 0 / 画面高缺失）返回 None，由调用方走回退口径。
    """
    if not regions or not height:
        return None
    rows = [r for r in regions if float(r.get("h") or 0) > 0]
    if not rows:
        return None
    anchor = max(rows, key=lambda r: float(r["w"]) * float(r["h"]))
    line_h = float(anchor["h"])
    font_size = max(14, min(int(round(height * 0.09)),
                            int(round(line_h * _FONT_PER_LINE))))
    bottom = float(anchor["y"]) + line_h
    # 上界 40% 画高：OCR 先验已保证候选框底边落在画面下部 40%（见
    # _cloud_ocr.locate_subtitle_region），故此处只是防御性钳制，
    # 不能收紧——竖屏字幕常在 72% 处，需 28% 画高的下边距才能对齐
    margin_v = max(8, min(int(round(height * 0.4)), int(round(height - bottom))))
    return font_size, margin_v


def _write_ass(segments: list[dict], workdir: Path, cfg,
               regions: list[dict] | None = None,
               width: int | None = None, height: int | None = None) -> str:
    """把译文句段写成 ASS 字幕（底部居中，白字 + 深色描边）。

    PlayRes 取**视频实际像素尺寸**：libass 把 PlayRes 空间映射到画幅，两者
    宽高比不一致时会对文字做非等比拉伸（曾硬编码 1280×720 → 竖屏 1080×1920
    上字幕被横向压扁、位置/边距全部失真）。PlayRes = 画幅后 1 单位 = 1 像素，
    坐标与字号所见即所得。

    字号/下边距动态匹配原字幕（regions=OCR 原始行框时按锚定行高与行底推导）；
    长句仍经 _fit_subtitle 缩号/折行防溢出（匹配字号为上限）。
    regions 缺省回退 cfg.subtitle_font_size / MarginV=40，并按画幅等比换算
    （保持 720 基准下的既有观感）。
    """
    play_res_x = int(width) if width else 1280
    play_res_y = int(height) if height else 720
    scale = play_res_y / 720.0  # 回退口径的等比换算系数（1 单位 = 1 像素）
    fallback = (
        max(14, int(round(getattr(cfg, "subtitle_font_size", 22) * scale))),
        max(8, int(round(40 * scale))),
    )
    font_size, margin_v = _match_original_style(regions or [], play_res_y) or fallback
    font_name = _pick_font()
    # 描边随字号缩放：PlayRes=画幅后 Outline 单位即像素，固定 1 在 1080p 竖屏上
    # 太细（原 720 空间被放大 2.67× 才显得够粗），按字号 4% 取整
    outline = max(1, int(round(font_size * 0.04)))
    margin_lr = max(20, int(round(play_res_x * 0.094)))  # ≈ 原 120/1280 的比例
    lines = [
        "[Script Info]", f"PlayResX: {play_res_x}", f"PlayResY: {play_res_y}",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, "
        "BorderStyle, Outline, Shadow, Alignment, MarginV, MarginL, MarginR",
        # Alignment=2: 底部居中；MarginV 由原字幕行底位置推导
        # 字体由 _pick_font 运行时探测（CJK 缺失时显式回落，避免豆腐乱码）
        f"Style: Sub,{font_name},{font_size},&H00FFFFFF,&H00000000,"
        f"1,{outline},0,2,{margin_v},{margin_lr},{margin_lr}",
        "[Events]",
        "Format: Layer, Start, End, Style, Text",
    ]

    def ts(sec: float) -> str:
        h, rem = divmod(max(0.0, sec), 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):01d}:{int(m):02d}:{s:05.2f}"

    for seg in segments:
        fs, text = _fit_subtitle(str(seg["text"]).replace("\n", " "), font_size,
                                 play_res_x, margin_lr)
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
    "超级大横带"误伤前景主体。外扩量覆盖字形边缘/描边/阴影/底块
    （黑底块常在白字外侧超出 OCR 框十余像素，垂直外扩需足够）。
    """
    if not regions:
        return []
    out: list[dict] = []
    for r in regions:
        x0 = int(r["x"]) - 14
        y0 = int(r["y"]) - 16
        x1 = int(r["x"] + r["w"]) + 14
        y1 = int(r["y"] + r["h"]) + 16
        if width:
            x0, x1 = max(0, x0), min(int(width), x1)
        if height:
            y0, y1 = max(0, y0), min(int(height), y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue  # 退化框（噪声）跳过
        out.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
    return out


def _watermark_region(
    bbox: tuple[float, float, float, float] | None,
    width: int,
    height: int,
) -> list[dict]:
    """Convert a normalized fixed watermark box to a clipped pixel rectangle."""
    if bbox is None:
        return []
    x0, y0, x1, y1 = bbox
    left, top = round(x0 * width), round(y0 * height)
    right, bottom = round(x1 * width), round(y1 * height)
    if right <= left or bottom <= top:
        raise ValueError("watermark_bbox must define a non-empty region")
    return [{"x": left, "y": top, "w": right - left, "h": bottom - top}]


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


def _mix_timed_audio(segments: list[dict], dubs: list[str], duration_s: float,
                     out_path: str, *,
                     max_tempo: float | None = None) -> tuple[str, dict]:
    """把逐句 TTS 按原始时间戳拼回完整音轨，并产出时长对齐质量统计。

    segments: [{"begin": float, "end": float, "text": str, "index": int}, ...]
    dubs: 对应的 TTS 音频路径列表（与 segments 等长同序）
    max_tempo: 无损感变速上限；None 时取 config.localizer.dub_max_tempo

    分级变速（tempo = 配音时长 / 原句时长）：
    - tempo ≤ 1.02：原速（2% 以内不压，避免徒增变速伪影）；
    - 1.02 < tempo ≤ max_tempo：atempo 压缩到句时长内（计入 compressed）；
    - tempo > max_tempo：不强行 2 倍速（变调刺耳），自然语速播放（接受与
      下句轻微重叠），计入 overflow 并跟踪最差溢出比与句段 index。
    返回 ``(out_path, stats)``，stats 含 segments/compressed/overflow/
    max_overflow_ratio/worst_index（无溢出时 ratio=0.0、worst_index=None）。
    """
    if max_tempo is None:
        max_tempo = load_config().localizer.dub_max_tempo
    # 统一转为 wav + 静音间隔，再用 amix 合成
    tmp_dir = Path(out_path).parent
    wavs: list[tuple[float, str]] = []  # (delay_ms, wav_path)
    compressed = 0
    overflow = 0
    max_overflow_ratio = 0.0
    worst_index = None
    # wav 命名按 zip 位置（而非 seg['index']）：调用方传入的 segments
    # 已是过滤后的 tts_segs（index 可能非连续/缺失），位置必唯一防覆盖
    for pos, (seg, dub) in enumerate(zip(segments, dubs)):
        wav_p = str(tmp_dir / f"dub_{pos:04d}.wav")
        cmd = ["ffmpeg", "-y", "-i", dub]
        seg_dur = seg["end"] - seg["begin"]
        filters = []
        if seg_dur > 0.3:
            tempo = _media.probe_media(dub)[0] / seg_dur
            if 1.02 < tempo <= max_tempo:
                # 无损感域内压缩（已 ≤ max_tempo，无需再钳 2.0）
                filters.append(f"atempo={min(tempo, 2.0):.4f}")
                compressed += 1
            elif tempo > max_tempo:
                # 超上限：不硬压（保自然语速），接受溢出并统计最差句
                overflow += 1
                if tempo > max_overflow_ratio:
                    max_overflow_ratio = tempo
                    worst_index = seg.get("index", pos)
        # 响度归一（上游 f1709f7）：逐句拉平到 -18 LUFS，混音后音量稳定
        filters.append("loudnorm=I=-18:TP=-2:LRA=7")
        cmd += ["-filter:a", ",".join(filters)]
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
    stats = {
        "segments": len(wavs),
        "compressed": compressed,
        "overflow": overflow,
        "max_overflow_ratio": round(max_overflow_ratio, 3),
        "worst_index": worst_index,
    }
    return out_path, stats


def _build_timed_audio(segments: list[dict], dubs: list[str], duration_s: float,
                       out_path: str) -> str:
    """_mix_timed_audio 的薄封装：仅取路径，返回值仍为 str（旧调用方/demo 依赖）。"""
    return _mix_timed_audio(segments, dubs, duration_s, out_path)[0]


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
