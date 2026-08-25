"""localize_video 技能：全云端视频本地化流水线（单视频同步）。

流水线：ffmpeg 提音轨 → 百炼 Paraformer ASR（句级时间戳）
→ 阿里云 OCR 定位原字幕区（双匹配：文本冲突以 ASR 为准）
→ LongCat 翻译 → ffmpeg 擦除原字幕区 → CosyVoice 克隆音色逐句配音
→ 混音 + ASS 字幕烧录。

一切智能环节走云 API；无 key 显式 degraded，绝不静默降级。
失败契约遵循 HTTP 依赖类：r.ok=True + metrics.degraded + data.failure_category。
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import requests  # noqa: F401  保留模块级引用：测试 fixture 打桩用
from pydantic import BaseModel, Field, field_validator

from flowmind.config import load_config
from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.errors import is_retriable
from flowmind.skill import skill
from flowmind.skills import _cloud_asr, _cloud_ocr, _cloud_tts, _llm_translate
from flowmind.skills import _media
from flowmind.skills._image_backend import resolve_api_key  # noqa: F401

_VERSION = "0.1.0"

KEY_DASHSCOPE = "DASHSCOPE_API_KEY"
KEY_LONGCAT = "LONGCAT_API_KEY"


class LocalizeVideoInput(BaseModel):
    """单视频本地化入参。"""

    video_path: str = Field(..., min_length=1, description="视频路径或可公网访问 URL")
    target_lang: str | None = Field(default=None, description="目标语言；None=读 config 默认")
    source_lang: str | None = Field(default=None, description="源语言；None=zh")
    voice_id: str | None = Field(
        default=None,
        description="克隆音色 ID（voice_clone_enroll 注册所得）；None=不配音只换字幕",
    )
    output_path: str | None = Field(
        default=None, description="输出 mp4 路径；None=输入同目录 <stem>_localized.mp4"
    )
    keep_background_audio: bool = Field(default=False, description="原声 -12dB 保留为背景")

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
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


@skill(id="localize_video", name="全云视频本地化流水线", version=_VERSION)
def localize_video(inp: LocalizeVideoInput) -> SkillOutput[LocalizeVideoReport]:
    """ASR → LLM 翻译 → 原字幕擦除 → 克隆 TTS 配音 → 合成。

    双匹配策略：OCR 负责定位原字幕区域（供擦除）；翻译文本一律以 ASR 为准。
    """
    cfg = load_config().localizer
    dashscope_key = resolve_api_key(KEY_DASHSCOPE)
    longcat_key = resolve_api_key(KEY_LONGCAT)

    # ── 预检（确定性，不调云）──
    is_url = inp.video_path.startswith(("http://", "https://"))
    src = inp.video_path
    if not is_url and not Path(src).exists():
        return _fail(inp, f"视频文件不存在: {src}", "video")
    if not is_url:
        ext = Path(src).suffix.lower()
        if ext not in {e.lower() for e in cfg.allowed_extensions}:
            return _fail(inp, f"扩展名 {ext} 不在允许列表 {cfg.allowed_extensions}", "video")
    if not dashscope_key:
        return _fail(inp, f"未设置环境变量 {KEY_DASHSCOPE}（ASR/TTS/OCR 需要）", "environment")
    if not longcat_key:
        return _fail(inp, f"未设置环境变量 {KEY_LONGCAT}（翻译需要）", "environment")

    workdir = Path("._lv_work") / os.urandom(4).hex()
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        # ── 1) 提取音轨 ──
        audio_path = _media.extract_audio(
            src, str(workdir / "audio.wav"),
            sample_rate=int(getattr(cfg, "asr_sample_rate", 16000)),
        )

        # ── 2) ASR（句级时间戳；本地 wav 经 WS 流式直推云端，无需公网 URL）──
        segments = _cloud_asr.transcribe_local(audio_path, api_key=dashscope_key)
        if not segments:
            return _ok_shell(inp, workdir, duration=_media.probe_duration(src),
                             msg="ASR 未识别到任何语音（可能无人声）",
                             erased=False, voice=None)
        duration_s = _media.probe_duration(src) if not is_url else segments[-1]["end"]

        # ── 3) OCR 定位原字幕区（离线抽样 N 帧，非逐帧实时——见 _locate_region）──
        region = _locate_region(src, duration_s, workdir, dashscope_key, cfg)

        # ── 4) LongCat 翻译 ──
        translated = _llm_translate.translate_segments(
            segments,
            target_lang=inp.target_lang or cfg.target_lang_default,
            source_lang=inp.source_lang or cfg.source_lang_default,
            api_key=longcat_key,
        )

        # ── 5) 擦除原字幕区 + 烧译文字幕 ──
        ass_path = _write_ass(translated, workdir, cfg)
        out_path = inp.output_path or str(
            Path(src).with_stem(Path(src).stem + "_localized").with_suffix(".mp4"))
        erased_video = _media.burn_subtitles(
            src if not is_url else _ensure_local(src, workdir),
            str(workdir / "subbed.mp4"), ass_path,
            erase_regions=[region] if region else None,
        )

        # ── 6) 克隆 TTS 逐句配音（可选）──
        voice_used = None
        if inp.voice_id:
            dubs = _cloud_tts.synthesize_segments(
                translated, voice_id=inp.voice_id,
                out_dir=str(workdir / "dubs"), api_key=dashscope_key,
            )
            dub_track = _concat_wavs(dubs, str(workdir / "dub.wav"))
            _media.mix_audio(erased_video, dub_track,
                             out_path, keep_background=inp.keep_background_audio)
            voice_used = inp.voice_id
        else:
            # 不配音：直接把擦除+字幕后的产物落到目标路径
            # shutil.move 而非 Path.replace：workdir 与输出可能跨文件系统
            if out_path != erased_video:
                shutil.move(erased_video, out_path)

        chain = ReasoningChain(
            conclusion=(
                f"本地化完成：{len(translated)} 句、"
                f"{'已擦除' if region else '未检出'}原字幕区、"
                f"{'克隆配音 ' + voice_used if voice_used else '未配音'}"
            ),
            triggered_rules=[], evidence=[],
            causal_analysis=(
                f"Paraformer ASR {len(segments)} 句 → LongCat 翻译 → "
                f"阿里云 OCR 定位 → CosyVoice 合成 → ffmpeg 合成"
            ),
            risk_note=(
                "译文为 LLM 生成，正式投放前建议人工抽查；"
                "克隆音色需已通过 voice_clone_enroll 注册。"
            ),
        )
        return SkillOutput(
            data=LocalizeVideoReport(
                output_path=out_path,
                duration_seconds=round(duration_s, 3),
                asr_segment_count=len(segments),
                subtitle_region_erased=region is not None,
                voice_used=voice_used,
            ),
            reasoning=[chain], confidence=0.9, sample_size=len(segments),
        )
    except _cloud_asr.ASRError as exc:
        return _fail(inp, str(exc), exc.category, retriable=exc.retriable)
    except _cloud_tts.TTSError as exc:
        return _fail(inp, str(exc), exc.category, retriable=exc.retriable)
    except _cloud_ocr.OCRError as exc:
        return _fail(inp, str(exc), exc.category, retriable=exc.retriable)
    except _llm_translate.LLMTtranslateError as exc:
        return _fail(inp, str(exc), exc.category, retriable=exc.retriable)
    except _media.MediaError as exc:
        return _fail(inp, f"媒体处理失败: {type(exc).__name__}", "video")


def _locate_region(src: str, duration_s: float, workdir: Path,
                   key: str, cfg) -> dict | None:
    """离线抽样 N 帧（均匀铺开，非逐帧实时）→ 云 OCR → 聚合底部字幕 bbox。

    字幕条位置整部片子通常固定，抽样即可定位区域；帧数走 cfg.ocr_frame_count。
    """
    n = max(1, int(getattr(cfg, "ocr_frame_count", 5)))
    frames = []
    for i in range(n):
        frac = (i + 0.5) / n  # 均匀取中点，避开片头片尾黑屏
        p = str(workdir / f"frame_{i:02d}.png")
        try:
            frames.append(_media.extract_frame(src, duration_s * frac, p))
        except _media.MediaError:
            continue
    if not frames:
        return None
    height = int(getattr(cfg, "video_height_hint", 1080))
    return _cloud_ocr.locate_subtitle_region(frames, api_key=key,
                                             frame_height=height)


def _write_ass(segments: list[dict], workdir: Path, cfg) -> str:
    """把译文句段写成 ASS 字幕（bottom_safe）。"""
    font_size = getattr(cfg, "subtitle_font_size", 22)
    lines = [
        "[Script Info]", "PlayResX: 1280", "PlayResY: 720",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, "
        "BorderStyle, Outline, Shadow, Alignment, MarginV",
        f"Style: Sub,Noto Sans CJK SC,{font_size},&H00FFFFFF,&H00000000,"
        "1,2,0,2,40",
        "[Events]",
        "Format: Layer, Start, End, Text",
    ]

    def ts(sec: float) -> str:
        h, rem = divmod(max(0.0, sec), 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):01d}:{int(m):02d}:{s:05.2f}"

    for seg in segments:
        text = str(seg["text"]).replace("\n", " ")
        lines.append(f"Dialogue: 0,{ts(seg['begin'])},{ts(seg['end'])},{text}")
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


def _concat_wavs(wavs: list[str], out_path: str) -> str:
    """拼接逐句 wav 为整轨（句间静音对齐交给 ffmpeg concat）。"""
    lst = Path(out_path).with_suffix(".txt")
    lst.write_text("\n".join(f"file '{w}'" for w in wavs), encoding="utf-8")
    rc, _, err = _media.run_ffmpeg([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-c", "copy", out_path,
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
