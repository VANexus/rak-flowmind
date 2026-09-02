"""localize_video 技能：视频本地化流水线（单视频同步，本地优先）。

流水线：ffmpeg 提音轨 → ASR 句级时间戳（本地 faster-whisper GPU 优先，
auto 回落百炼 Paraformer）→ OCR 定位原字幕区（本地 RapidOCR 优先，
auto 回落 qwen3.5-ocr）→ LongCat 翻译 → ffmpeg 擦除原字幕区
→ CosyVoice 克隆音色逐句配音 → 混音 + ASS 字幕烧录。

后端由 config 后端开关控制（local/cloud/auto）；auto 两端都不可用时
显式 degraded，绝不静默降级。
失败契约遵循 HTTP 依赖类：r.ok=True + metrics.degraded + data.failure_category。
"""
from __future__ import annotations

import os
from pathlib import Path

import requests
from pydantic import BaseModel, Field, field_validator

from flowmind.config import load_config
from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.errors import is_retriable
from flowmind.skill import skill
from flowmind.skills import (
    _cloud_asr,
    _cloud_ocr,
    _cloud_tts,
    _llm_translate,
    _local_asr,
    _local_ocr,
    _media,
)
from flowmind.skills._secrets import get_api_key

_VERSION = "0.1.0"

KEY_SPEECH = "AI_SPEECH_API_KEY"
KEY_LLM = "AI_LLM_API_KEY"


class LocalizeVideoInput(BaseModel):
    """单视频本地化入参。"""

    video_path: str = Field(..., min_length=1, description="视频路径或可公网访问 URL")
    target_lang: str | None = Field(default=None, description="目标语言；None=读 config 默认")
    source_lang: str | None = Field(default=None, description="源语言；None=zh")
    voice_id: str | None = Field(
        default=None,
        description="配音音色（预设音色名，如 longanhuan_v3.6）；"
                    "None=读 config.localize_voice，配置为空则不配音",
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
    asr_backend: str = "cloud"            # 实际使用的 ASR 后端（local/cloud）
    ocr_backend: str = "cloud"            # 实际使用的 OCR 后端（local/cloud）
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


@skill(id="localize_video", name="视频本地化流水线（本地优先）", version=_VERSION)
def localize_video(inp: LocalizeVideoInput) -> SkillOutput[LocalizeVideoReport]:
    """ASR → LLM 翻译 → 原字幕擦除 → 克隆 TTS 配音 → 合成。

    双匹配策略：OCR 负责定位原字幕区域（供擦除）；翻译文本一律以 ASR 为准。
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
    if not speech_key:
        return _fail(inp, f"未设置环境变量 {KEY_SPEECH}（ASR/TTS/OCR 需要）", "environment")
    if not llm_key:
        return _fail(inp, f"未设置环境变量 {KEY_LLM}（翻译需要）", "environment")

    workdir = Path("._lv_work") / os.urandom(4).hex()
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        # ── 1) 提取音轨 ──
        audio_path = _media.extract_audio(
            src, str(workdir / "audio.wav"),
            sample_rate=cfg.asr_sample_rate,
        )

        # ── 2) ASR（本地优先：faster-whisper GPU；auto 回落 dashscope 流式）──
        asr_backend, asr_fn = _select_asr(cfg, speech_key)
        source_lang = inp.source_lang or cfg.source_lang_default
        segments = asr_fn(audio_path, source_lang)
        duration_s, width, height = (None, None, None)
        if not is_url:
            duration_s, width, height = _media.probe_media(src)
        if not segments:
            return _ok_shell(inp, workdir, duration=duration_s or 0.0,
                             msg="ASR 未识别到任何语音（可能无人声）",
                             erased=False, voice=None)
        if duration_s is None:
            duration_s = segments[-1]["end"]

        # ── 3) OCR 定位原字幕区（本地优先：RapidOCR CPU；auto 回落 qwen3.5-ocr）──
        ocr_backend, ocr_fn = _select_ocr(cfg, speech_key)
        region = _locate_region(src, duration_s, workdir, ocr_fn, cfg,
                                width=width, height=height)

        # ── 3.5) 长句拆分（避免"一坨字幕"覆盖全视频）──
        print(f"[LV] segments={len(segments)}")
        display_segments = _split_long_segments(segments)
        print(f"[LV] display_segments={len(display_segments)}")

        # ── 4) LongCat 翻译（用拆分后的短句，翻译更稳定）──
        try:
            translated = _llm_translate.translate_segments(
                display_segments,
                target_lang=inp.target_lang or cfg.target_lang_default,
                source_lang=inp.source_lang or cfg.source_lang_default,
                api_key=llm_key,
            )
        except Exception as e:
            print(f"[LV] TRANSLATE ERROR: {type(e).__name__}: {e}")
            raise
        print(f"[LV] translated={len(translated)}")

        # ── 5) 擦除原字幕区 + 烧译文字幕 ──
        ass_path = _write_ass(translated, workdir, cfg)
        out_path = inp.output_path or str(
            Path(src).with_stem(Path(src).stem + "_localized").with_suffix(".mp4"))
        regions = region or []
        # 横条擦除：把多个竖检测区合并为一个底部横带，避免竖条马赛克
        erase_regs = _to_horizontal_bar(regions, width, height)
        print(f"[LV] erase_regs={erase_regs}")
        try:
            erased_video = _media.burn_subtitles(
                src if not is_url else _ensure_local(src, workdir),
                str(workdir / "subbed.mp4"), ass_path,
                erase_regions=erase_regs or None,
            )
        except Exception as e:
            print(f"[LV] BURN ERROR: {type(e).__name__}: {e}")
            raise
        print(f"[LV] erased_video={erased_video}")

        # ── 6) TTS 逐句配音 ─ 移除原声，避免音画不同步 ──
        voice_used = None
        eff_voice = inp.voice_id or cfg.localize_voice or ""
        print(f"[LV] eff_voice={eff_voice!r}")
        if eff_voice:
            # TTS 需要 index 字段；过滤过短句（≤1字，避免 TTS API 报错）
            tts_segs = [
                {**s, "index": i} for i, s in enumerate(translated)
                if len(str(s.get("text", "")).strip()) > 1
            ]
            print(f"[LV] tts_segs={len(tts_segs)} (filtered from {len(translated)})")
            if not tts_segs:
                # 全部过短，退回使用原始 translated
                tts_segs = [{**s, "index": i} for i, s in enumerate(translated)]
            try:
                dubs = _cloud_tts.synthesize_segments(
                    tts_segs, voice_id=eff_voice,
                    out_dir=str(workdir / "dubs"), api_key=speech_key,
                    target_model=cfg.localize_tts_model,
                )
            except Exception as e:
                print(f"[LV] TTS ERROR: {type(e).__name__}: {e}")
                raise
            print(f"[LV] dubs={len(dubs)}")
            # 按原始时间戳合成配音（带静音间隔，保证音画同步）
            dub_track = _build_timed_audio(translated, dubs, duration_s,
                                           str(workdir / "dub.wav"))
            _replace_audio(erased_video, dub_track, out_path)
            voice_used = eff_voice
        else:
            # 不配音：移除原声，仅保留擦除+字幕
            _strip_audio(erased_video, out_path)
        print(f"[LV] final out_path={out_path}")

        chain = ReasoningChain(
            conclusion=(
                f"本地化完成：{len(display_segments)} 句（ASR {len(segments)} 句拆分）、"
                f"{'已擦除' if regions else '未检出'}原字幕区、"
                f"{'克隆配音 ' + voice_used if voice_used else '未配音'}"
            ),
            triggered_rules=[], evidence=[],
            causal_analysis=(
                f"ASR[{asr_backend}] {len(segments)} 句 → 拆分为 {len(display_segments)} 句 → "
                f"LongCat 翻译 → OCR[{ocr_backend}] 定位 → qwen-audio TTS → ffmpeg 合成"
            ),
            risk_note="译文为 LLM 生成，正式投放前建议人工抽查。",
        )
        return SkillOutput(
            data=LocalizeVideoReport(
                output_path=out_path,
                duration_seconds=round(duration_s, 3),
                asr_segment_count=len(segments),
                subtitle_region_erased=bool(regions),
                voice_used=voice_used,
                asr_backend=asr_backend,
                ocr_backend=ocr_backend,
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
    时间按字数比例分配。
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
    return result


def _write_ass(segments: list[dict], workdir: Path, cfg) -> str:
    """把译文句段写成 ASS 字幕（底部居中，带半透明底条）。"""
    font_size = getattr(cfg, "subtitle_font_size", 22)
    lines = [
        "[Script Info]", "PlayResX: 1280", "PlayResY: 720",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, "
        "BorderStyle, Outline, Shadow, Alignment, MarginV, MarginL, MarginR",
        # Alignment=2: 底部居中；MarginV=40 留边距防贴底
        # 用 Noto Serif CJK SC（系统已装），不用 Sans（未装会导致不渲染）
        f"Style: Sub,Noto Serif CJK SC,{font_size},&H00FFFFFF,&H00000000,"
        "1,1,0,2,40,120,120",
        "[Events]",
        "Format: Layer, Start, End, Style, Text",
    ]

    def ts(sec: float) -> str:
        h, rem = divmod(max(0.0, sec), 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):01d}:{int(m):02d}:{s:05.2f}"

    for seg in segments:
        text = str(seg["text"]).replace("\n", "\\N")
        lines.append(f"Dialogue: 0,{ts(seg['begin'])},{ts(seg['end'])},Sub,{text}")
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


def _to_horizontal_bar(regions: list[dict], width: int, height: int) -> list[dict]:
    """把多个竖检测区合并为底部一条横带（避免竖条马赛克）。

    只覆盖画面底部约 25%（典型字幕区），不触及顶部标题/主体。
    """
    if not regions:
        return []
    min_x = max(0, min(r["x"] for r in regions) - 30)
    max_x = min(width, max(r["x"] + r["w"] for r in regions) + 30)
    # 横带固定在底部 25%：起点=75%高度，终点=98%高度
    min_y = int(height * 0.70)
    max_y = int(height * 0.98)
    return [{"x": min_x, "y": min_y, "w": max_x - min_x, "h": max_y - min_y}]


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

    segments: [{"begin": float, "end": float, "text": str}, ...]
    dubs: 对应的 TTS mp3 路径列表
    """
    # 统一转为 wav + 静音间隔，再用 amix 合成
    tmp_dir = Path(out_path).parent
    wavs: list[tuple[float, str]] = []  # (delay_ms, wav_path)
    for seg, dub in zip(segments, dubs):
        # mp3 → wav
        wav_p = str(tmp_dir / f"dub_{seg.get('index', 0):04d}.wav")
        cmd = ["ffmpeg", "-y", "-i", dub, "-ac", "1", "-ar", "16000", wav_p]
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
    af = f"{delays};{mix_inputs}amix=inputs={n}:duration=longest:dropout_transition=0[aout]"
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
