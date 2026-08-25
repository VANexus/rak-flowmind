"""ffmpeg 薄封装：视频/音频的机械媒体操作。

只做非 AI 的媒体处理（提音轨/探时长/抽帧/擦除区域/烧字幕/混音），
一切智能环节（ASR/TTS/翻译/OCR）走云 API——见 CLAUDE.md 云优先原则。
命令通过 run_ffmpeg/run_ffprobe 间接执行，测试可注入 fake。
"""
from __future__ import annotations

import json
import subprocess


class MediaError(Exception):
    """ffmpeg 操作失败。"""


def _run(cmd: list[str], timeout: float = 300.0) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def run_ffmpeg(cmd: list[str], **kw) -> tuple[int, str, str]:
    """执行 ffmpeg 命令（可注入替换）。"""
    return _run(cmd, **kw)


def run_ffprobe(cmd: list[str], **kw) -> tuple[int, str, str]:
    """执行 ffprobe 命令（可注入替换）。"""
    return _run(cmd, **kw)


def extract_audio(video_path: str, out_path: str, sample_rate: int = 16000) -> str:
    """提取音轨为 wav（单声道、指定采样率，供 ASR 用）。"""
    rc, _, err = run_ffmpeg([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sample_rate), out_path,
    ])
    if rc != 0:
        raise MediaError(f"提取音轨失败: {err[-200:]}")
    return out_path


def probe_duration(media_path: str) -> float:
    """ffprobe 探媒体时长（秒）。"""
    rc, out, err = run_ffprobe([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", media_path,
    ])
    if rc != 0:
        raise MediaError(f"探测时长失败: {err[-200:]}")
    duration = json.loads(out).get("format", {}).get("duration")
    if duration is None:
        raise MediaError("ffprobe 未返回时长")
    return float(duration)


def probe_resolution(media_path: str) -> tuple[int, int]:
    """ffprobe 探视频分辨率 (width, height)。"""
    rc, out, err = run_ffprobe([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", media_path,
    ])
    if rc != 0:
        raise MediaError(f"探测分辨率失败: {err[-200:]}")
    stream = (json.loads(out).get("streams") or [{}])[0]
    w, h = stream.get("width"), stream.get("height")
    if not w or not h:
        raise MediaError("ffprobe 未返回分辨率")
    return int(w), int(h)


def extract_frame(video_path: str, timestamp_s: float, out_path: str) -> str:
    """抽取指定时间点的一帧 PNG。"""
    rc, _, err = run_ffmpeg([
        "ffmpeg", "-y", "-ss", f"{timestamp_s:.3f}", "-i", video_path,
        "-frames:v", "1", out_path,
    ])
    if rc != 0:
        raise MediaError(f"抽帧失败: {err[-200:]}")
    return out_path


def burn_subtitles(
    video_path: str,
    out_path: str,
    ass_path: str | None,
    erase_regions: list[dict] | None = None,
) -> str:
    """擦除原字幕区域（delogo）+ 烧录 ASS 字幕。

    erase_regions 元素形如 {"x": int, "y": int, "w": int, "h": int}。
    """
    filters: list[str] = []
    for r in (erase_regions or []):
        filters.append(
            f"delogo=x={int(r['x'])}:y={int(r['y'])}:w={int(r['w'])}:h={int(r['h'])}"
        )
    if ass_path:
        filters.append(f"ass={ass_path}")
    cmd: list[str] = ["ffmpeg", "-y", "-i", video_path]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += [out_path]
    rc, _, err = run_ffmpeg(cmd)
    if rc != 0:
        raise MediaError(f"字幕合成失败: {err[-200:]}")
    return out_path


def mix_audio(
    video_path: str, dub_path: str, out_path: str, *, keep_background: bool = False
) -> str:
    """把配音轨合回视频。keep_background=True 时原声降为 -12dB 背景。"""
    if keep_background:
        afilter = "[1:a]volume=-12dB[bg];[bg][2:a]amix=inputs=2:duration=first[aout]"
        amap = ["-filter_complex", afilter, "-map", "0:v", "-map", "[aout]"]
        inputs = ["-i", video_path, "-i", dub_path]
    else:
        amap = ["-map", "0:v", "-map", "1:a"]
        inputs = ["-i", video_path, "-i", dub_path]
    cmd = ["ffmpeg", "-y", *inputs, *amap, "-c:v", "copy", out_path]
    rc, _, err = run_ffmpeg(cmd)
    if rc != 0:
        raise MediaError(f"音轨合成失败: {err[-200:]}")
    return out_path
