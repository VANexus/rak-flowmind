"""媒体层：ffprobe 元数据 / 场景切分 / 字幕 / 剪辑渲染（ffmpeg 子进程）。

约定：
- 所有 ffmpeg/ffprobe 子进程带 timeout（config.editing 无单独项时取 30min
  上限防挂死）；失败带 stderr 片段显式抛 MediaError（错误永不静默）。
- 渲染 v1 能力边界（见 docs/agent-video-editing.md §7）：视频裁剪 + 图片
  Ken Burns + xfade(fade/none) 转场 + 字幕烧录 + 口播音轨。音频策略 v1 =
  有口播用口播，否则静音轨（原声混音/BGM 自动选段留 v2）。
- mock=True：用 lavfi color 源产出真实可播放的占位 mp4（demo 冒烟用），
  不做字幕烧录。
"""
from __future__ import annotations

import json
import logging
import math
import re
import shutil
import subprocess
from pathlib import Path

from flowmind.editing.models import ShotInfo, TimelineDoc, TimelineShot

logger = logging.getLogger(__name__)

_SUBPROCESS_TIMEOUT_S = 1800.0   # 单条 ffmpeg 命令上限（30min，防挂死）
_PTS_RE = re.compile(r"pts_time:\s*([\d.]+)")


class MediaError(Exception):
    """媒体处理失败。category 语义与 errors.py 一致。"""

    def __init__(self, message: str, category: str = "environment",
                 retriable: bool = False) -> None:
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def ensure_ffmpeg() -> str:
    """确认 ffmpeg/ffprobe 可用，返回 ffmpeg 路径；缺失显式报错。"""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise MediaError(
            "未找到 ffmpeg/ffprobe（PATH 不可见）。请安装 ffmpeg 后重试。",
            category="environment")
    return ffmpeg


def classify_asset(path: str) -> str:
    """按扩展名判定素材类型 video/image/audio。无法识别显式报错。"""
    ext = Path(path).suffix.lower()
    if ext in (".mp4", ".mov", ".mkv", ".webm"):
        return "video"
    if ext in (".jpg", ".jpeg", ".png", ".webp"):
        return "image"
    if ext in (".mp3", ".wav", ".m4a", ".aac"):
        return "audio"
    raise MediaError(f"无法识别的素材扩展名：{ext}（{path}）",
                     category="video")


def _run(cmd: list[str], *, timeout_s: float = _SUBPROCESS_TIMEOUT_S) -> str:
    """执行命令，返回 stdout；非零退出带 stderr 片段抛 MediaError。"""
    logger.debug("media 子进程：%s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise MediaError(
            f"媒体子进程超时（>{int(timeout_s)}s）：{cmd[0]}",
            category="transient", retriable=True) from exc
    except FileNotFoundError as exc:
        raise MediaError(f"找不到可执行文件：{cmd[0]}", category="environment") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-600:]
        raise MediaError(
            f"媒体子进程失败（exit={proc.returncode}）：{cmd[0]} …{tail}",
            category="video")
    return proc.stdout or ""


def probe(path: str) -> dict:
    """ffprobe 元数据（format + streams 原始 dict）。"""
    ensure_ffmpeg()
    out = _run([
        "ffprobe", "-v", "error", "-show_format", "-show_streams",
        "-of", "json", path,
    ])
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe 输出不可解析：{exc}") from exc


def probe_summary(path: str) -> dict:
    """精简元数据：{duration_s,width,height,fps,has_audio}。图片给 0 时长。"""
    kind = classify_asset(path)
    data = probe(path)
    fmt_dur = float(data.get("format", {}).get("duration") or 0.0)
    width = height = 0
    fps = 0.0
    has_audio = False
    for st in data.get("streams", []):
        if st.get("codec_type") == "video" and not width:
            width = int(st.get("width") or 0)
            height = int(st.get("height") or 0)
            rate = st.get("avg_frame_rate") or st.get("r_frame_rate") or "0/1"
            try:
                num, den = rate.split("/")
                fps = float(num) / float(den) if float(den) else 0.0
            except (ValueError, ZeroDivisionError):
                fps = 0.0
            if not fmt_dur and st.get("duration"):
                fmt_dur = float(st["duration"])
        elif st.get("codec_type") == "audio":
            has_audio = True
    if kind == "image":
        fmt_dur = 0.0
    return {
        "kind": kind, "duration_s": fmt_dur, "width": width,
        "height": height, "fps": round(fps, 2), "has_audio": has_audio,
    }


def scene_shots(path: str, *, threshold: float = 0.3,
                max_shots: int = 12) -> list[ShotInfo]:
    """视频场景切分（ffmpeg select='gt(scene,T)' + showinfo 解析时间戳）。

    镜头数超上限时按时间均匀抽样边界（保节奏分布，不保每个镜头）。
    """
    ensure_ffmpeg()
    vf = f"select='gt(scene,{threshold})',showinfo"
    # showinfo 的探测输出在 stderr，成功路径也要回收，不能走 _run（只回 stdout）
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", path, "-vf", vf,
             "-an", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_SUBPROCESS_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise MediaError(
            f"场景切分子进程超时（>{int(_SUBPROCESS_TIMEOUT_S)}s）：{path}",
            category="transient", retriable=True) from exc
    # 有场景切分时 exit 仍为 0；非 0 视为素材损坏
    if proc.returncode != 0:
        raise MediaError(
            f"场景切分失败（exit={proc.returncode}）：{(proc.stderr or '')[-400:]}",
            category="video")
    times = [float(m) for m in _PTS_RE.findall(proc.stderr or "")]
    summary = probe_summary(path)
    duration = summary["duration_s"]
    if duration <= 0:
        raise MediaError(f"视频时长非法（{duration}）：{path}", category="video")
    # 去重 + 去掉贴边时间点
    times = sorted({round(t, 3) for t in times if 0.05 < t < duration - 0.05})
    if len(times) >= max_shots:   # 边界数 ≥ 上限：均匀抽样 max_shots-1 个
        step = len(times) / (max_shots - 1)
        times = [times[int(i * step)] for i in range(max_shots - 1)]
    bounds = [0.0] + times + [duration]
    return [
        ShotInfo(index=i, start_s=bounds[i], end_s=bounds[i + 1],
                 score=1.0 if 0 < i < len(bounds) - 1 else 0.0)
        for i in range(len(bounds) - 1)
    ]


def extract_frames(path: str, times: list[float], out_dir: Path, *,
                   width: int = 480, prefix: str = "frame") -> list[tuple[float, Path]]:
    """在指定时刻抽帧（jpeg），返回 [(t, 路径)]——vision 看图输入。"""
    ensure_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[tuple[float, Path]] = []
    for i, t in enumerate(times):
        dst = out_dir / f"{prefix}_{i:02d}.jpg"
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, t):.3f}", "-i", path, "-frames:v", "1",
            "-vf", f"scale={width}:-2", "-q:v", "4", str(dst),
        ], timeout_s=120)
        out.append((t, dst))
    return out


def extract_preview(video: str, out_dir: Path, *,
                    count: int = 6) -> list[Path]:
    """成片均匀抽帧（自检/给用户过目）：preview/frame_XX.jpg。"""
    info = probe_summary(video)
    duration = info["duration_s"]
    times = [duration * (i + 0.5) / count for i in range(count)]
    return [p for _, p in extract_frames(
        video, times, out_dir, width=360, prefix="preview")]


def build_srt(shots: list[TimelineShot]) -> str:
    """按镜头时间轴生成 SRT 文本（同 section 的连续镜头字幕不重复展开）。"""

    def ts(sec: float) -> str:
        ms = int(round(sec * 1000))
        h, ms = divmod(ms, 3600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines: list[str] = []
    idx = 1
    for shot in shots:
        text = (shot.subtitle_text or "").strip()
        if not text:
            continue
        lines += [str(idx), f"{ts(shot.global_start_s)} --> {ts(shot.global_end_s)}",
                  text, ""]
        idx += 1
    return "\n".join(lines)


# ── v1.3 设计层：ASS 动态字幕/设计卡 ──────────────────────────────

_ACCENT = "&H3DD8FF&"   # 关键词高亮（金 #FFD83D → ASS &HBBGGRR&）


def _ass_time(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int(sec % 3600 // 60)
    s = sec % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _ass_escape(text: str) -> str:
    return (str(text).replace("{", "（").replace("}", "）")
            .replace("\r", " ").replace("\n", "\\N"))


def _mark_emphasis(text: str, words: list[str]) -> str:
    """关键词金色高亮：命中词包上 {\c...\b1}…{\r}（从长到短防嵌套误切）。"""
    out = _ass_escape(text)
    for w in sorted({w for w in words if w and w in text}, key=len, reverse=True):
        w_esc = _ass_escape(w)
        if w_esc in out:
            out = out.replace(w_esc, f"{{\\c{_ACCENT}\\b1}}{w_esc}{{\\r}}", 1)
    return out


def build_ass(timeline: TimelineDoc) -> str:
    """v1.3 设计层：ASS 动态字幕 + 开场/片尾设计卡文本（libass 烧录）。

    设计语言：钩子镜字幕居中放大弹入、正文底部弹入、关键词金色高亮、
    右上角进度角标；设计卡大字缩放弹入 + 副标题淡入。时间全部用全局轴
    （global_start/end 已按转场折算，字幕与成片逐镜对齐）。
    """
    w, h = timeline.width, timeline.height
    font = "Microsoft YaHei"
    sub_size = max(28, round(h * 0.030))
    hook_size = max(40, round(h * 0.046))
    card_main = max(52, round(h * 0.062))
    card_sub = max(26, round(h * 0.028))
    badge_size = max(18, round(h * 0.020))
    margin_v = round(h * 0.16)          # 标准下沿位（源片已去字幕，无打架）
    badge_m = round(w * 0.030)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: SUB,{font},{sub_size},&H00FFFFFF,&H00FFFFFF,&H00201610,&H96000000,-1,0,0,0,100,100,0,0,1,3,1,2,40,40,{margin_v},1
Style: SUB_HOOK,{font},{hook_size},&H00FFFFFF,&H00FFFFFF,&H00201610,&H96000000,-1,0,0,0,100,100,0,0,1,4,1,5,40,40,0,1
Style: CARD_MAIN,{font},{card_main},&H00FFFFFF,&H00FFFFFF,&H00201610,&H96000000,-1,0,0,0,100,100,1,0,1,4,2,5,0,0,0,1
Style: CARD_SUB,{font},{card_sub},&H00E8E8E8,&H00E8E8E8,&H00201610,&H96000000,0,0,0,0,100,100,0,0,1,2,1,5,0,0,0,1
Style: BADGE,{font},{badge_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H78000000,-1,0,0,0,100,100,0,0,3,2,0,9,{badge_m},{badge_m},{badge_m},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []

    def ev(style: str, start: float, end: float, text: str) -> None:
        if end > start:
            events.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},{style},,"
                f"0,0,0,,{text}")

    # ── 设计卡文本（卡镜头 = mode card，文字动画走 ASS）──
    for card, shot in ((timeline.title_card, "card_title"),
                       (timeline.end_card, "card_end")):
        if card is None:
            continue
        window = next((s for s in timeline.shots
                       if s.mode == "card" and s.id == shot), None)
        if window is None:
            continue
        s0, s1 = window.global_start_s, window.global_end_s
        # 主标题按字数自适应缩字号（CJK 全宽 ≈1.0em：13 字 × 119px 实测
        # 满宽溢出；0.62 系数低估字宽，首版渲染验证后改 0.98）
        n = max(6, len(card.main))
        fs_main = min(card_main, round(w * 0.92 / n))
        ev("CARD_MAIN", s0 + 0.05, s1 - 0.05,
           f"{{\\fs{fs_main}\\pos({w // 2},{round(h * 0.44)})\\fscx42\\fscy42"
           f"\\t(0,320,\\fscx100\\fscy100)\\fad(250,250)}}"
           f"{_ass_escape(card.main)}")
        if card.sub:
            ev("CARD_SUB", s0 + 0.30, s1 - 0.10,
               f"{{\\pos({w // 2},{round(h * 0.44) + round(h * 0.075)})"
               f"\\fad(450,250)}}{_ass_escape(card.sub)}")

    # ── 逐镜字幕 + 角标 ──
    for shot in timeline.shots:
        if shot.mode == "card":
            continue
        s0, s1 = shot.global_start_s, shot.global_end_s
        text = (shot.subtitle_text or "").strip()
        if text:
            mark = _mark_emphasis(text, shot.subtitle_emphasis)
            if shot.hook:
                mark = (f"{{\\pos({w // 2},{round(h * 0.74)})"
                        f"\\fscx70\\fscy70\\t(0,200,\\fscx100\\fscy100)"
                        f"\\fad(140,140)}}{mark}")
                ev("SUB_HOOK", s0 + 0.10, s1 - 0.05, mark)
            else:
                mark = f"{{\\fscx84\\fscy84\\t(0,160,\\fscx100\\fscy100)\\fad(120,120)}}{mark}"
                ev("SUB", s0 + 0.08, s1 - 0.06, mark)
        if shot.badge:
            ev("BADGE", s0 + 0.10, s1 - 0.05,
               f"{{\\fad(150,150)}}{_ass_escape(shot.badge)}")
    return header + "\n".join(events) + "\n"


def _escape_filter_path(path: str) -> str:
    """ffmpeg filter 字符串里的文件路径转义（Windows 盘符冒号需反斜杠）。"""
    return path.replace("\\", "/").replace(":", "\\:")


def _x264_args(preset: str, bitrate: str) -> list[str]:
    return ["-c:v", "libx264", "-preset", preset, "-b:v", bitrate,
            "-pix_fmt", "yuv420p"]


def render(timeline: TimelineDoc, *, workdir: Path,
           subtitle_file: Path | None, mock: bool = False,
           transition_duration_s: float = 0.5,
           subtitle_font: str = "Noto Sans CJK SC",
           subtitle_font_size: int = 22,
           x264_preset: str = "medium",
           video_bitrate: str = "6M") -> Path:
    """按时间线渲染成片，返回 output.mp4 路径。

    subtitle_file 为字幕文件（v1.3 起为 ASS 动态字幕；None=不烧字幕）。
    """
    ensure_ffmpeg()
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / "output.mp4"
    dur = max(1.0, timeline.duration_s)

    if mock:
        _render_mock(timeline, out, duration=dur)
        return out

    segs = _normalize_segments(timeline, workdir)
    joined = _join_segments(segs, timeline, workdir,
                            transition_duration_s=transition_duration_s)
    _burn_and_mux(joined, timeline, subtitle_file, out,
                  subtitle_font=subtitle_font,
                  subtitle_font_size=subtitle_font_size,
                  x264_preset=x264_preset, video_bitrate=video_bitrate)
    return out


def _render_mock(timeline: TimelineDoc, out: Path, *, duration: float) -> None:
    """mock 渲染：lavfi color 源 + 静音轨，产出真实可播放的占位 mp4。"""
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i",
        f"color=c=0x1f6feb:s={timeline.width}x{timeline.height}:"
        f"d={duration}:r={timeline.fps}",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-shortest", *_x264_args("ultrafast", "2M"), "-c:a", "aac",
        str(out),
    ], timeout_s=300)


def _normalize_segments(timeline: TimelineDoc, workdir: Path) -> list[Path]:
    """Pass 1：每个镜头归一化为同参数中间段（统一画幅/帧率/编码，-an）。

    段长必须与时间线逐帧一致（时间线时长已在 pipeline 量化到帧网格），
    否则 xfade 的 offset 恒等式失效、尾镜被截（离线渲染回归实测）。
    v1.3：card 模式渲染深色设计卡底；视频镜头支持缓推缓拉（zoompan）；
    全部实拍段统一轻调色（对比度/饱和度微调，观感一致）。
    """
    segs: list[Path] = []
    vf_fit = (
        "scale={w}:{h}:force_original_aspect_ratio=decrease,"
        "pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    grade = "eq=contrast=1.04:saturation=1.08"
    for i, shot in enumerate(timeline.shots):
        seg = workdir / f"seg_{i:03d}.mp4"
        if shot.mode == "card":
            # v1.3 设计卡底：深蓝黑 + 轻暗角，文字由 ASS 全局烧录
            frames = max(2, round(shot.duration_s * timeline.fps))
            _run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i",
                f"color=c=0x0E1220:s={timeline.width}x{timeline.height}"
                f":r={timeline.fps}:d={shot.duration_s:.3f}",
                "-vf", f"vignette=angle=PI/4.6,format=yuv420p,setsar=1,"
                       f"fps={timeline.fps}",
                "-frames:v", str(frames), "-an",
                *_x264_args("veryfast", "2M"), str(seg),
            ])
            segs.append(seg)
            continue
        if shot.mode == "image_kenburns":
            kb_vf, frames = _kenburns_vf(shot, timeline)
            # 单帧输入 + zoompan d=frames 产出恰好 frames 帧（-frames:v 封底）。
            # 不能用 -loop 1 -t D：zoompan 会把每个输入帧都 hold d 帧，输出
            # 长度爆炸；也不能在 zoompan 之后再过 fps 滤镜——zoompan 已按目标
            # 帧率输出，实测下游 fps 会丢 1 帧（60→59），破坏段长恒等
            _run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", shot.source_path,
                "-vf", kb_vf + ","
                + vf_fit.format(w=timeline.width, h=timeline.height)
                + "," + grade,
                "-an", "-frames:v", str(frames),
                *_x264_args("veryfast", "4M"), str(seg),
            ])
        else:
            # v1.1 变速/定格：输出时长 = duration_s（含定格）；
            # 内容段 = duration_s − freeze_s，源窗口 = 内容段 × speed，
            # setpts 提速/慢放，tpad 克隆尾帧做定格
            content_s = max(0.2, shot.duration_s - shot.freeze_s)
            src_window = content_s * shot.speed
            clip_vf = vf_fit.format(
                w=timeline.width, h=timeline.height) + f",fps={timeline.fps}"
            if abs(shot.speed - 1.0) > 1e-3:
                clip_vf = f"setpts=PTS/{shot.speed}," + clip_vf
            if shot.freeze_s > 0:
                clip_vf += (f",tpad=stop_mode=clone:"
                            f"stop_duration={shot.freeze_s:.3f}")
            # v1.3 缓推缓拉：scale eval=frame 逐帧缩放，表达式用输入帧时间
            # t 推进（PTS 不动）。不用 zoompan——它按图片设计会把输出 PTS
            # 重写成 25fps 基准：30fps 源实测被拉长 1.2 倍且下游帧队列爆炸
            # （buffers queued → ENOMEM，离线 25fps 素材踩不到、真机必踩）。
            # 变量尺寸帧由下游 scale+pad 归一，段长与源严格一致（150帧实测）
            z0, z1 = shot.zoom_start or 1.0, shot.zoom_end or 1.0
            if abs(z1 - z0) > 1e-3:
                window = max(0.1, src_window)
                zexpr = f"{z0:.4f}+({z1 - z0:+.4f})*min(t/{window:.3f},1)"
                clip_vf = (f"scale=eval=frame:"
                           f"w='trunc(iw*({zexpr})/2)*2':"
                           f"h='trunc(ih*({zexpr})/2)*2'," + clip_vf)
            clip_vf += "," + grade
            _run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{shot.source_in:.3f}",
                "-t", f"{src_window:.3f}",
                "-i", shot.source_path, "-vf", clip_vf, "-an",
                *_x264_args("veryfast", "4M"), str(seg),
            ])
        segs.append(seg)
    return segs


def _kenburns_vf(shot: TimelineShot,
                 timeline: TimelineDoc) -> tuple[str, int]:
    """图片 Ken Burns（zoompan；先放大 2 倍防抖）。pan 决定运动方向。

    返回 (滤镜串, 目标帧数)：调用方须以 -frames:v 封底输出帧数，
    保证段落时长恰为 duration_s（zoompan d=frames 只对单帧输入安全）。
    """
    frames = max(2, int(shot.duration_s * timeline.fps))
    w, h = timeline.width, timeline.height
    # 输出分辨率给 2 倍再由下游 scale/pad 归一；zoom 0→0.15 缓动
    if shot.pan == "out":
        zoom = f"if(eq(on,1),1.15,max(1.15-0.15*on/{frames},1.0))"
    else:   # in / left / right / none 均用渐进放大
        zoom = f"min(1+0.15*on/{frames},1.15)"
    if shot.pan == "left":
        x = f"(iw-iw/zoom)*on/{frames}"
    elif shot.pan == "right":
        x = f"(iw-iw/zoom)*(1-on/{frames})"
    else:
        x = "iw/2-(iw/zoom/2)"
    y = "ih/2-(ih/zoom/2)"
    vf = (f"scale=8000:-2,zoompan=z='{zoom}':x='{x}':y='{y}'"
          f":d={frames}:s={w * 2}x{h * 2}:fps={timeline.fps}")
    return vf, frames


def transition_steps(shots: list[TimelineShot], *,
                     transition_duration_s: float,
                     fps: int) -> list[float]:
    """每个镜头前的转场耗时（xfade 重叠量）：首镜 0，其余按转场类型。

    none = 1 帧 epsilon（见 _join_segments 注释：duration=0 会触发截断）；
    step 以相邻两段半时长为上限防越界。时间线/字幕与成片对齐依赖此函数。

    全 none 时 _join_segments 走 concat demuxer（零重叠），steps 必须全 0，
    否则 SRT/时间线按重叠折算会逐镜漂移（16 镜实测尾部漂 0.5s）。
    """
    if all(s.transition == "none" for s in shots):
        return [0.0] * len(shots)
    eps = 1.0 / max(1, fps)
    steps = [0.0] * len(shots)
    for k in range(1, len(shots)):
        step = (transition_duration_s
                if shots[k].transition != "none" else eps)
        steps[k] = min(step, shots[k - 1].duration_s * 0.5,
                       shots[k].duration_s * 0.5)
    return steps


def effective_global_starts(shots: list[TimelineShot], *,
                            transition_duration_s: float,
                            fps: int) -> list[float]:
    """折算转场重叠后的成片时间轴起点（与 _join_segments 输出一致）。

    output 总时长 = Σduration − Σstep；SRT/timeline 用此对齐成片。
    """
    steps = transition_steps(
        shots, transition_duration_s=transition_duration_s, fps=fps)
    starts: list[float] = []
    cursor = 0.0
    for i, shot in enumerate(shots):
        cursor -= steps[i]
        starts.append(round(cursor, 3))
        cursor += shot.duration_s
    return starts


def _join_segments(segs: list[Path], timeline: TimelineDoc, workdir: Path,
                   *, transition_duration_s: float) -> Path:
    """Pass 2：拼接（none=concat demuxer；fade=xfade 链）。"""
    joined = workdir / "joined.mp4"
    if len(segs) == 1:
        shutil.copyfile(segs[0], joined)
        return joined
    shots = timeline.shots
    if all(s.transition == "none" for s in shots):
        lst = workdir / "concat.txt"
        lst.write_text(
            "\n".join(f"file '{p.resolve().as_posix()}'" for p in segs),
            encoding="utf-8")
        _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
              "-f", "concat", "-safe", "0", "-i", str(lst),
              *_x264_args("veryfast", "4M"), str(joined)])
        return joined

    # xfade 链：offset_k = Σd(0..k-1) − k·D
    durs = [s.duration_s for s in shots]
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for p in segs:
        cmd += ["-i", str(p)]
    filters: list[str] = []
    for i in range(len(segs)):
        # 段已在 Pass 1 归一（同画幅/帧率/SAR/像素格式），这里只保证像素
        # 格式；不要再过 fps/scale——实测对 zoompan 段再过 fps 会丢 1 帧，
        # 段长偏离时间线 → offset 恒等式失效 → 尾镜被截
        filters.append(f"[{i}:v]format=yuv420p[v{i}]")
    prev = "v0"
    offset = 0.0
    steps = transition_steps(
        shots, transition_duration_s=transition_duration_s,
        fps=timeline.fps)
    for k in range(1, len(segs)):
        offset += durs[k - 1]
        label = f"x{k}" if k < len(segs) - 1 else "vout"
        # "none"（硬切）不能传 duration=0 且 offset 顶到输入边界——ffmpeg 的
        # xfade 要求 offset 严格小于当前输入时长，等于边界会直接截断后续
        # 输入（实测 D=0/offset=2.0 输出停在 2.0s）。none 用 1 帧 epsilon
        # 交叉溶解，观感即硬切
        step = steps[k]
        offset -= step
        # offset 向下取整到微秒再输出：恒等式 offset+step == 前链输出时长，
        # 若把 offset 4.166667 按 .3f 舍入成 4.167，加 step(0.0333) 会超出
        # 前链时长 0.3ms，xfade 判定转场越过输入边界 → 整段截掉下一镜
        # （离线回归实测：尾镜 1.0s 全丢，成片 4.17s≠时间线 5.17s）。
        # 向下取整保证 offset+step 严格 ≤ 前链时长
        offset = math.floor(max(0.0, offset) * 1_000_000) / 1_000_000
        # v1.3 转场按镜头声明走（fade/zoomin/slideleft/circleopen/dissolve/
        # fadeblack/radial…——白名单见 models.TRANSITIONS，归一化已兜底）
        filters.append(
            f"[{prev}][v{k}]xfade=transition={shots[k].transition}:"
            f"duration={step:.4f}:"
            f"offset={offset:.6f}[{label}]")
        prev = label
    cmd += ["-filter_complex", ";".join(filters), "-map", "[vout]",
            *_x264_args("veryfast", "4M"), str(joined)]
    _run(cmd)
    return joined


def _burn_and_mux(joined: Path, timeline: TimelineDoc,
                  subtitle_file: Path | None,
                  out: Path, *, subtitle_font: str, subtitle_font_size: int,
                  x264_preset: str, video_bitrate: str) -> None:
    """Pass 3：字幕烧录（可选，v1.3 起烧 ASS 动态字幕）+ 音轨。

    音频策略：口播优先垫底（apad），BGM 存在时 atrim 到成片长 + 淡入淡出，
    与口播 amix（BGM 压到 0.30 闪避档）；无口播时 BGM 0.85 作主音轨。
    """
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(joined)]
    if timeline.voiceover_path:
        cmd += ["-i", timeline.voiceover_path]
    elif not timeline.bgm_path:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
    if timeline.bgm_path:
        cmd += ["-i", timeline.bgm_path]
    if subtitle_file is not None:
        # v1.3 烧 ASS（libass）：字体由 ASS 内部样式表决定，force_style 不再
        # 传（会整体覆盖样式表）；subtitles filter 支持 .ass 扩展名自动识别
        vf = f"subtitles=filename='{_escape_filter_path(str(subtitle_file))}'"
        # 注意：-vf 是输出选项，必须放在全部输入之后（放输入之间会被解析成
        # 输入选项，ffmpeg 直接报错）
        cmd += ["-vf", vf]
    dur = max(0.5, timeline.duration_s)
    if timeline.voiceover_path and timeline.bgm_path:
        fade_out = max(0.0, dur - 1.5)
        cmd += ["-filter_complex",
                "[1:a]apad[vo];"
                f"[2:a]atrim=0:{dur:.3f},asetpts=PTS-STARTPTS,volume=0.30,"
                f"afade=t=in:d=0.8,afade=t=out:st={fade_out:.3f}:d=1.5[bg];"
                "[vo][bg]amix=inputs=2:duration=first:normalize=0[aout]",
                "-map", "0:v", "-map", "[aout]"]
    elif timeline.bgm_path:
        fade_out = max(0.0, dur - 1.5)
        cmd += ["-filter_complex",
                f"[1:a]atrim=0:{dur:.3f},asetpts=PTS-STARTPTS,volume=0.85,"
                f"afade=t=in:d=0.8,afade=t=out:st={fade_out:.3f}:d=1.5[aout]",
                "-map", "0:v", "-map", "[aout]"]
    else:
        cmd += ["-map", "0:v", "-map", "1:a"]
        if timeline.voiceover_path:
            # 口播常短于成片：apad 把音轨垫静音到视频长度，-shortest 只在
            # 口播超长侧截断——裸 -shortest 会把视频尾段切掉（v1.1 缺陷）
            cmd += ["-af", "apad"]
    cmd += ["-shortest", *_x264_args(x264_preset, video_bitrate),
            "-c:a", "aac", "-b:a", "192k", str(out)]
    _run(cmd)
