"""ffmpeg 薄封装：视频/音频的机械媒体操作。

只做非 AI 的媒体处理（提音轨/探时长/抽帧/等比缩放/擦除区域/烧字幕/混音/裁到时长窗口），
一切智能环节（ASR/TTS/翻译/OCR）走云 API——见 CLAUDE.md 云优先原则。
命令通过 run_ffmpeg/run_ffprobe 间接执行，测试可注入 fake；
图片的探尺寸 / 等比缩放走惰性 PIL（不经命令行，故无 fake 注入点）。

**ffmpeg 从哪来**：默认按 ``PATH`` 找；本机 ffmpeg 常是「按需下载、不进全局 PATH」（ADR-080
的已知实况），故支持配置 ``FLOWMIND_MEDIA_BIN_DIR``（通常写在 ``.env``）——配了就**优先**从该
目录解析 ffmpeg/ffprobe 的**绝对路径**（不改系统 PATH，也不动父进程环境）。未配 ⇒ 原样按 PATH。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import NamedTuple

# ffmpeg/ffprobe 所在的 bin 目录（env 键名；值形如 D:\App\ffmpeg\bin 或 /usr/local/bin）
_BIN_DIR_ENV = "FLOWMIND_MEDIA_BIN_DIR"


class MediaError(Exception):
    """ffmpeg 操作失败。"""


def _resolve_cmd(cmd: list[str]) -> list[str]:
    """把 ``cmd[0]``（ffmpeg/ffprobe）解析成**绝对路径**：配了 ``FLOWMIND_MEDIA_BIN_DIR`` 就优先用它。

    为什么解析绝对路径而不是改 PATH：``subprocess`` 在 Windows 上走 ``CreateProcess``，「传 env」
    时 PATH 搜索语义有歧义；直接给绝对路径最确定，也不污染父进程环境。找不到 ⇒ 原样返回，让
    「本机没有 ffmpeg」那条路径照常如实报错（**不静默换工具**）。
    """
    from flowmind.config import ensure_env_loaded  # 惰性：该键通常写在 .env 里，此处不保证已载

    ensure_env_loaded()
    bin_dir = os.environ.get(_BIN_DIR_ENV, "").strip()
    if not bin_dir or not cmd:
        return cmd
    found = shutil.which(cmd[0], path=bin_dir)
    return [found, *cmd[1:]] if found else cmd


def _run(cmd: list[str], timeout: float = 300.0) -> tuple[int, str, str]:
    # 显式 UTF-8 解码：父进程 locale 被翻成 C 时，跟随 locale 解码非 ASCII 输出会崩溃
    proc = subprocess.run(
        _resolve_cmd(cmd), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )
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


def probe_media(media_path: str) -> tuple[float, int, int]:
    """ffprobe 一次拿时长(秒)与分辨率 (width, height)。"""
    rc, out, err = run_ffprobe([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", "-select_streams", "v:0", media_path,
    ])
    if rc != 0:
        raise MediaError(f"探测媒体信息失败: {err[-200:]}")
    data = json.loads(out)
    duration = (data.get("format") or {}).get("duration")
    streams = data.get("streams") or [{}]
    w, h = streams[0].get("width"), streams[0].get("height")
    if duration is None:
        raise MediaError("ffprobe 未返回时长")
    return float(duration), int(w or 0), int(h or 0)


def extract_frame(video_path: str, timestamp_s: float, out_path: str) -> str:
    """抽取指定时间点的一帧 PNG。"""
    rc, _, err = run_ffmpeg([
        "ffmpeg", "-y", "-ss", f"{timestamp_s:.3f}", "-i", video_path,
        "-frames:v", "1", out_path,
    ])
    if rc != 0:
        raise MediaError(f"抽帧失败: {err[-200:]}")
    return out_path


class ClipFit(NamedTuple):
    """裁到镜头窗口的结果（ADR-080 阶段 5「剪辑引擎」的裁切实况）。

    ``path`` 是**实际可用的片段路径**——未裁切时 = 入参原路径（调用方据此决定要不要落派生件）。
    ``sourceMs`` = 源片段实测时长、``outMs`` = 裁/补后时长、``shortfallMs`` = 源片段**比窗口短**多少
    （>0 且 ``path != video_path`` ⇒ 已按**末帧定格**补齐并把画面拖满窗口）。
    """

    path: str
    sourceMs: int
    outMs: int
    shortfallMs: int
    trimmed: bool


def clip_to_window(
    video_path: str,
    out_path: str,
    window_ms: int,
    *,
    source_ms: int | None = None,
    freeze_pad: bool = True,
) -> ClipFit:
    """把片段裁到**镜头窗口**（``window_ms``）：超长裁掉、不足按末帧定格补齐。

    为什么需要它：图生视频模型给的时长只是「建议值附近」，与 ``timeline`` 锁定的镜头窗口**不是同一个数**
    ——多出来的部分会**顶到下一镜**（观众看到上一镜画面多停一截），不足的部分会让该镜**提前黑/卡**
    （后面镜头往前挤）。故出片后在本侧用 ffmpeg 把每镜**裁到窗口**（ADR-080）。

    行为（三段，各自如实记进 ``ClipFit``）：
      ① 源 = 窗口 ⇒ **原样返回**（不重编码、不落派生件）；
      ② 源 > 窗口 ⇒ ``-t`` **裁掉**尾部，``outMs = window_ms``、``shortfallMs = 0``；
      ③ 源 < 窗口 ⇒ ``freeze_pad`` 时用 ``tpad=stop_mode=clone`` **末帧定格**补齐到窗口
         （``outMs = window_ms``、``shortfallMs = 源与窗口之差``）；``freeze_pad=False`` 则**不补**
         （``outMs = source_ms``），但 ``shortfallMs`` **照实报出**——补不补是策略，短不短是事实。

    编码：走 ``libx264 + yuv420p`` 重编码（`-c copy` 只能按关键帧切，给不出精确窗口）。
    片段的**自带音轨丢弃**（``-an``）——成片音轨由 ``timeline`` 锁定的 tts/bgm 轨铺，I2V 片段本就不带口播；
    真要保留片段音轨得另立参数位（当前不需要）。

    ``source_ms`` 给定则不探（调用方已知时省一次 ffprobe）；未给即 ``probe_media`` 实测。
    ffmpeg/ffprobe 缺失或命令失败 ⇒ 抛 ``MediaError``（调用方**如实告警并保留原片段**，绝不假装裁过）。
    """
    if window_ms <= 0:
        raise MediaError(f"镜头窗口必须 > 0（当前 {window_ms}ms）")
    if source_ms is None:
        try:
            source_ms = int(round(probe_media(video_path)[0] * 1000))
        except FileNotFoundError as exc:  # ffprobe 不在（与 _dsp_qc 同口径：裸 FileNotFoundError）
            raise MediaError("本机没有 ffprobe——无法实测片段时长，裁切不可用") from exc
    if source_ms <= 0:
        raise MediaError("源片段时长未知（0）——不猜，跳过裁切")

    if source_ms == window_ms:
        return ClipFit(video_path, source_ms, source_ms, 0, False)

    if source_ms > window_ms:
        _cut(video_path, out_path, window_ms, freeze_s=None)
        return ClipFit(out_path, source_ms, window_ms, 0, True)

    shortfall = window_ms - source_ms
    if not freeze_pad:
        return ClipFit(video_path, source_ms, source_ms, shortfall, False)
    _cut(video_path, out_path, window_ms, freeze_s=shortfall / 1000.0)
    return ClipFit(out_path, source_ms, window_ms, shortfall, True)


def _cut(video_path: str, out_path: str, window_ms: int, *, freeze_s: float | None) -> None:
    """落一次裁切/定格命令（``freeze_s`` 给定即先 ``tpad`` 克隆末帧补足）。"""
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if freeze_s:
        cmd += ["-vf", f"tpad=stop_mode=clone:stop_duration={freeze_s:.3f}"]
    cmd += ["-t", f"{window_ms / 1000.0:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", out_path]
    try:
        rc, _, err = run_ffmpeg(cmd)
    except FileNotFoundError as exc:  # ffmpeg 不在（实测为裸 FileNotFoundError）
        raise MediaError("本机没有 ffmpeg——裁切不可用（如实跳过，不假装裁过）") from exc
    if rc != 0:
        raise MediaError(
            f"{'末帧定格补齐' if freeze_s else '裁到镜头窗口'}失败: {err[-200:]}")


def _pil():  # noqa: ANN202  惰性导入 Pillow（类型不便标注）
    """惰性导入 Pillow（图片处理用）。未装即如实报错——不静默跳过。"""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover  运行时由 rapidocr 带入
        raise MediaError("未安装 Pillow——图片探尺寸/缩放需要它") from exc
    return Image


def probe_image_size(image_path: str) -> tuple[int, int]:
    """图片像素尺寸 ``(width, height)``（读不出即抛 MediaError）。

    不用 ``probe_media``：那条走 ffprobe 且**要求 format.duration**，而图片没有时长。
    """
    Image = _pil()
    try:
        with Image.open(image_path) as im:
            return int(im.width), int(im.height)
    except (OSError, ValueError) as exc:
        raise MediaError(f"读取图片尺寸失败: {exc}") from exc


def scale_image_to_fit(
    image_path: str, out_path: str, max_w: int, max_h: int, *, quality: int = 92,
) -> str:
    """等比**缩小**到长宽均不超过 ``(max_w, max_h)``，落 ``out_path``；已不超则原样返回入参。

    **只缩不放**：放大既不解决上游口径问题，又只会更糊。``max_w`` / ``max_h`` 传 0 视为不限制。
    返回值是**实际可用路径**（未缩时 = ``image_path``），调用方据此决定是否落派生件。
    """
    if not (max_w and max_h):
        return image_path
    w, h = probe_image_size(image_path)
    if w <= max_w and h <= max_h:
        return image_path
    ratio = min(max_w / w, max_h / h)
    tw, th = max(1, round(w * ratio)), max(1, round(h * ratio))
    Image = _pil()
    save_kw: dict = {"quality": quality} if out_path.lower().endswith((".jpg", ".jpeg")) else {}
    try:
        with Image.open(image_path) as im:
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            im.resize((tw, th), Image.LANCZOS).save(out_path, **save_kw)
    except (OSError, ValueError) as exc:
        raise MediaError(f"图片等比缩放失败: {exc}") from exc
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
    video_path: str, dub_path: str, out_path: str, *, keep_background: bool = False,
    bg_path: str | None = None,
) -> str:
    """把配音轨合回视频。keep_background=True 时背景音降为 -12dB 混入。

    bg_path：背景音源（原始视频的音轨 wav）。擦除链输出的视频是**无声**的，
    不能用 [0:a] 当背景——必须显式传原始音轨；缺省时回退旧行为（视频自带音轨）。
    配音 0.9 倍预留峰值余量 + 显式 normalize=0，避免 amix 归一化衰减/削波。
    """
    if keep_background:
        if bg_path and os.path.exists(bg_path):
            # 背景来自独立音轨：video + dub + bg 三输入
            afilter = (
                "[2:a]volume=-12dB[bg];[1:a]volume=0.9[dub];"
                "[bg][dub]amix=inputs=2:duration=first:normalize=0[aout]"
            )
            inputs = ["-i", video_path, "-i", dub_path, "-i", bg_path]
            amap = ["-filter_complex", afilter, "-map", "0:v", "-map", "[aout]"]
        else:
            # 视频自带音轨（旧行为兜底）
            afilter = (
                "[0:a]volume=-12dB[bg];[1:a]volume=0.9[dub];"
                "[bg][dub]amix=inputs=2:duration=first:normalize=0[aout]"
            )
            inputs = ["-i", video_path, "-i", dub_path]
            amap = ["-filter_complex", afilter, "-map", "0:v", "-map", "[aout]"]
    else:
        amap = ["-map", "0:v", "-map", "1:a"]
        inputs = ["-i", video_path, "-i", dub_path]
    cmd = ["ffmpeg", "-y", *inputs, *amap, "-c:v", "copy", out_path]
    rc, _, err = run_ffmpeg(cmd)
    if rc != 0:
        raise MediaError(f"音轨合成失败: {err[-200:]}")
    return out_path
