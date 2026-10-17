"""dsp 质检的**测量侧**（ADR-079）：抽帧 + 客观量。

**分工（用户 2026-10-14 拍板）**：质检的**判定**（画面 / 主体一致性 / 运动质量 / 镜头意图）与
**重试 / 换图调度**在**前端质检层**（复用 `AI_VISION_MODEL` 接线）；后端只负责它**只有它做得了**的那半：

  1. **抽帧**（ffmpeg 在**服务端**侧；前端仓不碰 ffmpeg，故这一步只能落在这里——真 ffmpeg 行为
     由 `examples/dsp_trim_realcheck.py` 复核：10%/50%/90% 三帧真抽出 PNG、时长未知只抽首帧）；
  2. **客观量**（PIL 本地算，**不需要 numpy**）：分辨率和平均亮度（判**黑帧**）、
     相邻帧差异（判**冻结帧**——整段不动的画面在生成里很常见，且靠肉眼在成片里才看得见）。

**只测量，不判定**：本模块**不下** 「这张帧好不好」的结论——那是视觉模型的事。
`blackFrames` / `frozenPairs` 是**计数与阈值**，不是判决（阈值写在下面常量里、可读可改）。

**失败即抛**（`MediaError`）：抽帧失败 / ffmpeg 缺失 / PIL 缺失 / 帧少于一帧——调用方据此
**如实告警并放弃本次测量**（**绝不伪造帧或假统计**）。缺 ffmpeg 时 `subprocess` 抛的是裸
`FileNotFoundError`（实测），故 `sample_frames` 显式收成 `MediaError`，让上游只处理一种异常。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from flowmind.skills import _media as media

# 抽帧位置（视频时长的比例）：开头 / 中间 / 结尾各一帧——盖住「起手崩坏 / 中段漂移 / 收尾融化」。
FRAME_RATIOS: tuple[float, ...] = (0.1, 0.5, 0.9)

# 客观量的**阈值**（只用于计数，不是判决；写在码里便于事后核）：
BLACK_LUMA_MAX = 16.0      # 平均亮度 < 该值视为**黑帧**
FROZEN_DIFF_MAX = 1.5      # 相邻帧平均绝对差 < 该值视为**疑似冻结**（画面几乎没动）


@dataclass(frozen=True)
class FrameSample:
    """抽帧结果：本地帧文件路径 + 对应的时长比例。"""
    paths: list[str] = field(default_factory=list)
    ratios: list[float] = field(default_factory=list)


def sample_frames(
    video_path: str,
    out_dir: str,
    duration_s: float,
    ratios: tuple[float, ...] = FRAME_RATIOS,
) -> FrameSample:
    """按 `ratios` 从视频里抽帧（PNG），返回本地路径。

    ⚠️ 时长未知（`duration_s <= 0`）时**只抽第一帧**（等比例位置算不出来，如实退化，不猜）。
    ffmpeg 缺失 / 抽帧失败 ⇒ 抛 `MediaError`（调用方如实告警，不伪造）。
    """
    picked: list[str] = []
    used: list[float] = []
    spots = ratios if duration_s > 0 else (0.0,)
    for i, r in enumerate(spots):
        ts = max(0.0, duration_s * r) if duration_s > 0 else 0.0
        out = f"{out_dir}/frame-{i}.png"
        try:
            picked.append(media.extract_frame(video_path, ts, out))
        except FileNotFoundError as exc:  # ffmpeg 不在（实测为裸 FileNotFoundError）
            raise media.MediaError(
                "本机没有 ffmpeg——质检抽帧不可用（如实跳过本次测量，不伪造帧）") from exc
        used.append(r)
    if not picked:
        raise media.MediaError("一帧都没抽出来")
    return FrameSample(paths=picked, ratios=used)


def frame_stats(frame_paths: list[str]) -> dict[str, object]:
    """PIL 本地算**客观量**（无需 numpy）：分辨率 / 每帧平均亮度 / 相邻帧差异。

    返回（键名进 JSON，保持稳定）：
      ``width`` / ``height`` / ``frames``（帧数）/ ``meanLuma``（逐帧平均亮度）/
      ``meanDiff``（相邻帧平均绝对差）/ ``blackFrames``（亮度 < 阈值的帧数）/
      ``frozenPairs``（差异 < 阈值的相邻对数）。
    解析不出图 / 帧为空 ⇒ 抛 `MediaError`（调用方如实告警）。
    """
    if not frame_paths:
        raise media.MediaError("没有帧可测")
    try:
        from PIL import Image, ImageChops, ImageStat  # noqa: PLC0415  惰性（同 _media 口径）
    except ImportError as exc:
        raise media.MediaError("未安装 Pillow——质检客观量需要它") from exc

    lumas: list[float] = []
    diffs: list[float] = []
    size = (0, 0)
    prev = None
    for path in frame_paths:
        try:
            img = Image.open(path).convert("L")
        except Exception as exc:  # noqa: BLE001  坏帧 / 非图：如实报，不猜
            raise media.MediaError(f"帧读不出来（{path}）：{exc}") from exc
        size = img.size
        lumas.append(float(ImageStat.Stat(img).mean[0]))
        if prev is not None:
            diffs.append(float(ImageStat.Stat(ImageChops.difference(prev, img)).mean[0]))
        prev = img

    return {
        "width": int(size[0]),
        "height": int(size[1]),
        "frames": len(lumas),
        "meanLuma": [round(v, 2) for v in lumas],
        "meanDiff": [round(v, 2) for v in diffs],
        "blackFrames": sum(1 for v in lumas if v < BLACK_LUMA_MAX),
        "frozenPairs": sum(1 for v in diffs if v < FROZEN_DIFF_MAX),
    }
