"""本地字幕区擦除：LaMa inpainting（simple-lama-inpainting 封装）。

delogo 横带的升级替代：LaMa 对字幕区做图像级修复，复杂背景
（网页/渐变/纹理）不再出现竖向拉丝。逐帧流程：

    ffmpeg 抽帧 → 逐帧掩码（ROI 并集内白字笔画精修，前景不误伤）
    → LaMa 逐帧修复 → ffmpeg 按源帧率重编码（无声视频，配音在后级拼接）

模型 big-lama（~200MB）首次使用经 torch hub 自动下载到
HF 缓存（/srv/data/models，走代理）；懒加载，GPU fp32 单帧
约 0.3~0.6s（Pascal 6.1 实测口径），8GB 显存安全。
失败显式抛 InpaintError（category/retriable 与 errors.py 对齐），
绝不静默回落 delogo——回落由调用方按 backend 配置决策。
"""
from __future__ import annotations

from pathlib import Path

from flowmind.skills import _media


class InpaintError(Exception):
    """LaMa 修复失败。category/retriable 语义与 errors.py 对齐。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def available() -> bool:
    """simple-lama-inpainting 可导入即视为可用（模型首次用时才下载）。"""
    try:
        import simple_lama_inpainting  # noqa: F401
        return True
    except ImportError:
        return False


def erase_regions(src: str, regions: list[dict], out_path: str, workdir: str) -> str:
    """对视频矩形区域逐帧 LaMa 修复，输出无声视频。

    经 _inpaint_frames_adapter 薄适配层执行，测试可 monkeypatch 替换。
    """
    if not regions:
        raise InpaintError("擦除区列表为空（应先走 OCR 定位）", category="video")
    return _inpaint_frames_adapter(src, regions, out_path, workdir)


def _inpaint_frames_adapter(
    src: str, regions: list[dict], out_path: str, workdir: str,
) -> str:
    """逐帧修复主体：抽帧 → 掩码 → LaMa → 重编码。"""
    try:
        from PIL import Image
        from simple_lama_inpainting import SimpleLama
    except ImportError as exc:
        raise InpaintError(
            "未安装 simple-lama-inpainting（conda env update -f environment.yml）",
            category="environment",
        ) from exc

    duration_s, width, height = _media.probe_media(src)
    fps = _probe_fps(src)
    if not (width and height):
        raise InpaintError(f"探测视频尺寸失败: {src}", category="video")

    frames_dir = Path(workdir) / "frames"
    out_dir = Path(workdir) / "inpainted_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 全量抽帧（-vsync 0 逐帧精确对应，帧率由重编码阶段还原）
    rc, _, err = _media.run_ffmpeg([
        "ffmpeg", "-y", "-i", src, "-vsync", "0",
        str(frames_dir / "f_%06d.png"),
    ])
    if rc != 0:
        raise InpaintError(f"抽帧失败: {err[-200:]}", category="video")

    # 2) 掩码：ROI 基准（矩形并集，调用方 _prep_erase_regions 已外扩）
    #    + 逐帧白字精修——掩码贴住"这一帧实际出现的文字笔画"，
    #    落在 ROI 内的前景亮区（产品/人脸/白墙）不再被整块误伤。
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    roi = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(roi)
    for r in regions:
        draw.rectangle([
            max(0, int(r["x"]) - 4), max(0, int(r["y"]) - 4),
            min(width, int(r["x"] + r["w"]) + 4),
            min(height, int(r["y"] + r["h"]) + 4),
        ], fill=255)
    roi_np = np.asarray(roi) > 0
    roi_w = max((int(r["w"]) for r in regions), default=0)

    def _frame_mask(img: Image.Image) -> Image.Image:
        """逐帧掩码：白字笔画 + 17px 膨胀，限制在 ROI 内。

        - 白字判定：灰度 >=200（典型字幕白字+暗描边）
        - 实心块过滤：填充率(area/bbox_area)>0.5 且 bbox 宽 >80% ROI 宽的
          组件是前景实体（白墙/产品面），不是文字笔画——删除；
          亮背景下文字与背景粘连成高填充大组件被删后 sum≈0，
          自然触发整框回落（该场景下正确行为）
        - 帧内无可辨识白字（彩色字/白底白字）→ 回落整块 ROI
        """
        gray = np.asarray(img.convert("L"))
        text = (gray >= 200) & roi_np
        try:
            from scipy import ndimage
            lab, n = ndimage.label(text)
            if n:
                slices = ndimage.find_objects(lab)
                for i, sl in enumerate(slices, start=1):
                    if sl is None:
                        continue
                    bh = sl[0].stop - sl[0].start
                    bw = sl[1].stop - sl[1].start
                    comp = lab[sl] == i
                    fill = comp.sum() / (bh * bw)
                    if fill > 0.5 and bw > roi_w * 0.8:
                        text[sl][comp] = False
        except ImportError:
            pass
        if int(text.sum()) < gray.size * 0.0005:
            return roi
        dilated = Image.fromarray((text * 255).astype("uint8"))
        dilated = dilated.filter(ImageFilter.MaxFilter(17))
        keep = (np.asarray(dilated) > 0) & roi_np  # 不越出 ROI
        return Image.fromarray((keep * 255).astype("uint8"))

    # 3) LaMa 逐帧修复（模型首个帧时懒下载；修复结果存 jpg 省盘）
    try:
        lama = SimpleLama()
        frame_paths = sorted(frames_dir.glob("f_*.png"))
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            result = lama(img, _frame_mask(img))
            result.save(str(out_dir / fp.name), quality=95)
    except Exception as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda" in msg and "error" in msg:
            raise InpaintError(
                f"LaMa GPU 修复失败（显存不足?）: {type(exc).__name__}",
                category="transient", retriable=True,
            ) from exc
        if "connection" in msg or "download" in msg or "fetch" in msg:
            raise InpaintError(
                f"big-lama 模型下载失败（需代理）: {type(exc).__name__}",
                category="environment",
            ) from exc
        raise InpaintError(
            f"LaMa 修复失败: {type(exc).__name__}", category="transient",
        ) from exc

    # 4) 按源帧率重编码（无声；配音/混音在后级）
    rc, _, err = _media.run_ffmpeg([
        "ffmpeg", "-y", "-framerate", fps,
        "-i", str(out_dir / "f_%06d.png"),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", out_path,
    ])
    if rc != 0:
        raise InpaintError(f"重编码失败: {err[-200:]}", category="video")
    # 抽帧中间目录很大（1080p 每秒约 40MB），成功后即清理；失败保留供排查
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    return out_path


def _probe_fps(src: str) -> str:
    """探测源视频帧率（返回 "num/den" 原始形式，供 -framerate 使用）。"""
    rc, out, err = _media.run_ffprobe([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", src,
    ])
    if rc != 0:
        raise InpaintError(f"探测帧率失败: {err[-200:]}", category="video")
    import json
    try:
        data = json.loads(out)
        rate = data["streams"][0]["r_frame_rate"]
        if not rate or rate == "0/0":
            raise ValueError(rate)
        return rate
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
        raise InpaintError(f"帧率解析失败: {exc}", category="video") from exc
