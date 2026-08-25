"""LamaInpaintingBackend — LaMa (Large Mask Inpainting) via simple_lama_inpainting。

LaMa 是 Facebook Research 的 SOTA 图像修复模型,特点是:
  - 大面积 mask 也能补得自然(基于 Fourier Convolutions 的全局感受野)
  - 单帧推理 ~0.3s on GPU, ~2s on CPU for 1080p
  - 首次运行会从 GitHub release 下载 big-lama.pt (~200MB), 缓存到 torch hub

Pipeline 集成:
  - PipelineConfig.inpaint_backend = "lama" 切换到此 backend
  - OpenCVInpaintBackend 的 algorithm/radius 参数会被忽略 (LaMa 内部决定)

依赖:
  pip install simple_lama_inpainting   # ~200MB 模型首次自动下载

环境变量:
  LAMA_MODEL  : 自定义 .pt 路径 (跳过下载)
  LAMA_MODEL_URL : 自定义下载 URL
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from video_localization_engine.inpainting.base import InpaintingBackend
from video_localization_engine.types.video import FramePacket


# 设备选择策略: 优先 CUDA, 否则 CPU; 可通过 LAMA_DEVICE 环境变量覆盖
def _pick_device():
    override = os.environ.get("LAMA_DEVICE", "").strip().lower()
    if override in ("cpu", "cuda"):
        import torch
        if override == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(override)
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LamaInpaintingBackend(InpaintingBackend):
    """LaMa inpainting backend。

    构造参数:
      - device: torch.device (默认按 CUDA 可用性自动选择)
      - strict: True → simple_lama_inpainting 缺包时 raise ImportError;
                False → 构造成功但 is_available() 返回 False (graceful degrade)
    """

    def __init__(self, device: Optional[object] = None, strict: bool = False, **kwargs):
        # PipelineConfig 会一并传 algorithm/radius (opencv 的字段) — LaMa 没有,
        # 静默忽略保证 PipelineConfig.inpaint_algorithm 字段无需拆分
        self.strict = strict
        self._device = device or _pick_device()
        self._impl = None  # 延迟到首次 inpaint 调用再装载 (避免 import 时间影响测试启动)
        self._load_error: Optional[BaseException] = None
        self._ignored_kwargs = sorted(kwargs.keys())

    @property
    def name(self) -> str:
        return "lama"

    def is_available(self) -> bool:
        if self._impl is not None:
            return True
        try:
            import simple_lama_inpainting  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_loaded(self):
        """首次调用时实例化 SimpleLama (会下载 ~200MB 模型首次启动)。"""
        if self._impl is not None:
            return
        try:
            from simple_lama_inpainting import SimpleLama
            self._impl = SimpleLama(device=self._device)
        except ImportError as e:
            self._load_error = e
            if self.strict:
                raise
            self._impl = None  # gracefully degrade

    def inpaint(self, packet: FramePacket, mask: np.ndarray) -> np.ndarray:
        """用 LaMa 修复 FramePacket.image 的 mask 区域。

        Args:
            packet: FramePacket, 取 .image (BGR uint8 HxWx3)
            mask:   0/255 二值 mask, HxW 单通道 (与 image 同尺寸)

        Returns:
            BGR uint8 HxWx3 修复图; 与 image 同尺寸。
            若 backend 不可用, 回退为 packet.image.copy() (保持 pipeline 不崩)。
        """
        self._ensure_loaded()
        if self._impl is None:
            return packet.image.copy()
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if mask.max() <= 1:
            mask = mask * 255
        bgr = packet.image
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 image, got shape {bgr.shape}")
        h, w = bgr.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        # BGR -> RGB, uint8 -> float32 [0,1]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            from PIL import Image as PILImage
            img_pil = PILImage.fromarray(rgb)
            mask_pil = PILImage.fromarray(mask)
            out_pil = self._impl(img_pil, mask_pil)
            out_rgb = np.asarray(out_pil)
        except Exception as e:
            # GPU/CPU 推理失败不应让 pipeline 整体挂掉 — 回退到原图
            return bgr.copy()
        if out_rgb.shape[:2] != (h, w):
            out_rgb = cv2.resize(out_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        return out_bgr.astype(np.uint8)
