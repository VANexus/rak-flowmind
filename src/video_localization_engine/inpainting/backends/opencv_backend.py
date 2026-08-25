"""OpenCVInpaintBackend — 整个仓库唯一调用 cv2.inpaint 的地方。

支持算法:
  - "telea" (Fast Marching Method, 默认, 较快)
  - "ns"    (Navier-Stokes, 平滑)

Future backends (同 build() 签名):
  - LaMaInpaintBackend
  - ProPainterInpaintBackend
  - DiffusionInpaintBackend
"""
from __future__ import annotations

import cv2

from video_localization_engine.inpainting.base import InpaintingBackend
from video_localization_engine.types.video import FramePacket


class OpenCVInpaintBackend(InpaintingBackend):
    """cv2.inpaint backend — Phase D 默认 backend。"""

    def __init__(self, algorithm: str = "telea", radius: int = 3):
        if algorithm not in ("telea", "ns"):
            raise ValueError(f"algorithm must be 'telea' or 'ns', got {algorithm!r}")
        if radius < 1:
            raise ValueError(f"radius must be >= 1, got {radius}")
        self.algorithm = algorithm
        self.radius = radius

    @property
    def name(self) -> str:
        return "opencv"

    def is_available(self) -> bool:
        return hasattr(cv2, "inpaint")

    def inpaint(self, packet: FramePacket, mask):
        import numpy as np
        # mask 必须是 uint8 0/255
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if mask.max() <= 1:
            mask = mask * 255
        flag = cv2.INPAINT_TELEA if self.algorithm == "telea" else cv2.INPAINT_NS
        return cv2.inpaint(packet.image, mask, self.radius, flag)
