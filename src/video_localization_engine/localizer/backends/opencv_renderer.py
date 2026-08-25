"""OpenCVRenderer — cv2.putText 简单字幕渲染。

后续可替换:
  - PIL/PillowRenderer (支持中文字体)
  - FFmpegAssRenderer (ass 字幕烧录)
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from video_localization_engine.localizer.protocols import RendererBackend
from video_localization_engine.types.detection import BBox


class OpenCVRenderer(RendererBackend):
    """cv2.putText 后端。

    参数:
      font_scale: 字体大小 (默认 1.0)
      color: 字色 (BGR)
      stroke_color: 描边色 (BGR)
      thickness: 字粗
      stroke_thickness: 描边粗 (默认 thickness+2)
      font: cv2 字体常量 (默认 FONT_HERSHEY_SIMPLEX)
    """

    def __init__(self, font_scale: float = 1.0,
                 color: Tuple[int, int, int] = (255, 255, 255),
                 stroke_color: Tuple[int, int, int] = (0, 0, 0),
                 thickness: int = 2,
                 stroke_thickness: int = 4,
                 font: int = cv2.FONT_HERSHEY_SIMPLEX):
        self.font_scale = font_scale
        self.color = color
        self.stroke_color = stroke_color
        self.thickness = thickness
        self.stroke_thickness = stroke_thickness
        self.font = font

    @property
    def name(self) -> str:
        return "opencv"

    def render(self, text: str, bbox: BBox,
               frame_size: Tuple[int, int], **kwargs) -> np.ndarray:
        h, w = frame_size
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        if not bbox or not text:
            return canvas
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # cv2.putText baseline 在左下, anchor 用 (x1, y2 - pad)
        ((tw, th), baseline) = cv2.getTextSize(
            text, self.font, self.font_scale, self.thickness)
        pad = 4
        # 居中 bbox: 水平 + 垂直
        cx = x1 + max(pad, (x2 - x1 - tw) // 2)
        cy = y1 + max(pad + th, (y2 - y1 + th) // 2)
        # 描边
        cv2.putText(canvas, text, (cx, cy), self.font, self.font_scale,
                    (*self.stroke_color, 255), self.stroke_thickness, cv2.LINE_AA)
        # 主字
        cv2.putText(canvas, text, (cx, cy), self.font, self.font_scale,
                    (*self.color, 255), self.thickness, cv2.LINE_AA)
        return canvas
