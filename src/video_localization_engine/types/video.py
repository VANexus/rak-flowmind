"""L1 VideoMeta + FramePacket.

这些对象是 L1 输出, 整个 VLE 流转的"画面"基础。
不绑定分辨率, 不绑定坐标系 (统一以原始像素坐标流通)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class Orientation(str, Enum):
    """视频方向。"""
    LANDSCAPE = "landscape"   # w > h
    PORTRAIT = "portrait"     # w < h
    SQUARE = "square"         # w == h


@dataclass(frozen=True)
class VideoMeta:
    """视频元信息 (L1 输出)。整个 pipeline 只读。"""
    source_path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_ms: int
    orientation: Orientation
    has_audio: bool
    # 可选上游 hint, 不假设有
    content_type_hint: Optional[str] = None
    source_locale: Optional[str] = None  # "zh", "en", None=unknown

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)


@dataclass
class FramePacket:
    """单帧包 — L1 输出, L2-L6 输入。"""
    frame_id: int                       # 0-based
    timestamp_ms: int                   # 绝对时间
    image: np.ndarray                   # HxWx3 BGR uint8 (与 cv2 一致)
    meta: VideoMeta                     # 引用同一 VideoMeta

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]