"""L2/L3 中间产物: TextCandidate + RegionProposal。

设计原则:
- 坐标统一用原始像素 (不归一化)。
- polygon 是任意 4+ 点 (PaddleOCR 是 4 点, 但允许 detector 输出任意形状)。
- 所有 "像不像字幕" 的判断都不在这里, 这里只携带客观观测。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

Point = Tuple[float, float]
Polygon = List[Point]
BBox = Tuple[float, float, float, float]  # x_min, y_min, x_max, y_max


@dataclass
class TextCandidate:
    """L3 detector 单次输出 — 一个文字块。

    detector 不判断字幕, 只识别文字位置和内容。
    """
    polygon: Polygon
    text: str
    confidence: float
    language: Optional[str] = None     # "zh", "en", etc.
    detector_id: str = "unknown"       # "paddleocr_ch", "dbnet", ...
    # 衍生字段 (由 detector 或 wrapper 计算, 不在 detector 内部硬编码阈值)
    char_count: Optional[int] = None
    bbox: Optional[BBox] = None        # 从 polygon 推出来
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.bbox is None:
            xs = [p[0] for p in self.polygon]
            ys = [p[1] for p in self.polygon]
            self.bbox = (min(xs), min(ys), max(xs), max(ys))
        if self.char_count is None:
            self.char_count = len(self.text.strip())


@dataclass
class FrameTextCandidates:
    """一帧的所有 detector 输出。"""
    frame_id: int
    timestamp_ms: int
    width: int
    height: int
    candidates: List[TextCandidate] = field(default_factory=list)

    def __iter__(self):
        return iter(self.candidates)

    def __len__(self):
        return len(self.candidates)


@dataclass
class RegionProposal:
    """L2 region_policy 输出 — 字幕可能区域。

    proposal 是"候选区", weight 表示政策对这个区域是字幕的置信度 (0-1)。
    不在 proposal 里的区域不代表不是字幕, 只是这个 policy 没主张。
    """
    polygon: Polygon
    weight: float                                # 0.0 - 1.0
    source: str                                  # "policy:bottom_horizontal" / "ocr_density" 等
    description: Optional[str] = None