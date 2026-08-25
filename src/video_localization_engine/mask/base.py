"""L5 MaskBackend 抽象基类。

业务层 (MaskGenerator) 只持有 MaskBackend 引用;具体 rasterize 算法由 backend 实现。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np

from video_localization_engine.mask.artifact import MaskArtifact, DilationSpec
from video_localization_engine.types.instance import SubtitleInstance
from video_localization_engine.types.video import FramePacket


class MaskBackend(ABC):
    """Mask backend 抽象。

    业务层不直接调用此 backend 的 rasterize;而是 backend.build 返回 MaskArtifact
    (只填 polygons/mask_path),后续 InpaintingEngine 调 artifact.materialize() 物化。
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def build(self, instances: List[SubtitleInstance], packet: FramePacket,
              dilation_px: int = 0,
              dilation_xy: Optional[Tuple[int, int]] = None) -> List[MaskArtifact]:
        """输入一帧 packet + 跨帧稳定 instance 列表,返回每 instance 各自的 MaskArtifact。

        多个 instance 各自独立 artifact — orchestrator 端决定是否 union。
        instance.observed_frame_ids[i] == packet.frame_id 时, 用 instance.observed_polygons[i]。

        dilation 参数优先级:
          dilation_xy = (dx, dy) → 用 x/y 独立 dilation
          dilation_xy = None → 回退到 dilation_px (int), 等价 (n, n)
        """
