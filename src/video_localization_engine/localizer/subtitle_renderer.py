"""L7.4 SubtitleRenderer 门面 — 调度 RendererBackend。"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from video_localization_engine.localizer.protocols import RendererBackend
from video_localization_engine.localizer.registries import RendererRegistry
from video_localization_engine.types.detection import BBox


class SubtitleRenderer:
    """字幕渲染门面 — 输入文本+bbox+帧尺寸, 输出 RGBA 渲染图。"""

    def __init__(self, backend_name: str = "opencv", **backend_kwargs):
        backend_cls = RendererRegistry.get(backend_name)
        self.backend: RendererBackend = backend_cls(**backend_kwargs)
        self.backend_name = backend_name

    def render(self, text: str, bbox: BBox,
               frame_size: Tuple[int, int], **kwargs) -> np.ndarray:
        return self.backend.render(text, bbox, frame_size, **kwargs)
