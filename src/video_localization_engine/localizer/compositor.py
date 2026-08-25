"""L7.5 Compositor 门面 — 调度 CompositorBackend。"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from video_localization_engine.localizer.protocols import CompositorBackend
from video_localization_engine.localizer.registries import CompositorRegistry


class Compositor:
    """合成门面 — 帧序列 + 音频 → mp4。"""

    def __init__(self, backend_name: str = "ffmpeg", **backend_kwargs):
        backend_cls = CompositorRegistry.get(backend_name)
        self.backend: CompositorBackend = backend_cls(**backend_kwargs)
        self.backend_name = backend_name

    def composite(self, frames: List[np.ndarray], audio: Optional[np.ndarray],
                  sample_rate: int, fps: float, output_path: str) -> str:
        return self.backend.composite(frames, audio, sample_rate, fps, output_path)

    @property
    def is_available(self) -> bool:
        if hasattr(self.backend, "is_available"):
            return self.backend.is_available()
        return True
