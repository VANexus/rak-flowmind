"""L7.2 TtsEngine 门面 — 调度 TtsBackend。"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from video_localization_engine.localizer.protocols import TtsBackend
from video_localization_engine.localizer.registries import TtsRegistry


class TtsEngine:
    """TTS 门面 — 输入文本 + locale, 返回 mono float32 samples。"""

    def __init__(self, backend_name: str = "mock", **backend_kwargs):
        backend_cls = TtsRegistry.get(backend_name)
        self.backend: TtsBackend = backend_cls(**backend_kwargs)
        self.backend_name = backend_name

    def synth(self, text: str, target_locale: str,
              sample_rate: int, **kwargs) -> np.ndarray:
        return self.backend.synth(text, target_locale, sample_rate, **kwargs)

    @property
    def supported_locales(self) -> List[str]:
        return self.backend.supported_locales
