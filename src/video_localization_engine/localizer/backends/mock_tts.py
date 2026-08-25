"""MockTtsBackend — 测试用, 输出静音 (零采样) numpy 数组。

不依赖 Edge-TTS / ElevenLabs / Coqui 等外部 TTS API。
时长估算 = len(text) * 0.06s/字 (粗略中英文), 保证不空。
"""
from __future__ import annotations

import numpy as np

from video_localization_engine.localizer.protocols import TtsBackend


class MockTtsBackend(TtsBackend):
    """返回全零 mono float32 samples。

    配置参数:
      - sec_per_char: 每字估算秒数 (默认 0.06, 中英文平均)
      - min_duration_sec: 最小时长 (默认 0.5s, 避免空)
    """

    def __init__(self, sec_per_char: float = 0.06, min_duration_sec: float = 0.5):
        self.sec_per_char = sec_per_char
        self.min_duration_sec = min_duration_sec

    @property
    def name(self) -> str:
        return "mock"

    def synth(self, text: str, target_locale: str,
              sample_rate: int, **kwargs) -> np.ndarray:
        n_sec = max(self.min_duration_sec, len(text) * self.sec_per_char)
        n = max(1, int(n_sec * sample_rate))
        return np.zeros(n, dtype=np.float32)
