"""L7.3 TimingAligner — TTS 音频时长 vs 原字幕时长的对齐。

策略:
  - 短于目标: 尾部补静音
  - 长于目标且 ≤ 20%: 重采样 (调速)
  - 长于目标 > 20%: 截断
"""
from __future__ import annotations

import numpy as np


class TimingAligner:
    """时长对齐器。"""

    def __init__(self, max_speed_change: float = 0.20,
                 sample_rate: int = 24000):
        self.max_speed_change = max_speed_change
        self.sample_rate = sample_rate

    def align(self, audio: np.ndarray, target_duration_ms: int) -> np.ndarray:
        if target_duration_ms <= 0:
            return np.zeros(1, dtype=audio.dtype)
        cur_ms = int(len(audio) / max(self.sample_rate, 1) * 1000)
        target_n = max(1, int(target_duration_ms / 1000 * self.sample_rate))
        if cur_ms <= target_duration_ms:
            # 补静音
            pad_n = max(0, target_n - len(audio))
            if pad_n == 0:
                return audio
            return np.concatenate([audio, np.zeros(pad_n, dtype=audio.dtype)])
        # cur_ms > target_ms
        ratio = cur_ms / target_duration_ms
        if ratio <= 1 + self.max_speed_change:
            # 调速: 线性插值
            idx = np.linspace(0, len(audio) - 1, target_n).astype(np.int64)
            return audio[idx]
        # 截断
        return audio[:target_n]
