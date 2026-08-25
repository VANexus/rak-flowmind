"""L7 backend 协议 (interfaces only — implementations live in localizer/backends/).

业务层 (Translator / TtsEngine / SubtitleRenderer / Compositor 门面) 只持有 backend 引用;
具体实现 (DeepL / Edge-TTS / PIL / ffmpeg) 必须满足 Protocol 才能被注册。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from video_localization_engine.types.detection import BBox


# ============================================================
# L7.1 Translator
# ============================================================
class TranslatorBackend(ABC):
    """翻译 backend: 输入源文 + 源/目标 locale, 返回目标语言文本。"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def translate(self, text: str, source_locale: str,
                  target_locale: str, **kwargs) -> str:
        """返回翻译后文本。"""
        ...

    def batch_translate(self, texts: List[str], source_locale: str,
                        target_locale: str, **kwargs) -> List[str]:
        """默认逐条调 translate;backend 可覆盖做 batch 优化。"""
        return [self.translate(t, source_locale, target_locale, **kwargs) for t in texts]


# ============================================================
# L7.2 TTS
# ============================================================
class TtsBackend(ABC):
    """TTS backend: 输入文本 + 目标语言, 返回 mono float32 音频。"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def synth(self, text: str, target_locale: str,
              sample_rate: int, **kwargs) -> np.ndarray:
        """返回 mono float32, 长度由 backend 决定 (不对齐, 由 TimingAligner 后续处理)。"""
        ...

    @property
    def supported_locales(self) -> List[str]:
        """backend 支持的 locale 列表 (默认空, 表示支持任意)。"""
        return []


# ============================================================
# L7.3 Renderer
# ============================================================
class RendererBackend(ABC):
    """字幕渲染 backend: 输入文本 + bbox + 帧尺寸, 返回 RGBA 渲染图。

    渲染图尺寸 == 帧尺寸, 仅 bbox 区域 alpha>0; 业务层做 alpha blend。
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def render(self, text: str, bbox: BBox,
               frame_size: Tuple[int, int], **kwargs) -> np.ndarray:
        """返回 HxWx4 uint8 RGBA。"""
        ...


# ============================================================
# L7.4 Compositor
# ============================================================
class CompositorBackend(ABC):
    """合成 backend: 输入帧序列 + 音频 + fps, 输出最终 mp4。"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def composite(self, frames: List[np.ndarray], audio: Optional[np.ndarray],
                  sample_rate: int, fps: float, output_path: str) -> str:
        """帧序列 + 音频 → mp4 文件, 返回 output_path。"""
        ...
