"""L3 TextDetector 抽象基类。

不假设 OCR 算法, 不假设语言, 不假设输出格式。
所有 OCR 引擎都要实现这个接口。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from video_localization_engine.types.detection import TextCandidate
from video_localization_engine.types.video import FramePacket


class TextDetector(ABC):
    """文字检测器抽象基类。

    输出: 每一帧的所有 TextCandidate (多文字块)。
    不做: 字幕判断, 跨帧追踪。
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def supported_languages(self) -> List[str]: ...

    @abstractmethod
    def detect(self, packet: FramePacket) -> List[TextCandidate]: ...

    def warmup(self) -> None:
        """可选: 预热模型 (e.g. PaddleOCR 首次推理要 30s)"""
        pass