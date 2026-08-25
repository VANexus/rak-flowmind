"""L3 TextDetector Registry."""
from __future__ import annotations

from video_localization_engine.detector.base import TextDetector
from video_localization_engine.utils.registry import RegistryBase


class TextDetectorRegistry(RegistryBase[type[TextDetector]]):
    pass


# 默认注册
from video_localization_engine.detector.paddle_backend import PaddleOCRDetector
TextDetectorRegistry.register("paddleocr_ch", PaddleOCRDetector)
TextDetectorRegistry.register("paddleocr_en", lambda: PaddleOCRDetector(lang="en"))