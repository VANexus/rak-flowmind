"""L3: Text detectors — output TextCandidates."""
from video_localization_engine.detector.base import TextDetector
from video_localization_engine.detector.paddle_backend import PaddleOCRDetector
from video_localization_engine.detector.registry import TextDetectorRegistry

__all__ = [
    "TextDetector",
    "PaddleOCRDetector",
    "TextDetectorRegistry",
]
