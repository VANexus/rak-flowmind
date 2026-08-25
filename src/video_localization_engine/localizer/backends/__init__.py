"""localizer backends — 各 backend 实现。"""
from video_localization_engine.localizer.backends.deepseek_translator import DeepSeekTranslator
from video_localization_engine.localizer.backends.edge_tts_backend import EdgeTtsBackend
from video_localization_engine.localizer.backends.ffmpeg_compositor import FFmpegCompositor
from video_localization_engine.localizer.backends.mock_translator import MockTranslator
from video_localization_engine.localizer.backends.mock_tts import MockTtsBackend
from video_localization_engine.localizer.backends.opencv_renderer import OpenCVRenderer
from video_localization_engine.localizer.backends.pillow_renderer import PillowRenderer

__all__ = [
    "DeepSeekTranslator",
    "EdgeTtsBackend",
    "FFmpegCompositor",
    "MockTranslator",
    "MockTtsBackend",
    "OpenCVRenderer",
    "PillowRenderer",
]
