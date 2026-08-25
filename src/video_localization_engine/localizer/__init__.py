"""L7: Localizer — Translator + TTS + TimingAligner + Renderer + Compositor。

数据流:
  SubtitleTrack (.vle.json) → Translator → TtsEngine → TimingAligner
  → SubtitleRenderer → Compositor → 目标语言视频
"""
from video_localization_engine.localizer.compositor import Compositor
from video_localization_engine.localizer.debug_writer import write_localize_artifacts
from video_localization_engine.localizer.protocols import (
    CompositorBackend,
    RendererBackend,
    TranslatorBackend,
    TtsBackend,
)
from video_localization_engine.localizer.registries import (
    CompositorRegistry,
    RendererRegistry,
    TranslatorRegistry,
    TtsRegistry,
)
from video_localization_engine.localizer.subtitle_renderer import SubtitleRenderer
from video_localization_engine.localizer.timing_aligner import TimingAligner
from video_localization_engine.localizer.tts_engine import TtsEngine
from video_localization_engine.localizer.translator import Translator
from video_localization_engine.localizer.types import (
    AudioSegment,
    LocalizedSubtitle,
    LocalizedTrack,
)

# 默认注册 — orchestrator 无需手动 register
from video_localization_engine.localizer.backends.deepseek_translator import DeepSeekTranslator
from video_localization_engine.localizer.backends.edge_tts_backend import EdgeTtsBackend
from video_localization_engine.localizer.backends.ffmpeg_compositor import FFmpegCompositor
from video_localization_engine.localizer.backends.mock_translator import MockTranslator
from video_localization_engine.localizer.backends.mock_tts import MockTtsBackend
from video_localization_engine.localizer.backends.opencv_renderer import OpenCVRenderer
from video_localization_engine.localizer.backends.pillow_renderer import PillowRenderer

# 默认注册在 video_localization_engine.localizer.registries 中完成 (导入即注册)
# 这里仅 re-export 各 backend 类以保持向后兼容
# Future:
# CompositorRegistry.register("moviepy", MoviePyCompositor)

__all__ = [
    "TranslatorBackend", "TtsBackend", "RendererBackend", "CompositorBackend",
    "TranslatorRegistry", "TtsRegistry", "RendererRegistry", "CompositorRegistry",
    "Translator", "TtsEngine", "TimingAligner", "SubtitleRenderer", "Compositor",
    "AudioSegment", "LocalizedSubtitle", "LocalizedTrack",
    "MockTranslator", "MockTtsBackend", "EdgeTtsBackend",
    "OpenCVRenderer", "PillowRenderer", "FFmpegCompositor",
    "DeepSeekTranslator",
    "write_localize_artifacts",
]
