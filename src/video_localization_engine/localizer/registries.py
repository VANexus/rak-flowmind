"""L7 4 个 backend Registry — backend 可替换。"""
from __future__ import annotations

from video_localization_engine.localizer.backends import (
    DeepSeekTranslator,
    EdgeTtsBackend,
    FFmpegCompositor,
    MockTranslator,
    MockTtsBackend,
    OpenCVRenderer,
    PillowRenderer,
)
from video_localization_engine.localizer.protocols import (
    CompositorBackend,
    RendererBackend,
    TranslatorBackend,
    TtsBackend,
)
from video_localization_engine.utils.registry import RegistryBase


class TranslatorRegistry(RegistryBase[type[TranslatorBackend]]):
    pass


class TtsRegistry(RegistryBase[type[TtsBackend]]):
    pass


class RendererRegistry(RegistryBase[type[RendererBackend]]):
    pass


class CompositorRegistry(RegistryBase[type[CompositorBackend]]):
    pass


# 默认注册 — orchestrator 无需手动 register
TranslatorRegistry.register("mock", MockTranslator)
TranslatorRegistry.register("deepseek", DeepSeekTranslator)
TtsRegistry.register("mock", MockTtsBackend)
TtsRegistry.register("edge_tts", EdgeTtsBackend)
RendererRegistry.register("opencv", OpenCVRenderer)
RendererRegistry.register("pillow", PillowRenderer)
CompositorRegistry.register("ffmpeg", FFmpegCompositor)
# Future:
# TranslatorRegistry.register("deepl", DeepLTranslator)
# CompositorRegistry.register("moviepy", MoviePyCompositor)
