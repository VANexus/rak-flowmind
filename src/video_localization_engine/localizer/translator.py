"""L7.1 Translator 门面 — 调度 TranslatorBackend。"""
from __future__ import annotations

from typing import List, Optional

from video_localization_engine.localizer.protocols import TranslatorBackend
from video_localization_engine.localizer.registries import TranslatorRegistry


class Translator:
    """翻译门面 — 输入源文本, 输出目标文本。"""

    def __init__(self, backend_name: str = "mock", **backend_kwargs):
        backend_cls = TranslatorRegistry.get(backend_name)
        self.backend: TranslatorBackend = backend_cls(**backend_kwargs)
        self.backend_name = backend_name

    def translate(self, text: str, source_locale: str,
                  target_locale: str, **kwargs) -> str:
        return self.backend.translate(text, source_locale, target_locale, **kwargs)

    def translate_batch(self, texts: List[str], source_locale: str,
                        target_locale: str, **kwargs) -> List[str]:
        return self.backend.batch_translate(
            texts, source_locale, target_locale, **kwargs)
