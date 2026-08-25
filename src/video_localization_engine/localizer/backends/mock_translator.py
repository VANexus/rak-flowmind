"""MockTranslator — 测试用, 不依赖任何外部翻译 API。"""
from __future__ import annotations

from video_localization_engine.localizer.protocols import TranslatorBackend


class MockTranslator(TranslatorBackend):
    """返回 '[target_locale] ' + text。

    用于:
      - 单元测试 (验证 translator 调用链)
      - 集成测试 (不调用 DeepL / Claude 等)
      - 同一段视频多语种演示 (locale 标签清晰可见)
    """

    def __init__(self, prefix_template: str = "[{locale}] "):
        self.prefix_template = prefix_template

    @property
    def name(self) -> str:
        return "mock"

    def translate(self, text: str, source_locale: str,
                  target_locale: str, **kwargs) -> str:
        prefix = self.prefix_template.format(locale=target_locale)
        return f"{prefix}{text}"
