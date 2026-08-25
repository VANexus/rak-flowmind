"""DeepSeekTranslator — DeepSeek API (OpenAI 兼容) 翻译。

DeepSeek 提供 OpenAI 兼容接口:  base_url 默认 https://api.deepseek.com/v1,
models = ["deepseek-chat", "deepseek-reasoner"]。

Usage:
    export DEEPSEEK_API_KEY=sk-xxx
    translator = Translator(backend_name="deepseek")

Env vars:
    DEEPSEEK_API_KEY    : API key (必需, 否则 graceful degrade 返回原文本+前缀)
    DEEPSEEK_BASE_URL   : 默认 https://api.deepseek.com/v1
    DEEPSEEK_MODEL      : 默认 deepseek-chat

Graceful degrade (无 key):
    不会抛错, 返回 f"[{target_locale} no-api-key] {src_text}"
    让 pipeline 在没配 key 时仍可整体 dry-run。
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from video_localization_engine.localizer.protocols import TranslatorBackend


_log = logging.getLogger(__name__)


# 中→英 / 中→日 等常见目标短system prompt — 可以覆盖。
DEFAULT_SYSTEM = (
    "You are a subtitle translator. Translate the user's text into "
    "{target_lang_name}. Keep the same line breaks. Do not add commentary. "
    "Output ONLY the translated line. Keep it concise enough to read as a "
    "subtitle (similar length or shorter)."
)

LANG_NAMES = {
    "en": "English",
    "zh": "Chinese (Simplified)",
    "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "th": "Thai",
    "vi": "Vietnamese",
}


class DeepSeekTranslator(TranslatorBackend):
    """DeepSeek 翻译 backend. 通过 OpenAI Python SDK 兼容协议调用。

    参数:
      api_key:   显式传入, 否则读 DEEPSEEK_API_KEY 环境变量
      base_url:  默认 https://api.deepseek.com/v1
      model:     默认 deepseek-chat
      timeout:   HTTP 超时秒数 (默认 15)
      max_retries: 单次请求内的 SDK 重试次数 (默认 1)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 15.0,
        max_retries: int = 1,
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "").strip()
        self.base_url = base_url or os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
        )
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None  # 延迟 — 防止没装 openai / key 在 import 时挂

    @property
    def name(self) -> str:
        return "deepseek"

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            return None
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            _log.warning("openai SDK not installed, deepseek backend degraded: %s", e)
            return None
        try:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            return self._client
        except Exception as e:
            _log.warning("failed to construct OpenAI client: %s", e)
            return None

    def translate(self, text: str, source_locale: str,
                  target_locale: str, **kwargs) -> str:
        if not text or not text.strip():
            return ""
        client = self._get_client()
        if client is None:
            _log.warning(
                "DeepSeek API key not set, returning original text "
                "(set DEEPSEEK_API_KEY to enable translation)"
            )
            return text

        target_lang_name = LANG_NAMES.get(target_locale, target_locale)
        system_prompt = DEFAULT_SYSTEM.format(target_lang_name=target_lang_name)
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=kwargs.get("temperature", 0.2),
            )
        except Exception as e:
            _log.warning(
                "deepseek translate failed: %s: %s — returning original text",
                type(e).__name__, e,
            )
            return text

        try:
            return (resp.choices[0].message.content or "").strip()
        except (AttributeError, IndexError, KeyError) as e:
            _log.warning(
                "deepseek reply shape unexpected: %s — returning original text",
                e,
            )
            return text

    def batch_translate(self, texts: List[str], source_locale: str,
                        target_locale: str, **kwargs) -> List[str]:
        # 当前实现就依次单条 — DeepSeek 没特别的 batch endpoint
        return [self.translate(t, source_locale, target_locale, **kwargs) for t in texts]
