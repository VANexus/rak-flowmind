"""Phase E+ — DeepSeekTranslator (L4 翻译 backend) 单测。

覆盖:
  - Protocol 合规
  - 无 API key 时返回原文本 (绝不能返回配置/错误提示字符串)
  - API 调用失败时返回原文本
  - 注册到 TranslatorRegistry
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from video_localization_engine.localizer import (
    DeepSeekTranslator,
    TranslatorRegistry,
)
from video_localization_engine.localizer.protocols import TranslatorBackend


# 任何这些子串出现在返回文本里都视为 "配置/错误提示字符串" — 会被当作字幕渲染
# 故 deepseek fallback 必须避免这些
FORBIDDEN_SUBSTRINGS = (
    "api-key",
    "API key",
    "DeepSeek",
    "Deepseek",
    "deepseek",
    "请配置",
    "no-api-key",
    "error",
    "[",
    "]",  # 方括号包裹的前缀 (e.g. "[zh no-api-key] ..." / "[zh error] ...")
)


def _assert_clean_fallback(result: str, original: str, ctx: str):
    """断言 result == original, 且不含任何配置/错误提示子串。"""
    assert result == original, (
        f"{ctx}: expected exactly original text {original!r}, got {result!r}"
    )
    lowered = result.lower()
    for bad in FORBIDDEN_SUBSTRINGS:
        assert bad.lower() not in lowered, (
            f"{ctx}: forbidden substring {bad!r} found in result {result!r}"
        )


def test_DeepSeekTranslator_protocol_conformance():
    """DeepSeekTranslator 满足 TranslatorBackend Protocol。"""
    assert isinstance(DeepSeekTranslator(), TranslatorBackend)
    print("✓ DeepSeekTranslator protocol: OK")


def test_deepseek_translator_no_api_key_returns_original(monkeypatch):
    """无 API key 时 translate() 必须返回原始 text (中文不变), 不能含任何配置提示。

    这是优先级 7 的 bug 修复 — 之前返回 "[zh no-api-key] 你好" 被当作字幕渲染。
    """
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    t = DeepSeekTranslator()
    assert t.api_key == "", f"expected empty key, got {t.api_key!r}"

    result = t.translate("你好", "zh", "en")
    _assert_clean_fallback(result, "你好", "no-api-key zh→en")
    print(f"✓ no-api-key zh→en: {result!r}")

    # 英文也一样
    result_en = t.translate("Hello World", "en", "zh")
    _assert_clean_fallback(result_en, "Hello World", "no-api-key en→zh")
    print(f"✓ no-api-key en→zh: {result_en!r}")


def test_DeepSeekTranslator_empty_text():
    """空文本短路返回 '', 不调用网络。"""
    t = DeepSeekTranslator()
    assert t.translate("", "en", "zh") == ""
    assert t.translate("   ", "en", "zh") == ""
    print("✓ DeepSeekTranslator empty text: OK")


def test_DeepSeekTranslator_registered():
    """DeepSeekTranslator 已注册到 TranslatorRegistry('deepseek')。"""
    assert "deepseek" in TranslatorRegistry.available(), (
        f"Available: {TranslatorRegistry.available()}"
    )
    cls = TranslatorRegistry.get("deepseek")
    assert cls is DeepSeekTranslator
    print(f"✓ DeepSeekTranslator registered: keys={TranslatorRegistry.available()}")


def test_DeepSeekTranslator_batch_translate_falls_back(monkeypatch):
    """batch_translate 序列调用 translate, 无 key 时逐条返回原文本。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    t = DeepSeekTranslator()
    out = t.batch_translate(["你好", "世界"], "zh", "en")
    assert out == ["你好", "世界"], f"expected original texts, got {out}"
    for s in out:
        _assert_clean_fallback(s, s, "batch no-api-key")
    print(f"✓ DeepSeekTranslator batch fallback: {out}")


def test_DeepSeekTranslator_api_error_returns_original(monkeypatch):
    """API 调用失败时返回原文本, 不能返回 "[zh error] ..." 这种错误提示。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    t = DeepSeekTranslator(api_key="sk-fake")  # 给个假 key 让 _get_client 走远一点

    class _BoomClient:
        def chat(self):  # noqa: ANN001
            raise RuntimeError("network unreachable")

        completions = property(lambda self: self.chat())

    with patch.object(t, "_get_client", return_value=_BoomClient()):
        result = t.translate("你好", "zh", "en")
    _assert_clean_fallback(result, "你好", "API error zh→en")
    print(f"✓ API error fallback: {result!r}")


def test_DeepSeekTranslator_reply_shape_bad_returns_original(monkeypatch):
    """API 返回结构异常时 (没有 choices 等) 也返回原文本。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    t = DeepSeekTranslator(api_key="sk-fake")

    class _WeirdReply:
        choices = []  # 触发 IndexError → fallback

    class _WeirdClient:
        class completions:
            @staticmethod
            def create(**kwargs):
                return _WeirdReply()

    with patch.object(t, "_get_client", return_value=_WeirdClient):
        result = t.translate("你好世界", "zh", "en")
    _assert_clean_fallback(result, "你好世界", "bad reply shape")
    print(f"✓ bad reply shape fallback: {result!r}")