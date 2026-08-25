"""Phase E+ — EdgeTtsBackend 网络失败稳健性单测。

覆盖:
  - 永远超时的网络: graceful degrade 返回静音, 不抛
  - 空文本短路
  - 注册到 TtsRegistry
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from video_localization_engine.localizer import (
    EdgeTtsBackend,
    TtsRegistry,
)
from video_localization_engine.localizer.protocols import TtsBackend


def test_EdgeTtsBackend_protocol_conformance():
    """EdgeTtsBackend 满足 TtsBackend Protocol。"""
    assert isinstance(EdgeTtsBackend(), TtsBackend)
    print("✓ EdgeTtsBackend protocol: OK")


def test_EdgeTtsBackend_graceful_degrade_on_network_error():
    """Mock 一个永远超时的连接 — 验证返回静音数组 + 不抛。"""
    backend = EdgeTtsBackend(timeout_sec=1, max_retries=1, fallback_silence_sec=0.5)

    def _always_timeout(text, voice):
        # 模拟 edge_tts 网络挂死
        raise TimeoutError("simulated WebSocket read timeout (WSL2 unreachable)")

    with patch.object(backend, "_synth_with_retry", side_effect=_always_timeout):
        # 此处必须不抛
        wav = backend.synth("你好世界", "zh-CN", sample_rate=24000)

    assert isinstance(wav, np.ndarray), f"expected ndarray, got {type(wav)}"
    assert wav.dtype == np.float32, f"expected float32, got {wav.dtype}"
    assert wav.ndim == 1, f"expected 1D, got shape {wav.shape}"
    n_expected = int(0.5 * 24000)  # fallback_silence_sec=0.5
    assert wav.shape[0] == n_expected, (
        f"expected {n_expected} samples (0.5s @ 24kHz), got {wav.shape[0]}"
    )
    # 静音 → 全 0
    assert np.all(wav == 0.0), f"silence should be all-zero, got non-zero values"
    print(f"✓ EdgeTtsBackend graceful degrade: shape={wav.shape} dtype={wav.dtype}")


def test_EdgeTtsBackend_strict_raises_on_failure():
    """strict=True 时, 网络失败应当抛 RuntimeError (而不是吞掉)。"""
    backend = EdgeTtsBackend(strict=True, timeout_sec=1, max_retries=1)

    def _always_timeout(text, voice):
        raise TimeoutError("simulated")

    with patch.object(backend, "_synth_with_retry", side_effect=_always_timeout):
        try:
            backend.synth("hi", "en-US", sample_rate=24000)
            raised = False
        except RuntimeError as e:
            raised = True
            assert "edge-tts" in str(e).lower()
        assert raised, "strict=True should raise RuntimeError"
    print("✓ EdgeTtsBackend strict mode: OK")


def test_EdgeTtsBackend_empty_text_short_circuit():
    """空文本应直接返回 0 长度, 不进网络。"""
    backend = EdgeTtsBackend()
    assert backend.synth("", "en", sample_rate=24000).shape == (0,)
    assert backend.synth("   ", "en", sample_rate=24000).shape == (0,)
    print("✓ EdgeTtsBackend empty text: OK")


def test_EdgeTtsBackend_registered():
    """EdgeTtsBackend 已注册到 TtsRegistry('edge_tts')。"""
    assert "edge_tts" in TtsRegistry.available(), (
        f"Available: {TtsRegistry.available()}"
    )
    cls = TtsRegistry.get("edge_tts")
    assert cls is EdgeTtsBackend
    print(f"✓ EdgeTtsBackend registered: keys={TtsRegistry.available()}")


def test_EdgeTtsBackend_voice_pick():
    """_pick_voice 根据 locale 选默认 voice, 找不到回落 en。"""
    backend = EdgeTtsBackend()
    assert backend._pick_voice("zh") == "zh-CN-XiaoxiaoNeural"
    assert backend._pick_voice("zh-CN") == "zh-CN-XiaoxiaoNeural"
    assert backend._pick_voice("en") == "en-US-AriaNeural"
    assert backend._pick_voice("ja-JP") == "ja-JP-NanamiNeural"
    # 兜底: 未知 locale 用 en
    assert backend._pick_voice("xx-XX") == "en-US-AriaNeural"
    print("✓ EdgeTtsBackend voice pick: OK")
