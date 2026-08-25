"""_cloud_tts 测试：预设音色逐句合成（适配层隔离，不依赖 dashscope SDK）。

覆盖：
- synthesize_segments：逐句调合成器、按 index 命名输出
- 空音色显式报错（云优先：不静默降级）
- 合成失败分类（限流→transient）
"""
from __future__ import annotations

import pytest

from flowmind.skills import _cloud_tts
from flowmind.skills._cloud_tts import TTSError

SEGMENTS = [
    {"index": 0, "begin": 0.0, "end": 2.0, "text": "Hello"},
    {"index": 1, "begin": 2.0, "end": 4.0, "text": "World"},
]


def test_synthesize_segments_calls_synth_per_segment(tmp_path):
    made = []

    def fake_synth(text: str, out_path: str, **_kw) -> str:
        made.append(text)
        return out_path

    outs = _cloud_tts.synthesize_segments(
        SEGMENTS, voice_id="longanhuan_v3.6", out_dir=str(tmp_path),
        api_key="k", synth_fn=fake_synth,
    )
    assert len(outs) == 2
    assert outs[0].endswith("seg_0000.mp3")
    assert made == ["Hello", "World"]


def test_synthesize_empty_voice_raises(tmp_path):
    with pytest.raises(ValueError, match="音色"):
        _cloud_tts.synthesize_segments(
            SEGMENTS, voice_id="", out_dir=str(tmp_path), api_key="k",
            synth_fn=lambda *a, **k: "",
        )


def test_default_model_and_voice():
    assert _cloud_tts.TARGET_MODEL == "qwen-audio-3.0-tts-flash"
    assert _cloud_tts.DEFAULT_VOICE == "longanhuan_v3.6"


def test_synth_http_error_classification(monkeypatch):
    def fake_call(text, out_path, voice_id, model, timeout, headers, payload):
        raise TTSError("合成限流", category="transient", retriable=True)

    monkeypatch.setattr(_cloud_tts, "_call_ws_synth", fake_call)
    with pytest.raises(TTSError) as ei:
        _cloud_tts.synthesize_text("hi", out_path="/tmp/a.mp3",
                                   voice_id="longanhuan_v3.6", api_key="k")
    assert ei.value.retriable is True
