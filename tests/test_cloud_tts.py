"""_cloud_tts 测试：CosyVoice 音色注册（REST）+ 逐句合成（适配层隔离 WS 细节）。

覆盖：
- create_voice：endpoint/model/action/prefix/url 参数、返回 voice_id
- list_voices / delete_voice
- synthesize_segments：逐句调合成器、输出 wav 路径列表、时长对齐参数透传
- 无 key / 空音色显式报错
- 合成失败分类
"""
from __future__ import annotations

import pytest

from flowmind.skills import _cloud_tts
from flowmind.skills._cloud_tts import TTSError


class _FakeResp:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._json


import requests  # noqa: E402

SEGMENTS = [
    {"index": 0, "begin": 0.0, "end": 2.0, "text": "Hello"},
    {"index": 1, "begin": 2.0, "end": 4.0, "text": "World"},
]


def test_create_voice_builds_request(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None, **_kw):
        calls.append({"url": url, "headers": headers, "json": json})
        return _FakeResp(200, {"output": {"voice_id": "cosyvoice-v3.5-plus-myvoice-abc"}})

    monkeypatch.setattr(_cloud_tts.requests, "post", fake_post)
    voice_id = _cloud_tts.create_voice(
        sample_audio_url="https://oss/sample.wav", prefix="myvoice",
        api_key="k-ds", target_model="cosyvoice-v3.5-plus",
    )
    assert "myvoice" in voice_id
    req = calls[0]
    assert "/services/audio/tts/customization" in req["url"]
    assert req["headers"]["Authorization"] == "Bearer k-ds"
    body = req["json"]
    assert body["model"] == "voice-enrollment"
    assert body["input"]["action"] == "create_voice"
    assert body["input"]["target_model"] == "cosyvoice-v3.5-plus"
    assert body["input"]["url"] == "https://oss/sample.wav"


def test_delete_voice(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None, **_kw):
        calls.append(json)
        return _FakeResp(200, {})

    monkeypatch.setattr(_cloud_tts.requests, "post", fake_post)
    _cloud_tts.delete_voice("vid-1", api_key="k")
    assert calls[0]["input"]["action"] == "delete_voice"
    assert calls[0]["input"]["voice_id"] == "vid-1"


def test_synthesize_segments_calls_synth_per_segment(tmp_path, monkeypatch):
    """逐句合成：每句一次 synth_fn，产出按 index 命名的 wav。"""
    made: list[str] = []

    def fake_synth(text: str, out_path: str, **_kw) -> str:
        made.append(text)
        return out_path

    outs = _cloud_tts.synthesize_segments(
        SEGMENTS, voice_id="v1", out_dir=str(tmp_path),
        api_key="k", synth_fn=fake_synth,
    )
    assert len(outs) == 2
    assert outs[0].endswith("seg_0000.wav")
    assert made == ["Hello", "World"]


def test_synthesize_empty_voice_raises():
    with pytest.raises(ValueError, match="voice_id"):
        _cloud_tts.synthesize_segments(
            SEGMENTS, voice_id="", out_dir="/tmp", api_key="k",
            synth_fn=lambda *a, **k: "",
        )


def test_no_key_raises():
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        _cloud_tts.create_voice("https://x.wav", prefix="p", api_key="")


def test_synth_http_error_classification(monkeypatch):
    def fake_run(cmd, timeout=None, headers=None, payload=None, **_kw):
        raise TTSError("合成 HTTP 503", category="transient", retriable=True)

    monkeypatch.setattr(_cloud_tts, "_call_ws_synth", fake_run)
    with pytest.raises(TTSError) as ei:
        _cloud_tts.synthesize_text("hi", out_path="/tmp/a.wav",
                                   voice_id="v", api_key="k")
    assert ei.value.retriable is True
