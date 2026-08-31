"""_cloud_asr 本地流式转写测试（WS 适配层隔离，不依赖 dashscope SDK）。

覆盖：
- transcribe_local：本地 wav 直推 → 句段列表（毫秒→秒）
- 无 key 显式报错
- WS 适配层异常分类（限流→transient / 其他→video）
- 文件不存在报错
"""
from __future__ import annotations

import pytest

from flowmind.skills import _cloud_asr
from flowmind.skills._cloud_asr import ASRError


def test_transcribe_local_calls_stream_adapter(monkeypatch, tmp_path):
    """本地文件经 _stream_recognize 推流，返回句段并做毫秒→秒换算。"""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFFfake")
    raw_sentences = [
        {"begin_time": 0, "end_time": 2100, "text": "大家好"},
        {"begin_time": 2200, "end_time": 4800, "text": "欢迎观看"},
        {"begin_time": 4900, "end_time": 5000, "text": ""},  # 空句剔除
    ]
    captured = {}

    def fake_stream(wav_path: str, *, api_key: str, model: str,
                    sample_rate: int = 16000, language_hints: list[str] | None = None):
        captured["wav"] = wav_path
        captured["key"] = api_key
        captured["model"] = model
        captured["sample_rate"] = sample_rate
        captured["language_hints"] = language_hints
        return raw_sentences

    monkeypatch.setattr(_cloud_asr, "_stream_recognize", fake_stream)
    segs = _cloud_asr.transcribe_local(
        str(wav), api_key="k", model="paraformer-realtime-8k-v1", sample_rate=8000)
    assert captured["wav"] == str(wav)
    assert captured["model"] == "paraformer-realtime-8k-v1"
    assert captured["sample_rate"] == 8000
    assert len(segs) == 2
    assert segs[0] == {"index": 0, "begin": 0.0, "end": 2.1, "text": "大家好"}
    assert segs[1]["end"] == pytest.approx(4.8)


def test_transcribe_local_no_key_raises(tmp_path):
    wav = tmp_path / "b.wav"
    wav.write_bytes(b"x")
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        _cloud_asr.transcribe_local(str(wav), api_key="")


def test_transcribe_local_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(_cloud_asr, "_stream_recognize",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("不该到这")))
    with pytest.raises(ASRError) as ei:
        _cloud_asr.transcribe_local(str(tmp_path / "no.wav"), api_key="k")
    assert ei.value.category == "video"


def test_stream_adapter_error_classification(monkeypatch, tmp_path):
    """适配层抛出的限流错保留 transient 分类。"""
    wav = tmp_path / "c.wav"
    wav.write_bytes(b"x")

    def throttled(*a, **kw):
        raise ASRError("限流", category="transient", retriable=True)

    monkeypatch.setattr(_cloud_asr, "_stream_recognize", throttled)
    with pytest.raises(ASRError) as ei:
        _cloud_asr.transcribe_local(str(wav), api_key="k")
    assert ei.value.retriable is True


def test_real_stream_adapter_without_sdk(monkeypatch, tmp_path):
    """未装 dashscope 时适配层给出明确 environment 错。"""
    import builtins

    real_import = builtins.__import__

    def no_dashscope(name, *a, **kw):
        if name.startswith("dashscope"):
            raise ImportError("No module named 'dashscope'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_dashscope)
    with pytest.raises(ASRError) as ei:
        _cloud_asr._stream_recognize("/x.wav", api_key="k", model="m")
    assert ei.value.category == "environment"
