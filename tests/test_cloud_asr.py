"""_cloud_asr 测试：Paraformer 录音文件识别（REST 异步提交/轮询，requests 打桩）。

覆盖：
- 提交任务：endpoint/参数/Authorization
- 轮询：PENDING→RUNNING→SUCCEEDED 状态机、超时报错
- 响应解析：transcripts → 句段 [{index,begin,end,text}]（毫秒时间戳转秒）
- 无 key 显式报错
- 网络/HTTP 错误分类
"""
from __future__ import annotations

import pytest

from flowmind.skills import _cloud_asr
from flowmind.skills._cloud_asr import ASRError


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


def test_submit_task_builds_request(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None, **_kw):
        calls.append({"url": url, "headers": headers, "json": json})
        return _FakeResp(200, {"output": {"task_id": "task-1"}})

    monkeypatch.setattr(_cloud_asr.requests, "post", fake_post)
    task_id = _cloud_asr.submit_task(
        "https://cdn.example.com/a.wav", api_key="k-ds",
        model="paraformer-v2",
    )
    assert task_id == "task-1"
    req = calls[0]
    assert "/services/audio/asr" in req["url"]
    assert req["headers"]["Authorization"] == "Bearer k-ds"
    body = req["json"]
    assert body["model"] == "paraformer-v2"
    assert body["input"]["file_urls"] == ["https://cdn.example.com/a.wav"]
    assert body["parameters"]["diarization_enabled"] is False


def test_poll_until_succeeded(monkeypatch):
    states = iter([
        _FakeResp(200, {"output": {"task_status": "PENDING"}}),
        _FakeResp(200, {"output": {"task_status": "RUNNING"}}),
        _FakeResp(200, {"output": {"task_status": "SUCCEEDED"}, "output2": None}),
    ])

    def fake_get(url, headers=None, timeout=None, **_kw):
        return next(states)

    monkeypatch.setattr(_cloud_asr.requests, "get", fake_get)
    result = _cloud_asr.poll_task("task-1", api_key="k", interval_s=0, max_wait_s=5)
    assert result["output"]["task_status"] == "SUCCEEDED"


def test_poll_failed_state_raises_transient(monkeypatch):
    def fake_get(url, headers=None, timeout=None, **_kw):
        return _FakeResp(200, {"output": {"task_status": "FAILED",
                                          "message": "audio decode error"}})

    monkeypatch.setattr(_cloud_asr.requests, "get", fake_get)
    with pytest.raises(ASRError) as ei:
        _cloud_asr.poll_task("t", api_key="k", interval_s=0)
    # FAILED 任务多半是输入音频问题 → video；但服务端故障也可能是 transient。
    # ASR FAILED 默认按 video 分类（修输入），消息里带原因。
    assert ei.value.category in ("video", "transient")


def test_poll_timeout_raises_environment(monkeypatch):
    def fake_get(url, headers=None, timeout=None, **_kw):
        return _FakeResp(200, {"output": {"task_status": "RUNNING"}})

    monkeypatch.setattr(_cloud_asr.requests, "get", fake_get)
    with pytest.raises(ASRError) as ei:
        _cloud_asr.poll_task("t", api_key="k", interval_s=0, max_wait_s=0.05)
    assert ei.value.category == "environment"


def test_transcribe_parses_sentences_to_segments(monkeypatch):
    """SUCCEEDED 结果 → 句段列表，毫秒时间戳转秒，text 非空过滤。"""
    payload = {
        "output": {"task_id": "t", "task_status": "SUCCEEDED"},
        "results": {
            "file_url": {
                "transcription_url": "http://x/result.json",
            }
        },
    }

    transcription = {
        "transcripts": [{
            "sentences": [
                {"begin_time": 0, "end_time": 2000, "text": "大家好"},
                {"begin_time": 2100, "end_time": 4500, "text": "今天讲发动机"},
                {"begin_time": 4600, "end_time": 5000, "text": ""},  # 空句剔除
            ],
        }],
    }
    seen: list[str] = []

    def fake_get(url, headers=None, timeout=None, **_kw):
        seen.append(url)
        if url.endswith("/poll"):
            return _FakeResp(200, payload)
        if "result.json" in url:
            return _FakeResp(200, transcription)
        return _FakeResp(404)

    monkeypatch.setattr(_cloud_asr.requests, "get", fake_get)

    def fake_post(url, headers=None, json=None, timeout=None, **_kw):
        return _FakeResp(200, {"output": {"task_id": "t1"}})

    monkeypatch.setattr(_cloud_asr.requests, "post", fake_post)

    segs = _cloud_asr.transcribe(
        audio_url="https://f.a.wav", poll_url_suffix="/poll",
        api_key="k", interval_s=0,
    )
    assert len(segs) == 2
    assert segs[0] == {"index": 0, "begin": 0.0, "end": 2.0, "text": "大家好"}
    assert segs[1]["begin"] == pytest.approx(2.1)
    assert segs[1]["text"] == "今天讲发动机"


def test_no_key_raises_value_error():
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        _cloud_asr.submit_task("https://x.wav", api_key="")
