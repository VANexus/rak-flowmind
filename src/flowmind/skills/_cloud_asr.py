"""阿里百炼 Paraformer 语音识别封装。

两条路径：
- transcribe_local：**本地 wav 经 WebSocket 流式直推云端**（首选——无需公网 URL）。
- transcribe：录音文件识别 REST 异步（提交 URL → 轮询 → 取结果），适合已有媒资 URL。

云优先原则：ASR 全走云端；无 key 显式报错。
统一输出句段 [{index, begin, end, text}]（秒），供翻译与 TTS 逐句对齐。
key 由调用方从 env 读出后传入，不进 config 文件。
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

DEFAULT_BASE = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_MODEL = "paraformer-v2"


class ASRError(Exception):
    """ASR 调用失败。category/retriable 语义与 errors.py 对齐。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def submit_task(
    audio_url: str, *, api_key: str, api_base: str = DEFAULT_BASE,
    model: str = DEFAULT_MODEL,
) -> str:
    """提交录音识别任务，返回 task_id。"""
    if not api_key:
        raise ValueError(
            "收到空 API key。请检查环境变量 DASHSCOPE_API_KEY 是否设置。"
            "云优先原则：ASR 必须走云端，不做本地降级。"
        )
    url = f"{api_base.rstrip('/')}/services/audio/asr"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "input": {"file_urls": [audio_url]},
        # 单说话人营销视频：关掉说话人分离，保留默认标点
        "parameters": {"diarization_enabled": False},
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30.0)
    except requests.exceptions.RequestException as exc:
        raise ASRError("ASR 提交失败", category="environment") from exc
    if resp.status_code >= 500:
        raise ASRError(f"ASR 提交 HTTP {resp.status_code}", category="transient", retriable=True)
    if resp.status_code >= 400:
        raise ASRError(f"ASR 提交 HTTP {resp.status_code}（检查音频 URL 可访问性）", category="video")
    data = resp.json()
    task_id = (data.get("output") or {}).get("task_id")
    if not task_id:
        raise ASRError("ASR 提交响应缺 task_id")
    return str(task_id)


def poll_task(
    task_id: str, *, api_key: str, api_base: str = DEFAULT_BASE,
    interval_s: float = 2.0, max_wait_s: float = 600.0,
) -> dict:
    """轮询任务至终态，返回最终响应 JSON。FAILED/超时抛 ASRError。"""
    url = f"{api_base.rstrip('/')}/services/audio/asr"
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = time.monotonic() + max_wait_s
    while True:
        try:
            resp = requests.get(url, headers=headers,
                                params={"task_id": task_id}, timeout=30.0)
        except requests.exceptions.RequestException as exc:
            raise ASRError("ASR 轮询网络错误", category="environment") from exc
        if resp.status_code >= 500:
            raise ASRError(f"ASR 轮询 HTTP {resp.status_code}",
                           category="transient", retriable=True)
        if resp.status_code >= 400:
            raise ASRError(f"ASR 轮询 HTTP {resp.status_code}", category="video")
        data = resp.json()
        status = str((data.get("output") or {}).get("task_status", "")).upper()
        if status == "SUCCEEDED":
            return data
        if status == "FAILED":
            reason = str((data.get("output") or {}).get("message", ""))[:200]
            raise ASRError(f"ASR 任务失败: {reason}", category="video")
        if time.monotonic() > deadline:
            raise ASRError("ASR 任务超时未完成", category="environment",
                           retriable=False)
        if interval_s > 0:
            time.sleep(interval_s)


def fetch_transcription(transcription_url: str) -> list[dict]:
    """拉取转写 JSON，解析为句段列表 [{index, begin, end, text}]（秒）。"""
    try:
        resp = requests.get(transcription_url, timeout=60.0)
    except requests.exceptions.RequestException as exc:
        raise ASRError("转写结果拉取失败", category="environment") from exc
    if resp.status_code >= 400:
        raise ASRError(f"转写结果 HTTP {resp.status_code}", category="transient",
                       retriable=True)
    sentences: list[dict] = []
    for transcript in resp.json().get("transcripts", []):
        for s in transcript.get("sentences", []):
            text = str(s.get("text", "")).strip()
            if not text:
                continue
            sentences.append({
                "begin": int(s["begin_time"]) / 1000.0,
                "end": int(s["end_time"]) / 1000.0,
                "text": text,
            })
    return [
        {"index": i, "begin": s["begin"], "end": s["end"], "text": s["text"]}
        for i, s in enumerate(sentences)
    ]


def transcribe(
    audio_url: str, *, api_key: str, api_base: str = DEFAULT_BASE,
    model: str = DEFAULT_MODEL, interval_s: float = 2.0,
    max_wait_s: float = 600.0, poll_url_suffix: str | None = None,
) -> list[dict]:
    """一站式：提交 → 轮询 → 拉取并解析句段。"""
    task_id = submit_task(audio_url, api_key=api_key, api_base=api_base, model=model)
    base = f"{api_base.rstrip('/')}/services/audio/asr"
    suffix = poll_url_suffix or ""
    final = _poll_with_suffix(task_id, api_key=api_key, url=f"{base}{suffix}",
                              interval_s=interval_s, max_wait_s=max_wait_s)
    transcription_url = _extract_transcription_url(final)
    return fetch_transcription(transcription_url)


def _poll_with_suffix(
    task_id: str, *, api_key: str, url: str,
    interval_s: float, max_wait_s: float,
) -> dict:
    """poll_task 的 URL 可注入版本（测试用 suffix 区分端点）。"""
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = time.monotonic() + max_wait_s
    while True:
        try:
            resp = requests.get(url, headers=headers,
                                params={"task_id": task_id}, timeout=30.0)
        except requests.exceptions.RequestException as exc:
            raise ASRError("ASR 轮询网络错误", category="environment") from exc
        if resp.status_code >= 400:
            cat = "transient" if resp.status_code >= 500 else "video"
            raise ASRError(f"ASR 轮询 HTTP {resp.status_code}", category=cat)
        data = resp.json()
        output = data.get("output") or {}
        status = str(output.get("task_status", "")).upper()
        if status == "SUCCEEDED":
            return data
        if status == "FAILED":
            raise ASRError(f"ASR 任务失败: {str(output.get('message', ''))[:200]}",
                           category="video")
        if time.monotonic() > deadline:
            raise ASRError("ASR 任务超时未完成", category="environment")
        if interval_s > 0:
            time.sleep(interval_s)


def _extract_transcription_url(final_response: dict) -> str:
    """从 SUCCEEDED 响应里挖 transcription_url（兼容 results/file_url 两种结构）。"""
    out = final_response.get("output") or {}
    url = out.get("transcription_url")
    if url:
        return str(url)
    results = final_response.get("results") or {}
    for v in results.values():
        u = v.get("transcription_url")
        if u:
            return str(u)
    raise ASRError("SUCCEEDED 响应缺 transcription_url")


# ── 本地流式路径（WebSocket 直推，无需公网 URL）──

STREAM_MODEL = "paraformer-realtime-v2"


def transcribe_local(
    wav_path: str, *, api_key: str, model: str = STREAM_MODEL,
) -> list[dict]:
    """本地音频文件流式推给云端识别，返回句段列表（秒）。

    经 _stream_recognize 适配层（dashscope Recognition WS），测试可替换。
    """
    if not api_key:
        raise ValueError(
            "收到空 API key。请检查环境变量 DASHSCOPE_API_KEY 是否设置。"
            "云优先原则：ASR 必须走云端，不做本地降级。"
        )
    if not Path(wav_path).exists():
        raise ASRError(f"音频文件不存在: {wav_path}", category="video")
    sentences = _stream_recognize(wav_path, api_key=api_key, model=model)
    out: list[dict] = []
    for s in sentences:
        text = str(s.get("text", "")).strip()
        if not text:
            continue
        out.append({
            "begin": int(s["begin_time"]) / 1000.0,
            "end": int(s["end_time"]) / 1000.0,
            "text": text,
        })
    return [
        {"index": i, "begin": s["begin"], "end": s["end"], "text": s["text"]}
        for i, s in enumerate(out)
    ]


def _stream_recognize(wav_path: str, *, api_key: str, model: str) -> list[dict]:
    """dashscope Recognition WS 薄适配层：本地文件二进制分片上云。

    返回原始句 [{begin_time, end_time, text}]（毫秒）。
    生产实现用 dashscope SDK；测试通过 monkeypatch 替换本函数。
    懒 import：不装 dashscope 不影响包导入。
    """
    try:
        from dashscope.audio.asr import Recognition  # type: ignore
    except ImportError as exc:
        raise ASRError(
            "未安装 dashscope SDK（uv add dashscope 后可用）",
            category="environment",
        ) from exc

    sentences: list[dict] = []
    partial: dict[int, str] = {}  # sentence_id → 累积文本
    final_times: dict[int, tuple[int, int]] = {}

    def on_event(result):
        try:
            for sent in result.get_sentence():
                sid = id(sent) if not isinstance(sent, dict) else sent.get("sentence_id", 0)
                text = sent.get("text", "") if isinstance(sent, dict) else getattr(sent, "text", "")
                begin = sent.get("begin_time") if isinstance(sent, dict) else getattr(sent, "begin_time", None)
                end = sent.get("end_time") if isinstance(sent, dict) else getattr(sent, "end_time", None)
                if begin is not None and end is not None:
                    final_times[sid] = (int(begin), int(end))
                if text:
                    partial[sid] = text
        except Exception:  # 单事件解析失败不中断整条流
            pass

    def on_error(err):
        msg = str(err)
        if "throttl" in msg.lower() or "429" in msg:
            raise ASRError("ASR 限流", category="transient", retriable=True) from err
        raise ASRError(f"ASR 流式失败: {type(err).__name__}", category="video") from err

    recognition = Recognition(
        model=model,
        format="wav",
        sample_rate=16000,
        callback=type("_CB", (), {
            "on_event": staticmethod(on_event),
            "on_error": staticmethod(on_error),
            "on_complete": staticmethod(lambda: None),
        })(),
    )
    try:
        recognition.start()
        chunk_size = 3200  # 100ms @16kHz 16bit mono
        with open(wav_path, "rb") as f:
            f.seek(44)  # 跳过 wav 头
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                recognition.send_audio_frame(chunk)
        recognition.stop()
    except ASRError:
        raise
    except Exception as exc:
        raise ASRError(f"ASR 流式异常: {type(exc).__name__}", category="video") from exc

    return [
        {"begin_time": t[0], "end_time": t[1], "text": partial[sid]}
        for sid, t in sorted(final_times.items(), key=lambda kv: kv[1][0])
        if sid in partial
    ]
