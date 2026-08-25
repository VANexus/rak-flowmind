"""阿里百炼 Paraformer 录音文件识别封装（REST 异步：提交 → 轮询 → 取结果）。

云优先原则：ASR 全走云端；无 key 显式报错。
输入音频需可公网访问的 URL；返回句段 [{index, begin, end, text}]（秒），
供翻译与 TTS 逐句对齐使用。key 由调用方从 env 读出后传入，不进 config 文件。

接口形态参考百炼「录音文件识别」：
- POST {base}/services/audio/asr  提交任务，body: {model, input:{file_urls}, parameters}
- GET  {base}/services/audio/asr?task_id=...  轮询 task_status
- SUCCEEDED 后响应携带 transcription_url，GET 该 URL 得
  {transcripts:[{sentences:[{begin_time,end_time,text}]}]}（毫秒时间戳）
"""
from __future__ import annotations

import time

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
