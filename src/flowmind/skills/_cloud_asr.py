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


class ASRError(Exception):
    """ASR 调用失败。category/retriable 语义与 errors.py 对齐。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


# ── 本地流式路径（WebSocket 直推，无需公网 URL）──

STREAM_MODEL = "paraformer-realtime-8k-v1"


def transcribe_local(
    wav_path: str, *, api_key: str, model: str = STREAM_MODEL,
    sample_rate: int = 8000,
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
    sentences = _stream_recognize(wav_path, api_key=api_key, model=model,
                                  sample_rate=sample_rate)
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


def _stream_recognize(
    wav_path: str, *, api_key: str, model: str, sample_rate: int = 8000,
) -> list[dict]:
    """dashscope Recognition WS 薄适配层：本地文件 callback 流式直推。

    返回原始句 [{begin_time, end_time, text}]（毫秒，仅 sentence_end 完结句）。
    注意：SDK 1.27 的同步 call(file) 模式有 headers 缺失 bug（str(result) 即炸），
    必须走 start/send_audio_frame/stop 回调流式模式；callback 里也不得对
    result 做字符串化（同样触发该 bug）。测试通过 monkeypatch 替换本函数。
    """
    try:
        import dashscope
        from dashscope.audio.asr import Recognition  # type: ignore
    except ImportError as exc:
        raise ASRError(
            "未安装 dashscope SDK（uv add dashscope 后可用）",
            category="environment",
        ) from exc

    if not Path(wav_path).exists():
        raise ASRError(f"音频文件不存在: {wav_path}", category="video")

    # 关键：显式赋给 SDK 全局 key。仅设进程 env 时，WS 接收线程可能拿不到认证，
    # 表现为整条流零事件静默失败。
    dashscope.api_key = api_key

    final_sentences: list[dict] = []
    last_sentence: list[dict] = []   # 无 sentence_end 字段的模型（8k-v1）：当前未完成短语
    error_holder: list[str] = []
    completed = False
    _seen_final: set[tuple] = set()  # 8k-v1 去重：(begin, end)

    class _Callback:
        def on_open(self): ...
        def on_event(self, result):
            try:
                s = result.get_sentence()
                if not isinstance(s, dict):
                    return
                if s.get("sentence_end"):
                    final_sentences.append(s)      # v2 等模型：显式完结标记
                elif s.get("end_time") is not None:
                    # 8k-v1：无 sentence_end，但 end_time 存在 = 该短语已确定；
                    # 同一短语可能被重复推送，按 (begin, end) 去重后累积。
                    key = (s.get("begin_time"), s.get("end_time"))
                    if key not in _seen_final:
                        _seen_final.add(key)
                        final_sentences.append(s)
                else:
                    # 无 end_time = 仍在修订中的中间假设，只保留最新一条
                    last_sentence.clear()
                    last_sentence.append(s)
            except Exception:  # 单事件解析失败不中断整条流
                pass
        def on_error(self, response):
            # 不触碰 response 的字符串化（SDK headers bug）
            code = getattr(response, "code", None)
            message = getattr(response, "message", None)
            error_holder.append(f"{code}: {message}" if message else str(code))
        def on_complete(self):
            nonlocal completed
            completed = True
            # 兜底：模型全程未发出任何确定短语（全为中间假设）时，取最后一条作为单句。
            # 正常路径下 8k-v1 的确定短语已在 on_event 中累积到 final_sentences。
            if not final_sentences and last_sentence:
                final_sentences.append(last_sentence[-1])
        def on_close(self): ...

    recognition = Recognition(
        model=model,
        format="wav",
        sample_rate=sample_rate,
        callback=_Callback(),
    )
    try:
        recognition.start()
        # 分片节奏按近实时模拟：3200B=200ms@8kHz/16bit/mono，间隔 60ms。
        # 实测灌大包（9600B/20ms）时服务端会静默丢弃整条流（零事件）。
        with open(wav_path, "rb") as f:
            while True:
                chunk = f.read(3200)
                if not chunk:
                    break
                recognition.send_audio_frame(chunk)
                time.sleep(0.06)
        recognition.stop()
        # on_complete/on_error 在 SDK 接收线程里异步触发，等它收尾
        # （实测 8k 模型 stop 后约 1-2 秒才派发完最后的事件）
        for _ in range(40):
            if completed or error_holder:
                break
            time.sleep(0.1)
    except ASRError:
        raise
    except FileNotFoundError as exc:
        raise ASRError(f"音频文件不存在: {wav_path}", category="video") from exc
    except Exception as exc:  # noqa: BLE001  SDK 异常形态多变，统一分类
        msg = str(exc).lower()
        if "throttl" in msg or "429" in msg:
            raise ASRError("ASR 限流", category="transient", retriable=True) from exc
        raise ASRError(f"ASR 流式异常: {type(exc).__name__}", category="video") from exc

    if error_holder:
        err_text = "; ".join(error_holder)[:200]
        if "throttl" in err_text.lower() or "429" in err_text:
            raise ASRError("ASR 限流", category="transient", retriable=True)
        raise ASRError(f"ASR 流式失败: {err_text}", category="video")

    return [
        {"begin_time": int(s["begin_time"]), "end_time": int(s["end_time"]),
         "text": str(s.get("text", ""))}
        for s in final_sentences
    ]
