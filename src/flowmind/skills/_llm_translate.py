"""LongCat 翻译客户端：Anthropic 兼容 /v1/messages 协议（httpx 直调）。

云优先原则：翻译必须走云端 LLM；无 key 显式报错，不静默降级。
输入是 ASR 句段（index/begin/end/text），输出同结构译文——时间轴字段原样保留，
供后续字幕对齐与 TTS 逐句合成使用。
key 由调用方从环境变量读出后传入，本模块不直接读 env、不进 config 文件。
"""
from __future__ import annotations

import json

import httpx
import requests  # noqa: F401  保留模块级引用：测试 fixture 经 <mod>.requests 打桩拦截

DEFAULT_BASE = "https://api.longcat.chat/anthropic"
DEFAULT_MODEL = "LongCat-2.0"

_SYSTEM_PROMPT = (
    "你是专业视频字幕本地化译者。把用户给出的字幕句段从源语言翻译成目标语言。"
    "严格要求：1) 只输出 JSON 数组，不要解释/前缀/代码块；"
    "2) 每个元素保留 index 字段不变，text 换成译文；"
    "3) 口语化、简洁、适合字幕阅读（每句不超过原文长度的 1.5 倍）；"
    "4) 不软化、不替换、不 euphemize 技术术语与品牌名，必要时附原文括注。"
)


class LLMTtranslateError(Exception):
    """翻译调用失败。category/retriable 语义与 errors.py 一致。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def translate_segments(
    segments: list[dict],
    *,
    target_lang: str,
    source_lang: str = "zh",
    api_key: str,
    api_base: str = DEFAULT_BASE,
    model: str = DEFAULT_MODEL,
    batch_size: int = 3,
    timeout_s: float = 180.0,
    client: httpx.Client | None = None,
) -> list[dict]:
    """分批翻译句段，返回同结构列表（text 为译文，begin/end/index 原样）。"""
    if not api_key:
        raise ValueError(
            "收到空 API key。请检查环境变量 AI_LLM_API_KEY 是否设置。"
            "云优先原则：翻译必须走云 LLM，不做本地降级。"
        )
    out: list[dict] = []
    for start in range(0, len(segments), max(1, batch_size)):
        batch = segments[start:start + max(1, batch_size)]
        out.extend(_translate_batch(
            batch, target_lang=target_lang, source_lang=source_lang,
            api_key=api_key, api_base=api_base, model=model,
            timeout_s=timeout_s, client=client,
        ))
    return out


def _translate_batch(
    batch: list[dict], *, target_lang: str, source_lang: str,
    api_key: str, api_base: str, model: str, timeout_s: float,
    client: httpx.Client | None,
) -> list[dict]:
    # LLM 只见批内局部序号（0-based），返回后映射回全局 index
    local = [{"index": i, "text": s["text"]} for i, s in enumerate(batch)]
    user_payload = {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "segments": local,
    }
    body = {
        "model": model,
        "max_tokens": 8192,
        "system": _SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }
    url = f"{api_base.rstrip('/')}/v1/messages"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        if client is not None:
            resp = client.post(url, headers=headers, json=body)
        else:
            with httpx.Client(timeout=timeout_s) as c:
                resp = c.post(url, headers=headers, json=body)
    except requests.exceptions.Timeout as exc:
        raise LLMTtranslateError("LLM 超时", category="environment", retriable=False) from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        cat = "transient" if status >= 500 else "video"
        raise LLMTtranslateError(f"LLM HTTP {status}", category=cat,
                                 retriable=status >= 500) from exc

    if resp.status_code >= 500:
        raise LLMTtranslateError(f"LLM HTTP {resp.status_code}",
                                 category="transient", retriable=True)
    if resp.status_code >= 400:
        raise LLMTtranslateError(f"LLM HTTP {resp.status_code}", category="video")

    data = resp.json()
    # Anthropic 兼容协议：content 是块数组；推理模型的 thinking 块在前，
    # 取第一个 type=text 的块（无则结构异常）
    text = next(
        (blk.get("text") for blk in data.get("content", [])
         if isinstance(blk, dict) and blk.get("type") == "text"),
        None,
    )
    if not text:
        raise LLMTtranslateError("LLM 返回结构异常")

    return _parse_reply(text, batch)


def _parse_reply(content: str, batch: list[dict]) -> list[dict]:
    """解析 LLM 的 JSON 数组回复（批内局部 index），映射回原句段结构。"""
    text = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        items = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMTtranslateError("LLM 译文不是合法 JSON") from exc
    if not isinstance(items, list) or len(items) != len(batch):
        raise LLMTtranslateError("LLM 译文章数与原文不一致")
    by_local_index = {it.get("index"): it.get("text", "") for it in items}
    out: list[dict] = []
    for local_i, seg in enumerate(batch):
        if local_i not in by_local_index or not str(by_local_index[local_i]).strip():
            raise LLMTtranslateError(f"LLM 缺少第 {local_i} 句的译文")
        out.append({**seg, "text": str(by_local_index[local_i]).strip()})
    return out
