"""通用 LLM 客户端：Anthropic 兼容 /v1/messages 协议（httpx 直调，返回 JSON dict）。

云优先原则：LLM 必须走云端 API；无 key 显式报错，不静默降级。
供 content_* 系列技能（文案生成 / 思路设计 / 审计复核）复用；视频本地化的
翻译走 `_llm_translate.translate_segments`（保留原实现，不迁到本模块）。

key 由调用方从环境变量读出后传入，本模块不直接读 env、不进 config 文件。
"""
from __future__ import annotations

import json

import httpx
import requests  # noqa: F401  保留模块级引用：测试 fixture 经 <mod>.requests 打桩拦截

DEFAULT_BASE = "https://api.longcat.chat/anthropic"
DEFAULT_MODEL = "LongCat-2.0"


class LLMClientError(Exception):
    """LLM 调用失败。category/retriable 语义与 errors.py 一致（供错误分类与重试决策）。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def llm_json(
    *,
    prompt: str,
    system: str,
    api_key: str,
    api_base: str = DEFAULT_BASE,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    timeout_s: float = 60.0,
    client: httpx.Client | None = None,
) -> dict:
    """调 Anthropic 兼容 /v1/messages，把首个 type=text 块解析为 JSON dict。

    - 输入 prompt 视为不透明内容，由调用方负责放进合适结构的 user 消息。
    - 只接受 JSON 对象（dict）回复；数组/非 JSON 抛 LLMClientError。
    - 错误分类：超时=environment、5xx=transient(可重试)、4xx=video、结构异常=unknown。
    """
    if not api_key:
        raise ValueError(
            "收到空 API key。请检查环境变量 AI_LLM_API_KEY 是否设置。"
            "云优先原则：LLM 必须走云 API，不做本地降级。"
        )

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
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
        raise LLMClientError("LLM 超时", category="environment", retriable=False) from exc
    except httpx.TimeoutException as exc:
        raise LLMClientError("LLM 超时", category="environment", retriable=False) from exc
    except httpx.HTTPError as exc:
        raise LLMClientError(f"LLM 连接失败：{type(exc).__name__}", category="environment", retriable=False) from exc

    if resp.status_code >= 500:
        raise LLMClientError(f"LLM HTTP {resp.status_code}", category="transient", retriable=True)
    if resp.status_code >= 400:
        raise LLMClientError(f"LLM HTTP {resp.status_code}", category="video", retriable=False)

    data = resp.json()
    # Anthropic 兼容协议：content 是块数组；推理模型的 thinking 块在前，
    # 取第一个 type=text 的块（无则结构异常）
    text = next(
        (blk.get("text") for blk in data.get("content", [])
         if isinstance(blk, dict) and blk.get("type") == "text"),
        None,
    )
    if not text:
        raise LLMClientError("LLM 返回结构异常（无 type=text 块）", category="unknown", retriable=False)

    return _parse_json_object(text)


def _parse_json_object(content: str) -> dict:
    """解析 LLM 回复为 JSON 对象；容忍代码围栏与首尾空白。"""
    text = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    if text.startswith("```"):
        # 只处理整体围栏；正文内的代码块不在此层处理
        text = text[3:].removesuffix("```")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMClientError("LLM 回复不是合法 JSON", category="unknown", retriable=False) from exc
    if not isinstance(parsed, dict):
        raise LLMClientError("LLM 回复不是 JSON 对象", category="unknown", retriable=False)
    return parsed
