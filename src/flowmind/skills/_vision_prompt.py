"""视觉提示词反推客户端：Anthropic 兼容 /v1/messages 携带图片块，反推生成式提示词。

供 ``image_prompt_reverse`` 技能复用。key 由调用方从环境变量读出后传入，
本模块不直接读 env、不进 config 文件（与 _llm_client 保持一致）。
"""
from __future__ import annotations

import json

import httpx
import requests  # noqa: F401  保留模块级引用：测试 fixture 经 <mod>.requests 打桩拦截


class VisionError(Exception):
    """视觉反推失败。category/retriable 语义与 errors.py 一致。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


_SYSTEM_PROMPT = (
    "你是专业的 AI 生图提示词工程师。观察用户提供的参考图片，反推出可用于"
    "图像生成模型的英文提示词（prompt）、风格标签与默认负面词。只输出 JSON 对象："
    '{"prompt": "...", "style_tags": ["..."], "negative_prompt": "..."}。'
    "prompt 需包含主体、构图、光线、风格、材质、色彩；"
    "style_tags 用 2-5 个英文风格关键词；"
    "negative_prompt 给出通用负面词（模糊/畸形/文字/水印等）。"
)


def reverse_prompt(
    *,
    image_url: str,
    hint: str | None,
    api_key: str,
    api_base: str,
    model: str,
    timeout_s: float = 45.0,
    client: httpx.Client | None = None,
) -> dict:
    """POST /v1/messages，携带 image 块，反推提示词为 JSON dict。

    返回 ``{prompt, style_tags, negative_prompt}``。失败抛 VisionError。
    """
    if not api_key:
        raise ValueError(
            "收到空 API key。请检查环境变量 LONGCAT_API_KEY 是否设置。"
            "云优先原则：提示词反推必须走视觉云 LLM，不做本地降级。"
        )

    text = "请反推这张图片的生成式提示词。"
    if hint and hint.strip():
        text += f" 补充说明：{hint.strip()}"

    body = {
        "model": model,
        "max_tokens": 1024,
        "system": _SYSTEM_PROMPT,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "url", "url": image_url}},
                {"type": "text", "text": text},
            ],
        }],
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
        raise VisionError("提示词反推超时", category="environment", retriable=False) from exc
    except httpx.TimeoutException as exc:
        raise VisionError("提示词反推超时", category="environment", retriable=False) from exc
    except httpx.HTTPError as exc:
        raise VisionError(f"提示词反推连接失败：{type(exc).__name__}", category="environment", retriable=False) from exc

    if resp.status_code >= 500:
        raise VisionError(f"提示词反推 HTTP {resp.status_code}", category="transient", retriable=True)
    if resp.status_code >= 400:
        raise VisionError(f"提示词反推 HTTP {resp.status_code}", category="video", retriable=False)

    data = resp.json()
    text = next(
        (blk.get("text") for blk in data.get("content", [])
         if isinstance(blk, dict) and blk.get("type") == "text"),
        None,
    )
    if not text:
        raise VisionError("提示词反推返回结构异常（无 type=text 块）", category="unknown", retriable=False)

    parsed = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        obj = json.loads(parsed)
    except json.JSONDecodeError:
        # LLM 常在 JSON 前后夹说明文字：截取首个 { 到最后一个 } 再试一次
        lo, hi = parsed.find("{"), parsed.rfind("}")
        if lo == -1 or hi <= lo:
            raise VisionError("提示词反推回复不是合法 JSON", category="unknown", retriable=False) from None
        try:
            obj = json.loads(parsed[lo:hi + 1])
        except json.JSONDecodeError as exc:
            raise VisionError("提示词反推回复不是合法 JSON", category="unknown", retriable=False) from exc
    if not isinstance(obj, dict):
        raise VisionError("提示词反推回复不是 JSON 对象", category="unknown", retriable=False)
    return obj