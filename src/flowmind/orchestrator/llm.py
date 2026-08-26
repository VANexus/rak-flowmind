"""编排器共享 LLM 调用模块。

Planner 和 Summarizer 共享相同的 Anthropic Messages API 调用逻辑。
抽离到这里消除重复，同时保持模块级 _call_llm / _parse_json 名字
供测试 mock（planner._call_llm / summarizer._call_llm）。
"""
from __future__ import annotations

import json

import httpx

from flowmind.config import get_config
from flowmind.skills._secrets import get_api_key


def _parse_json(text: str) -> dict:
    """从 LLM 文本输出中提取 JSON（容忍 markdown 代码围栏）。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # 首行 ``` 或 ```json，末行 ```
        if lines and lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        cleaned = "\n".join(lines).strip()
    return json.loads(cleaned)


def _call_llm(prompt: str, system: str | None = None) -> dict:
    """调用 Anthropic-compatible Messages API，返回解析后的 JSON。

    Args:
        prompt: 完整 user prompt 字符串。
        system: 可选的 system prompt（Summarizer 使用，Planner 不用）。

    Returns:
        LLM 返回的 JSON 对象（dict）。

    Raises:
        ValueError: API 调用失败或响应解析失败。
    """
    orch_cfg = get_config().orchestrator
    api_key = get_api_key(orch_cfg.llm_key_env)

    payload = {
        "model": orch_cfg.llm_model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system

    try:
        r = httpx.post(
            f"{orch_cfg.llm_base_url.rstrip('/')}/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=60.0,
        )
    except httpx.HTTPError as exc:
        raise ValueError(f"LLM 调用失败: {exc}") from exc

    if r.status_code != 200:
        raise ValueError(f"LLM 返回 {r.status_code}: {r.text[:200]}")

    try:
        data = r.json()
        text = data["content"][0]["text"]
        return _parse_json(text)
    except Exception as exc:
        raise ValueError(f"LLM 响应解析失败: {exc}") from exc
