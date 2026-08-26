"""Summarizer 节点：调 LLM 汇总各步骤结果为最终输出。

职责：
- 用 build_summarizer_prompt() 构建汇总 prompt
- 经 _call_llm() 调 Anthropic-compatible API 拿结构化汇总
- 返回 {"output", "summary", "cot"}

与 planner.py 共享相同的 _call_llm / _parse_json 模式（Anthropic Messages API）。
"""
from __future__ import annotations

import json

import httpx

from flowmind.config import get_config
from flowmind.orchestrator.prompts import build_summarizer_prompt
from flowmind.skills._secrets import get_api_key


def _parse_json(text: str) -> dict:
    """从 LLM 文本输出中提取 JSON（容忍 markdown 代码围栏）。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        cleaned = "\n".join(lines).strip()
    return json.loads(cleaned)


def _call_llm(prompt: str) -> dict:
    """调用 Anthropic-compatible Messages API，返回解析后的 JSON。

    Args:
        prompt: 完整 user prompt 字符串。

    Returns:
        LLM 返回的 JSON 对象（dict）。

    Raises:
        ValueError: API 调用失败或响应解析失败。
    """
    orch_cfg = get_config().orchestrator
    api_key = get_api_key(orch_cfg.llm_key_env)

    try:
        r = httpx.post(
            f"{orch_cfg.llm_base_url.rstrip('/')}/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": orch_cfg.llm_model,
                "max_tokens": 4096,
                "system": "你是一个结果汇总器，根据执行步骤生成最终输出。只输出 JSON。",
                "messages": [{"role": "user", "content": prompt}],
            },
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


def summarize_results(step_results: list[dict]) -> dict:
    """汇总所有步骤结果为最终输出。

    Args:
        step_results: 各步骤执行结果列表。

    Returns:
        {"output": str, "summary": str, "cot": str}

    Raises:
        ValueError: 无 API key 或 LLM 调用/解析失败。
    """
    orch_cfg = get_config().orchestrator
    api_key = get_api_key(orch_cfg.llm_key_env)
    if not api_key:
        raise ValueError(
            f"未配置 LLM API key（环境变量 {orch_cfg.llm_key_env} 为空）——"
            "请设置环境变量或 .env 后重试"
        )

    prompt = build_summarizer_prompt(step_results)
    return _call_llm(prompt)
