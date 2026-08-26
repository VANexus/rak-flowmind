"""Planner 节点：调 LLM 规划技能调用序列。

职责：
- 用 build_planner_prompt() 构建 prompt
- 经 _call_llm() 调 Anthropic-compatible API 拿 JSON 计划
- 截断到 max_steps
- 无 API key 时 raise ValueError（云优先原则，绝不静默降级）

_call_llm() 独立为模块级函数，便于测试 mock（拦截真实 LLM 调用）。
"""
from __future__ import annotations

import json

import httpx

from flowmind.config import get_config
from flowmind.orchestrator.prompts import build_planner_prompt
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


def plan_task(*, goal: str, skill_group: str | None, max_steps: int) -> dict:
    """规划技能调用序列。

    Args:
        goal: 用户自然语言目标。
        skill_group: 可选的技能分组过滤。
        max_steps: 最大步骤数。

    Returns:
        {"steps": [...], "cot": "..."} 计划字典。

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

    prompt = build_planner_prompt(goal=goal, skill_group=skill_group, max_steps=max_steps)
    plan = _call_llm(prompt)

    # 截断到 max_steps
    steps = plan.get("steps", [])
    if len(steps) > max_steps:
        steps = steps[:max_steps]

    return {
        "steps": steps,
        "cot": plan.get("cot", ""),
    }
