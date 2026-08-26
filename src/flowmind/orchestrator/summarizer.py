"""Summarizer 节点：调 LLM 汇总各步骤结果为最终输出。

职责：
- 用 build_summarizer_prompt() 构建汇总 prompt
- 经 _call_llm() 调 Anthropic-compatible API 拿结构化汇总
- 返回 {"output", "summary", "cot"}

与 planner.py 共享相同的 _call_llm / _parse_json 模式（Anthropic Messages API）。
实际实现抽离到 flowmind.orchestrator.llm，这里 re-export 保持名字不变。
"""
from __future__ import annotations

from flowmind.config import get_config
from flowmind.orchestrator.llm import _call_llm, _parse_json  # noqa: F401
from flowmind.orchestrator.prompts import build_summarizer_prompt
from flowmind.skills._secrets import get_api_key

# Summarizer 专用 system prompt（传递给共享 _call_llm）
_SUMMARIZER_SYSTEM = "你是一个结果汇总器，根据执行步骤生成最终输出。只输出 JSON。"


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
    return _call_llm(prompt, system=_SUMMARIZER_SYSTEM)
