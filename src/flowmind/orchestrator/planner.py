"""Planner 节点：调 LLM 规划技能调用序列。

职责：
- 用 build_planner_prompt() 构建 prompt
- 经 _call_llm() 调 Anthropic-compatible API 拿 JSON 计划
- 截断到 max_steps
- 无 API key 时 raise ValueError（云优先原则，绝不静默降级）

_call_llm() 独立为模块级函数，便于测试 mock（拦截真实 LLM 调用）。
实际实现抽离到 flowmind.orchestrator.llm，这里 re-export 保持名字不变。
"""
from __future__ import annotations

from flowmind.config import get_config
from flowmind.orchestrator.llm import _call_llm, _parse_json  # noqa: F401
from flowmind.orchestrator.prompts import build_planner_prompt
from flowmind.skills._secrets import get_api_key


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
