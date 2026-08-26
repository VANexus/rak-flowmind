"""Executor 节点：调用 invoke() 执行单步技能。

职责：
- 从 step dict 提取 skill_id + params
- 调 invoke() 执行，返回 SkillResult
- 不处理错误（错误恢复由 Recovery 节点负责）
"""
from __future__ import annotations

from flowmind.skill import invoke


def execute_step(step: dict):
    """执行单个技能步骤。

    Args:
        step: {"skill": str, "input": dict, "reason": str}

    Returns:
        SkillResult — invoke() 的返回值，透传给下游。
    """
    skill_id = step["skill"]
    params = step.get("input", {})
    return invoke(skill_id, params)
