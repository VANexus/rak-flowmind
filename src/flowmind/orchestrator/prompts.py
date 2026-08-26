"""LLM prompt 模板：Planner 与 Summarizer 的 system/user prompt 构建。"""
from __future__ import annotations

import json

from flowmind.skill import registry


def _skill_summaries() -> str:
    """生成所有技能的摘要列表（供 Planner 参考）。"""
    reg = registry()
    lines = []
    for spec in reg.values():
        lines.append(f"- {spec.id}: {spec.name} — {spec.description}")
    return "\n".join(lines)


def build_planner_prompt(*, goal: str, skill_group: str | None, max_steps: int) -> str:
    """构建 Planner 的 user prompt。

    Args:
        goal: 用户自然语言目标。
        skill_group: 可选的技能分组过滤。
        max_steps: 最大步骤数。

    Returns:
        完整 prompt 字符串。
    """
    skills = _skill_summaries()
    group_hint = f"\n用户选择了能力分组: {skill_group}，请优先使用该分组下的技能。" if skill_group else ""
    return f"""你是一个任务编排器。根据用户目标，规划一系列技能调用的执行计划。

可用技能：
{skills}{group_hint}

用户目标：{goal}

请输出 JSON 格式的执行计划（不要包含 markdown 代码围栏）：
{{
  "steps": [
    {{"skill": "<skill_id>", "input": {{<参数对象>}}, "reason": "<为什么需要这步>"}}
  ],
  "cot": "<你的整体规划思路>"
}}

约束：
- 最多 {max_steps} 步
- 每步的 skill 必须是上方列表中的有效 skill_id
- 步骤顺序要合理（前置步骤先执行）
- 如果目标无法完成，返回空 steps 数组并说明原因
"""


def build_summarizer_prompt(step_results: list[dict]) -> str:
    """构建 Summarizer 的 user prompt。

    Args:
        step_results: 各步骤执行结果列表。

    Returns:
        完整 prompt 字符串。
    """
    results_json = json.dumps(step_results, ensure_ascii=False, default=str, indent=2)
    return f"""你是一个结果汇总器。根据以下步骤的执行结果，生成最终输出。

执行结果：
{results_json}

请输出 JSON 格式（不要包含 markdown 代码围栏）：
{{
  "output": "<最终输出给用户的内容>",
  "summary": "<简要说明完成了什么>",
  "cot": "<你的汇总思路>"
}}

如果部分步骤失败，在 output 中说明已完成的部分和未完成的部分。
"""
