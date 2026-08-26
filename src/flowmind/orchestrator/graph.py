"""编排器：组装 Planner → Executor → Recovery → Summarizer。

当前用顺序执行循环实现核心编排逻辑。未来可迁移为 LangGraph StateGraph
以获得条件分支 / 并行 / 持久化 checkpoint 能力。
"""
from __future__ import annotations

from flowmind.config import get_config
from flowmind.orchestrator.executor import execute_step
from flowmind.orchestrator.planner import plan_task
from flowmind.orchestrator.recovery import decide_recovery
from flowmind.orchestrator.summarizer import summarize_results


def run_orchestrator(*, goal: str, skill_group: str | None, include_reasoning: bool) -> dict:
    """运行编排器：规划 → 执行 → 汇总。

    Args:
        goal: 用户自然语言目标。
        skill_group: 可选技能分组过滤。
        include_reasoning: 是否附带完整推理链。

    Returns:
        {"output": Any, "reasoning": list, "degraded": bool, "error": str|None}
    """
    cfg = get_config().orchestrator

    # 1. 规划
    try:
        plan = plan_task(goal=goal, skill_group=skill_group, max_steps=cfg.max_plan_steps)
    except Exception as exc:
        return {
            "output": None,
            "reasoning": [],
            "degraded": False,
            "error": f"规划失败: {exc}",
        }

    if not plan["steps"]:
        return {
            "output": None,
            "reasoning": [],
            "degraded": False,
            "error": "无法规划执行步骤",
        }

    # 2. 逐步执行
    step_results: list[dict] = []
    degraded = False
    reasoning: list[str] = [f"规划 CoT: {plan['cot']}"] if include_reasoning else []

    for step in plan["steps"]:
        retries_left = cfg.max_retries_per_step
        result = execute_step(step)

        # 失败时进入恢复循环
        while not result.ok:
            recovery = decide_recovery(result, retries_left=retries_left)

            if recovery["action"] == "retry" and retries_left > 0:
                retries_left -= 1
                result = execute_step(step)
                continue
            elif recovery["action"] == "skip":
                degraded = True
                step_results.append({"skill": step["skill"], "ok": False, "skipped": True})
                if include_reasoning:
                    reasoning.append(f"跳过 {step['skill']}: {recovery['reason']}")
                break
            else:
                # fail 或重试次数耗尽
                return {
                    "output": None,
                    "reasoning": reasoning,
                    "degraded": True,
                    "error": recovery["reason"],
                }

        else:
            # 步骤成功（while 正常结束，未 break）
            step_results.append({
                "skill": step["skill"],
                "ok": result.ok,
                "data": result.data,
            })
            if include_reasoning and result.reasoning:
                reasoning.extend([r.causal_analysis for r in result.reasoning])

    # 3. 汇总
    try:
        summary = summarize_results(step_results)
    except Exception as exc:
        return {
            "output": None,
            "reasoning": reasoning,
            "degraded": True,
            "error": f"汇总失败: {exc}",
        }

    if include_reasoning:
        reasoning.append(f"汇总 CoT: {summary.get('cot', '')}")

    return {
        "output": summary.get("output"),
        "reasoning": reasoning,
        "degraded": degraded,
        "error": None,
    }
