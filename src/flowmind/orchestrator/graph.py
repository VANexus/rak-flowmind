"""编排器：组装 Planner → Executor → Recovery → Summarizer。

基于 LangGraph StateGraph 实现，获得：
- 条件分支（recovery 决策路由：retry / skip / fail）
- 节点级可观测性
- 未来可扩展并行执行 / checkpoint 持久化

每个节点调用独立的 skill 函数（plan_task / execute_step / decide_recovery /
summarizer_results），保持模块级名字便于测试 mock。
"""
from __future__ import annotations

import operator
from typing import Any, Annotated

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from flowmind.config import get_config
from flowmind.orchestrator.executor import execute_step
from flowmind.orchestrator.planner import plan_task
from flowmind.orchestrator.recovery import decide_recovery
from flowmind.orchestrator.summarizer import summarize_results


class _OrchestratorState(TypedDict):
    """编排器内部状态（LangGraph StateGraph 的 TypedDict）。

    step_results /  reasoning 用 operator.add reducer 实现跨节点累加
    （LangGraph 默认行为是替换，累加必须显式声明 reducer）。
    """
    goal: str
    skill_group: str | None
    include_reasoning: bool
    plan: dict | None
    step_results: Annotated[list[dict], operator.add]
    degraded: bool
    reasoning: Annotated[list[str], operator.add]
    error: str | None
    output: Any
    current_step_idx: int
    retries_left: int
    last_result: Any  # SkillResult | None


def _planner_node(state: _OrchestratorState) -> dict:
    """规划节点：调 LLM 生成执行计划。"""
    cfg = get_config().orchestrator
    try:
        plan = plan_task(
            goal=state["goal"],
            skill_group=state["skill_group"],
            max_steps=cfg.max_plan_steps,
        )
    except Exception as exc:
        return {"plan": {"steps": [], "cot": ""}, "error": f"规划失败: {exc}"}

    if not plan["steps"]:
        return {"plan": plan, "error": "无法规划执行步骤"}

    reasoning = []
    if state["include_reasoning"] and plan.get("cot"):
        reasoning.append(f"规划 CoT: {plan['cot']}")

    return {"plan": plan, "reasoning": reasoning}


def _executor_node(state: _OrchestratorState) -> dict:
    """执行节点：执行当前步骤，成功时推进索引。"""
    step = state["plan"]["steps"][state["current_step_idx"]]
    result = execute_step(step)

    if result.ok:
        step_results = [{
            "skill": step["skill"],
            "ok": result.ok,
            "data": result.data,
        }]
        reasoning = []
        if state["include_reasoning"] and result.reasoning:
            reasoning.extend([r.causal_analysis for r in result.reasoning])
        # 成功：推进到下一步，重置重试计数
        return {
            "last_result": result,
            "step_results": step_results,
            "reasoning": reasoning,
            "current_step_idx": state["current_step_idx"] + 1,
            "retries_left": get_config().orchestrator.max_retries_per_step,
        }

    # 失败：保留 last_result 供 recovery 决策，不推进
    return {"last_result": result}


def _recovery_node(state: _OrchestratorState) -> dict:
    """恢复节点：根据失败结果决定 retry / skip / fail。"""
    result = state["last_result"]
    recovery = decide_recovery(result, retries_left=state["retries_left"])
    action = recovery["action"]

    if action == "retry" and state["retries_left"] > 0:
        # 重试：减重试次数，保持当前步骤索引
        return {"retries_left": state["retries_left"] - 1}

    if action == "skip":
        # 跳过：记录跳过的步骤，推进索引，标记 degraded
        step = state["plan"]["steps"][state["current_step_idx"]]
        reasoning = []
        if state["include_reasoning"]:
            reasoning.append(f"跳过 {step['skill']}: {recovery['reason']}")
        return {
            "step_results": [{"skill": step["skill"], "ok": False, "skipped": True}],
            "reasoning": reasoning,
            "degraded": True,
            "current_step_idx": state["current_step_idx"] + 1,
            "retries_left": get_config().orchestrator.max_retries_per_step,
        }

    # fail 或重试次数耗尽
    return {"degraded": True, "error": recovery["reason"]}


def _summarizer_node(state: _OrchestratorState) -> dict:
    """汇总节点：调 LLM 汇总步骤结果为最终输出。"""
    try:
        summary = summarize_results(state["step_results"])
    except Exception as exc:
        return {"error": f"汇总失败: {exc}"}

    reasoning = []
    if state["include_reasoning"] and summary.get("cot"):
        reasoning.append(f"汇总 CoT: {summary.get('cot', '')}")

    return {"output": summary.get("output"), "reasoning": reasoning}


def _route_after_planner(state: _OrchestratorState) -> str:
    """规划后路由：无步骤 → 结束；有步骤 → 执行。"""
    if state.get("error") or not state["plan"]["steps"]:
        return END
    return "executor"


def _route_after_executor(state: _OrchestratorState) -> str:
    """执行后路由：成功 → 下一步或汇总；失败 → 恢复。"""
    if not state["last_result"].ok:
        return "recovery"
    # 成功：还有步骤 → 继续执行；否则 → 汇总
    if state["current_step_idx"] < len(state["plan"]["steps"]):
        return "executor"
    return "summarizer"


def _route_after_recovery(state: _OrchestratorState) -> str:
    """恢复后路由：失败 → 结束；否则 → 下一步或汇总。"""
    if state.get("error"):
        return END
    # retry 或 skip 后：还有步骤 → 继续执行；否则 → 汇总
    if state["current_step_idx"] < len(state["plan"]["steps"]):
        return "executor"
    return "summarizer"


def _build_graph() -> Any:
    """构建 LangGraph StateGraph。"""
    builder = StateGraph(_OrchestratorState)
    builder.add_node("planner", _planner_node)
    builder.add_node("executor", _executor_node)
    builder.add_node("recovery", _recovery_node)
    builder.add_node("summarizer", _summarizer_node)

    builder.add_edge(START, "planner")
    # 路由函数返回目标节点名字符串；dict 形式显式映射（含 END）
    builder.add_conditional_edges("planner", _route_after_planner, {
        "executor": "executor",
        END: END,
    })
    builder.add_conditional_edges("executor", _route_after_executor, {
        "executor": "executor",
        "summarizer": "summarizer",
        "recovery": "recovery",
    })
    builder.add_conditional_edges("recovery", _route_after_recovery, {
        "executor": "executor",
        "summarizer": "summarizer",
        END: END,
    })
    builder.add_edge("summarizer", END)

    return builder.compile()


# 编译后的图（懒加载单例）
_graph = None


def _get_graph() -> Any:
    """获取编译后的图（懒加载单例）。"""
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


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
    initial_state: _OrchestratorState = {
        "goal": goal,
        "skill_group": skill_group,
        "include_reasoning": include_reasoning,
        "plan": None,
        "step_results": [],
        "degraded": False,
        "reasoning": [],
        "error": None,
        "output": None,
        "current_step_idx": 0,
        "retries_left": cfg.max_retries_per_step,
        "last_result": None,
    }

    graph = _get_graph()
    result = graph.invoke(initial_state)

    return {
        "output": result.get("output"),
        "reasoning": result.get("reasoning", []),
        "degraded": result.get("degraded", False),
        "error": result.get("error"),
    }
