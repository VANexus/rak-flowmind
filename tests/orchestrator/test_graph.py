"""LangGraph 编排器测试：验证 Planner → Executor → Recovery → Summarizer 组装。

测试通过 mock 各节点拦截真实执行，聚焦图组装逻辑。
"""
from __future__ import annotations

from unittest import mock

from flowmind.contracts import (
    ReliabilityMetrics,
    SkillError,
    SkillResult,
    new_trace,
)


def test_run_orchestrator_returns_output():
    """正常流程：规划 → 执行成功 → 汇总，返回 output + degraded=False。"""
    from flowmind.orchestrator.graph import run_orchestrator

    with (
        mock.patch("flowmind.orchestrator.graph.plan_task") as mock_plan,
        mock.patch("flowmind.orchestrator.graph.execute_step") as mock_exec,
        mock.patch("flowmind.orchestrator.graph.summarize_results") as mock_sum,
    ):
        mock_plan.return_value = {
            "steps": [{"skill": "inventory_risk", "input": {"sku": "A"}, "reason": "查"}],
            "cot": "规划",
        }
        mock_exec.return_value = SkillResult(
            ok=True, skill="inventory_risk", version="1.0", trace=new_trace(),
            data={"result": "ok"},
            metrics=ReliabilityMetrics(latency_ms=10, confidence=0.9, sample_size=1),
        )
        mock_sum.return_value = {"output": "完成了", "summary": "ok", "cot": "汇总"}

        result = run_orchestrator(goal="查库存", skill_group="data", include_reasoning=False)

    assert result["output"] == "完成了"
    assert result["degraded"] is False
    assert result["error"] is None


def test_run_orchestrator_marks_degraded_on_skip():
    """步骤失败 + 不可重试 → skip → degraded=True。"""
    from flowmind.orchestrator.graph import run_orchestrator

    with (
        mock.patch("flowmind.orchestrator.graph.plan_task") as mock_plan,
        mock.patch("flowmind.orchestrator.graph.execute_step") as mock_exec,
        mock.patch("flowmind.orchestrator.graph.summarize_results") as mock_sum,
        mock.patch("flowmind.orchestrator.graph.decide_recovery") as mock_recovery,
    ):
        mock_plan.return_value = {
            "steps": [{"skill": "x", "input": {}, "reason": "t"}],
            "cot": "",
        }
        mock_exec.return_value = SkillResult(
            ok=False, skill="x", version="1.0", trace=new_trace(),
            error=SkillError(code="INTERNAL", message="fail", retriable=False),
            metrics=ReliabilityMetrics(latency_ms=0, confidence=0, sample_size=0),
        )
        mock_recovery.return_value = {"action": "skip", "reason": "不可重试"}
        mock_sum.return_value = {"output": "部分完成", "summary": "", "cot": ""}

        result = run_orchestrator(goal="test", skill_group=None, include_reasoning=False)

    assert result["degraded"] is True


def test_run_orchestrator_empty_steps_returns_error():
    """规划返回空步骤 → 返回 error，不执行。"""
    from flowmind.orchestrator.graph import run_orchestrator

    with mock.patch("flowmind.orchestrator.graph.plan_task") as mock_plan:
        mock_plan.return_value = {"steps": [], "cot": ""}
        result = run_orchestrator(goal="空", skill_group=None, include_reasoning=False)

    assert result["error"] is not None
    assert "无法规划" in result["error"]
