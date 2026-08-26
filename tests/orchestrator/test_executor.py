"""Executor 节点测试：验证 invoke() 调用与参数透传。

测试通过 mock `invoke` 拦截真实技能执行，不需要真实后端。
"""
from __future__ import annotations

from unittest import mock

from flowmind.contracts import ReliabilityMetrics, SkillResult, new_trace


def test_execute_step_returns_skill_result():
    """execute_step 调 invoke() 并返回其 SkillResult。"""
    from flowmind.orchestrator.executor import execute_step

    mock_result = SkillResult(
        ok=True, skill="inventory_risk", version="1.0",
        trace=new_trace(), data={"result": "ok"},
        metrics=ReliabilityMetrics(latency_ms=0, confidence=1.0, sample_size=1),
    )
    with mock.patch("flowmind.orchestrator.executor.invoke", return_value=mock_result):
        step = {"skill": "inventory_risk", "input": {"sku": "A001"}, "reason": "查库存"}
        result = execute_step(step)

    assert result.ok is True


def test_execute_step_passes_correct_args():
    """execute_step 把 step["skill"] / step["input"] 透传给 invoke()。"""
    from flowmind.orchestrator.executor import execute_step

    with mock.patch("flowmind.orchestrator.executor.invoke") as mock_invoke:
        step = {"skill": "inventory_risk", "input": {"sku": "A001"}, "reason": "查库存"}
        execute_step(step)

    mock_invoke.assert_called_once_with("inventory_risk", {"sku": "A001"})
