"""Planner 节点测试：验证 LLM 规划调用、截断、无 key 报错。

测试通过 mock `_call_llm` 拦截真实 LLM 调用，不需要 API key。
"""
from __future__ import annotations

from unittest import mock

import pytest


def test_plan_task_returns_steps_and_cot():
    """plan_task 调 LLM 解析 JSON，返回 steps + cot 结构。"""
    from flowmind.orchestrator import planner

    fake_llm = {
        "steps": [
            {"skill": "inventory_risk", "input": {"sku": "A1"}, "reason": "计算库存风险"},
        ],
        "cot": "先算库存风险再输出",
    }
    with (
        mock.patch.object(planner, "_call_llm", return_value=fake_llm) as m,
        mock.patch("flowmind.orchestrator.planner.get_api_key", return_value="fake-key"),
    ):
        result = planner.plan_task(goal="分析库存", skill_group=None, max_steps=5)

    assert result["cot"] == "先算库存风险再输出"
    assert len(result["steps"]) == 1
    assert result["steps"][0]["skill"] == "inventory_risk"
    # 验证 prompt 被透传
    m.assert_called_once()
    assert "分析库存" in m.call_args.args[0]


def test_plan_task_truncates_to_max_steps():
    """LLM 返回步数超过 max_steps 时，截断到 max_steps。"""
    from flowmind.orchestrator import planner

    fake_llm = {
        "steps": [{"skill": f"s{i}", "input": {}, "reason": f"r{i}"} for i in range(10)],
        "cot": "many steps",
    }
    with (
        mock.patch.object(planner, "_call_llm", return_value=fake_llm),
        mock.patch("flowmind.orchestrator.planner.get_api_key", return_value="fake-key"),
    ):
        result = planner.plan_task(goal="test", skill_group=None, max_steps=3)

    assert len(result["steps"]) == 3


def test_plan_task_raises_without_api_key():
    """无 API key 时 raise ValueError（云优先原则，绝不静默降级）。"""
    from flowmind.orchestrator import planner

    with mock.patch("flowmind.orchestrator.planner.get_api_key", return_value=None):
        with pytest.raises(ValueError, match="API key"):
            planner.plan_task(goal="test", skill_group=None, max_steps=5)
