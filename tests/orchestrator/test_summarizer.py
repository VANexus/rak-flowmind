"""Summarizer 节点测试：验证 LLM 汇总调用与输出结构。

测试通过 mock `_call_llm` 拦截真实 LLM 调用，不需要 API key。
"""
from __future__ import annotations

from unittest import mock


def test_summarize_returns_output():
    """summarize_results 调 LLM 返回 output/summary/cot 结构。"""
    from flowmind.orchestrator.summarizer import summarize_results

    mock_llm = {
        "output": "完成了文案创作",
        "summary": "基于选题写了文案",
        "cot": "汇总思路",
    }
    with (
        mock.patch("flowmind.orchestrator.summarizer._call_llm", return_value=mock_llm),
        mock.patch("flowmind.orchestrator.summarizer.get_api_key", return_value="fake-key"),
    ):
        result = summarize_results([{"skill": "x", "ok": True}])

    assert result["output"] == "完成了文案创作"
    assert result["summary"] == "基于选题写了文案"
    assert result["cot"] == "汇总思路"


def test_summarize_handles_all_failures():
    """全部失败时，summarizer 仍返回 LLM 的汇总输出。"""
    from flowmind.orchestrator.summarizer import summarize_results

    mock_llm = {
        "output": "所有步骤都失败了",
        "summary": "无法完成",
        "cot": "没有成功步骤",
    }
    with (
        mock.patch("flowmind.orchestrator.summarizer._call_llm", return_value=mock_llm),
        mock.patch("flowmind.orchestrator.summarizer.get_api_key", return_value="fake-key"),
    ):
        result = summarize_results([{"skill": "x", "ok": False}])

    assert "失败" in result["output"] or "无法" in result["output"]
