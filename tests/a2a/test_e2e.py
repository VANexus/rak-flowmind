"""A2A 端到端测试：验证 Agent Card 发现 → Task 提交 → 编排 → 结果完整流。

测试通过 mock `run_orchestrator` 拦截真实编排，聚焦 HTTP 层 + 类型映射。
"""
from __future__ import annotations

from unittest import mock

from starlette.testclient import TestClient

from flowmind.a2a.server import FlowMindA2AServer


def _make_client() -> TestClient:
    server = FlowMindA2AServer(base_url="http://localhost:8001")
    return TestClient(server.get_app())


def test_full_a2a_flow_happy_path():
    """完整 A2A 流：发现 Agent Card → 提交任务 → 编排 → 返回 completed。"""
    client = _make_client()

    # 1. 发现 Agent Card
    resp = client.get("/.well-known/agent.json")
    assert resp.status_code == 200
    card = resp.json()
    # TestClient 用 testserver 作 base_url，card url 指向 /a2a 端点
    assert card["url"].endswith("/a2a")

    # 2. 提交任务（mock 编排器）
    with mock.patch("flowmind.a2a.server.run_orchestrator") as mock_orch:
        mock_orch.return_value = {
            "output": {"text": "完成了"},
            "reasoning": [],
            "degraded": False,
            "error": None,
        }
        resp = client.post("/a2a", json={
            "jsonrpc": "2.0",
            "id": "req-1",
            "method": "tasks/send",
            "params": {
                "id": "task-1",
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "帮我写篇文案"}],
                },
                "metadata": {},
            },
        })

    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["id"] == "task-1"
    assert result["status"]["state"] == "completed"


def test_a2a_flow_with_reasoning():
    """A2A 请求附带 include_reasoning=true → 结果含 history。"""
    client = _make_client()

    with mock.patch("flowmind.a2a.server.run_orchestrator") as mock_orch:
        mock_orch.return_value = {
            "output": {"text": "ok"},
            "reasoning": ["规划: 先写文案", "汇总: 完成"],
            "degraded": False,
            "error": None,
        }
        resp = client.post("/a2a", json={
            "jsonrpc": "2.0",
            "id": "req-2",
            "method": "tasks/send",
            "params": {
                "id": "task-2",
                "message": {"role": "user", "parts": [{"type": "text", "text": "test"}]},
                "metadata": {"include_reasoning": True},
            },
        })

    result = resp.json()["result"]
    assert "history" in result  # reasoning 附带在 history 中
    assert len(result["history"]) == 2


def test_a2a_flow_degraded():
    """A2A 任务部分完成 → status.degraded=True。"""
    client = _make_client()

    with mock.patch("flowmind.a2a.server.run_orchestrator") as mock_orch:
        mock_orch.return_value = {
            "output": {"text": "部分完成"},
            "reasoning": [],
            "degraded": True,
            "error": None,
        }
        resp = client.post("/a2a", json={
            "jsonrpc": "2.0",
            "id": "req-3",
            "method": "tasks/send",
            "params": {
                "id": "task-3",
                "message": {"role": "user", "parts": [{"type": "text", "text": "test"}]},
                "metadata": {},
            },
        })

    result = resp.json()["result"]
    assert result["status"]["state"] == "completed"
    assert result["status"]["degraded"] is True


def test_a2a_flow_failed():
    """编排器返回 error → status.state=failed + status.message。"""
    client = _make_client()

    with mock.patch("flowmind.a2a.server.run_orchestrator") as mock_orch:
        mock_orch.return_value = {
            "output": None,
            "reasoning": [],
            "degraded": True,
            "error": "规划失败: 无 API key",
        }
        resp = client.post("/a2a", json={
            "jsonrpc": "2.0",
            "id": "req-4",
            "method": "tasks/send",
            "params": {
                "id": "task-4",
                "message": {"role": "user", "parts": [{"type": "text", "text": "test"}]},
                "metadata": {},
            },
        })

    result = resp.json()["result"]
    assert result["status"]["state"] == "failed"
    assert "规划失败" in result["status"]["message"]
