"""A2A server 测试：端点路由存在性。"""
from __future__ import annotations


def test_server_exposes_agent_card_endpoint():
    from flowmind.a2a.server import FlowMindA2AServer

    server = FlowMindA2AServer(base_url="http://localhost:8001")
    app = server.get_app()
    # Starlette app 应有路由
    routes = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/.well-known/agent.json" in routes


def test_server_exposes_a2a_endpoint():
    from flowmind.a2a.server import FlowMindA2AServer

    server = FlowMindA2AServer(base_url="http://localhost:8001")
    app = server.get_app()
    routes = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/a2a" in routes
