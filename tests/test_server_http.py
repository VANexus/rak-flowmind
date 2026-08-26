"""HTTP 服务器扩展测试：验证 A2A 路由挂载到 MCP app。"""


def test_http_server_mounts_a2a_routes():
    from flowmind.server_http import build_app

    app = build_app()
    routes = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/.well-known/agent.json" in routes
    assert "/a2a" in routes
