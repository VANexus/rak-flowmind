"""flowmind-mcp-http 入口：以 Streamable HTTP 传输启动 MCP + A2A 服务。

供任意 HTTP 客户端消费——技能逻辑与密钥全部留在 flowmind，Web 端零密钥。
MCP 与 A2A 共享 8001 端口，各占不同路径前缀。

启动：uv run flowmind-mcp-http
配置（环境变量）：
  FLOWMIND_MCP_HOST    默认 127.0.0.1
  FLOWMIND_MCP_PORT    默认 8001
端点：
  MCP: http://<host>:<port>/mcp （Streamable HTTP，JSON-RPC over POST）
  A2A: http://<host>:<port>/a2a （JSON-RPC over POST）
  Agent Card: http://<host>:<port>/.well-known/agent.json
"""
from __future__ import annotations

import os

from starlette.applications import Starlette

from flowmind.a2a.server import FlowMindA2AServer
from flowmind.server import mcp


def build_app() -> Starlette:
    """构建组合 app（MCP + A2A），返回 Starlette 应用。"""
    # FastMCP v1.28 以 streamable_http_app() 暴露底层 Starlette app
    mcp_app = mcp.streamable_http_app()

    # 挂载 A2A 路由
    a2a_server = FlowMindA2AServer()
    a2a_app = a2a_server.get_app()

    # 将 A2A 路由合并到 MCP app
    for route in a2a_app.routes:
        if hasattr(route, "path"):
            mcp_app.routes.append(route)

    return mcp_app


def main() -> None:
    host = os.environ.get("FLOWMIND_MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("FLOWMIND_MCP_PORT", "8001"))
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
