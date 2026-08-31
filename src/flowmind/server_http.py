"""flowmind-mcp-http 入口：以 Streamable HTTP 传输启动 MCP + A2A 服务。

供任意 HTTP 客户端消费——技能逻辑与密钥全部留在 flowmind，Web 端零密钥。
MCP 与 A2A 共享 8001 端口，各占不同路径前缀。

同端口额外暴露 REST 发现 API（/api/v1/manifest、/api/v1/health），供前端运行时
发现技能，不再硬编码技能清单。

启动：uv run flowmind-mcp-http
配置（环境变量）：
  FLOWMIND_MCP_HOST    默认 127.0.0.1
  FLOWMIND_MCP_PORT    默认 8001
端点：
  MCP: http://<host>:<port>/mcp （Streamable HTTP，JSON-RPC over POST）
  A2A: http://<host>:<port>/a2a （JSON-RPC over POST）
  Agent Card: http://<host>:<port>/.well-known/agent.json
  http://<host>:<port>/mcp            （Streamable HTTP，JSON-RPC over POST）
  http://<host>:<port>/api/v1/manifest （技能清单）
  http://<host>:<port>/api/v1/health   （健康探针）"""
from __future__ import annotations

import os

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware

from flowmind.a2a.server import FlowMindA2AServer
from flowmind.server import mcp
from flowmind.server_rest import register_rest_routes


def _add_cors_middleware() -> None:
    """为 FastMCP 的 Starlette 应用挂载 CORS 中间件。

    前端 (cross-dashboard) 与后端 (flowmind) 分属不同端口，浏览器会
    阻止跨域 fetch。此中间件放行浏览器发现端点的跨域请求。

    实现方式：mcp.run() 内部会调用 self.streamable_http_app() 创建
    Starlette 实例，我们在实例层面 patch 这个方法，让它在创建后
    自动挂载 CORSMiddleware——零侵入路由代码。

    配置（环境变量）：
      FLOWMIND_CORS_ORIGINS   逗号分隔的允许来源，默认覆盖常用本地端口
    """
    allowed = os.environ.get(
        "FLOWMIND_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:8787,http://127.0.0.1:8787,"
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    allowed = [o.strip() for o in allowed if o.strip()]

    original = mcp.streamable_http_app

    def streamable_http_app_with_cors():
        app = original()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
            # 必须显式暴露 mcp-session-id，否则浏览器 fetch 读不到该响应头，
            # MCP SDK 的 StreamableHTTPClientTransport 就无法捕获会话 ID，
            # 导致后续 tools/call 报 "Missing session ID"。
            expose_headers=["mcp-session-id", "Mcp-Session-Id"],
        )
        return app

    mcp.streamable_http_app = streamable_http_app_with_cors  # type: ignore[method-assign]


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
    # 在 run 之前挂载 REST 发现路由（与 /mcp 同端口）
    register_rest_routes(mcp)
    # 在 run 之前挂载 CORS 中间件（让前端跨域 fetch 发现端点）
    _add_cors_middleware()
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
