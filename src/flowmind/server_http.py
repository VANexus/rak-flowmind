"""flowmind-mcp-http 入口：以 Streamable HTTP 传输启动 MCP 服务。

供任意 HTTP 客户端消费（如 cross-dashboard 的 Next.js MCP Client）——技能逻辑
与密钥全部留在 flowmind，Web 端零密钥。复用 server.py 的 FastMCP 实例（工具已注册），
只切换 transport，框架层零改动。

启动：uv run flowmind-mcp-http
配置（环境变量）：
  FLOWMIND_MCP_HOST    默认 127.0.0.1
  FLOWMIND_MCP_PORT    默认 8001
端点：http://<host>:<port>/mcp （Streamable HTTP，JSON-RPC over POST）
"""
from __future__ import annotations

import os

from flowmind.server import mcp


def main() -> None:
    host = os.environ.get("FLOWMIND_MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("FLOWMIND_MCP_PORT", "8001"))
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
