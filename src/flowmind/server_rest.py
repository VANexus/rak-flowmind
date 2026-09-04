"""REST 发现 API：把 discover / build_manifest 暴露为 HTTP 端点。

Agent / 前端借此在运行时发现技能（不再硬编码）。复用 server.py 的
FastMCP 实例，通过 v1 的 ``custom_route`` 装饰器把路由挂到同一个 Starlette 应用
（与 /mcp 同端口 8001），无需新依赖、无需新进程。

端点：
  GET /api/v1/manifest        → 完整技能清单（含 schema）
  GET /api/v1/manifest/{id}   → 单个技能；未知 id 返回 404

（健康探针 /api/v1/health 在 server_tasks.py，与任务通道同模块。）
"""
from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse

from flowmind import build_manifest, discover
from flowmind.skill import registry


def register_rest_routes(mcp) -> None:
    """向 FastMCP 实例注册三条 REST 发现路由。

    必须在 ``mcp.run()`` 之前调用，这样路由才会被挂到 Starlette 应用上。
    """

    @mcp.custom_route("/api/v1/manifest", methods=["GET"])
    async def manifest_list(request: Request) -> JSONResponse:  # noqa: ARG001
        """返回完整技能清单。"""
        return JSONResponse(build_manifest())

    @mcp.custom_route("/api/v1/manifest/{skill_id}", methods=["GET"])
    async def manifest_detail(request: Request) -> JSONResponse:
        """返回单个技能的发现信息；未知 id 返回 404。"""
        skill_id = request.path_params["skill_id"]
        try:
            entry = discover(skill_id)
        except KeyError:
            available = sorted(registry().keys())
            return JSONResponse(
                {"error": "unknown_skill", "available": available},
                status_code=404,
            )
        return JSONResponse(entry)
