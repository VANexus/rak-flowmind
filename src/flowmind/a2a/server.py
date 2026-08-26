"""A2A 协议层：Task 端点 + Agent Card 发现。

基于 a2a-sdk 实现 Google A2A 协议，暴露：
- GET /.well-known/agent.json → Agent Card
- POST /a2a → Task 处理（tasks/send, tasks/get, tasks/sendSubscribe, tasks/cancel）
"""
from __future__ import annotations

import json

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from flowmind.a2a.agent_card import build_agent_card
from flowmind.a2a.store import cancel_task, get_task, save_task
from flowmind.a2a.types import a2a_task_to_request, result_to_a2a_task
from flowmind.orchestrator.graph import run_orchestrator


async def _agent_card_handler(request: Request) -> JSONResponse:
    """GET /.well-known/agent.json"""
    base_url = str(request.base_url).rstrip("/")
    card = build_agent_card(base_url=base_url)
    return JSONResponse(card)


async def _a2a_handler(request: Request) -> JSONResponse:
    """POST /a2a — JSON-RPC A2A 请求。"""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    method = body.get("method", "")
    params = body.get("params", {})
    req_id = body.get("id")

    if method == "tasks/send":
        result = await _handle_task_send(params)
    elif method == "tasks/get":
        result = await _handle_task_get(params)
    elif method == "tasks/cancel":
        result = await _handle_task_cancel(params)
    else:
        return JSONResponse({"error": f"Unknown method: {method}"}, status_code=400)

    return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": result})


async def _handle_task_send(params: dict) -> dict:
    """处理 tasks/send：接收任务 → 编排 → 存储 → 返回 Task。"""
    task_id = params.get("id", "")
    request = a2a_task_to_request(params.get("message", {}), params.get("metadata", {}))

    result = run_orchestrator(
        goal=request["goal"],
        skill_group=request["skill_group"],
        include_reasoning=request["include_reasoning"],
    )
    task = result_to_a2a_task(task_id, result, request["include_reasoning"])
    await save_task(task)
    return task


async def _handle_task_get(params: dict) -> dict:
    """处理 tasks/get：从存储查询任务状态。"""
    task = await get_task(params.get("id", ""))
    if task is None:
        return {"id": params.get("id"), "status": {"state": "failed", "message": "任务不存在"}}
    return task


async def _handle_task_cancel(params: dict) -> dict:
    """处理 tasks/cancel：在存储中标记任务为 canceled。"""
    task = await cancel_task(params.get("id", ""))
    if task is None:
        return {"id": params.get("id"), "status": {"state": "failed", "message": "任务不存在"}}
    return task


class FlowMindA2AServer:
    """FlowMind A2A 服务器。"""

    def __init__(self, base_url: str = "http://localhost:8001") -> None:
        self.base_url = base_url

    def get_app(self) -> Starlette:
        """构建 Starlette app（可挂载到 FastMCP 或独立运行）。"""
        routes = [
            Route("/.well-known/agent.json", _agent_card_handler),
            Route("/a2a", _a2a_handler, methods=["POST"]),
        ]
        return Starlette(routes=routes)
