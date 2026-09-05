"""任务 REST 通道：把 TaskManager 暴露为 HTTP 端点（与 /mcp 同端口同进程）。

长任务（分钟级 GPU 流水线）走本通道：POST 提交立即 202 → GET 轮询状态 →
GET 流式下载产物；轻技能走 MCP tools/call（localize_status 等只读查询）。
任务状态由 TaskManager 落 PG（服务重启不丢），进度经 MQTT 推送。

端点：
  POST /api/v1/tasks                        提交批量本地化任务（202）
  GET  /api/v1/tasks/{task_id}              查询任务状态（200 / 404）
  GET  /api/v1/tasks/{task_id}/download     流式下载产物（FileResponse）
  GET  /api/v1/health                       健康探针（版本 + 组件状态）

状态码约定：
  202 受理（task_ids 在响应体；队列中途满 → 202 + warning 部分受理）
  400 非 JSON 体；422 入参校验失败（含全部视频扩展名被拒）
  429 队列满（TaskQueueFull 背压，一个都没受理）
  404 任务不存在 / 产物不存在
  健康探针恒 200——组件故障只反映在 status/components 字段（失败不 500）。

提交语义与 localize_submit 技能一致（复用其入参校验与任务参数整形），
但不经 invoke() 信封——REST 层直接调 manager.submit，TaskQueueFull
可按异常类型精确映射 429（errors.py 无独立错误码，见 localize_submit
模块 docstring 的决策记录）。

并发纪律：async 端点内的 store/manager 同步 DB 调用（submit / get /
health 的 PG 往返）一律包 ``anyio.to_thread.run_sync``（await 形态），
避免阻塞事件循环拖累同端口的 /mcp Streamable HTTP 通道。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import anyio
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse

from flowmind.config import load_config
from flowmind.skill import registry
from flowmind.skills.localize_submit import (
    TASK_SKILL_ID,
    SubmitInput,
    _split_paths,
    _task_args,
)
from flowmind.tasks import TaskQueueFull, vectors
from flowmind.tasks.manager import get_task_manager

logger = logging.getLogger(__name__)

_VERSION = "0.2.0"

# manager 初始化失败后 degraded 结果的缓存时长（秒）：期内不重试构造，
# 防探针重试风暴（构造含 PG 连接超时 + recover，失败场景下每次探针
# 重试都吃完整连接耗时，拖慢事件循环）。
_HEALTH_DEGRADED_TTL = 30.0
_health_degraded: tuple[float, dict[str, str]] | None = None


def register_task_routes(mcp) -> None:
    """向 FastMCP 实例注册任务 REST 路由。

    必须在 ``mcp.run()`` 之前调用；路由与 /mcp 同 Starlette 应用同端口。
    TaskManager 沿用既有惰性单例（get_task_manager），首次访问即建
    store 连接 + 启动恢复 + GC 线程，与 MCP 通道共享同一实例。
    """

    @mcp.custom_route("/api/v1/tasks", methods=["POST"])
    async def tasks_submit(request: Request) -> JSONResponse:
        """提交批量本地化任务（localize_submit 同形状 JSON）。

        队列背压：一个都没受理 → 429 queue_full；中途满 → 202 部分受理
        （已受理 task_ids 保留 + warning，transient 可稍后重提剩余）。
        """
        try:
            body = json.loads(await request.body())
        except ValueError as exc:
            return JSONResponse(
                {"error": "invalid_json", "detail": str(exc)}, status_code=400)
        try:
            inp = SubmitInput.model_validate(body)
        except ValidationError as exc:
            return JSONResponse(
                {"error": "validation", "detail": str(exc)}, status_code=422)
        accepted, rejected = _split_paths(
            inp.videos, load_config().localizer.allowed_extensions)
        manager = await anyio.to_thread.run_sync(get_task_manager)
        # 鉴权占位中间件实装后从凭证解析 tenant_id 写入 request.state；
        # 现为 no-op 不设置 → None（管道已接通，store.tenant_id 列就绪）
        tenant_id = getattr(request.state, "tenant_id", None)
        task_ids: list[str] = []
        for video in accepted:
            try:
                task_ids.append(await anyio.to_thread.run_sync(
                    manager.submit, TASK_SKILL_ID, _task_args(video, inp),
                    tenant_id))
            except TaskQueueFull as exc:
                if not task_ids:
                    return JSONResponse(
                        {"error": "queue_full", "detail": str(exc),
                         "rejected_count": len(rejected),
                         "rejected_paths": rejected},
                        status_code=429)
                return JSONResponse(
                    {
                        "task_ids": task_ids,
                        "accepted": len(task_ids),
                        "rejected_count": len(rejected),
                        "rejected_paths": rejected,
                        "warning": (
                            f"队列已满，仅受理前 {len(task_ids)}/{len(accepted)}"
                            f" 个视频；其余可稍后重提"
                        ),
                    },
                    status_code=202,
                )
        return JSONResponse(
            {
                "task_ids": task_ids,
                "accepted": len(task_ids),
                "rejected_count": len(rejected),
                "rejected_paths": rejected,
                "skill_id": TASK_SKILL_ID,
            },
            status_code=202,
        )

    @mcp.custom_route("/api/v1/tasks/{task_id}", methods=["GET"])
    async def task_get(request: Request) -> JSONResponse:
        """查询单个任务状态（TaskStore 行的 JSON 视图）。"""
        manager = await anyio.to_thread.run_sync(get_task_manager)
        rec = await anyio.to_thread.run_sync(
            manager.get_task, request.path_params["task_id"])
        if rec is None:
            return JSONResponse({"error": "unknown_task"}, status_code=404)
        return JSONResponse(rec)

    @mcp.custom_route("/api/v1/tasks/{task_id}/download", methods=["GET"])
    async def task_download(request: Request) -> FileResponse | JSONResponse:
        """流式下载任务产物。

        路径穿越防护：``file`` 参数只允许匹配该任务 output_paths 中的
        **basename 白名单**（用户输入从不与任何目录拼接），``../`` 等穿越
        序列必然落空白名单 → 404，物理上不可能触达产物之外的文件。
        """
        task_id = request.path_params["task_id"]
        name = request.query_params.get("file", "").strip()
        if not name:
            return JSONResponse(
                {"error": "missing_file_param"}, status_code=400)
        manager = await anyio.to_thread.run_sync(get_task_manager)
        rec = await anyio.to_thread.run_sync(manager.get_task, task_id)
        if rec is None:
            return JSONResponse({"error": "unknown_task"}, status_code=404)
        output_paths = [str(p) for p in (rec.get("output_paths") or []) if p]
        matched = next(
            (p for p in output_paths if Path(p).name == name), None)
        if matched is None or not Path(matched).is_file():
            return JSONResponse({"error": "file_not_found"}, status_code=404)
        return FileResponse(matched, filename=name)

    @mcp.custom_route("/api/v1/health", methods=["GET"])
    async def health(request: Request) -> JSONResponse:  # noqa: ARG001
        """健康探针：版本 + 各组件尽力检查（组件故障不 500）。

        pg 是任务引擎硬依赖（error → status=degraded）；mqtt / milvus
        为增值通道（disabled / unverified / connecting 均不算故障）。
        探针风暴防护：manager 初始化失败的结果缓存
        ``_HEALTH_DEGRADED_TTL``（30s），期内直接返回缓存的 degraded
        状态、不重试构造；缓存过期后重试（PG 恢复即自愈，成功即清缓存）。
        """
        global _health_degraded
        components: dict[str, str] = {}
        cached = _health_degraded
        if cached is not None and (
                time.monotonic() - cached[0] < _HEALTH_DEGRADED_TTL):
            logger.debug("健康检查：manager 初始化失败缓存命中（30s 内不重试构造）")
            components.update(cached[1])
        else:
            try:
                manager = await anyio.to_thread.run_sync(get_task_manager)
                _health_degraded = None  # 构造成功：清缓存
                components["pg"] = await anyio.to_thread.run_sync(
                    manager.store.health_status)
                components["mqtt"] = manager.events.status()
            except Exception as exc:  # noqa: BLE001  探针绝不 500
                logger.warning("健康检查：任务引擎初始化失败: %s", exc)
                components["pg"] = "error"
                components["mqtt"] = "unknown"
                _health_degraded = (time.monotonic(), dict(components))
        components["milvus"] = await anyio.to_thread.run_sync(
            vectors.health_status)
        return JSONResponse(
            {
                "status": "ok" if components["pg"] == "ok" else "degraded",
                "version": _VERSION,
                "skill_count": len(registry()),
                "components": components,
            }
        )
