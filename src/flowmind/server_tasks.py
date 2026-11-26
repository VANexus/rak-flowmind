"""任务 REST 通道：把 TaskManager 暴露为 HTTP 端点（与 /mcp 同端口同进程）。

长任务（分钟级 GPU 流水线）走本通道：POST 提交立即 202 → GET 轮询状态 →
GET 流式下载产物；轻技能走 MCP tools/call（localize_status 等只读查询）。
任务状态由 TaskManager 落 PG（服务重启不丢），进度经 MQTT 推送。

端点：
  POST /api/v1/tasks                        提交任务（202；两种 body，见下）
  GET  /api/v1/tasks/{task_id}              查询任务状态（200 / 404）
  DELETE /api/v1/tasks/{task_id}            取消任务（200 / 404 / 409 已终态）
  POST /api/v1/tasks/{task_id}/retry        重提终态任务（202 / 404 / 409 不可重提）
  GET  /api/v1/tasks/{task_id}/download     流式下载产物（FileResponse）
  GET  /api/v1/health                       健康探针（版本 + 组件状态）

POST /api/v1/tasks 两种 body：
  ① {"skill_id": "...", "args": {...}}      通用异步提交（任意注册技能，
     args 按该技能 pydantic 入参模型前置校验）——前端统一异步任务
     通道的入口，20 个技能全部可经此挂进任务引擎；
  ② {"videos": [...], ...}                  批量本地化（localize_submit 同形状，
     向后兼容，等价于 ① 的 skill_id=localize_video 逐视频展开）。

状态码约定：
  202 受理（task_ids 在响应体；队列中途满 → 202 + warning 部分受理）
  400 非 JSON 体；422 入参校验失败（含全部视频扩展名被拒）
  429 队列满（TaskQueueFull 背压，一个都没受理）
  404 任务不存在 / 技能不存在 / 产物不存在（跨租户与不存在同形，反枚举）
  409 任务已终态不可取消 / 不可重提（succeeded 或非终态重提）
  健康探针恒 200——组件故障只反映在 status/components 字段（失败不 500）。

提交语义与 localize_submit / localize_retry 技能一致（复用其入参校验与
任务参数整形），但不经 invoke() 信封——REST 层直接调 manager.submit /
manager.cancel，TaskQueueFull 可按异常类型精确映射 429（errors.py 无独立
错误码，见 localize_submit 模块 docstring 的决策记录）。

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


def _safe_errors(exc: ValidationError) -> list[dict]:
    """ValidationError → 可 JSON 序列化的错误列表（ctx 里的异常转字符串）。"""
    out: list[dict] = []
    for e in exc.errors(include_url=False):
        item = {k: v for k, v in e.items() if k in ("type", "loc", "msg", "input")}
        if "ctx" in e and isinstance(e["ctx"], dict):
            item["ctx"] = {k: str(v) for k, v in e["ctx"].items()}
        out.append(item)
    return out


def _get_task_for_request(manager, task_id: str, include_args: bool):
    """to_thread 入口（anyio.run_sync 不收 kwargs，包一层位置参数）。"""
    return manager.get_task(task_id, include_args=include_args)

# manager 初始化失败后 degraded 结果的缓存时长（秒）：期内不重试构造，
# 防探针重试风暴（构造含 PG 连接超时 + recover，失败场景下每次探针
# 重试都吃完整连接耗时，拖慢事件循环）。
_HEALTH_DEGRADED_TTL = 30.0
_health_degraded: tuple[float, dict[str, str]] | None = None

# 健康全量结果缓存（含 ok 态）：auth-exempt 探针风暴不再逐次打 PG/Milvus
_HEALTH_CACHE_TTL = 5.0
_health_cache: tuple[float, dict] | None = None


def register_task_routes(mcp) -> None:
    """向 FastMCP 实例注册任务 REST 路由。

    必须在 ``mcp.run()`` 之前调用；路由与 /mcp 同 Starlette 应用同端口。
    TaskManager 沿用既有惰性单例（get_task_manager），首次访问即建
    store 连接 + 启动恢复 + GC 线程，与 MCP 通道共享同一实例。
    """

    @mcp.custom_route("/api/v1/tasks", methods=["POST"])
    async def tasks_submit(request: Request) -> JSONResponse:
        """提交任务（通用 ① / 批量本地化 ② 两种 body，见模块 docstring）。

        队列背压：一个都没受理 → 429 queue_full；中途满 → 202 部分受理
        （已受理 task_ids 保留 + warning，transient 可稍后重提剩余）。
        """
        try:
            body = json.loads(await request.body())
        except ValueError as exc:
            return JSONResponse(
                {"error": "invalid_json", "detail": str(exc)}, status_code=400)
        if isinstance(body, dict) and body.get("skill_id"):
            return await _submit_generic(body, request)
        return await _submit_localize_batch(body, request)

    async def _submit_generic(body: dict, request: Request) -> JSONResponse:
        """通用异步提交：{"skill_id", "args"} → 挂进任务引擎（前端统一通道）。

        前置校验：技能必须在册（404，附 available 清单——与 manifest 404
        语义对齐）；args 按该技能 pydantic 入参模型校验（422，errors() 同
        MCP VALIDATION 形状）。通过后与批量路径共用 TaskManager.submit
        （队列背压 / 租户归属 / fail-closed 全部同源）。
        """
        skill_id = str(body["skill_id"]).strip()
        spec = registry().get(skill_id)
        if spec is None:
            return JSONResponse(
                {"error": "unknown_skill",
                 "available": sorted(registry())},
                status_code=404)
        args = body.get("args") or {}
        if not isinstance(args, dict):
            return JSONResponse(
                {"error": "validation", "detail": "args 必须是对象"},
                status_code=422)
        try:
            spec.input_model.model_validate(args)
            # 步骤选择校验（2026-10-31）：pydantic 只校验名字是否合法，
            # 依赖/必需步骤的可执行性在这里收口——显式 422，绝不静默半跑
            from flowmind.pipeline_steps import resolve_selection

            resolve_selection(
                skill_id,
                steps=args.get("steps"),
                skip_steps=args.get("skip_steps"),
            )
        except ValidationError as exc:
            # 错误详情必须可 JSON 序列化：pydantic 的 ctx.error 是异常对象，
            # 直接塞进 JSONResponse 会 500（2026-10-31 实测）
            return JSONResponse(
                {"error": "validation",
                 "detail": _safe_errors(exc)},
                status_code=422)
        except ValueError as exc:
            return JSONResponse(
                {"error": "steps_invalid", "detail": str(exc)},
                status_code=422)
        manager = await anyio.to_thread.run_sync(get_task_manager)
        tenant_id = getattr(request.state, "tenant_id", None)
        try:
            task_id = await anyio.to_thread.run_sync(
                manager.submit, skill_id, args, tenant_id)
        except TaskQueueFull as exc:
            return JSONResponse(
                {"error": "queue_full", "detail": str(exc)}, status_code=429)
        logger.info("通用任务已提交：%s skill=%s（REST 统一通道）", task_id, skill_id)
        return JSONResponse(
            {"task_ids": [task_id], "accepted": 1,
             "rejected_count": 0, "rejected_paths": [],
             "skill_id": skill_id},
            status_code=202)

    async def _submit_localize_batch(body: dict, request: Request) -> JSONResponse:
        """批量本地化提交（localize_submit 同形状；向后兼容保留）。"""
        try:
            inp = SubmitInput.model_validate(body)
        except ValidationError as exc:
            return JSONResponse(
                {"error": "validation", "detail": str(exc)}, status_code=422)
        accepted, rejected = _split_paths(
            inp.videos, load_config().localizer.allowed_extensions)
        manager = await anyio.to_thread.run_sync(get_task_manager)
        # 租户归属：RakAuthMiddleware 校验 Bearer 后写入 request.state（并与
        # flowmind.auth 的上下文同源）；未启用鉴权时为 None——manager.submit
        # 会回退到上下文解析，并在“启用鉴权但无凭证”时拒写（fail-closed）。
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
        """查询单个任务状态（TaskStore 行的 JSON 视图）。

        轮询热路径（2026-10-31）：默认不返回 args（调用方自己提交的内容，
        每次 poll 重复解析/传输纯浪费）；retry 等确需原参的走 include=args。
        """
        manager = await anyio.to_thread.run_sync(get_task_manager)
        include_args = request.query_params.get("include") == "args"
        rec = await anyio.to_thread.run_sync(
            _get_task_for_request, manager, request.path_params["task_id"],
            include_args)
        if rec is None:
            return JSONResponse({"error": "unknown_task"}, status_code=404)
        return JSONResponse(rec)

    @mcp.custom_route("/api/v1/tasks/{task_id}", methods=["DELETE"])
    async def task_cancel(request: Request) -> JSONResponse:
        """取消任务（协作取消 + CAS 终态，语义同 localize_cancel 技能）。

        与读路径同租户作用域（manager.cancel 内部 _visible）：不存在 /
        跨租户 → 同形 404（反枚举）；已终态 → 409 not_cancellable（幂等
        语义在技能信封里是 degraded，REST 用显式状态码让前端免二次查询）。
        """
        manager = await anyio.to_thread.run_sync(get_task_manager)
        task_id = request.path_params["task_id"]
        rec = await anyio.to_thread.run_sync(manager.get_task, task_id)
        if rec is None:
            return JSONResponse({"error": "unknown_task"}, status_code=404)
        if rec["status"] in ("succeeded", "failed", "cancelled", "interrupted"):
            return JSONResponse(
                {"error": "not_cancellable",
                 "detail": f"任务已是终态 {rec['status']}，无需取消"},
                status_code=409)
        ok = await anyio.to_thread.run_sync(manager.cancel, task_id)
        if not ok:
            # 并发竞态：查后被人先写终态 → 与 409 同语义返回当前状态
            return JSONResponse(
                {"error": "not_cancellable", "detail": "任务已被并发写入终态"},
                status_code=409)
        return JSONResponse({"task_id": task_id, "cancelled": True})

    @mcp.custom_route("/api/v1/tasks/{task_id}/retry", methods=["POST"])
    async def task_retry(request: Request) -> JSONResponse:
        """重提任务（复制原 args_json 重新 submit，语义同 localize_retry）。

        准入：failed / cancelled / interrupted 可重提；succeeded 拒绝
        （409，无重提价值）；queued / running 拒绝（409，先取消再重提）。
        队列背压同提交路径：TaskQueueFull → 429。
        """
        manager = await anyio.to_thread.run_sync(get_task_manager)
        task_id = request.path_params["task_id"]
        rec = await anyio.to_thread.run_sync(manager.get_task, task_id)
        if rec is None:
            return JSONResponse({"error": "unknown_task"}, status_code=404)
        status = rec["status"]
        if status == "succeeded":
            return JSONResponse(
                {"error": "not_retryable",
                 "detail": "任务已成功，无需重提"},
                status_code=409)
        if status in ("queued", "running"):
            return JSONResponse(
                {"error": "not_retryable",
                 "detail": f"任务 {status} 中，先取消再重提"},
                status_code=409)
        tenant_id = getattr(request.state, "tenant_id", None)
        try:
            new_id = await anyio.to_thread.run_sync(
                manager.submit, rec["skill_id"], rec.get("args") or {},
                tenant_id)
        except TaskQueueFull as exc:
            return JSONResponse(
                {"error": "queue_full", "detail": str(exc)}, status_code=429)
        return JSONResponse(
            {"original_task_id": task_id, "new_task_id": new_id,
             "skill_id": rec["skill_id"], "original_status": status},
            status_code=202)

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
        探针风暴防护（2026-10-31 扩展）：该端点 auth-exempt，K8s/网关/
        消费方的高频探测会直接变成 DB+gRPC 负载——**全量结果 5s 进程内
        缓存**（含 ok 态），窗口内直接回缓存；manager 初始化失败另有
        30s 专属缓存（期内不重试构造）。
        """
        global _health_degraded, _health_cache
        now = time.monotonic()
        cached = _health_cache
        if cached is not None and now - cached[0] < _HEALTH_CACHE_TTL:
            return JSONResponse(cached[1])
        components: dict[str, str] = {}
        degraded = _health_degraded
        if degraded is not None and (
                now - degraded[0] < _HEALTH_DEGRADED_TTL):
            logger.debug("健康检查：manager 初始化失败缓存命中（30s 内不重试构造）")
            components.update(degraded[1])
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
        body = {
            "status": "ok" if components["pg"] == "ok" else "degraded",
            "version": _VERSION,
            "skill_count": len(registry()),
            "components": components,
        }
        _health_cache = (time.monotonic(), body)
        return JSONResponse(body)
