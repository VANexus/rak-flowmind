"""独立 REST 后端服务：把技能注册表暴露为普通 HTTP API（无需 MCP/A2A 协议）。

与 flowmind-mcp-http（默认 8001，MCP + A2A）并存，默认监听 8002，互不影响。
供 Web 前端 / 统一后端等普通 HTTP 客户端直接消费；技能逻辑与密钥全部留在
flowmind 侧，客户端零密钥。

入口：flowmind-api（pyproject [project.scripts]）
配置（环境变量，全部带默认值）：
  FLOWMIND_API_HOST             默认 127.0.0.1
  FLOWMIND_API_PORT             默认 8002
  FLOWMIND_API_WORKERS          job 线程池大小，默认 1（GPU 类技能串行防 OOM）
  FLOWMIND_API_JOB_TTL_SECONDS  终态 job 回收 TTL，默认 3600
  FLOWMIND_API_MAX_FINISHED     终态 job 保留上限，默认 100（淘汰最旧）
  FLOWMIND_API_MAX_PENDING      queued+running 上限，默认 100（超限 429）
  FLOWMIND_API_INVOKE_WAIT_JOBS 置 1/true 后 sync invoke 先等 job lane 空闲（默认关）
  FLOWMIND_CORS_ORIGINS         逗号分隔的允许来源

端点：
  GET  /api/v1/health                     健康探针（含 job 队列水位）
  GET  /api/v1/manifest                   完整技能清单（含 input/output schema）
  GET  /api/v1/manifest/{skill_id}        单个技能发现信息
  POST /api/v1/skills/{skill_id}/invoke   同步调用（响应体 = SkillResult 信封）
  POST /api/v1/jobs                       提交异步 job（立即返回 job_id）
  GET  /api/v1/jobs/{job_id}              查询 job 状态与结果
  GET  /api/v1/jobs                       job 列表（created_at 倒序）

HTTP 状态码约定（SkillResult 信封是唯一契约，状态码只做传输层归类，
由 error.code 映射；Agent 解析信封，网关/重试读状态码）：
  成功（含 degraded=True，ok 恒为 True）→ 200
  NOT_FOUND → 404；VALIDATION → 422；INTERNAL → 500（响应体都是完整信封）
  请求体非合法 JSON → 400 {"error":"invalid_json"}（唯一无信封场景）

并发契约（调用方必读）：
  sync /invoke 不进 job lane、不与 job 互斥——localize_video 等分钟级 GPU
  技能一律走 /jobs；sync invoke 仅用于轻量技能，或确认 job lane 空闲时。
  FLOWMIND_API_INVOKE_WAIT_JOBS=1 可开启护栏（全有全无：所有 sync invoke
  与 job 共享同一信号量，轻量调用也会排在 GPU job 之后）。

Job 语义：
  status=failed 仅表示 runner 层意外崩溃（防御性保留）；技能级失败
  （result.ok=false，含 VALIDATION）时 job 仍是 succeeded，失败细节看
  result.error——args 不做入队前预校验，校验由 invoke() 统一产出。
  job 为内存态，服务重启即丢：轮询方对 unknown_job 与被 TTL 回收做
  同类处理（重新提交）。
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import anyio
import uvicorn
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

import flowmind.skills  # noqa: F401  触发技能注册
from flowmind import build_manifest, discover
from flowmind.contracts import SkillResult, new_trace
from flowmind.skill import invoke, registry

_VERSION = "0.1.0"  # 与 server_rest 的 health 保持一致

_TERMINAL = ("succeeded", "failed")

# error.code → HTTP 状态码映射（信封之外只做传输层归类）
_STATUS_BY_ERROR_CODE = {"NOT_FOUND": 404, "VALIDATION": 422, "INTERNAL": 500}


def _env_int(name: str, default: int) -> int:
    """读环境变量整数（缺失/非法一律回落默认值）。"""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _cors_origins() -> list[str]:
    """CORS 允许来源。默认串与 server_http._add_cors_middleware 对齐
    （不直接 import server_http——其顶部会实例化 FastMCP 并注册全部
    MCP tool，import 副作用过大）。"""
    raw = os.environ.get(
        "FLOWMIND_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:8787,http://127.0.0.1:8787,"
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


def _invoke_waits_jobs() -> bool:
    """sync invoke 是否等待 job lane（FLOWMIND_API_INVOKE_WAIT_JOBS 护栏）。"""
    return os.environ.get("FLOWMIND_API_INVOKE_WAIT_JOBS", "").lower() in ("1", "true")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _envelope_response(result: SkillResult) -> JSONResponse:
    """SkillResult 信封 → JSONResponse（状态码按 error.code 映射）。"""
    status = _STATUS_BY_ERROR_CODE.get(result.error.code, 500) if result.error else 200
    return JSONResponse(result.model_dump(mode="json"), status_code=status)


def _run_invoke(
    skill_id: str,
    raw_args: Any,
    trace_id: str | None = None,
    lane: threading.Semaphore | None = None,
) -> SkillResult:
    """同步调用技能（阻塞，供线程执行）。invoke 永不抛异常，本函数也不抛。

    lane 非空时先独占 job lane（FLOWMIND_API_INVOKE_WAIT_JOBS 护栏），
    与 JobManager 的 worker 共享同一信号量。trace_id 由调用方透传
    （sync invoke 的 X-FlowMind-Trace-Id 头）；不传则执行时刻新生成。
    """
    with (lane if lane is not None else nullcontext()):
        return invoke(skill_id, raw_args, new_trace(source="rest-api", trace_id=trace_id))


@dataclass
class JobRecord:
    """一个异步 job 的完整状态（内存态，服务重启即丢）。"""
    job_id: str
    skill_id: str
    args: dict
    status: str                    # queued | running | succeeded | failed
    created_at: str                # ISO8601 UTC
    started_at: str | None = None
    finished_at: str | None = None
    result: dict | None = None     # 终态时 = SkillResult.model_dump(mode="json")
    job_error: str | None = None   # runner 层意外异常（status=failed 时填充）


class JobStore:
    """内存 job 存储：单锁保护；惰性 sweep（不起后台线程）。

    sweep 只回收终态 job（queued/running 永不回收——running 回收会撕裂
    轮询，queued 回收会丢任务）；终态超限按 finished_at 淘汰最旧。
    """

    def __init__(self, ttl_seconds: int, max_finished: int, max_pending: int):
        self._ttl = max(1, ttl_seconds)
        self._max_finished = max(1, max_finished)
        self._max_pending = max(1, max_pending)
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}

    def submit(self, rec: JobRecord) -> bool:
        """入队；pending（queued+running）超限返回 False（调用方回 429）。"""
        with self._lock:
            self._sweep()
            pending = sum(1 for j in self._jobs.values() if j.status not in _TERMINAL)
            if pending >= self._max_pending:
                return False
            self._jobs[rec.job_id] = rec
            return True

    def get(self, job_id: str) -> JobRecord | None:
        """按 id 取记录（返回浅拷贝，锁外可安全序列化）。"""
        with self._lock:
            self._sweep()
            rec = self._jobs.get(job_id)
            return replace(rec) if rec is not None else None

    def list_jobs(self) -> list[JobRecord]:
        """全部记录（浅拷贝），created_at 倒序。"""
        with self._lock:
            self._sweep()
            recs = [replace(r) for r in self._jobs.values()]
        recs.sort(key=lambda r: r.created_at, reverse=True)
        return recs

    def counts(self) -> dict[str, int]:
        """pending 水位（health 探针用）。"""
        with self._lock:
            return {
                "queued": sum(1 for j in self._jobs.values() if j.status == "queued"),
                "running": sum(1 for j in self._jobs.values() if j.status == "running"),
            }

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is not None and rec.status == "queued":
                rec.status = "running"
                rec.started_at = _now_iso()

    def mark_finished(self, job_id: str, result: dict | None, job_error: str | None = None) -> None:
        """落终态：有 job_error → failed（runner 层崩溃），否则 succeeded。"""
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return
            rec.status = "failed" if job_error else "succeeded"
            rec.finished_at = _now_iso()
            rec.result = result
            rec.job_error = job_error

    def _sweep(self) -> None:
        """惰性回收（调用方必须已持锁）。"""
        now = datetime.now(timezone.utc)
        expired = [
            jid for jid, j in self._jobs.items()
            if j.status in _TERMINAL and j.finished_at is not None
            and (now - datetime.fromisoformat(j.finished_at)).total_seconds() > self._ttl
        ]
        for jid in expired:
            del self._jobs[jid]
        finished = sorted(
            (j for j in self._jobs.values() if j.status in _TERMINAL),
            key=lambda j: j.finished_at or "",
        )
        excess = len(finished) - self._max_finished
        for j in finished[:excess] if excess > 0 else []:
            del self._jobs[j.job_id]


class JobManager:
    """异步 job 执行器：线程池 + 存储 + lane 信号量（与 sync-invoke 护栏共享）。

    workers 默认 1：GPU 类技能（localize_video / 本地 TTS / LaMa）在单卡上
    必须串行，否则 OOM。
    """

    def __init__(self, workers: int, ttl_seconds: int, max_finished: int, max_pending: int):
        self.workers = max(1, workers)
        self.lane = threading.Semaphore(self.workers)
        self.store = JobStore(ttl_seconds, max_finished, max_pending)
        self._pool = ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix="flowmind-job",
        )

    def submit(self, skill_id: str, args: dict) -> str | None:
        """提交 job，返回 job_id；pending 超限返回 None（调用方回 429）。"""
        job_id = uuid.uuid4().hex
        rec = JobRecord(
            job_id=job_id, skill_id=skill_id, args=args,
            status="queued", created_at=_now_iso(),
        )
        if not self.store.submit(rec):
            return None
        try:
            self._pool.submit(self._worker, job_id)
        except RuntimeError as exc:
            # 执行器已关闭等极端情况：错误永不静默，落成 failed 供轮询方看到
            self.store.mark_finished(job_id, None, job_error=str(exc))
        return job_id

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _worker(self, job_id: str) -> None:
        rec = self.store.get(job_id)
        if rec is None:  # 极端边界：入库即被清理（仅极小 TTL 下可能）
            return
        try:
            with self.lane:
                self.store.mark_running(job_id)
                result = _run_invoke(rec.skill_id, rec.args)
                self.store.mark_finished(job_id, result.model_dump(mode="json"))
        except Exception as exc:  # 兜底：runner 层意外异常（错误永不静默）
            self.store.mark_finished(job_id, None, job_error=str(exc))


def _job_to_dict(rec: JobRecord) -> dict:
    """job 记录 → 响应 dict（锁外执行；result 已是 JSON-safe dict）。"""
    return {
        "job_id": rec.job_id,
        "skill_id": rec.skill_id,
        "status": rec.status,
        "created_at": rec.created_at,
        "started_at": rec.started_at,
        "finished_at": rec.finished_at,
        "result": rec.result,
        "job_error": rec.job_error,
    }


async def _health(request: Request) -> JSONResponse:
    """健康探针：与 server_rest 同构字段 + job 队列水位。"""
    manager: JobManager = request.app.state.job_manager
    return JSONResponse({
        "status": "ok",
        "skill_count": len(registry()),
        "version": _VERSION,
        "jobs": {**manager.store.counts(), "workers": manager.workers},
    })


async def _manifest_list(request: Request) -> JSONResponse:
    return JSONResponse(build_manifest())  # noqa: ARG001


async def _manifest_detail(request: Request) -> JSONResponse:
    try:
        entry = discover(request.path_params["skill_id"])
    except KeyError:
        return JSONResponse(
            {"error": "unknown_skill", "available": sorted(registry())},
            status_code=404,
        )
    return JSONResponse(entry)


async def _invoke(request: Request) -> JSONResponse:
    """同步调用。无需 registry 预查——未知技能由 invoke() 产出 NOT_FOUND
    信封，统一走状态码映射。"""
    try:
        raw = json.loads(await request.body())
    except ValueError as exc:  # JSONDecodeError / UTF-8 解码错误
        return JSONResponse({"error": "invalid_json", "detail": str(exc)}, status_code=400)
    manager: JobManager = request.app.state.job_manager
    lane = manager.lane if _invoke_waits_jobs() else None
    trace_id = request.headers.get("x-flowmind-trace-id")
    result = await anyio.to_thread.run_sync(
        _run_invoke, request.path_params["skill_id"], raw, trace_id, lane,
    )
    return _envelope_response(result)


async def _jobs_submit(request: Request) -> JSONResponse:
    """提交异步 job。未知技能前置检查（不留孤儿 queued job）；
    args 不预校验（校验由 invoke 统一产出 VALIDATION 信封）。"""
    try:
        body = json.loads(await request.body())
    except ValueError as exc:
        return JSONResponse({"error": "invalid_json", "detail": str(exc)}, status_code=400)
    if (
        not isinstance(body, dict)
        or not isinstance(body.get("skill_id"), str)
        or ("args" in body and not isinstance(body.get("args"), dict))
    ):
        return JSONResponse({"error": "invalid_request"}, status_code=400)
    skill_id = body["skill_id"]
    if skill_id not in registry():
        return JSONResponse(
            {"error": "unknown_skill", "available": sorted(registry())},
            status_code=404,
        )
    manager: JobManager = request.app.state.job_manager
    job_id = manager.submit(skill_id, body.get("args") or {})
    if job_id is None:
        return JSONResponse({"error": "too_many_jobs"}, status_code=429)
    rec = manager.store.get(job_id)
    return JSONResponse(
        {
            "job_id": job_id,
            "skill_id": skill_id,
            "status": "queued",
            "created_at": rec.created_at if rec else None,
        },
        status_code=202,
    )


async def _job_get(request: Request) -> JSONResponse:
    manager: JobManager = request.app.state.job_manager
    rec = manager.store.get(request.path_params["job_id"])
    if rec is None:
        return JSONResponse({"error": "unknown_job"}, status_code=404)
    return JSONResponse(_job_to_dict(rec))


async def _jobs_list(request: Request) -> JSONResponse:
    manager: JobManager = request.app.state.job_manager
    recs = manager.store.list_jobs()
    return JSONResponse({"jobs": [_job_to_dict(r) for r in recs], "count": len(recs)})


_routes = [
    Route("/api/v1/health", _health, methods=["GET"]),
    Route("/api/v1/manifest", _manifest_list, methods=["GET"]),
    Route("/api/v1/manifest/{skill_id}", _manifest_detail, methods=["GET"]),
    Route("/api/v1/skills/{skill_id}/invoke", _invoke, methods=["POST"]),
    Route("/api/v1/jobs", _jobs_submit, methods=["POST"]),
    Route("/api/v1/jobs", _jobs_list, methods=["GET"]),
    Route("/api/v1/jobs/{job_id}", _job_get, methods=["GET"]),
]


def create_app() -> Starlette:
    """构建独立 REST 服务 app（main 与 demo 共用）。"""
    manager = JobManager(
        workers=_env_int("FLOWMIND_API_WORKERS", 1),
        ttl_seconds=_env_int("FLOWMIND_API_JOB_TTL_SECONDS", 3600),
        max_finished=_env_int("FLOWMIND_API_MAX_FINISHED", 100),
        max_pending=_env_int("FLOWMIND_API_MAX_PENDING", 100),
    )

    @asynccontextmanager
    async def _lifespan(app: Starlette) -> AsyncIterator[None]:
        yield
        manager.shutdown()  # 退出时不再接收新 job，未启动的取消

    app = Starlette(routes=_routes, lifespan=_lifespan)
    app.state.job_manager = manager
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["x-flowmind-trace-id"],
    )
    return app


def main() -> None:
    """flowmind-api 入口：独立 REST 服务（默认 127.0.0.1:8002）。"""
    # 服务进程启动即加载 .env（与 skills/_secrets 同约定：API key 不进 toml/commit，
    # 只落 gitignored 的 .env）。部分技能（如 marketing_image_gen）直接读
    # os.environ，不经 _secrets.get_api_key，因此服务层必须先兜底加载。
    # load_dotenv 不覆盖已加载变量——真实环境变量仍优先。
    project_root = Path(__file__).resolve().parents[2]  # src/flowmind/ 上溯两级
    load_dotenv(project_root.parent / ".env")
    load_dotenv(project_root / ".env")
    uvicorn.run(
        create_app(),
        host=os.environ.get("FLOWMIND_API_HOST", "127.0.0.1"),
        port=_env_int("FLOWMIND_API_PORT", 8002),
    )


if __name__ == "__main__":
    main()
