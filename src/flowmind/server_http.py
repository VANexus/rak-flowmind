"""mcp-base-gpu 单端口服务入口（唯一 HTTP 入口）：MCP + 任务 REST 双通道。

同端口同进程暴露两类通道，技能逻辑与密钥全部留在服务端，客户端零密钥：
- MCP（Streamable HTTP，``/mcp``）：7 个 localize_* 工具，轻技能
  tools/call 同步返回（localize_status / localize_search 等只读查询）。
- 任务 REST（``/api/v1/tasks``）：POST 提交 → 202 轮询 → 产物下载，
  分钟级 GPU 长任务专用（TaskManager 落 PG，重启不丢）。

端点总览：
  /mcp                                    MCP Streamable HTTP（JSON-RPC）
  /api/v1/manifest                        技能清单（发现 API）
  /api/v1/manifest/{skill_id}             单技能 schema
  POST /api/v1/tasks                      提交批量本地化任务（202 / 429）
  GET  /api/v1/tasks/{task_id}            任务状态（200 / 404）
  GET  /api/v1/tasks/{task_id}/download   产物流式下载
  GET  /api/v1/health                     健康探针（版本 + 组件状态）

启动：conda run -n flowmind mcp-base-gpu
配置（环境变量）：
  FLOWMIND_MCP_HOST    默认 127.0.0.1（集群部署 0.0.0.0）
  FLOWMIND_MCP_PORT    默认 8002（ECO-ADR-0006：本服务 :8002，前置 Go 网关 :8090）
  FLOWMIND_CORS_ORIGINS  逗号分隔的允许来源
  FLOWMIND_AUTH_SECRET   调用方凭证校验密钥（与 go-kernel 的
                       KERNEL_TRUSTED_SECRET 同值；**空 = 鉴权关闭**，
                       见 flowmind.auth）
基础设施（PG / MQTT / Milvus / 嵌入服务）env 优先、config.toml 兜底，
见 flowmind.config.InfraConfig 与 .env.example。
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware

from flowmind.auth import TokenError, use_caller, verify_token
from flowmind.config import get_config
from flowmind.server import mcp
from flowmind.server_rest import register_rest_routes
from flowmind.server_tasks import register_task_routes

logger = logging.getLogger(__name__)

# 联邦优雅注销回调（_start_federation 装配）。由 lifespan shutdown 调用：
# uvicorn 优雅停机完成后会恢复默认 handler 并 re-raise SIGTERM——进程被
# 信号直接终止、解释器关闭流程（atexit）不执行，lifespan shutdown 是
# SIGTERM 路径下唯一可靠的注销窗口。
_federation_stop: Callable[[], None] | None = None


class RakAuthMiddleware:
    """调用方凭证校验中间件（ECO-ADR-0011 A 阶段实装，替早前的 no-op 占位）。

    为何是**纯 ASGI** 而非 BaseHTTPMiddleware：BaseHTTPMiddleware 把下游
    放到另一个 task 里跑，中间件内设的 contextvar 不一定可见；租户上下文
    必须让技能层读到，所以直接写 ASGI 入口（同一 task 上下文，
    anyio.to_thread 会再拷贝一层到 worker 线程）。

    行为：
      * 密钥未配（dev / 独立部署）→ 原样放行，不绑上下文
        （``tenant_scope()`` 返回 (False, None)，不过滤）；
      * 豁免前缀（默认 /api/v1/health 与 /api/v1/manifest）→ 匿名可达，
        否则集群 K8s 探针与发现面会死；
      * 其余请求：校验 ``Authorization: Bearer``，失败按 code 回 401
        （**不回显凭证本身**，只回失败类别）；成功则绑进上下文 +
        ``scope["state"]["tenant_id"]``（供 REST 端点取用）。

    不信任任何自填身份头：``X-Rak-Tenant`` 在此仅作诊断参考，归属一律
    以已校验 token 为准（ADR-0011 §4c）。
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        cfg = get_config().auth
        if not cfg.secret or _auth_exempt(scope.get("path", ""), cfg.exempt_prefixes):
            await self.app(scope, receive, send)
            return

        token = _bearer_token(scope.get("headers") or [])
        if token is None:
            await _unauthorized(send, "missing_token",
                                "缺少 Authorization: Bearer 凭证")
            return
        try:
            caller = verify_token(token)
        except TokenError as exc:
            await _unauthorized(send, exc.code, str(exc))
            return

        scope.setdefault("state", {})["tenant_id"] = caller.tenant_id
        scope["state"]["principal"] = caller.principal
        with use_caller(caller):
            await self.app(scope, receive, send)


def _auth_exempt(path: str, prefixes: list[str]) -> bool:
    """路径是否命中豁免前缀（健康探针 / 发现面）。"""
    return any(path == p or path.startswith(p.rstrip("/") + "/")
               for p in (prefixes or []))


def _bearer_token(headers) -> str | None:
    """从 ASGI 头部取 Bearer 凭证（头部名为小写 bytes）。"""
    for name, value in headers:
        if name != b"authorization":
            continue
        raw = value.decode("latin-1").strip()
        scheme, _, rest = raw.partition(" ")
        if scheme.lower() == "bearer" and rest.strip():
            return rest.strip()
        return None
    return None


async def _unauthorized(send, code: str, detail: str) -> None:
    """统一 401 JSON（与 REST 侧错误形状一致；detail 绝不包含凭证）。"""
    body = json.dumps({"error": code, "detail": detail}).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": 401,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"www-authenticate", b'Bearer error="invalid_token"'),
        ],
    })
    await send({"type": "http.response.body", "body": body})


def _cors_origins() -> list[str]:
    """CORS 允许来源（前端跨域 fetch 发现端点与任务通道用）。"""
    raw = os.environ.get(
        "FLOWMIND_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:8787,http://127.0.0.1:8787,"
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


def _transport_security():
    """SDK 的 DNS-rebinding 防护（transport_security.py 会对非白名单 Host 返回
    421）。白名单 = localhost 变体 + 联邦宣告地址（FLOWMIND_FEDERATION_URL，
    即网关 mcpclient 实际访问的 host）+ FLOWMIND_MCP_ALLOWED_HOSTS 追加项。
    网关是唯一 MCP 消费方，且已有独立的应用层鉴权；这里只放行部署面宣告的主机。"""
    from urllib.parse import urlsplit

    from mcp.server.transport_security import TransportSecuritySettings

    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    fed = os.environ.get("FLOWMIND_FEDERATION_URL", "").strip()
    if fed:
        sp = urlsplit(fed if "://" in fed else "http://" + fed)
        if sp.hostname:
            hosts.append(f"{sp.hostname}:*")
            hosts.append(sp.hostname)
    extra = os.environ.get("FLOWMIND_MCP_ALLOWED_HOSTS", "")
    hosts += [h.strip() for h in extra.split(",") if h.strip()]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=hosts)


def _add_middlewares() -> None:
    """在 FastMCP 的 Starlette 应用实例层挂载鉴权占位 + CORS 中间件。

    实现方式：mcp.run() 内部会调用 self.streamable_http_app() 创建
    Starlette 实例，我们在实例层面 patch 这个方法，让它在创建后自动
    add_middleware——零侵入路由代码。Starlette 的 add_middleware 是
    前插语义：最后添加的在最外层，故 CORS 需在鉴权占位之后添加
    （浏览器预检 OPTIONS 不经过鉴权逻辑）。
    """
    original = mcp.streamable_http_app

    def streamable_http_app_with_middleware():
        # 先注入 Host 白名单（DNS-rebinding 防护）：网关以 FLOWMIND_FEDERATION_URL
        # 宣告的地址访问本服务，默认的 localhost-only 白名单会 421 拒之门外。
        mcp.settings.transport_security = _transport_security()
        app = original()
        app.add_middleware(RakAuthMiddleware)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_cors_origins(),
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
            # 必须显式暴露 mcp-session-id，否则浏览器 fetch 读不到该响应头，
            # MCP SDK 的 StreamableHTTPClientTransport 就无法捕获会话 ID，
            # 导致后续 tools/call 报 "Missing session ID"。
            expose_headers=["mcp-session-id", "Mcp-Session-Id"],
        )
        return app

    mcp.streamable_http_app = streamable_http_app_with_middleware  # type: ignore[method-assign]


def _load_dotenv() -> None:
    """服务进程启动即加载 .env（API key 与基础设施地址只落 gitignored 的 .env）。

    加载顺序（load_dotenv 不覆盖已加载变量——真实环境变量仍优先，
    先加载者胜出）：仓库根 .env 先载（repo 内开发/部署的主配置），
    再补父目录 .env（worktree 场景外层共享配置兑底）。

    非 editable 布局（包被安装进 site-packages，parents[2] 下无
    pyproject.toml）时仓库根定位失效，.env 自动加载不可用——记
    warning 提示改用真实环境变量注入配置。
    """
    project_root = Path(__file__).resolve().parents[2]  # src/flowmind/ 上溯两级
    if not (project_root / "pyproject.toml").is_file():
        logger.warning(
            "非 editable 布局（%s 下无 pyproject.toml）：仓库根/父目录 .env "
            "不会自动加载，请改用真实环境变量注入配置", project_root)
    load_dotenv(project_root / ".env")      # 仓库根优先
    load_dotenv(project_root.parent / ".env")  # 父目录兑底（不覆盖已加载）


def _start_federation(port: int) -> None:
    """联邦自注册装配（FLOWMIND_FEDERATION_REGISTER=1 开启，默认关闭）。

    开启后进程向 MCP 网关联邦注册表声明自身（PG + MQTT 双通道）并维持
    心跳；未开启或任何失败均静默跳过——联邦是增值能力，绝不阻塞服务
    启动。详见 flowmind.federation 包 docstring。
    """
    global _federation_stop
    try:
        from flowmind.federation import start_federation

        registrar = start_federation(port=port)
        if registrar is not None:
            _federation_stop = registrar.stop
    except Exception as exc:  # noqa: BLE001  联邦能力绝不阻塞服务启动
        logger.warning("联邦自注册启动失败（服务继续独立运行）: %s", exc)


def _wrap_streamable_lifespan() -> None:
    """把联邦优雅注销挂进 FastMCP 的 Starlette lifespan 关闭链。

    根因（端到端验证发现）：uvicorn 收到 SIGTERM 完成优雅停机后，会在
    capture_signals 中恢复默认信号 handler 并 re-raise 信号——进程被信
    号直接终止，解释器关闭流程（含 atexit）不执行，atexit 注册的联邦
    注销永远不触发（SIGINT/CTRL+C 走 KeyboardInterrupt 异常路径不受
    影响）。lifespan shutdown 在 re-raise 之前执行，是 SIGTERM 路径下
    唯一可靠的注销窗口；注销置于 session manager 关闭之前——网关先摘
    除工具再停服务，无能力真空期。atexit 兜底保留（非 uvicorn 宿主或
    程序内退出仍走解释器关闭），HeartbeatWorker.stop 幂等，双触发无害。
    """
    original = mcp.streamable_http_app

    def streamable_http_app_with_lifecycle():
        app = original()
        orig_lifespan = app.router.lifespan_context

        @contextlib.asynccontextmanager
        async def lifespan(app):
            async with orig_lifespan(app):
                yield
                if _federation_stop is not None:
                    try:
                        # 同步阻塞数秒（MQTT QoS1 确认 + PG offline），
                        # 仅发生在关闭路径，不影响运行期
                        _federation_stop()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("联邦注销异常（忽略，进程继续退出）: %s", exc)

        app.router.lifespan_context = lifespan
        return app

    mcp.streamable_http_app = streamable_http_app_with_lifecycle  # type: ignore[method-assign]


def main() -> None:
    """mcp-base-gpu 入口：单端口 8002（MCP + 任务 REST 双通道）。

    端口 8002 适配网关架构：Go MCP 网关（go-kernel，:8080）对外承接
    /mcp 与 /api/v1/tasks，本服务作为其后端（静态配置或联邦动态注册，
    后者由 FLOWMIND_FEDERATION_REGISTER 开启）。
    FLOWMIND_MCP_PORT 仍可覆盖（单独直连部署时自定义端口）。
    """
    _load_dotenv()
    host = os.environ.get("FLOWMIND_MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("FLOWMIND_MCP_PORT", "8002"))
    if port != 8002:
        logger.warning(
            "生效端口 %s != 8002：网关约定后端端口为 8002，"
            "请检查 FLOWMIND_MCP_PORT/.env 是否为残留旧值（如 8001）", port)
    mcp.settings.host = host
    mcp.settings.port = port
    # 在 run 之前注册路由（与 /mcp 同 Starlette 应用同端口）
    register_rest_routes(mcp)   # /api/v1/manifest 技能发现
    register_task_routes(mcp)   # /api/v1/tasks 任务通道 + /api/v1/health
    _add_middlewares()          # CORS + 鉴权占位
    # 联邦装配顺序（先启动，后条件包裹）：start_federation 先判定开关并
    # 尝试注册，仅在拿到注销回调（联邦确已启动）时才包裹 lifespan——
    # 未启用（默认）/启动失败时不碰 FastMCP lifespan 链，启动路径与
    # 不包裹时严格等价，主流程零变化。
    _start_federation(port)      # 联邦自注册（默认关；失败静默）
    if _federation_stop is not None:
        _wrap_streamable_lifespan()  # 联邦优雅注销挂进 lifespan（SIGTERM 路径）
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
