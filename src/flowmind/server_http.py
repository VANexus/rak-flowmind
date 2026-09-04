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
  FLOWMIND_MCP_PORT    默认 8002（前置 Go 网关占 8080，本服务作为其后端）
  FLOWMIND_CORS_ORIGINS  逗号分隔的允许来源
基础设施（PG / MQTT / Milvus / 嵌入服务）env 优先、config.toml 兜底，
见 flowmind.config.InfraConfig 与 .env.example。
"""
from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response

from flowmind.server import mcp
from flowmind.server_rest import register_rest_routes
from flowmind.server_tasks import register_task_routes

logger = logging.getLogger(__name__)

# 联邦优雅注销回调（_start_federation 装配）。由 lifespan shutdown 调用：
# uvicorn 优雅停机完成后会恢复默认 handler 并 re-raise SIGTERM——进程被
# 信号直接终止、解释器关闭流程（atexit）不执行，lifespan shutdown 是
# SIGTERM 路径下唯一可靠的注销窗口。
_federation_stop: Callable[[], None] | None = None


class AuthPlaceholderMiddleware(BaseHTTPMiddleware):
    """多租户鉴权占位中间件（当前 no-op，请求原样放行）。

    这是接入既有登录授权后端时的扩展点：对接时在 dispatch() 中
    校验 Authorization 头（JWT / 签名 / 会话票据），校验失败直接返回
    401 JSONResponse；通过后从凭证解析 tenant_id 写入
    ``request.state.tenant_id``——下游任务 REST 端点与技能层即可按租户
    隔离（TaskStore.tenant_id 列已预留）。本中间件挂在应用外层，
    MCP 与 REST 两条通道经同一入口，鉴权策略对两者同时生效。
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        return await call_next(request)


def _cors_origins() -> list[str]:
    """CORS 允许来源（前端跨域 fetch 发现端点与任务通道用）。"""
    raw = os.environ.get(
        "FLOWMIND_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:8787,http://127.0.0.1:8787,"
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


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
        app = original()
        app.add_middleware(AuthPlaceholderMiddleware)
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
    mcp.settings.host = host
    mcp.settings.port = port
    # 在 run 之前注册路由（与 /mcp 同 Starlette 应用同端口）
    register_rest_routes(mcp)   # /api/v1/manifest 技能发现
    register_task_routes(mcp)   # /api/v1/tasks 任务通道 + /api/v1/health
    _add_middlewares()          # CORS + 鉴权占位
    _wrap_streamable_lifespan()  # 联邦优雅注销挂进 lifespan（SIGTERM 路径）
    _start_federation(port)     # 联邦自注册（默认关；失败静默）
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
