"""异步任务基础设施：PostgreSQL 持久化 + MQTT 进度推送 + GPU 管控 + Milvus 向量化。

mcp-base-gpu SaaS 化的任务引擎（阶段 2）。模块边界（import 无环）：

- 本包 ``__init__`` 只定义**异常**与**任务运行时上下文**（contextvar），
  顶层不 import store/manager/events/vectors——manager 会引入技能注册表，
  而 skills/localize_video 需要反向 import 本包（CancelledError / 上下文）。
- 子模块按需导入：``from flowmind.tasks.manager import get_task_manager``

基础设施事实记录（2026-09-04 探测，凭证绝不入库）：

- PostgreSQL：集群 app 业务用户无 CREATEDB/CREATE SCHEMA 权限（42501），
  库 ``mcp_base_gpu`` 已由管理员建好（owner=app）。连接串从 env
  ``FLOWMIND_PG_DSN`` 读；未设置时回落开发机 mesh 的
  ``RAK_PG_HOST/RAK_PG_PORT/RAK_PG_APP_USER/RAK_PG_APP_PASS``。
  经 PgBouncer 事务模式：短连接 + autocommit，不用 advisory lock /
  服务端 prepared statement。
- MQTT（EMQX）：明文 1883。host 从 ``FLOWMIND_MQTT_HOST`` 读，未设置回落
  ``RAK_MQTT_HOST``；两者都缺 → 发布器静默禁用（纯落库降级）。
- Milvus 2.6.6：``FLOWMIND_MILVUS_URI``（默认开发机 mesh NodePort，
  集群内部注入 http://milvus.agentic.svc:19530）。
- BGE embedding（BAAI/bge-base-zh-v1.5，768 维）：集群预期端点
  100.121.213.4:31997 当前**连接拒绝**（无该部署），客户端按 TEI 风格
  /embed 与 OpenAI 兼容 /v1/embeddings 双形状自适应；开发机联调用本机
  同形状服务（见 skills/_bge_embed.py 模块注释）。

本阶段临时 pip 安装的验证依赖（阶段 5 正式进 environment.yml）：
psycopg2-binary / paho-mqtt / pymilvus。
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ── 任务状态机 ──
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_INTERRUPTED = "interrupted"  # 服务重启时由 recover_running() 标记

TERMINAL_STATUSES = frozenset(
    {STATUS_SUCCEEDED, STATUS_FAILED, STATUS_CANCELLED, STATUS_INTERRUPTED})
NON_TERMINAL_STATUSES = frozenset({STATUS_QUEUED, STATUS_RUNNING})


class CancelledError(RuntimeError):
    """任务被协作式取消：流水线在阶段边界检查到 cancel flag 后抛出。

    命名沿用任务领域语义（非 asyncio.CancelledError）；由
    localize_video 入口转为 failure_category="cancelled" 的 degraded
    信封，TaskManager 据此落 cancelled 终态。
    """

    def __init__(self, message: str = "任务已取消", task_id: str = ""):
        super().__init__(message)
        self.task_id = task_id


class TaskQueueFull(RuntimeError):
    """pending（queued+running）达到 max_pending_tasks 上限。

    调用方（REST/MCP 层）应映射为 429 Too Many Requests（背压语义）。
    """


class TaskStoreError(RuntimeError):
    """PostgreSQL 任务存储不可用/操作失败（连接、建表、SQL 错误）。"""


@dataclass
class TaskContext:
    """任务执行运行时上下文（TaskManager 经 contextvar 注入流水线）。

    直连 invoke()（无 TaskManager）时 current_task_context() 为 None，
    技能入口使用默认 no-op 回调与默认 workdir——对外签名不变。
    """

    task_id: str
    workdir: Path | None = None
    progress_cb: Callable[[str, float, str], None] = field(
        default=lambda stage, pct, message: None)  # noqa: E501 (stage, pct, message)
    cancel_check: Callable[[], bool] = field(default=lambda: False)


_ctx: ContextVar[TaskContext | None] = ContextVar("flowmind_task_context", default=None)


def current_task_context() -> TaskContext | None:
    """当前线程绑定的任务上下文；无 TaskManager 时为 None。"""
    return _ctx.get()


def set_task_context(ctx: TaskContext) -> Token:
    """绑定上下文（worker 线程内调用），返回 token 供 finally 复位。"""
    return _ctx.set(ctx)


def reset_task_context(token: Token) -> None:
    """复位上下文（worker 线程复用前必须清理，防止串任务）。"""
    _ctx.reset(token)
