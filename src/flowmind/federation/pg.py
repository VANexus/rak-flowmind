"""联邦注册表 PostgreSQL 持久层（功能包侧，pgx/psycopg2 短连接模式）。

与 Go 侧 go-kernel/internal/pgstore 同 DDL（``federation_backends`` 表；
改动必须双侧同步）。功能包只需写自身一行：注册 upsert / 心跳刷新 /
注销置 offline——工具清单（federation_tools）与调用记账
（federation_usage）由网关侧维护。

PgBouncer 事务模式适配（与 tasks/store.py 同口径硬约束）：
- 每次操作新建短连接（connect → 单语句 → close），无跨语句会话状态；
- autocommit 单语句事务，快进快出，不占 PgBouncer 服务端连接槽；
- 不用 advisory lock / LISTEN-NOTIFY / 服务端 prepared statement。

降级铁律（联邦能力绝不影响功能包主流程）：
- PG 未配置（FLOWMIND_PG_DSN / config infra.pg_dsn / RAK_PG_* 均空）
  → 永久禁用，注册降级为仅 MQTT 通道；
- 连接/写入失败只记日志返回 False，绝不抛异常、绝不阻塞。
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# 与 Go pgstore.go ddl[0] 逐字段一致（双侧同步约定）
_DDL_BACKENDS = """
CREATE TABLE IF NOT EXISTS federation_backends (
    backend_id     TEXT PRIMARY KEY,
    version        TEXT NOT NULL DEFAULT '',
    url            TEXT NOT NULL,
    transport      TEXT NOT NULL DEFAULT 'streamable-http',
    prefix         TEXT NOT NULL DEFAULT '',
    capabilities   JSONB NOT NULL DEFAULT '{}',
    status         TEXT NOT NULL DEFAULT 'active',
    registered_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_heartbeat TIMESTAMPTZ,
    auth_config    JSONB NOT NULL DEFAULT '{}'
)
"""

_SQL_REGISTER = """
INSERT INTO federation_backends
    (backend_id, version, url, transport, prefix, capabilities, status, last_heartbeat)
VALUES (%s, %s, %s, %s, %s, %s, 'active', now())
ON CONFLICT (backend_id) DO UPDATE SET
    version        = EXCLUDED.version,
    url            = EXCLUDED.url,
    transport      = EXCLUDED.transport,
    prefix         = EXCLUDED.prefix,
    capabilities   = EXCLUDED.capabilities,
    status         = 'active',
    last_heartbeat = now()
"""

_SQL_HEARTBEAT = (
    "UPDATE federation_backends SET last_heartbeat = now() WHERE backend_id = %s"
)
_SQL_UNREGISTER = (
    "UPDATE federation_backends SET status = 'offline' WHERE backend_id = %s"
)


def _resolve_conn_spec() -> tuple[str | None, dict[str, Any]] | None:
    """连接规格解析（与 tasks/store.py 同口径：FLOWMIND_PG_DSN → config → RAK_PG_*）。

    返回 (dsn, kwargs)：dsn 非空走 psycopg2.connect(dsn)；否则走
    psycopg2.connect(**kwargs)。均未配置返回 None（本模块永久禁用）。
    """
    from flowmind.config import get_config

    dsn = (os.environ.get("FLOWMIND_PG_DSN", "").strip()
           or get_config().infra.pg_dsn.strip())
    if dsn:
        return dsn, {}
    host = os.environ.get("RAK_PG_HOST", "").strip()
    user = os.environ.get("RAK_PG_APP_USER", "").strip()
    password = os.environ.get("RAK_PG_APP_PASS", "")
    if not (host and user):
        return None
    try:
        port = int(os.environ.get("RAK_PG_PORT", "5432"))
    except ValueError:
        port = 5432
    dbname = (os.environ.get("FLOWMIND_PG_DB", "").strip()
              or get_config().infra.pg_db)
    return None, {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "dbname": dbname,
        "connect_timeout": 5,
    }


class FederationPGStore:
    """联邦注册表 PG 写入器（线程安全：短连接无共享可变状态）。"""

    def __init__(self) -> None:
        spec = _resolve_conn_spec()
        self._enabled = spec is not None
        self._dsn, self._kwargs = spec if spec else (None, {})
        self._ready_lock = threading.Lock()
        self._ready = False
        if not self._enabled:
            logger.info("联邦注册表 PG 未配置（FLOWMIND_PG_DSN / RAK_PG_* 均空）"
                        "——注册降级为仅 MQTT 通道")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def upsert_backend(self, *, backend_id: str, version: str, url: str,
                       transport: str, prefix: str, capabilities: str) -> bool:
        """注册/刷新自身元数据（幂等 upsert；registered_at 由数据库保留）。"""
        return self._execute(
            _SQL_REGISTER,
            (backend_id, version, url, transport, prefix, capabilities),
            context=f"register {backend_id}",
        )

    def update_heartbeat(self, backend_id: str) -> bool:
        """刷新心跳（服务端 now() 统一时钟；行不存在 no-op——register QoS1 会补）。"""
        return self._execute(_SQL_HEARTBEAT, (backend_id,),
                             context=f"heartbeat {backend_id}")

    def set_offline(self, backend_id: str) -> bool:
        """注销置 offline（行保留供审计，与 Go 侧 SetBackendStatus 语义一致）。"""
        return self._execute(_SQL_UNREGISTER, (backend_id,),
                             context=f"unregister {backend_id}")

    # ── 内部 ──

    def _connect(self):
        """短借连接（psycopg2 懒 import，与 tasks/store.py 同口径）。"""
        import psycopg2

        if self._dsn:
            return psycopg2.connect(self._dsn, connect_timeout=5)
        return psycopg2.connect(**self._kwargs)

    def _ensure_ready(self, conn) -> None:
        """幂等建表（进程内一次；失败不置位，下次操作重试）。"""
        if self._ready:
            return
        with self._ready_lock:
            if self._ready:
                return
            with conn.cursor() as cur:
                cur.execute(_DDL_BACKENDS)
            self._ready = True
            logger.info("联邦注册表就绪（federation_backends）")

    def _execute(self, sql: str, params: tuple, *, context: str) -> bool:
        """执行单条写语句（短连接 + autocommit；失败只日志返回 False）。"""
        if not self._enabled:
            return False
        try:
            conn = self._connect()
        except Exception as exc:  # noqa: BLE001  联邦通道失败绝不外泄
            logger.warning("联邦注册表 PG 连接失败（%s，降级重试）: %s", context, exc)
            return False
        try:
            conn.autocommit = True
            self._ensure_ready(conn)
            with conn.cursor() as cur:
                cur.execute(sql, params)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("联邦注册表 PG 写入失败（%s，降级）: %s", context, exc)
            return False
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
