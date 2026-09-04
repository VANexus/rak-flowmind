"""PostgreSQL 任务存储（TaskStore）——TaskManager 的持久层。

PgBouncer 事务模式适配（硬约束）：
- 每次操作新建短连接（connect → 单语句 → close），无跨语句会话状态；
- autocommit 单语句事务，快进快出，不占 PgBouncer 服务端连接槽；
- 不用 advisory lock / LISTEN-NOTIFY / 服务端 prepared statement
  （psycopg2 默认客户端侧 prepare，天然兼容）；
- 连接失败重试 1 次（瞬时抖动），仍失败抛 TaskStoreError（错误永不静默）。

决策记录：
- 建库：app 业务用户无 CREATEDB 权限（42501），``mcp_base_gpu`` 库已由
  管理员建好（owner=app）；本模块只做**幂等建表**（IF NOT EXISTS）。
- 线程安全：无共享可变连接（每操作短借连接），仅建表标志用锁保护。
- TTL GC 语义：manager 只清 workdir 不删 DB 行（行保留供审计与状态查询）。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

from flowmind.tasks import TaskStoreError

logger = logging.getLogger(__name__)

_SCHEMA_NAME = "public"
_TABLE_NAME = "tasks"

# 幂等建表 + 查询索引（status 过滤 + created_at 排序是 list_tasks 的固定形态）
_DDL = [
    f"""
    CREATE TABLE IF NOT EXISTS {_SCHEMA_NAME}.{_TABLE_NAME} (
        task_id     TEXT PRIMARY KEY,
        skill_id    TEXT NOT NULL,
        args_json   TEXT NOT NULL,
        status      TEXT NOT NULL,
        stage       TEXT,
        progress    REAL DEFAULT 0,
        error       TEXT,
        created_at  TIMESTAMPTZ DEFAULT now(),
        started_at  TIMESTAMPTZ,
        finished_at TIMESTAMPTZ,
        tenant_id   TEXT,
        output_paths TEXT
    )
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_tasks_status_created
        ON {_SCHEMA_NAME}.{_TABLE_NAME} (status, created_at DESC)
    """,
]

_SELECT_COLS = (
    "task_id, skill_id, args_json, status, stage, progress, error, "
    "created_at, started_at, finished_at, tenant_id, output_paths"
)


def _resolve_conn_spec() -> tuple[str | None, dict[str, Any]]:
    """连接规格解析（配置源顺序：env → config.toml → RAK_PG_* mesh 兜底）。

    FLOWMIND_PG_DSN 优先，其次 config ``infra.pg_dsn``；均未配置则回落
    RAK_PG_HOST/RAK_PG_APP_USER/RAK_PG_APP_PASS（库名：FLOWMIND_PG_DB 或
    config ``infra.pg_db``）。返回 (dsn, kwargs)：dsn 非空则
    psycopg2.connect(dsn)；否则 psycopg2.connect(**kwargs)（密码等特殊
    字符安全，不走 DSN 转义）。均未配置 → TaskStoreError（配置缺失属
    部署错误，显式失败）。
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
        raise TaskStoreError(
            "PostgreSQL 未配置：设置 FLOWMIND_PG_DSN（或 config infra.pg_dsn），或提供 "
            "RAK_PG_HOST / RAK_PG_PORT / RAK_PG_APP_USER / RAK_PG_APP_PASS "
            "（source ~/.agents/skills/rak/.env）")
    try:
        port = int(os.environ.get("RAK_PG_PORT", "5432"))
    except ValueError as exc:
        raise TaskStoreError(f"RAK_PG_PORT 非法: {os.environ.get('RAK_PG_PORT')!r}") from exc
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


class TaskStore:
    """PG 任务存储：CRUD + 进度更新 + 启动恢复 + 终态查询。"""

    def __init__(self) -> None:
        self._dsn, self._conn_kwargs = _resolve_conn_spec()
        self._ready_lock = threading.Lock()
        self._ready = False

    # ── 连接与建表 ──

    def _connect(self):
        """短借连接（重试 1 次）。psycopg2 懒 import（阶段 5 进 environment.yml）。"""
        try:
            import psycopg2
        except ImportError as exc:
            raise TaskStoreError(
                "未安装 psycopg2（本阶段验证临时安装：pip install psycopg2-binary；"
                "阶段 5 进 environment.yml）") from exc
        last: Exception | None = None
        for attempt in (1, 2):
            try:
                if self._dsn:
                    return psycopg2.connect(self._dsn)
                return psycopg2.connect(**self._conn_kwargs)
            except Exception as exc:  # noqa: BLE001  瞬时抖动重试一次
                last = exc
                logger.debug("PG 连接失败（第 %s 次）: %s", attempt, exc)
                time.sleep(0.3)
        raise TaskStoreError(f"PostgreSQL 连接失败（重试 2 次）: {last}") from last

    def _ensure_ready(self) -> None:
        """幂等建表（进程内一次；并发下双检锁）。"""
        if self._ready:
            return
        with self._ready_lock:
            if self._ready:
                return
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    for stmt in _DDL:
                        cur.execute(stmt)
                conn.commit()
                self._ready = True
            finally:
                conn.close()
            logger.info("TaskStore 就绪（表 %s.%s）", _SCHEMA_NAME, _TABLE_NAME)

    # ── 写路径 ──

    def create_task(self, task_id: str, skill_id: str, args: dict,
                    tenant_id: str | None = None) -> None:
        """落 queued 任务（task_id 冲突属调用方 bug，直接抛）。"""
        self._ensure_ready()
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {_TABLE_NAME} "
                    "(task_id, skill_id, args_json, status, tenant_id) "
                    "VALUES (%s, %s, %s, 'queued', %s)",
                    (task_id, skill_id, json.dumps(args, ensure_ascii=False), tenant_id),
                )
            conn.commit()
        finally:
            conn.close()

    def update_progress(self, task_id: str, stage: str, progress: float) -> None:
        """进度落库（高频调用：单 UPDATE 短事务）。"""
        self._ensure_ready()
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {_TABLE_NAME} SET stage = %s, progress = %s "
                    "WHERE task_id = %s",
                    (stage, float(progress), task_id),
                )
            conn.commit()
        finally:
            conn.close()

    def set_status(self, task_id: str, status: str, *,
                   error: str | None = None,
                   output_paths: list[str] | None = None) -> None:
        """状态迁移：running 记 started_at；终态记 finished_at。

        error/output_paths 传 None 时保留原值（COALESCE），终态可携带。
        """
        self._ensure_ready()
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {_TABLE_NAME} SET status = %s, "
                    "started_at = CASE WHEN %s = 'running' THEN now() ELSE started_at END, "
                    "finished_at = CASE WHEN %s IN "
                    "('succeeded','failed','cancelled','interrupted') "
                    "THEN now() ELSE finished_at END, "
                    "error = COALESCE(%s, error), "
                    "output_paths = COALESCE(%s, output_paths) "
                    "WHERE task_id = %s",
                    (
                        status, status, status,
                        error,
                        json.dumps(output_paths, ensure_ascii=False) if output_paths is not None else None,
                        task_id,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def recover_running(self, include_queued: bool = True) -> int:
        """启动恢复：遗留 running（可选 queued）任务 → interrupted。

        queued 也一并恢复：进程重启后线程池已清空，遗留 queued 行永不执行，
        且会永久虚增 pending 水位（背压误判）。返回恢复行数。
        """
        self._ensure_ready()
        statuses = ("queued", "running") if include_queued else ("running",)
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {_TABLE_NAME} SET status = 'interrupted', finished_at = now() "
                    "WHERE status = ANY(%s)",
                    (list(statuses),),
                )
                recovered = cur.rowcount
            conn.commit()
        finally:
            conn.close()
        if recovered:
            logger.warning("启动恢复：%s 个遗留任务标为 interrupted", recovered)
        return recovered

    def delete_task(self, task_id: str) -> bool:
        """删除任务行（当前 GC 策略不调用；保留给管理接口）。"""
        self._ensure_ready()
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {_TABLE_NAME} WHERE task_id = %s", (task_id,))
                deleted = cur.rowcount > 0
            conn.commit()
            return deleted
        finally:
            conn.close()

    # ── 读路径 ──

    def get_task(self, task_id: str) -> dict | None:
        """单任务详情（args/output_paths 反序列化为结构）。"""
        self._ensure_ready()
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {_SELECT_COLS} FROM {_TABLE_NAME} WHERE task_id = %s",
                    (task_id,),
                )
                row = cur.fetchone()
        finally:
            conn.close()
        return _row_to_dict(row) if row else None

    def list_tasks(self, status: str | None = None, limit: int = 100) -> list[dict]:
        """任务列表（created_at 倒序）；status=None 查全部。"""
        self._ensure_ready()
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                if status:
                    cur.execute(
                        f"SELECT {_SELECT_COLS} FROM {_TABLE_NAME} WHERE status = %s "
                        "ORDER BY created_at DESC LIMIT %s",
                        (status, max(1, int(limit))),
                    )
                else:
                    cur.execute(
                        f"SELECT {_SELECT_COLS} FROM {_TABLE_NAME} "
                        "ORDER BY created_at DESC LIMIT %s",
                        (max(1, int(limit)),),
                    )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [_row_to_dict(r) for r in rows]

    def count_pending(self) -> int:
        """pending 水位（queued+running），submit 背压判断用。"""
        self._ensure_ready()
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT count(*) FROM {_TABLE_NAME} "
                    "WHERE status IN ('queued','running')")
                return int(cur.fetchone()[0])
        finally:
            conn.close()

    def health_status(self) -> str:
        """健康探针用：短连接探活（连接/建表失败 → error，绝不抛）。"""
        try:
            self.count_pending()
            return "ok"
        except Exception:  # noqa: BLE001  尽力检查，绝不抛
            return "error"


def _row_to_dict(row: tuple) -> dict:
    """DB 行 → API dict（时间转 ISO8601，JSON 串反序列化）。"""
    (task_id, skill_id, args_json, status, stage, progress, error,
     created_at, started_at, finished_at, tenant_id, output_paths) = row
    return {
        "task_id": task_id,
        "skill_id": skill_id,
        "args": json.loads(args_json) if args_json else {},
        "status": status,
        "stage": stage,
        "progress": float(progress or 0.0),
        "error": error,
        "created_at": created_at.isoformat() if created_at else None,
        "started_at": started_at.isoformat() if started_at else None,
        "finished_at": finished_at.isoformat() if finished_at else None,
        "tenant_id": tenant_id,
        "output_paths": json.loads(output_paths) if output_paths else None,
    }
