"""PostgreSQL 任务存储（TaskStore）——TaskManager 的持久层。

PgBouncer 事务模式适配（硬约束）：
- 进程内 ``ThreadedConnectionPool`` 借还连接（2026-10-31 SaaS 化改造：原先
  每操作新建 TCP 连接，在轮询风暴下是全链路最大单点）；池内连接对
  PgBouncer 而言仍是客户端槽，单语句短事务语义不变；
- autocommit 单语句事务，快进快出，不占 PgBouncer 服务端连接槽；
- 不用 LISTEN-NOTIFY / 服务端 prepared statement（psycopg2 默认客户端侧
  prepare，天然兼容）；advisory lock 仅用 **xact 级**（事务结束自动释放，
  事务池安全）——当前准入走单语句软水位，未用；
- 连接失败重试 1 次（瞬时抖动），仍失败抛 TaskStoreError（错误永不静默）。

决策记录：
- 建库：app 业务用户无 CREATEDB 权限（42501）；2026-09-08 起任务表迁入统一库
  ``rak`` 的 ``flowmind`` schema（ECO-ADR-0013，登录角色 flowmind 由管理员预建）。
  本模块只做**幂等建表**（IF NOT EXISTS）。
- 线程安全：连接获取/归还由池管理；建表标志用锁保护。
- TTL GC 语义：manager 只清 workdir 不删 DB 行（行保留供审计与状态查询）。
- 准入水位（2026-10-31）：``create_task_admitted`` 单语句完成「水位检查+插入」，
  READ COMMITTED 下并发提交可造成 ±N（N=瞬时并发数）的**软性超越**——GPU
  单 worker 排队模型下水位只是背压信号而非硬不变量，接受软超越换取零额外
  往返与无应用层锁。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from flowmind.tasks import TERMINAL_STATUSES, TaskStoreError

logger = logging.getLogger(__name__)

# 统一库 rak 的 flowmind 域（ECO-ADR-0013）：本表归 flowmind 仓独占，
# 与 go-kernel 的 mcp 域物理隔离（用户裁决：仓库独立、schema 分开）
_SCHEMA_NAME = "flowmind"
_TABLE_NAME = "tasks"

# 连接池上限：worker=1 + REST/MCP 并发 to_thread 峰值；PgBouncer 客户端槽预算内
_POOL_MIN = 2
_POOL_MAX = 8
# 池借连排队上限（秒）：高并发风暴下排队毫秒级；超时说明 PG 整体不可用
_LEASE_WAIT_SECONDS = 5.0

# 幂等建表 + 查询索引：
#   - (status, created_at DESC)：list_tasks 固定形态
#   - pending 部分索引：准入水位 count(*) 走 index-only scan（表百万行时仍是微秒级）
#   - (tenant_id, created_at DESC)：租户域查询/配额
#   - finished_at 部分索引：TTL GC 圈定
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
        output_paths TEXT,
        steps_json  TEXT
    )
    """,
    # 增量加列（幂等）：既有部署的 tasks 表没有 steps_json（2026-10-31 新增）
    f"""
    ALTER TABLE {_SCHEMA_NAME}.{_TABLE_NAME} ADD COLUMN IF NOT EXISTS steps_json TEXT
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_tasks_status_created
        ON {_SCHEMA_NAME}.{_TABLE_NAME} (status, created_at DESC)
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_tasks_pending
        ON {_SCHEMA_NAME}.{_TABLE_NAME} (created_at)
        WHERE status IN ('queued', 'running')
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_tasks_tenant_created
        ON {_SCHEMA_NAME}.{_TABLE_NAME} (tenant_id, created_at DESC)
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_tasks_finished
        ON {_SCHEMA_NAME}.{_TABLE_NAME} (finished_at)
        WHERE finished_at IS NOT NULL
    """,
]

_SELECT_COLS = (
    "task_id, skill_id, args_json, status, stage, progress, error, "
    "created_at, started_at, finished_at, tenant_id, output_paths, steps_json"
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
        self._pool = None
        self._pool_lock = threading.Lock()

    # ── 连接与建表 ──

    def _new_connection(self):
        """新建裸连接（池初始化/池内连接失效重建用；重试 1 次）。"""
        try:
            import psycopg2
        except ImportError as exc:
            raise TaskStoreError(
                "未安装 psycopg2（environment.yml 已含 psycopg2-binary；"
                "conda env update -n flowmind -f environment.yml 安装）") from exc
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

    def _get_pool(self):
        """惰性进程内连接池（双检锁；借还由 _pool_sem 排队，不抛 PoolError）。"""
        if self._pool is not None:
            return self._pool
        with self._pool_lock:
            if self._pool is None:
                try:
                    from psycopg2.pool import ThreadedConnectionPool
                except ImportError as exc:
                    raise TaskStoreError(
                        "未安装 psycopg2（environment.yml 已含 psycopg2-binary）") from exc
                if self._dsn:
                    self._pool = ThreadedConnectionPool(
                        _POOL_MIN, _POOL_MAX, self._dsn)
                else:
                    self._pool = ThreadedConnectionPool(
                        _POOL_MIN, _POOL_MAX, **self._conn_kwargs)
                # 借连排队闸：getconn 本身在池耗尽时立刻抛 PoolError（非阻塞），
                # 用同容量信号量让并发借连**礼貌等待**而非失败——高并发提交
                # 风暴下排队毫秒级，远优于 500。BoundedSemaphore 防 多次 release。
                self._pool_sem = threading.BoundedSemaphore(_POOL_MAX)
                logger.info("TaskStore 连接池就绪（min=%s max=%s）",
                            _POOL_MIN, _POOL_MAX)
            return self._pool

    @contextmanager
    def _lease(self):
        """从池借连接，用毕归还；池耗尽时排队等待（≤5s），异常连接丢弃重建。"""
        pool = self._get_pool()
        if not self._pool_sem.acquire(timeout=_LEASE_WAIT_SECONDS):
            raise TaskStoreError(
                f"PG 连接池等待超时（{_LEASE_WAIT_SECONDS}s，"
                f"并发 {_POOL_MAX} 连接全占用）")
        try:
            try:
                conn = pool.getconn()
            except Exception as exc:  # noqa: BLE001  池内连接初始化失败等
                logger.debug("PG 池借连失败（重试 1 次）: %s", exc)
                time.sleep(0.2)
                try:
                    conn = pool.getconn()
                except Exception as exc2:  # noqa: BLE001
                    raise TaskStoreError(f"PG 连接池借连失败: {exc2}") from exc2
            try:
                yield conn
            except Exception:
                try:
                    conn.rollback()
                except Exception:  # noqa: BLE001  坏连接：丢弃
                    pool.putconn(conn, close=True)
                    raise
                pool.putconn(conn)
                raise
            pool.putconn(conn)
        finally:
            self._pool_sem.release()

    def _ensure_ready(self) -> None:
        """幂等建表（进程内一次；并发下双检锁）。"""
        if self._ready:
            return
        with self._ready_lock:
            if self._ready:
                return
            with self._lease() as conn:
                with conn.cursor() as cur:
                    for stmt in _DDL:
                        cur.execute(stmt)
                conn.commit()
            self._ready = True
            logger.info("TaskStore 就绪（表 %s.%s）", _SCHEMA_NAME, _TABLE_NAME)

    # ── 写路径 ──

    def create_task(self, task_id: str, skill_id: str, args: dict,
                    tenant_id: str | None = None) -> None:
        """落 queued 任务（无水位检查；task_id 冲突属调用方 bug，直接抛）。"""
        self._ensure_ready()
        with self._lease() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"INSERT INTO {_TABLE_NAME} "
                        "(task_id, skill_id, args_json, status, tenant_id) "
                        "VALUES (%s, %s, %s, 'queued', %s)",
                        (task_id, skill_id,
                         json.dumps(args, ensure_ascii=False), tenant_id),
                    )
                conn.commit()
            except Exception as exc:  # noqa: BLE001  统一包装（错误永不裸抛）
                raise TaskStoreError(
                    f"create_task 失败 task={task_id}: {exc}") from exc

    def create_task_admitted(self, task_id: str, skill_id: str, args: dict,
                             tenant_id: str | None,
                             global_max: int, tenant_max: int) -> bool:
        """原子化准入：单语句完成「全局水位 + 租户水位检查 + 插入」。

        返回 False = 水位已满（调用方映射 429），True = 已落 queued。
        READ COMMITTED 下并发提交存在 ±N 软性超越（见模块 docstring）；
        tenant_id 为空（未启用鉴权）时跳过租户水位。

        取代旧的 Python 全局锁 + count_pending + INSERT 三步（提交全序列化
        于一把互斥锁，是 2026-10-31 之前的全局准入瓶颈）。
        """
        self._ensure_ready()
        with self._lease() as conn:
            try:
                with conn.cursor() as cur:
                    sql = (
                        f"INSERT INTO {_TABLE_NAME} "
                        "(task_id, skill_id, args_json, status, tenant_id) "
                        "SELECT %s, %s, %s, 'queued', %s WHERE "
                        f"(SELECT count(*) FROM {_TABLE_NAME} "
                        "WHERE status IN ('queued','running')) < %s"
                    )
                    params: list = [
                        task_id, skill_id,
                        json.dumps(args, ensure_ascii=False), tenant_id,
                        max(1, int(global_max)),
                    ]
                    if tenant_id:
                        sql += (
                            f" AND (SELECT count(*) FROM {_TABLE_NAME} "
                            "WHERE status IN ('queued','running') "
                            "AND tenant_id = %s) < %s"
                        )
                        params.extend([tenant_id, max(1, int(tenant_max))])
                    cur.execute(sql, params)
                    admitted = cur.rowcount == 1
                conn.commit()
                return admitted
            except Exception as exc:  # noqa: BLE001
                raise TaskStoreError(
                    f"create_task_admitted 失败 task={task_id}: {exc}") from exc

    def update_progress(self, task_id: str, stage: str, progress: float) -> None:
        """进度落库（高频调用：单 UPDATE 短事务；调用方负责按间隔合并）。"""
        self._ensure_ready()
        with self._lease() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"UPDATE {_TABLE_NAME} SET stage = %s, progress = %s "
                        "WHERE task_id = %s",
                        (stage, float(progress), task_id),
                    )
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                raise TaskStoreError(
                    f"update_progress 失败 task={task_id}: {exc}") from exc

    def update_steps(self, task_id: str, steps: list[dict]) -> None:
        """步骤全量快照落库（单行 JSON 覆盖；调用方只在状态变化时调用）。

        独立列而非逐行表：步骤快照是「当前面板」的读模型，前端每次都要全量，
        单行覆盖最省往返；历史审计走 MQTT 事件流（QoS1 + 终态 retain）。
        """
        self._ensure_ready()
        with self._lease() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"UPDATE {_TABLE_NAME} SET steps_json = %s WHERE task_id = %s",
                        (json.dumps(steps, ensure_ascii=False), task_id),
                    )
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                raise TaskStoreError(
                    f"update_steps 失败 task={task_id}: {exc}") from exc

    def set_status(self, task_id: str, status: str, *,
                   error: str | None = None,
                   output_paths: list[str] | None = None,
                   from_statuses: tuple[str, ...] | None = None) -> bool:
        """状态迁移（CAS 守卫）：running 记 started_at；终态记 finished_at。

        error/output_paths 传 None 时保留原值（COALESCE），终态可携带。

        from_statuses 非空时作为原子守卫（``AND status = ANY(%s)``）：
        仅当当前状态命中才迁移——终态 first-writer-wins，cancel 与 worker
        完成互相覆盖终态的竞态由单条 UPDATE 原子性消除。
        返回 True=本次迁移生效（rowcount==1）；False=状态不匹配（已被人
        先写，幂等不报错）；执行期错误仍抛 TaskStoreError（与 CAS False
        语义严格区分）。
        """
        self._ensure_ready()
        with self._lease() as conn:
            try:
                with conn.cursor() as cur:
                    sql = (
                        f"UPDATE {_TABLE_NAME} SET status = %s, "
                        "started_at = CASE WHEN %s = 'running' THEN now() ELSE started_at END, "
                        "finished_at = CASE WHEN %s IN "
                        "('succeeded','failed','cancelled','interrupted') "
                        "THEN now() ELSE finished_at END, "
                        "error = COALESCE(%s, error), "
                        "output_paths = COALESCE(%s, output_paths) "
                        "WHERE task_id = %s"
                    )
                    params: list = [
                        status, status, status,
                        error,
                        json.dumps(output_paths, ensure_ascii=False) if output_paths is not None else None,
                        task_id,
                    ]
                    if from_statuses:
                        sql += " AND status = ANY(%s)"
                        params.append(list(from_statuses))
                    cur.execute(sql, params)
                    migrated = cur.rowcount == 1
                conn.commit()
                return migrated
            except Exception as exc:  # noqa: BLE001
                raise TaskStoreError(
                    f"set_status 失败 task={task_id} -> {status}: {exc}") from exc

    def recover_running(self, include_queued: bool = False) -> int:
        """启动恢复：遗留 running（可选 queued）任务 → interrupted。

        默认只恢复 running（2026-10-31 起队列持久化：queued 行由 manager
        启动时重新入队执行，见 ``list_queued``——202 受理的任务不再因重启
        静默丢失）。include_queued=True 保留给「丢弃队列」的运维场景。
        """
        self._ensure_ready()
        statuses = ("queued", "running") if include_queued else ("running",)
        with self._lease() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"UPDATE {_TABLE_NAME} SET status = 'interrupted', finished_at = now() "
                        "WHERE status = ANY(%s)",
                        (list(statuses),),
                    )
                    recovered = cur.rowcount
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                raise TaskStoreError(f"recover_running 失败: {exc}") from exc
        if recovered:
            logger.warning("启动恢复：%s 个遗留任务标为 interrupted", recovered)
        return recovered

    def list_queued(self, limit: int = 100) -> list[dict]:
        """按 created_at 升序取 queued 任务（重启重入队用；FIFO 公平）。"""
        self._ensure_ready()
        with self._lease() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT {_SELECT_COLS} FROM {_TABLE_NAME} "
                        "WHERE status = 'queued' ORDER BY created_at LIMIT %s",
                        (max(1, int(limit)),),
                    )
                    rows = cur.fetchall()
            except Exception as exc:  # noqa: BLE001
                raise TaskStoreError(f"list_queued 失败: {exc}") from exc
        return [_row_to_dict(r) for r in rows]

    def delete_task(self, task_id: str) -> bool:
        """删除任务行（当前 GC 策略不调用；保留给管理接口）。"""
        self._ensure_ready()
        with self._lease() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {_TABLE_NAME} WHERE task_id = %s", (task_id,))
                    deleted = cur.rowcount > 0
                conn.commit()
                return deleted
            except Exception as exc:  # noqa: BLE001
                raise TaskStoreError(f"delete_task 失败 task={task_id}: {exc}") from exc

    # ── 读路径 ──

    def get_task(self, task_id: str, *, include_args: bool = True) -> dict | None:
        """单任务详情。

        include_args=False 时跳过 args_json 的 json 解析并不返回 args
        （轮询热路径：args 是调用方自己提交的已知内容，每次 poll 重复
        解析/传输是纯浪费；retry 等确需原参的调用方传 True）。
        """
        self._ensure_ready()
        with self._lease() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT {_SELECT_COLS} FROM {_TABLE_NAME} WHERE task_id = %s",
                        (task_id,),
                    )
                    row = cur.fetchone()
            except Exception as exc:  # noqa: BLE001
                raise TaskStoreError(f"get_task 失败 task={task_id}: {exc}") from exc
        if not row:
            return None
        return _row_to_dict(row, include_args=include_args)

    def list_tasks(self, status: str | None = None, limit: int = 100) -> list[dict]:
        """任务列表（created_at 倒序）；status=None 查全部。"""
        self._ensure_ready()
        with self._lease() as conn:
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
            except Exception as exc:  # noqa: BLE001
                raise TaskStoreError(
                    f"list_tasks 失败 status={status}: {exc}") from exc
        return [_row_to_dict(r) for r in rows]

    def count_pending(self) -> int:
        """pending 水位（queued+running）；走 pending 部分索引（index-only）。"""
        self._ensure_ready()
        with self._lease() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT count(*) FROM {_TABLE_NAME} "
                        "WHERE status IN ('queued','running')")
                    return int(cur.fetchone()[0])
            except Exception as exc:  # noqa: BLE001
                raise TaskStoreError(f"count_pending 失败: {exc}") from exc

    def list_expired(self, finished_before: datetime,
                     limit: int = 500) -> list[str]:
        """TTL GC 圈定：终态且 finished_at 早于 cutoff 的任务 id。

        单条 SQL 直接圈定（``status = ANY(终态) AND finished_at < cutoff``），
        替代旧的逐状态 limit=1000 全状态扫描；finished_at 升序返回
        （最老的先清），limit 防单轮清理过长（未清完的下一轮 GC 再接手）。
        """
        self._ensure_ready()
        with self._lease() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT task_id FROM {_TABLE_NAME} "
                        "WHERE status = ANY(%s) AND finished_at IS NOT NULL "
                        "AND finished_at < %s ORDER BY finished_at LIMIT %s",
                        (sorted(TERMINAL_STATUSES), finished_before,
                         max(1, int(limit))),
                    )
                    rows = cur.fetchall()
            except Exception as exc:  # noqa: BLE001
                raise TaskStoreError(
                    f"list_expired 失败 finished_before={finished_before}: {exc}") from exc
        return [r[0] for r in rows]

    def task_exists(self, task_id: str) -> bool:
        """task_id 是否存在 DB 行（GC 孤儿目录判定用；存在即不属孤儿）。"""
        self._ensure_ready()
        with self._lease() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT 1 FROM {_TABLE_NAME} WHERE task_id = %s LIMIT 1",
                        (task_id,),
                    )
                    return cur.fetchone() is not None
            except Exception as exc:  # noqa: BLE001
                raise TaskStoreError(f"task_exists 失败 task={task_id}: {exc}") from exc

    def health_status(self) -> str:
        """健康探针用：借池内连接探活（连接/建表失败 → error，绝不抛）。"""
        try:
            self.count_pending()
            return "ok"
        except Exception:  # noqa: BLE001  尽力检查，绝不抛
            return "error"


def _row_to_dict(row: tuple, *, include_args: bool = True) -> dict:
    """DB 行 → API dict（时间转 ISO8601；args 仅在需要时反序列化）。"""
    (task_id, skill_id, args_json, status, stage, progress, error,
     created_at, started_at, finished_at, tenant_id, output_paths,
     steps_json) = row
    out = {
        "task_id": task_id,
        "skill_id": skill_id,
        "status": status,
        "stage": stage,
        "progress": float(progress or 0.0),
        "error": error,
        "created_at": created_at.isoformat() if created_at else None,
        "started_at": started_at.isoformat() if started_at else None,
        "finished_at": finished_at.isoformat() if finished_at else None,
        "tenant_id": tenant_id,
        "output_paths": json.loads(output_paths) if output_paths else None,
        "steps": json.loads(steps_json) if steps_json else [],
    }
    if include_args:
        out["args"] = json.loads(args_json) if args_json else {}
    return out
