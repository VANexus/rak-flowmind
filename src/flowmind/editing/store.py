"""ProjectStore：剪辑项目持久层（PG 优先，文件回落）。

PG 表 ``flowmind.editing_projects``：与 tasks 表同 schema 域（ECO-ADR-0013），
短连接 + autocommit 单语句（PgBouncer 事务模式，模式复用 tasks/store.py 的
约束）。连接规格解析复用 tasks.store._resolve_conn_spec。

回落语义（allow_file_store_fallback=True 时）：PG 任一步失败 → 记 warning
并切到 ``data_dir/editing/<project_id>/project.json`` 文件存储；回落行为写入
ProjectRecord.storage="file"，对用户可见（不静默降级成"看起来像 PG"）。
生产（allow_file_store_fallback=False）PG 失败即显式抛 ProjectStoreError。

素材库索引同理：PG 表 ``flowmind.editing_assets``；文件模式落
``data_dir/editing/assets_index.json``。
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from flowmind.editing.models import AssetRecord, ProjectRecord

logger = logging.getLogger(__name__)

_SCHEMA = "flowmind"
_TABLE_PROJECTS = "editing_projects"
_TABLE_ASSETS = "editing_assets"

_DDL = [
    f"""
    CREATE TABLE IF NOT EXISTS {_SCHEMA}.{_TABLE_PROJECTS} (
        project_id  TEXT PRIMARY KEY,
        tenant_id   TEXT,
        state       TEXT NOT NULL,
        stage       TEXT NOT NULL,
        payload     TEXT NOT NULL,
        created_at  TIMESTAMPTZ DEFAULT now(),
        updated_at  TIMESTAMPTZ DEFAULT now()
    )
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_editing_projects_state_created
        ON {_SCHEMA}.{_TABLE_PROJECTS} (state, created_at DESC)
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {_SCHEMA}.{_TABLE_ASSETS} (
        asset_id    TEXT PRIMARY KEY,
        tenant_id   TEXT,
        kind        TEXT NOT NULL,
        path        TEXT NOT NULL,
        tags        TEXT NOT NULL DEFAULT '[]',
        project_id  TEXT,
        created_at  TIMESTAMPTZ DEFAULT now()
    )
    """,
]


class ProjectStoreError(RuntimeError):
    """剪辑项目存储不可用（PG 与文件回落均失败）。"""


class ProjectStore:
    """项目 + 素材库持久层。后端在首次操作时确定（pg 或 file）。"""

    def __init__(self, *, data_dir: str | None = None,
                 allow_file_fallback: bool | None = None) -> None:
        from flowmind.config import load_config
        cfg = load_config().editing
        self._data_dir = Path(
            data_dir or load_config().localizer.data_dir).expanduser().resolve()
        self._allow_fallback = (
            cfg.allow_file_store_fallback
            if allow_file_fallback is None else allow_file_fallback)
        self._backend: str | None = None      # None=未定 / "pg" / "file"
        self._conn_spec: tuple[str | None, dict[str, Any]] | None = None
        self._ddl_done = False
        self._lock = threading.Lock()
        self._file_root = self._data_dir / "editing"

    # ── 后端选择 ──

    def _ensure_backend(self) -> str:
        if self._backend is not None:
            return self._backend
        with self._lock:
            if self._backend is not None:
                return self._backend
            try:
                from flowmind.tasks.store import _resolve_conn_spec
                self._conn_spec = _resolve_conn_spec()
                self._connect_exec("SELECT 1")
                self._ensure_ddl()
                self._backend = "pg"
                logger.info("ProjectStore 后端：PostgreSQL")
            except Exception as exc:  # noqa: BLE001 PG 失败原因多样，统一评估回落
                if not self._allow_fallback:
                    raise ProjectStoreError(
                        f"PostgreSQL 不可用且文件回落已禁用（生产配置）：{exc}"
                    ) from exc
                logger.warning(
                    "ProjectStore 回落文件存储（开发/demo 语义，storage=file "
                    "会记录在项目里）：PG 失败原因 %s: %s",
                    type(exc).__name__, exc)
                self._backend = "file"
        return self._backend

    def _connect_exec(self, sql: str, params: tuple = ()) -> None:
        """短连接执行单语句（PgBouncer 事务模式；失败重试 1 次）。"""
        import psycopg2  # 懒 import：文件模式不需要装 psycopg2

        assert self._conn_spec is not None
        dsn, kwargs = self._conn_spec
        for attempt in (1, 2):
            try:
                conn = psycopg2.connect(dsn, **kwargs) if dsn else psycopg2.connect(**kwargs)
                try:
                    conn.autocommit = True
                    with conn.cursor() as cur:
                        cur.execute(sql, params)
                finally:
                    conn.close()
                return
            except Exception:  # noqa: BLE001 瞬时抖动重试 1 次后抛
                if attempt == 1:
                    continue
                raise

    def _ensure_ddl(self) -> None:
        if self._ddl_done:
            return
        for stmt in _DDL:
            self._connect_exec(stmt)
        self._ddl_done = True

    # ── 项目 ──

    def save_project(self, rec: ProjectRecord) -> None:
        """落库项目记录。

        task_ids 单调合并：流水线在 worker 里持续推进项目状态，而
        edit_project/edit_review 在 submit 后补记续跑任务 id——两侧并发
        写时 task_ids 只增不减（防后写方把先写方追加的任务历史冲掉）。
        state/stages 仍按 last-writer-wins（调用方约定：回存前先重载）。
        """
        existing = self.get_project(rec.project_id)
        if existing is not None:
            merged = list(existing.task_ids)
            merged += [t for t in rec.task_ids if t not in merged]
            rec.task_ids = merged
        rec.touch()
        backend = self._ensure_backend()
        if backend == "pg":
            sql = (
                f"INSERT INTO {_SCHEMA}.{_TABLE_PROJECTS} "
                "(project_id, tenant_id, state, stage, payload, updated_at) "
                "VALUES (%s,%s,%s,%s,%s,now()) "
                "ON CONFLICT (project_id) DO UPDATE SET "
                "tenant_id=EXCLUDED.tenant_id, state=EXCLUDED.state, "
                "stage=EXCLUDED.stage, payload=EXCLUDED.payload, "
                "updated_at=now()"
            )
            self._connect_exec(sql, (
                rec.project_id, rec.tenant_id, rec.state, rec.stage,
                rec.model_dump_json()))
            return
        path = self._file_project_path(rec.project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(rec.model_dump_json(), encoding="utf-8")
        tmp.replace(path)   # 原子替换，防半截 JSON

    def get_project(self, project_id: str) -> ProjectRecord | None:
        backend = self._ensure_backend()
        if backend == "pg":
            import psycopg2
            assert self._conn_spec is not None
            dsn, kwargs = self._conn_spec
            conn = psycopg2.connect(dsn, **kwargs) if dsn else psycopg2.connect(**kwargs)
            try:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT payload FROM {_SCHEMA}.{_TABLE_PROJECTS} "
                        "WHERE project_id=%s", (project_id,))
                    row = cur.fetchone()
            finally:
                conn.close()
            return ProjectRecord.model_validate_json(row[0]) if row else None
        path = self._file_project_path(project_id)
        if not path.exists():
            return None
        try:
            return ProjectRecord.model_validate_json(
                path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 半截/损坏文件显式报错
            raise ProjectStoreError(f"项目文件损坏 {path}: {exc}") from exc

    def list_projects(self, *, tenant_id: str | None = None,
                      limit: int = 50) -> list[ProjectRecord]:
        backend = self._ensure_backend()
        rows: list[ProjectRecord] = []
        if backend == "pg":
            import psycopg2
            assert self._conn_spec is not None
            dsn, kwargs = self._conn_spec
            conn = psycopg2.connect(dsn, **kwargs) if dsn else psycopg2.connect(**kwargs)
            try:
                conn.autocommit = True
                with conn.cursor() as cur:
                    # 租户过滤必须进 WHERE：先 LIMIT 后过滤会让整页被其他
                    # 租户占满，自己的项目被挤出结果
                    if tenant_id is not None:
                        cur.execute(
                            f"SELECT payload FROM {_SCHEMA}.{_TABLE_PROJECTS} "
                            "WHERE tenant_id = %s "
                            "ORDER BY created_at DESC LIMIT %s",
                            (tenant_id, limit))
                    else:
                        cur.execute(
                            f"SELECT payload FROM {_SCHEMA}.{_TABLE_PROJECTS} "
                            "ORDER BY created_at DESC LIMIT %s", (limit,))
                    rows = [ProjectRecord.model_validate_json(r[0])
                            for r in cur.fetchall()]
            finally:
                conn.close()
        elif self._file_root.exists():
            paths = sorted(self._file_root.glob("*/project.json"),
                           key=lambda x: x.stat().st_mtime, reverse=True)
            for p in paths:
                if len(rows) >= limit:
                    break
                try:
                    rec = ProjectRecord.model_validate_json(
                        p.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001 坏文件跳过并告警（列表路径）
                    logger.warning("跳过损坏项目文件 %s", p, exc_info=True)
                    continue
                if tenant_id is not None and rec.tenant_id != tenant_id:
                    continue
                rows.append(rec)
        return rows

    def _file_project_path(self, project_id: str) -> Path:
        return self._file_root / project_id / "project.json"

    # ── 素材库 ──

    def save_asset(self, rec: AssetRecord) -> None:
        backend = self._ensure_backend()
        if backend == "pg":
            sql = (
                f"INSERT INTO {_SCHEMA}.{_TABLE_ASSETS} "
                "(asset_id, tenant_id, kind, path, tags, project_id) "
                "VALUES (%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (asset_id) DO UPDATE SET "
                "tenant_id=EXCLUDED.tenant_id, kind=EXCLUDED.kind, "
                "path=EXCLUDED.path, tags=EXCLUDED.tags, "
                "project_id=EXCLUDED.project_id"
            )
            self._connect_exec(sql, (
                rec.id, rec.tenant_id, rec.kind, rec.path,
                json.dumps(rec.tags, ensure_ascii=False), rec.project_id))
            return
        index = self._file_asset_index()
        index[rec.id] = rec.model_dump()
        self._write_file_asset_index(index)

    def list_assets(self, *, tenant_id: str | None = None,
                    kind: str | None = None, limit: int = 200) -> list[AssetRecord]:
        backend = self._ensure_backend()
        rows: list[AssetRecord] = []
        if backend == "pg":
            import psycopg2
            assert self._conn_spec is not None
            dsn, kwargs = self._conn_spec
            conn = psycopg2.connect(dsn, **kwargs) if dsn else psycopg2.connect(**kwargs)
            try:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT asset_id, tenant_id, kind, path, tags, project_id, "
                        f"created_at FROM {_SCHEMA}.{_TABLE_ASSETS} "
                        "ORDER BY created_at DESC LIMIT %s", (limit,))
                    for r in cur.fetchall():
                        rows.append(AssetRecord(
                            id=r[0], tenant_id=r[1], kind=r[2], path=r[3],
                            tags=json.loads(r[4] or "[]"), project_id=r[5],
                            created_at=r[6].isoformat() if r[6] else "",
                        ))
            finally:
                conn.close()
        else:
            for item in self._file_asset_index().values():
                try:
                    rows.append(AssetRecord.model_validate(item))
                except Exception:  # noqa: BLE001 坏条目跳过
                    logger.warning("跳过损坏素材条目 %s", item, exc_info=True)
        if tenant_id is not None:
            rows = [r for r in rows if r.tenant_id == tenant_id]
        if kind is not None:
            rows = [r for r in rows if r.kind == kind]
        return rows

    def _file_asset_index_path(self) -> Path:
        return self._file_root / "assets_index.json"

    def _file_asset_index(self) -> dict[str, dict]:
        path = self._file_asset_index_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 索引损坏重建（登记语义允许）
            logger.warning("素材索引损坏，按空索引重建：%s", path, exc_info=True)
            return {}

    def _write_file_asset_index(self, index: dict[str, dict]) -> None:
        path = self._file_asset_index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(path)


# 惰性单例（与 TaskManager 同风格）：同进程共享一个后端判定结果。
_store_lock = threading.Lock()
_store: ProjectStore | None = None


def get_project_store() -> ProjectStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ProjectStore()
    return _store
