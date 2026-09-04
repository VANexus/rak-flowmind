"""Milvus 字幕分段向量库（pymilvus MilvusClient，懒 import）。

collection ``localize_segments``：每个本地化任务的 ASR 分段向量，
供跨任务语义检索（"找讲过 XX 的视频片段"）。

schema（768 维对齐 bge-base-zh-v1.5）：
    segment_id  INT64  PK（auto_id=False，生成侧保证唯一）
    task_id     VARCHAR(64)
    video_name  VARCHAR(512)
    seg_index   INT64
    start_sec   FLOAT
    end_sec     FLOAT
    text        VARCHAR(4096)
    vector      FLOAT_VECTOR dim=768
索引：HNSW + COSINE（余弦相似；bge 归一化向量下等价内积）。

URI：env ``FLOWMIND_MILVUS_URI`` → config ``infra.milvus_uri`` → 内置默认
（开发机 mesh NodePort http://100.121.213.4:31953；集群内部注入
http://milvus.agentic.svc:19530）。

失败语义：连接/写入/检索失败抛 VectorStoreError（重试 2 次），
绝不吞——调用方（流水线尾部向量化）自行决定降级。
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid

logger = logging.getLogger(__name__)

COLLECTION = "localize_segments"
DIM = 768

_OUTPUT_FIELDS = ["task_id", "video_name", "seg_index", "start_sec", "end_sec", "text"]

_client = None
_ensured = False
_lock = threading.Lock()


class VectorStoreError(Exception):
    """Milvus 向量库不可用/操作失败。"""


def _uri() -> str:
    """连接地址解析（配置源顺序：env → config.toml → 内置默认）。"""
    from flowmind.config import get_config

    return (os.environ.get("FLOWMIND_MILVUS_URI", "").strip()
            or get_config().infra.milvus_uri.strip()
            or "http://100.121.213.4:31953")


def _get_client():
    """惰性单例客户端（MilvusClient gRPC 连接线程安全复用）。"""
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise VectorStoreError(
                "未安装 pymilvus（本阶段验证临时安装：pip install pymilvus；"
                "阶段 5 进 environment.yml）") from exc
        try:
            _client = MilvusClient(uri=_uri(), timeout=10)
        except Exception as exc:
            raise VectorStoreError(
                f"Milvus 连接失败（{_uri()}）: {type(exc).__name__}: {exc}") from exc
        return _client


def ensure_collection() -> None:
    """幂等建 collection + HNSW 索引 + load（进程内一次）。"""
    global _ensured
    if _ensured:
        return
    with _lock:
        if _ensured:
            return
        from pymilvus import DataType, MilvusClient

        client = _get_client()
        if not client.has_collection(COLLECTION):
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("segment_id", DataType.INT64, is_primary=True)
            schema.add_field("task_id", DataType.VARCHAR, max_length=64)
            schema.add_field("video_name", DataType.VARCHAR, max_length=512)
            schema.add_field("seg_index", DataType.INT64)
            schema.add_field("start_sec", DataType.FLOAT)
            schema.add_field("end_sec", DataType.FLOAT)
            schema.add_field("text", DataType.VARCHAR, max_length=4096)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="vector", index_type="HNSW", metric_type="COSINE",
                params={"M": 16, "efConstruction": 200},
            )
            client.create_collection(COLLECTION, schema=schema, index_params=index_params)
            logger.info("Milvus collection 已创建：%s（dim=%s, HNSW/COSINE）", COLLECTION, DIM)
        client.load_collection(COLLECTION)  # 幂等（已 load 为 no-op）
        _ensured = True


def _escape(value: str) -> str:
    """Milvus 布尔表达式字符串转义（task_id/video_name 防注入）。"""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _new_segment_id() -> int:
    """63 位随机 int64 主键（uuid4 空间压缩，碰撞概率可忽略）。"""
    return uuid.uuid4().int & ((1 << 63) - 1)


def upsert_task_segments(task_id: str, segments: list[dict]) -> int:
    """按 task_id 幂等 upsert：先 delete 后 insert（重试安全）。

    segments 行字段：text / start_sec / end_sec / vector 必填；
    video_name / seg_index 可选（缺省 "" / 行号）。返回写入行数。
    """
    if not segments:
        return 0
    last: Exception | None = None
    for attempt in (1, 2):
        try:
            client = _get_client()
            ensure_collection()
            client.delete(COLLECTION, filter=f'task_id == "{_escape(task_id)}"')
            rows = []
            for i, seg in enumerate(segments):
                rows.append({
                    "segment_id": _new_segment_id(),
                    "task_id": task_id,
                    "video_name": str(seg.get("video_name", "")),
                    "seg_index": int(seg.get("seg_index", i)),
                    "start_sec": float(seg.get("start_sec", 0.0)),
                    "end_sec": float(seg.get("end_sec", 0.0)),
                    "text": str(seg.get("text", ""))[:4000],
                    "vector": [float(x) for x in seg["vector"]],
                })
            # 分批插入（单批上限留余量，避免超大任务一次性提交）
            for i in range(0, len(rows), 500):
                client.insert(COLLECTION, data=rows[i:i + 500])
            return len(rows)
        except Exception as exc:  # noqa: BLE001  统一重试
            last = exc
            logger.debug("Milvus upsert 失败（第 %s 次）: %s", attempt, exc)
            time.sleep(0.5)
    raise VectorStoreError(
        f"Milvus upsert 失败（重试 2 次）: {type(last).__name__}: {last}") from last


def search(query_vector: list[float], top_k: int = 5,
           task_id: str | None = None) -> list[dict]:
    """向量检索。task_id 限定单任务范围；返回 [{id, distance, task_id, ...}]。"""
    client = _get_client()
    ensure_collection()
    flt = f'task_id == "{_escape(task_id)}"' if task_id else None
    results = client.search(
        COLLECTION,
        data=[[float(x) for x in query_vector]],
        limit=max(1, int(top_k)),
        filter=flt,
        output_fields=_OUTPUT_FIELDS,
    )
    hits: list[dict] = []
    for batch in results:
        for hit in batch:
            entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
            hits.append({
                "id": hit.get("id") if isinstance(hit, dict) else None,
                "distance": hit.get("distance") if isinstance(hit, dict) else None,
                **{k: entity.get(k) for k in _OUTPUT_FIELDS},
            })
    return hits


def reset_cache_for_tests() -> None:
    """清空进程内客户端/建表标志（仅测试用；生产勿调）。"""
    global _client, _ensured
    with _lock:
        _client = None
        _ensured = False


def health_status() -> str:
    """健康探针用：连接状态尽力检查（绝不抛、绝不建新连接）。

    unverified = 尚未使用过（惰性连接未触发）；ok / error = 已有连接
    的实际可用性。连接建立本身可能阻塞，健康检查不代建。
    """
    if _client is None:
        return "unverified"
    try:
        _client.list_collections()
        return "ok"
    except Exception:  # noqa: BLE001  尽力检查，绝不抛
        return "error"
