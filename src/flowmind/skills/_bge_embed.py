"""BGE 向量嵌入客户端（BAAI/bge-base-zh-v1.5，768 维）。

API 形状探测记录（2026-09-04，curl 实测）：

- 集群当时无 embedding/bge 部署（全集群 kubectl 无相关 svc/pod，
  LiteLLM 网关模型列表为空）——集群侧部署就绪前，本模块属禁用态。
- 开发机联调：用本机缓存模型 ``/srv/data/models/bge-base-zh-v1.5`` 起同形状
  服务（127.0.0.1:31997），实现两种业界通用形状：
    1) TEI（text-embeddings-inference）风格：
       POST /embed  {"inputs": ["t1", "t2"]}  →  [[768 float], ...]
    2) OpenAI 兼容：
       POST /v1/embeddings {"input": [...], "model": "..."}
                                    →  {"data": [{"embedding": [...], ...}]}
- 客户端先按 TEI 形状请求；非 200 或形状不符时回退 OpenAI 形状，
  哪个先成功即锁定（进程内记忆，避免每次双重请求）。

配置（配置源顺序：env → config.toml；均空 = **显式禁用**）：
- ``FLOWMIND_EMBEDDING_BASE_URL`` / config ``infra.embedding_base_url``：
  服务基址。两者均空时 ``embed_texts`` 抛 EmbedError（不回内置默认地址）；
  开发机联调可指向本机同形状服务（如 http://127.0.0.1:31997），集群部署
  注入集群内服务地址。
- 超时 30s（批量文本嵌入是秒级操作）。

失败语义：抛 EmbedError（category 字段与 errors.py 语义对齐），绝不吞、
绝不静默返回空向量——调用方（流水线尾部向量化）自行决定降级；
未配置属显式禁用态（category=environment，retriable=False）。
"""
from __future__ import annotations

import os
import threading

import requests


class EmbedError(Exception):
    """向量嵌入失败。category 语义与 errors.py 对齐。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


EMBED_DIM = 768
_TIMEOUT = 30.0

# 进程内锁定的 API 形状："tei" | "openai" | None（未探测）
_shape_lock = threading.Lock()
_api_shape: str | None = None


def _base_url() -> str:
    """服务基址解析（配置源顺序：env → config.toml；均空 = 未配置/禁用）。"""
    from flowmind.config import get_config

    return (os.environ.get("FLOWMIND_EMBEDDING_BASE_URL", "").strip()
            or get_config().infra.embedding_base_url.strip()).rstrip("/")


def is_configured() -> bool:
    """嵌入服务是否已配置（env 与 config 均空 = False，显式禁用）。"""
    return bool(_base_url())


def _post_embeddings(texts: list[str], shape: str) -> list[list[float]]:
    """按指定形状请求并解析向量；形状不符/HTTP 错误抛 EmbedError。"""
    url = f"{_base_url()}/embed" if shape == "tei" else f"{_base_url()}/v1/embeddings"
    if shape == "tei":
        payload: dict = {"inputs": texts}
    else:
        payload = {"input": texts, "model": "BAAI/bge-base-zh-v1.5"}
    try:
        resp = requests.post(url, json=payload, timeout=_TIMEOUT)
    except requests.exceptions.Timeout as exc:
        raise EmbedError("embedding 服务超时（environment）",
                         category="environment", retriable=True) from exc
    except requests.exceptions.ConnectionError as exc:
        raise EmbedError("embedding 服务不可达（environment）",
                         category="environment") from exc
    if resp.status_code != 200:
        raise EmbedError(
            f"embedding 服务 HTTP {resp.status_code}: {resp.text[:200]}",
            category="transient", retriable=resp.status_code >= 500)
    data = resp.json()
    if shape == "tei":
        vecs = data if isinstance(data, list) else None
    else:
        items = data.get("data") if isinstance(data, dict) else None
        vecs = [item.get("embedding") for item in items] if items else None
    if not isinstance(vecs, list) or not vecs:
        raise EmbedError(f"embedding 响应形状不符（{shape}）", category="unknown")
    for v in vecs:
        if not isinstance(v, list) or len(v) != EMBED_DIM:
            raise EmbedError(
                f"embedding 维度不符：期望 {EMBED_DIM}，实际 {len(v) if isinstance(v, list) else '?'}",
                category="unknown")
    return vecs


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量文本 → 768 维向量列表（顺序与输入一致）。

    未配置（服务基址空）抛 EmbedError（显式禁用，不静默降级）。
    空列表直接返回 []。首次调用自动探测 API 形状并进程内锁定。
    """
    if not texts:
        return []
    if not is_configured():
        raise EmbedError(
            "未配置 FLOWMIND_EMBEDDING_BASE_URL，向量嵌入已禁用"
            "（设置 FLOWMIND_EMBEDDING_BASE_URL 或 config infra.embedding_base_url"
            " 后可用）",
            category="environment")
    global _api_shape
    with _shape_lock:
        shape = _api_shape
    if shape is None:
        # 形状探测：TEI 优先，失败回退 OpenAI；成功即锁定
        try:
            vecs = _post_embeddings(texts, "tei")
            shape = "tei"
        except EmbedError:
            vecs = _post_embeddings(texts, "openai")
            shape = "openai"
        with _shape_lock:
            _api_shape = shape
        return vecs
    return _post_embeddings(texts, shape)
