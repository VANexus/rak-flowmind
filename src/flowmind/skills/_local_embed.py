"""本地向量嵌入：sentence-transformers + BAAI/bge-small-zh-v1.5（GPU）。

架构升级（GPU 化）：feishu_kb 检索从 BM25+TF-IDF 双路升级为三路召回
（+ 向量余弦），FP32 小模型在 P104-100（Pascal 6.1）上无压力。
无云端等价路径 —— 库不可用时调用方回落双路并如实描述（auto 语义）。

lazy import：sentence-transformers/torch 未安装时 embed_available()=False；
显式 encode 调用抛 EmbedError(category="environment")。
模型实例进程内单例缓存。模型缓存/下载已配在 ~/.bashrc：HF_HOME=/srv/data/models + 代理 127.0.0.1:7890，import 即生效。
"""
from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_DEVICE = "cuda"


class EmbedError(Exception):
    """本地嵌入调用失败。category/retriable 语义与 errors.py 对齐。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def embed_available() -> bool:
    """sentence-transformers + torch 是否可导入（供 auto 判断；测试可替换）。"""
    try:
        import sentence_transformers  # noqa: F401  # 导入探测本身即目的，勿删
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# 进程内单例：key=(model, device)
_models: dict[tuple, object] = {}


def _get_model(model: str, device: str):
    key = (model, device)
    if key not in _models:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbedError(
                "未安装 sentence-transformers（environment.yml 已含，conda env update 后可用）",
                category="environment",
            ) from exc
        try:
            _models[key] = SentenceTransformer(model, device=device)
        except Exception as exc:
            raise EmbedError(
                f"嵌入模型加载失败（{model}@{device}）: {type(exc).__name__}",
                category="environment",
            ) from exc
    return _models[key]


def encode(
    texts: list[str], *,
    model: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
) -> np.ndarray:
    """文本列表 → L2 归一化嵌入矩阵 (n, dim) float32（余弦 = 内积）。"""
    if not texts:
        return np.zeros((0, 1), dtype="float32")
    st_model = _get_model(model, device)
    try:
        vecs = st_model.encode(
            list(texts), batch_size=32, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")
    except EmbedError:
        raise
    except Exception as exc:
        raise EmbedError(
            f"本地嵌入推理异常: {type(exc).__name__}", category="environment",
        ) from exc
