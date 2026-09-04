"""localize_search 技能演示 —— 跨任务字幕语义检索（BGE 嵌入 + Milvus 近邻）。

运行：PYTHONPATH=<repo>/src conda run -n flowmind python examples/localize_search_demo.py

展示：
1. discover() 自动字段发现
2. happy path：query → 命中 2 个字幕分段（全库范围）
3. task_id 过滤：限定单任务范围
4. 空态：Milvus 空库 → ok=True + 空 hits（合理空态，非错误）
5. error：嵌入服务不可用 / 检索服务不可用 → ok=False（核心能力缺失，
   不打 degraded，避免误导 Agent 为「无命中」）

mock 方式：patch 技能模块引用的 _bge_embed.embed_texts 与 vectors.search
（本文件内 fake 实现），不依赖 GPU 嵌入服务 / Milvus 集群。
"""

from __future__ import annotations

import flowmind.skills  # noqa: F401  触发 @skill 注册
import flowmind.skills.localize_search as lsearch
from flowmind.discover import field_names
from flowmind.skill import invoke
from flowmind.skills._bge_embed import EmbedError
from flowmind.tasks.vectors import VectorStoreError


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ── fake：嵌入 + 向量检索 ───────────────────────────────────────────

def _fake_embed_ok(texts: list[str]) -> list[list[float]]:
    return [[0.1] * 768 for _ in texts]


_RAW_HITS = [
    {
        "id": 101, "distance": 0.92, "task_id": "task-001",
        "video_name": "promo-a.mp4", "seg_index": 7,
        "start_sec": 42.5, "end_sec": 47.0,
        "text": "本产品支持一键跨语言配音",
    },
    {
        "id": 205, "distance": 0.87, "task_id": "task-002",
        "video_name": "promo-b.mp4", "seg_index": 12,
        "start_sec": 118.2, "end_sec": 124.9,
        "text": "配音支持泰语日语韩语等八个语种",
    },
]


def _install(embed, search) -> None:
    """patch 技能模块引用的 _bge_embed.embed_texts 与 vectors.search。"""
    lsearch._bge_embed.embed_texts = embed
    lsearch.vectors.search = search


def _print(r) -> None:
    print(f"  ok      : {r.ok}")
    print(f"  query   : {r.data.query}  top_k={r.data.top_k}"
          f"  范围={r.data.task_id or '全库'}")
    for h in r.data.hits:
        print(f"    • [{h.task_id}] {h.video_name} #{h.seg_index}"
              f" {h.start_sec:.1f}-{h.end_sec:.1f}s score={h.score:.2f}")
        print(f"      {h.text}")
    print(f"  count   : {r.data.count}")
    print(f"  推理    : {r.reasoning[0].conclusion}")


def main() -> None:
    section("0) discover('localize_search') —— Agent 自查字段")
    for p, names in field_names("localize_search").items():
        print(f"  {p}: {names}")

    section("1) Happy path：全库检索命中 2 个分段")
    seen_kwargs = {}

    def fake_search(query_vector, top_k=5, task_id=None):
        seen_kwargs.update(top_k=top_k, task_id=task_id,
                           dim=len(query_vector))
        return list(_RAW_HITS)

    _install(_fake_embed_ok, fake_search)
    r = invoke("localize_search", {"query": "找讲一键配音的视频片段"})
    _print(r)
    check(r.ok and r.data.count == 2, "应命中 2 个分段")
    check(seen_kwargs["dim"] == 768, "query 应嵌入为 768 维向量")
    check(seen_kwargs["top_k"] == 5 and seen_kwargs["task_id"] is None,
          "默认 top_k=5 且全库检索")
    h0 = r.data.hits[0]
    check(h0.task_id == "task-001" and h0.video_name == "promo-a.mp4"
          and h0.seg_index == 7 and abs(h0.start_sec - 42.5) < 1e-6
          and abs(h0.end_sec - 47.0) < 1e-6, "hit 字段应完整透出")
    check(abs(h0.score - 0.92) < 1e-6, "score 应取 Milvus distance")

    section("2) task_id 过滤：限定单任务范围")
    calls = []

    def fake_search_filtered(query_vector, top_k=5, task_id=None):
        calls.append(task_id)
        return [h for h in _RAW_HITS if h["task_id"] == task_id]

    _install(_fake_embed_ok, fake_search_filtered)
    r = invoke("localize_search",
               {"query": "配音语种", "top_k": 10, "task_id": "task-002"})
    _print(r)
    check(r.ok and r.data.count == 1 and r.data.hits[0].task_id == "task-002",
          "task_id 过滤应只回单任务命中")
    check(calls == ["task-002"], "过滤参数应透传 vectors.search")
    check(r.data.top_k == 10, "显式 top_k 应生效")

    section("3) 空态：Milvus 空库 → ok=True + 空 hits（合理空态）")
    _install(_fake_embed_ok, lambda qv, top_k=5, task_id=None: [])
    r = invoke("localize_search", {"query": "不存在的主题"})
    _print(r)
    check(r.ok and r.data.count == 0 and r.data.hits == [],
          "空库应 ok=True + 空 hits（不是错误）")
    check("无命中" in r.reasoning[0].risk_note, "risk_note 应解释空态成因")

    section("4) error：嵌入服务不可用 → ok=False（不打 degraded）")
    def fake_embed_down(texts):
        raise EmbedError("connection refused")

    _install(fake_embed_down, lambda qv, top_k=5, task_id=None: _RAW_HITS)
    r = invoke("localize_search", {"query": "任意"})
    print(f"  ok          : {r.ok}")
    print(f"  error.code  : {r.error.code if r.error else None}")
    print(f"  error.msg   : {r.error.message if r.error else None}")
    check(r.ok is False, "嵌入服务不可用应 ok=False（核心能力缺失）")
    check(r.error is not None and r.error.code == "INTERNAL",
          "异常经 invoke 兜底为 INTERNAL")
    check("向量嵌入服务不可用" in (r.error.message or ""),
          "message 应脱敏并说明嵌入服务不可用")

    section("5) error：检索服务不可用 → ok=False（不打 degraded）")
    def fake_search_down(query_vector, top_k=5, task_id=None):
        raise VectorStoreError("milvus unavailable")

    _install(_fake_embed_ok, fake_search_down)
    r = invoke("localize_search", {"query": "任意"})
    print(f"  ok          : {r.ok}")
    print(f"  error.code  : {r.error.code if r.error else None}")
    print(f"  error.msg   : {r.error.message if r.error else None}")
    check(r.ok is False, "检索服务不可用应 ok=False")
    check("向量检索服务不可用" in (r.error.message or ""),
          "message 应脱敏并说明检索服务不可用")

    print("\n✅ localize_search_demo 全部通过")


if __name__ == "__main__":
    main()
