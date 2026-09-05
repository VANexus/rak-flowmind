"""localize_search 技能：跨任务字幕语义检索（BGE 嵌入 + Milvus 近邻）。

数据来源：localize_video 成功路径尾部把 ASR 分段经 BGE（bge-base-zh-v1.5，
768 维）嵌入写入 Milvus collection ``localize_segments``（tasks/vectors，
FLOWMIND_VECTORIZE 默认开）。本技能把自然语言 query 嵌入为向量后做余弦
近邻检索，返回命中的字幕分段（任务 / 视频 / 时间轴 / 原文）。

典型用途：「找讲过 XX 的视频片段」「定位某产品卖点出现在哪个任务第几秒」。

错误语义（决策记录）：
- embedding / Milvus 服务不可用或未配置（地址均空 = 显式禁用）→ 异常上抛，
  invoke() 兑底为 ok=False 结构化错误，**不打 degraded**——检索是该技能
  的核心产出，降级返回空结果会误导 Agent 为「无命中」；异常消息脱敏
  （不泄漏内部 host/URI；未配置时消息指明待设置的环境变量）。
- Milvus 空库 → ok=True + 空 hits（合理空态，与「没有匹配」同语义）。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.skill import skill
from flowmind.skills import _bge_embed
from flowmind.tasks import vectors

_VERSION = "0.1.0"


# ── 入参 ──

class SearchInput(BaseModel):
    """检索入参。"""
    query: str = Field(..., min_length=1, description="自然语言查询（匹配字幕文本语义）")
    top_k: int = Field(default=5, ge=1, le=50, description="返回命中数上限（1-50）")
    task_id: str | None = Field(
        default=None, description="限定单任务范围；None=跨全部已向量化任务",
    )


# ── 出参 ──

class SearchHit(BaseModel):
    """单个命中分段。"""
    task_id: str
    video_name: str
    seg_index: int
    start_sec: float
    end_sec: float
    text: str
    score: float           # 余弦相似度（COSINE，越大越相似）


class SearchReport(BaseModel):
    """检索业务载荷。"""
    query: str
    top_k: int
    task_id: str | None    # 实际使用的范围过滤（None=全库）
    hits: list[SearchHit]
    count: int


# ── 入口 ──

@skill(id="localize_search", name="字幕语义检索", version=_VERSION)
def localize_search(inp: SearchInput) -> SkillOutput[SearchReport]:
    """query → BGE 嵌入 → Milvus 余弦近邻 → 命中分段列表。

    数据流：query → _bge_embed.embed_texts → vectors.search（COSINE，
    可选 task_id 过滤）→ SearchReport + ReasoningChain → SkillResult 信封。
    空库返回空 hits（ok=True 合理空态）；服务不可用抛结构化错误（ok=False）。
    """
    try:
        query_vecs = _bge_embed.embed_texts([inp.query])
    except Exception as exc:  # noqa: BLE001  统一脱敏上抛（错误永不静默）
        # 内层消息已脱敏且未配置时含待设置的环境变量名，透传提高可修性
        raise RuntimeError(
            f"向量嵌入服务不可用（{type(exc).__name__}: {exc}），检索失败"
        ) from exc

    try:
        raw_hits = vectors.search(
            query_vecs[0], top_k=inp.top_k, task_id=inp.task_id)
    except Exception as exc:  # noqa: BLE001  统一脱敏上抛（错误永不静默）
        raise RuntimeError(
            f"向量检索服务不可用（{type(exc).__name__}: {exc}），检索失败"
        ) from exc

    hits = [
        SearchHit(
            task_id=str(h.get("task_id") or ""),
            video_name=str(h.get("video_name") or ""),
            seg_index=int(h.get("seg_index") or 0),
            start_sec=float(h.get("start_sec") or 0.0),
            end_sec=float(h.get("end_sec") or 0.0),
            text=str(h.get("text") or ""),
            score=float(h.get("distance") or 0.0),
        )
        for h in raw_hits
    ]

    scope = f"任务 {inp.task_id}" if inp.task_id else "全库"
    conclusion = (
        f"检索「{inp.query}」命中 {len(hits)} 个字幕分段"
        f"（范围：{scope}，top_k={inp.top_k}）。"
    )
    if hits:
        best = max(hits, key=lambda h: h.score)
        risk_note = (
            f"最高分 {best.score:.3f}：{best.video_name} 第 {best.start_sec:.1f}-"
            f"{best.end_sec:.1f}s；score 为余弦相似度，建议结合阈值人工确认。"
        )
    else:
        risk_note = "无命中：可能是检索范围过窄、top_k 过小，或库内尚无相关字幕。"
    chain = ReasoningChain(
        conclusion=conclusion,
        triggered_rules=[], evidence=[],
        causal_analysis=(
            f"query 经 bge-base-zh-v1.5（768 维）嵌入 → Milvus HNSW/COSINE "
            f"近邻（{'task_id 过滤' if inp.task_id else '无过滤'}）"
        ),
        risk_note=risk_note,
    )
    return SkillOutput(
        data=SearchReport(
            query=inp.query,
            top_k=inp.top_k,
            task_id=inp.task_id,
            hits=hits,
            count=len(hits),
        ),
        reasoning=[chain],
        confidence=1.0 if hits else 0.6,
        sample_size=len(hits),
    )
