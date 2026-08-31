"""alibaba_product_recommend 技能：结合关键词趋势的今日推荐商品 TOP5。

按偏好（social=内容传播力 / alibaba=关键词热度与B端转化 / mix=综合）给商品打分排序，
推荐理由必须引用具体关键词（如「xxx 关键词热度 Top1」「yyy 关键词涨幅 Top1」）。
错误契约：普通 raise（invoke() 套信封为 INTERNAL）；无 key 显式报错。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import build_chain
from flowmind.skills._llm_client import llm_json
from flowmind.skills._secrets import get_api_key

_VERSION = "0.1.0"

Preference = Literal["social", "alibaba", "mix"]

_PREF_HINT = {
    "social": "社媒侧：优先内容传播力强、视觉表现好、易出爆款内容的商品",
    "alibaba": "国际站侧：优先关键词热度高、B端采购转化潜力大的商品",
    "mix": "综合：平衡内容传播力与B端转化潜力",
}


class RecommendInput(BaseModel):
    """推荐入参。"""
    preference: Preference = Field(description="偏好：social / alibaba / mix")
    products: list[dict] = Field(default_factory=list, description="商品摘要列表")
    trend_keywords: list[dict] = Field(default_factory=list, description="关键词趋势")
    longtail_keywords: list[dict] = Field(default_factory=list, description="长尾词")
    max_items: int = Field(default=5, ge=1, le=10)


class Recommendation(BaseModel):
    """单条推荐。"""
    product_id: str
    subject: str
    score: int
    reasons: list[str]


class RecommendPlan(BaseModel):
    """推荐业务载荷。"""
    preference: str
    recommendations: list[Recommendation]


_SYSTEM = (
    "你是跨境电商 B2B 运营专家，负责从商品池中挑选今日最值得上架/推广的商品。\n"
    "硬性要求：\n"
    "1) 按偏好给商品打分（score 1-100）并排序，只取前 N；\n"
    "2) 每个商品的 reasons 必须引用具体关键词趋势做依据，格式如「xxx 关键词热度 Top1」「yyy 关键词涨幅 Top1」；\n"
    '3) 只输出 JSON 对象：{"recommendations": [{"product_id": "...", "subject": "...", "score": 95, "reasons": ["..."]}]}。'
)


def _prompt(inp: RecommendInput) -> str:
    trend = ", ".join(f"{k.get('word')}(热度{k.get('heat')},涨幅{k.get('delta')})" for k in inp.trend_keywords[:30]) or "无"
    longtail = ", ".join(str(k.get("word") or k) for k in inp.longtail_keywords[:30]) or "无"
    products = "\n".join(
        f"- {p.get('product_id')}: {p.get('subject')} (关键词 {p.get('keywords') or []})"
        for p in inp.products[:100]
    ) or "无"
    return (
        f"偏好：{inp.preference}（{_PREF_HINT[inp.preference]}）\n"
        f"候选商品池：\n{products}\n\n"
        f"今日关键词趋势：{trend}\n"
        f"行业长尾词：{longtail}\n\n"
        f"请推荐 {inp.max_items} 个商品并给出理由，只输出 JSON 对象。"
    )


def _validate_reasons(reasons: list[str], trend_keywords: list[dict]) -> list[str]:
    """过滤空泛 reasons：至少包含「热度/涨幅/TopN/TOPN」关键词 + 引用 1 个 trend_keywords.word。"""
    import re
    if not reasons:
        return ["基于综合得分推荐（关键词匹配 + 商品质量综合评估）"]
    trend_words = {k.get("word") for k in trend_keywords if isinstance(k, dict)}
    trend_words.discard(None)
    hot_pattern = re.compile(r"热度|涨幅|Top\d+|TOP\d+|top\d+")

    def _valid(r: str) -> bool:
        if not hot_pattern.search(r):
            return False
        return any((w and w in r) for w in trend_words)

    filtered = [r for r in reasons if _valid(r)]
    if filtered:
        return filtered
    return ["基于综合得分推荐（关键词匹配 + 商品质量综合评估）"]


@skill(id="alibaba_product_recommend", name="今日推荐上架商品 TOP", version=_VERSION)
def alibaba_product_recommend(inp: RecommendInput) -> SkillOutput[RecommendPlan]:
    """按偏好基于关键词趋势给商品打分排序，输出 TOP 推荐 + 引用关键词的理由。

    数据流：入参校验 → 云 LLM 结构化推荐 → 裁剪上限 → RecommendPlan + 推理链。
    """
    cfg = load_config().content
    api_key = get_api_key(cfg.llm_api_key_env)
    if not api_key:
        raise ValueError(
            f"未设置环境变量 {cfg.llm_api_key_env}。云优先原则：推荐必须走云 LLM。"
        )
    if not inp.products:
        raise ValueError("商品池为空，无法推荐（请先拉取国际站在线商品）")

    reply = llm_json(
        prompt=_prompt(inp),
        system=_SYSTEM,
        api_key=api_key,
        api_base=cfg.llm_api_base,
        model=cfg.llm_model,
        max_tokens=cfg.llm_max_tokens,
        timeout_s=cfg.llm_timeout_s,
    )

    raw = reply.get("recommendations")
    recs: list[Recommendation] = []
    if isinstance(raw, list):
        for it in raw[: inp.max_items]:
            if not isinstance(it, dict):
                continue
            pid = str(it.get("product_id") or "").strip()
            subject = str(it.get("subject") or "").strip()
            if not pid:
                continue
            raw_reasons = [str(r) for r in (it.get("reasons") or []) if str(r).strip()]
            reasons = _validate_reasons(raw_reasons, inp.trend_keywords)
            recs.append(Recommendation(
                product_id=pid,
                subject=subject or pid,
                score=int(it.get("score") or 0),
                reasons=reasons,
            ))

    if not recs:
        raise ValueError("LLM 未返回有效的推荐结果")

    chain = build_chain(
        conclusion=f"按「{inp.preference}」偏好推荐 {len(recs)} 个商品",
        causal_analysis=f"结合 {len(inp.trend_keywords)} 个趋势词与 {len(inp.longtail_keywords)} 个长尾词给 {len(inp.products)} 个商品打分",
        risk_note="推荐理由基于关键词趋势快照，最终上架前请人工确认商品匹配与合规。",
    )
    return SkillOutput(
        data=RecommendPlan(preference=inp.preference, recommendations=recs),
        reasoning=[chain], confidence=0.85, sample_size=len(recs),
    )