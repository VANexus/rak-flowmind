"""b2b_longtail_keywords 技能：相关行业长尾词榜单（走云 LLM 结构化生成）。

基于趋势热门词扩展同行业长尾词并按小类分组，输出可直接用于 Listing/推广的候选词。
错误契约：普通 raise（invoke() 套信封为 INTERNAL）；无 key 显式报错。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import build_chain
from flowmind.skills._llm_client import llm_json
from flowmind.skills._secrets import get_api_key

_VERSION = "0.1.0"


class LongtailInput(BaseModel):
    """长尾词入参。"""
    industry: str = Field(min_length=1, max_length=100, description="行业（如 美妆个护）")
    seed_keywords: list[str] = Field(default_factory=list, max_length=20, description="来源热门词")
    limit: int = Field(default=20, ge=1, le=50, description="返回条数上限")


class LongtailKeyword(BaseModel):
    """单条长尾词。"""
    word: str
    category: str
    search_intent: str = ""


class LongtailPlan(BaseModel):
    """长尾词榜单业务载荷。"""
    industry: str
    keywords: list[LongtailKeyword]


_SYSTEM = (
    "你是跨境电商关键词研究专家，精通 B2B 外贸行业长尾关键词挖掘。\n"
    "基于用户给出的行业与热门词，扩展同行业长尾关键词并按小类分组。\n"
    "硬性要求：\n"
    "1) 长尾词要足够具体（2-5 个词组成），符合海外采购/搜索习惯；\n"
    "2) 每个词给 category（小类，如 包装/功效/场景）与 search_intent（搜索意图，如 informational/commercial/transactional）；\n"
    '3) 只输出 JSON 对象：{"keywords": [{"word": "...", "category": "...", "search_intent": "..."}]}。'
)


def _prompt(industry: str, seed_keywords: list[str], limit: int) -> str:
    seeds = "、".join(seed_keywords) if seed_keywords else "无"
    return (
        f"行业：{industry}\n来源热门词：{seeds}\n\n"
        f"请生成 {limit} 个该行业的长尾关键词，按小类分组。只输出 JSON 对象。"
    )


@skill(id="b2b_longtail_keywords", name="行业长尾词榜单", version=_VERSION)
def b2b_longtail_keywords(inp: LongtailInput) -> SkillOutput[LongtailPlan]:
    """基于热门词扩展同行业长尾词榜单（LLM 结构化生成，按小类分组）。

    数据流：入参校验 → 云 LLM 结构化生成 → 裁剪上限 → LongtailPlan + 推理链。
    """
    cfg = load_config().content
    api_key = get_api_key(cfg.llm_api_key_env)
    if not api_key:
        raise ValueError(
            f"未设置环境变量 {cfg.llm_api_key_env}。云优先原则：长尾词生成必须走云 LLM。"
        )

    limit = inp.limit
    reply = llm_json(
        prompt=_prompt(inp.industry, inp.seed_keywords, limit),
        system=_SYSTEM,
        api_key=api_key,
        api_base=cfg.llm_api_base,
        model=cfg.llm_model,
        max_tokens=cfg.llm_max_tokens,
        timeout_s=cfg.llm_timeout_s,
    )

    raw = reply.get("keywords")
    keywords: list[LongtailKeyword] = []
    if isinstance(raw, list):
        for it in raw[:limit]:
            if not isinstance(it, dict):
                continue
            word = str(it.get("word") or "").strip()
            if not word:
                continue
            keywords.append(LongtailKeyword(
                word=word,
                category=str(it.get("category") or "通用"),
                search_intent=str(it.get("search_intent") or ""),
            ))

    if not keywords:
        raise ValueError("LLM 未返回有效的长尾词列表")

    chain = build_chain(
        conclusion=f"为「{inp.industry}」生成 {len(keywords)} 个长尾关键词（按小类分组）",
        causal_analysis=f"以 {len(inp.seed_keywords)} 个热门词为种子，走云 LLM 扩展同行业长尾词",
        risk_note="长尾词由 AI 生成，建议结合平台后台数据验证搜索量后再投放。",
    )
    return SkillOutput(
        data=LongtailPlan(industry=inp.industry, keywords=keywords),
        reasoning=[chain],
        confidence=0.85,
        sample_size=len(keywords),
    )