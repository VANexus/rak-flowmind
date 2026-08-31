"""alibaba_listing_generate 技能：生成国际站商品标题 / 详情 / 主图提示词。

按国际站字段规则生成（标题最长字数、特殊符号禁用、爆款潜规则等），规则来自
AlibabaConfig.listing_rules（运营提供的字段规则接入前使用通用默认）。
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


class ListingGenerateInput(BaseModel):
    """Listing 生成入参。"""
    product_id: str = Field(min_length=1, max_length=64, description="商品 ID")
    subject: str = Field(default="", max_length=200, description="商品名（缺省时用 product_id）")
    keyword: str = Field(default="", max_length=100, description="核心关键词")
    preference: Preference = "alibaba"
    language: str = Field(default="en", max_length=10, description="生成语种")


class ListingPlan(BaseModel):
    """Listing 业务载荷。"""
    product_id: str
    title: str
    description: str
    keywords: list[str]
    image_prompt: str
    warnings: list[str] = Field(default_factory=list)


def _system(cfg) -> str:
    rules = "\n".join(f"{i + 1}) {r}" for i, r in enumerate(cfg.listing_rules))
    return (
        "你是阿里国际站（Alibaba.com）资深 B 端 Listing 撰写专家。\n"
        f"生成英文（或指定语种）商品标题、详情文案与主图提示词。严格遵守以下字段规则：\n{rules}\n"
        "硬性要求：\n"
        "1) title 为核心关键词前置、含属性+用途+场景，不超标题最长字数、禁用特殊符号；\n"
        "2) description 结构清晰（可含换行/短句），突出卖点与采购场景；\n"
        "3) keywords 给 1-3 个核心关键词（逗号分隔的数组）；\n"
        "4) image_prompt 给一段可用于 AI 生图的英文主图提示词（白底/专业商用摄影风格）；\n"
        '5) 只输出 JSON 对象：{"title": "...", "description": "...", "keywords": ["..."], "image_prompt": "..."}。'
    )


def _prompt(inp: ListingGenerateInput) -> str:
    return (
        f"商品：{inp.subject or inp.product_id}\n"
        f"核心关键词：{inp.keyword or '由你判断'}\n"
        f"渠道偏好：{inp.preference}\n"
        f"语种：{inp.language}\n\n"
        "请生成该商品的国际站 Listing（标题/详情/关键词/主图提示词），只输出 JSON 对象。"
    )


@skill(id="alibaba_listing_generate", name="国际站 Listing 生成", version=_VERSION)
def alibaba_listing_generate(inp: ListingGenerateInput) -> SkillOutput[ListingPlan]:
    """按国际站字段规则生成标题/详情/主图提示词（LLM 结构化输出）。

    数据流：入参校验 → 云 LLM 按字段规则生成 → 特殊符号清洗 → ListingPlan + 推理链。
    """
    cfg = load_config().alibaba
    llm_cfg = load_config().content
    api_key = get_api_key(llm_cfg.llm_api_key_env)
    if not api_key:
        raise ValueError(
            f"未设置环境变量 {llm_cfg.llm_api_key_env}。云优先原则：Listing 生成必须走云 LLM。"
        )

    reply = llm_json(
        prompt=_prompt(inp),
        system=_system(cfg),
        api_key=api_key,
        api_base=llm_cfg.llm_api_base,
        model=llm_cfg.llm_model,
        max_tokens=llm_cfg.llm_max_tokens,
        timeout_s=llm_cfg.llm_timeout_s,
    )

    raw_title = str(reply.get("title") or "").strip()
    title = _clean_title(raw_title, cfg.title_max_len)
    description = str(reply.get("description") or "").strip()
    image_prompt = str(reply.get("image_prompt") or "").strip()
    if not title or not description:
        raise ValueError("LLM 未返回有效的 title/description")

    warnings: list[str] = []
    if raw_title != title:
        warnings.append(f"标题已截断或清洗特殊符号：原始 {len(raw_title)} 字 → 合规 {len(title)} 字（≤{cfg.title_max_len}）")

    raw_kw = reply.get("keywords")
    keywords: list[str] = []
    if isinstance(raw_kw, list):
        for k in raw_kw[:3]:
            s = str(k).strip()
            if s:
                keywords.append(s[:50])
    if isinstance(raw_kw, list) and len(raw_kw) > 3:
        warnings.append(f"关键词超过上限：原始 {len(raw_kw)} 个 → 保留前 3 个（国际站规则 ≤3）")
    if len(keywords) < 1:
        warnings.append("关键词不足 1 个，请人工补充核心关键词")

    chain = build_chain(
        conclusion=f"为商品 {inp.product_id} 生成国际站 Listing（标题 {len(title)} 字符，详情 {len(description)} 字符）",
        causal_analysis=f"按 {len(cfg.listing_rules)} 条字段规则走云 LLM 生成；核心词={inp.keyword or '自动'}",
        risk_note="Listing 由 AI 生成，上架前请人工核对字段字数/符号是否符合国际站最新规则。",
    )
    return SkillOutput(
        data=ListingPlan(
            product_id=inp.product_id, title=title, description=description,
            keywords=keywords, image_prompt=image_prompt, warnings=warnings,
        ),
        reasoning=[chain], confidence=0.9, sample_size=1,
    )


def _clean_title(title: str, max_len: int) -> str:
    """清洗标题：去特殊符号 + 截断到最长字数。"""
    for ch in ("&", "|", "#", "*", "%", "（", "）", "(", ")"):
        title = title.replace(ch, " ")
    title = " ".join(title.split())
    return title[:max_len]