"""content_idea_design 技能：AI 生成多平台选题思路。

入参：platform（xhs/wechat/douyin）+ subject（产品/主题）+ count（1-6）。
出参：ContentIdeaPlan（平台 + 选题角度列表 + 每条的标题与理由）。

走云 LLM（Anthropic 兼容协议）；无 key / 调用失败 raise（invoke() 套信封为 INTERNAL）。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import (
    ContentPlatform,
    build_chain,
    idea_prompt,
)
from flowmind.skills._llm_client import llm_json
from flowmind.skills._secrets import get_api_key

_VERSION = "0.1.0"


class ContentIdeaInput(BaseModel):
    """思路设计入参。"""
    platform: ContentPlatform = Field(description="目标平台：xhs / wechat / douyin")
    subject: str = Field(min_length=1, max_length=200, description="产品 / 主题")
    count: int = Field(default=3, ge=1, le=6, description="选题条数（1-6）")


class IdeaAngle(BaseModel):
    """单条选题。"""
    angle: str
    title: str
    reason: str = ""


class ContentIdeaPlan(BaseModel):
    """思路设计业务载荷。"""
    platform: str
    subject: str
    ideas: list[IdeaAngle]
    prompt_source: str  # "llm" | "fallback"
    fallback: bool = False
    warning: str | None = None


@skill(id="content_idea_design", name="内容选题思路设计", version=_VERSION)
def content_idea_design(inp: ContentIdeaInput) -> SkillOutput[ContentIdeaPlan]:
    """基于产品/主题为指定平台 AI 生成 N 条选题思路（角度 + 标题 + 理由）。

    数据流：入参校验 → 云 LLM 结构化生成 → 解析 + 数量裁剪 → ContentIdeaPlan + 推理链。
    """
    cfg = load_config().content
    api_key = get_api_key(cfg.llm_api_key_env)
    if not api_key:
        raise ValueError(
            f"未设置环境变量 {cfg.llm_api_key_env}。云优先原则：思路设计必须走云 LLM。"
        )

    reply = llm_json(
        prompt=idea_prompt(inp.platform, inp.subject, inp.count),
        system="你是选题策划专家，输出严格 JSON。",
        api_key=api_key,
        api_base=cfg.llm_api_base,
        model=cfg.llm_model,
        max_tokens=cfg.llm_max_tokens,
        timeout_s=cfg.llm_timeout_s,
    )

    raw_ideas = reply.get("ideas")
    if not isinstance(raw_ideas, list) or not raw_ideas:
        raise ValueError("LLM 未返回有效的 ideas 数组")

    ideas: list[IdeaAngle] = []
    for it in raw_ideas[:cfg.max_ideas]:
        if not isinstance(it, dict):
            continue
        title = str(it.get("title") or "").strip()
        if not title:
            continue
        ideas.append(IdeaAngle(
            angle=str(it.get("angle") or "综合").strip()[:30],
            title=title[:80],
            reason=str(it.get("reason") or "").strip()[:200],
        ))
    if not ideas:
        raise ValueError("LLM 返回的 ideas 无法解析")

    chain = build_chain(
        conclusion=f"为「{inp.subject}」生成 {len(ideas)} 条 {inp.platform} 选题思路",
        causal_analysis=f"调用云 LLM 按 {inp.platform} 平台角度模板生成；收到 {len(raw_ideas)} 条原始候选，保留 {len(ideas)} 条",
        risk_note="AI 选题仅供参考；正式发布前建议人工校验标题真实性与平台合规。",
    )
    return SkillOutput(
        data=ContentIdeaPlan(
            platform=inp.platform, subject=inp.subject, ideas=ideas,
            prompt_source="llm", fallback=False,
        ),
        reasoning=[chain],
        confidence=0.9,
        sample_size=len(ideas),
    )
