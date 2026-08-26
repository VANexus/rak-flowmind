"""content_copywrite 技能：平台化文案生成。

入参：platform + subject + 可选 angle/tone/keywords。
出参：ContentCopyPlan（标题 + 正文 + 标签）。

走云 LLM；失败 raise（invoke() 套信封为 INTERNAL）。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import (
    ContentPlatform,
    build_chain,
    copy_system,
    copy_user,
)
from flowmind.skills._llm_client import llm_json
from flowmind.skills._secrets import get_api_key

_VERSION = "0.1.0"


class ContentCopyInput(BaseModel):
    """文案生成入参。"""
    platform: ContentPlatform = Field(description="目标平台：xhs / wechat / douyin")
    subject: str = Field(min_length=1, max_length=200, description="产品 / 主题")
    angle: str | None = Field(default=None, max_length=100, description="选题角度")
    tone: str | None = Field(default=None, max_length=200, description="语气 / 风格补充")
    keywords: list[str] | None = Field(default=None, max_length=10, description="需融入的关键词")


class ContentCopyPlan(BaseModel):
    """文案生成业务载荷。"""
    platform: str
    subject: str
    angle: str | None
    tone: str | None
    title: str
    body: str
    tags: list[str]


@skill(id="content_copywrite", name="平台化文案生成", version=_VERSION)
def content_copywrite(inp: ContentCopyInput) -> SkillOutput[ContentCopyPlan]:
    """按平台调性（小红书种草 / 公众号长文 / 抖音口播）生成文案：标题 + 正文 + 标签。

    数据流：入参校验 → 云 LLM 结构化生成 → 字段裁剪/标签上限 → ContentCopyPlan + 推理链。
    """
    cfg = load_config().content
    api_key = get_api_key(cfg.llm_api_key_env)
    if not api_key:
        raise ValueError(
            f"未设置环境变量 {cfg.llm_api_key_env}。云优先原则：文案生成必须走云 LLM。"
        )

    reply = llm_json(
        prompt=copy_user(inp.platform, inp.subject, inp.angle, inp.tone, inp.keywords),
        system=copy_system(inp.platform),
        api_key=api_key,
        api_base=cfg.llm_api_base,
        model=cfg.llm_model,
        max_tokens=cfg.llm_max_tokens,
        timeout_s=cfg.llm_timeout_s,
    )

    title = str(reply.get("title") or "").strip()
    body = str(reply.get("body") or "").strip()
    if not title or not body:
        raise ValueError("LLM 未返回有效的 title/body")

    raw_tags = reply.get("tags")
    tags: list[str] = []
    if isinstance(raw_tags, list):
        for t in raw_tags[:cfg.max_tags]:
            s = str(t).strip().lstrip("#")
            if s:
                tags.append(s[:40])
    if len(tags) > cfg.max_tags:
        tags = tags[:cfg.max_tags]

    body = body[: cfg.max_copy_length]

    chain = build_chain(
        conclusion=f"为「{inp.subject}」生成 {inp.platform} 平台文案（标题 {len(title)} 字，正文 {len(body)} 字，{len(tags)} 标签）",
        causal_analysis=f"调用云 LLM 按 {inp.platform} 平台风格模板生成；角度={inp.angle or '默认'}，关键词={inp.keywords or '无'}",
        risk_note="AI 文案未做平台合规审计，发布前建议跑 content_audit 复核。",
    )
    return SkillOutput(
        data=ContentCopyPlan(
            platform=inp.platform, subject=inp.subject, angle=inp.angle,
            tone=inp.tone, title=title, body=body, tags=tags,
        ),
        reasoning=[chain],
        confidence=0.9,
        sample_size=1,
    )
