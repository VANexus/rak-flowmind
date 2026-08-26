"""content_* 技能共享：平台元数据、提示词构建、推理链辅助。

平台模型：xhs（小红书） / wechat（微信公众号） / douyin（抖音）。
每个平台带：展示标签、内容风格引导、图片比例与像素尺寸。
"""
from __future__ import annotations

from typing import Literal

from flowmind.contracts import ReasoningChain

ContentPlatform = Literal["xhs", "wechat", "douyin"]

PLATFORMS: dict[str, dict] = {
    "xhs": {
        "label": "小红书",
        "aspect": "3:4",
        "pixels": (1080, 1440),
        "format_hint": "图文笔记 · 3:4",
        "tone": "种草笔记：口语化、真实体验感、善用 emoji 与换行、结尾互动引导",
    },
    "wechat": {
        "label": "微信公众号",
        "aspect": "16:9",
        "pixels": (1920, 1080),
        "format_hint": "长文 · 16:9 头图",
        "tone": "深度长文：结构清晰、有观点与论据、专业但不晦涩、适合收藏转发",
    },
    "douyin": {
        "label": "抖音",
        "aspect": "9:16",
        "pixels": (1080, 1920),
        "format_hint": "短视频 · 9:16 口播",
        "tone": "口播脚本：口语化短句、节奏快、开头 3 秒抓注意力、结尾行动引导",
    },
}

IDEA_ANGLES: dict[str, list[str]] = {
    "xhs": ["痛点 + 场景", "反常识 · 对比", "科普 · 攻略", "真实测评", "开箱种草", "避坑指南"],
    "wechat": ["深度 · 拆解", "观点 · 洞察", "数据 · 报告", "行业 · 趋势", "案例 · 复盘", "方法论清单"],
    "douyin": ["口播 · 反差", "口播 · 实测", "口播 · 测评", "剧情 · 反转", "快问快答", "挑战 · 互动"],
}


def idea_prompt(platform: str, subject: str, count: int) -> str:
    angles = "、".join(IDEA_ANGLES.get(platform, IDEA_ANGLES["xhs"]))
    return (
        f"产品/主题：{subject}\n目标平台：{PLATFORMS.get(platform, {}).get('label', platform)}\n"
        f"可用选题角度：{angles}\n\n"
        f"请为这条产品/主题设计 {count} 条选题思路，要求：\n"
        "1) 每条给出 angle（角度，如'痛点 + 场景'）、title（可直接用的标题）、reason（为什么好，1 句）；\n"
        "2) 标题贴合平台用户兴趣，避免标题党与夸大；\n"
        '3) 只输出 JSON 对象：{"ideas": [{"angle": "...", "title": "...", "reason": "..."}]}。'
    )


def copy_system(platform: str) -> str:
    tone = PLATFORMS.get(platform, {}).get("tone", "")
    return (
        "你是资深的新媒体内容创作专家，精通小红书、微信公众号、抖音的内容写法。\n"
        f"本次输出面向：{PLATFORMS.get(platform, {}).get('label', platform)}。\n"
        f"风格要求：{tone}\n"
        "硬性要求：\n"
        "1) 标题 15-30 字，抓人但不标题党；\n"
        "2) 正文贴合平台调性；\n"
        "3) 给出 3-6 个标签（tags，不带 # 号，小红书/抖音用话题词，公众号用主题词）；\n"
        "4) 禁用绝对化用语（最/第一/顶级/全网最低等）与医疗功效宣称（治疗/根治等）；\n"
        '5) 只输出 JSON 对象：{"title": "...", "body": "...", "tags": ["..."]}。'
    )


def copy_user(
    platform: str, subject: str, angle: str | None, tone: str | None, keywords: list[str] | None
) -> str:
    parts = [f"产品/主题：{subject}"]
    if angle:
        parts.append(f"选题角度：{angle}")
    if tone:
        parts.append(f"语气/风格补充：{tone}")
    if keywords:
        parts.append(f"需自然融入的关键词：{'、'.join(keywords)}")
    return "\n".join(parts)


def audit_system() -> str:
    return (
        "你是内容合规审计专家，熟悉《广告法》及小红书/微信公众号/抖音的平台规范。\n"
        "对用户给出的文案做二次复核（规则扫描之外），重点发现：\n"
        "1) 绝对化用语、医疗功效、金融收益承诺；\n"
        "2) 各平台特有的导流/诱导行为；\n"
        "3) 可能引发投诉或限流的表达；\n"
        '只输出 JSON 对象：{"findings": [{"category": "absolute|medical|advert|platform|finance", '
        '"severity": "error|warning", "message": "...", "suggestion": "..."}]}；'
        "无可疑项时 findings 为空数组。"
    )


def audit_user(platform: str, title: str, body: str, tags: list[str]) -> str:
    return (
        f"目标平台：{PLATFORMS.get(platform, {}).get('label', platform)}\n"
        f"标题：{title}\n正文：{body}\n标签：{'、'.join(tags)}\n\n请复核并输出 JSON。"
    )


def build_chain(
    conclusion: str,
    causal_analysis: str,
    risk_note: str,
    hits: list | None = None,
    evidence: list | None = None,
) -> ReasoningChain:
    """组装四段式推理链（文本字段全非空）。"""
    return ReasoningChain(
        conclusion=conclusion,
        triggered_rules=hits or [],
        evidence=evidence or [],
        causal_analysis=causal_analysis,
        risk_note=risk_note,
    )
