"""content_xhs_draft 技能：小红书内容草稿包生成。

小红书无公开发布 API → 生成结构化内容包供复制粘贴或导入第三方工具。
入参：标题 + 正文 + 标签 + 配图 URL。
出参：格式化草稿包（标题裁剪/标签补 #/平台规范提醒）+ JSON 导出。

纯计算类（无外部依赖），失败 raise。
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import build_chain

_VERSION = "0.1.0"

# 小红书平台限制
_TITLE_MAX = 20        # 标题 ≤ 20 字
_BODY_MAX = 1000       # 正文 ≤ 1000 字
_TAGS_MAX = 10         # 标签 ≤ 10 个
_IMAGES_MAX = 18       # 图片 ≤ 18 张


class XhsDraftInput(BaseModel):
    """小红书草稿包入参。"""
    title: str = Field(min_length=1, max_length=200, description="笔记标题")
    body: str = Field(min_length=1, max_length=10000, description="笔记正文")
    tags: list[str] = Field(default_factory=list, max_length=20, description="标签（可带或不带 #）")
    image_urls: list[str] = Field(default_factory=list, max_length=30, description="配图 URL 列表")
    topic: str | None = Field(default=None, max_length=100, description="话题 / 品牌词")


class XhsDraftResult(BaseModel):
    """小红书草稿包业务载荷。"""
    title: str                         # 裁剪后标题（≤20 字）
    body: str                          # 裁剪后正文（≤1000 字）
    tags: list[str]                    # 规范化标签（带 #）
    images: list[str]                  # 图片 URL 列表
    warnings: list[str]                # 平台规范提醒
    content_json: str                  # 完整 JSON（可导入第三方工具）
    char_count: int                    # 正文字数
    within_limits: bool                # 是否全部在限制内


@skill(id="content_xhs_draft", name="小红书草稿包", version=_VERSION)
def content_xhs_draft(inp: XhsDraftInput) -> SkillOutput[XhsDraftResult]:
    """生成小红书内容草稿包：裁剪 + 标签规范化 + 平台规范提醒 + JSON 导出。

    数据流：入参校验 → 标题/正文裁剪 → 标签补 # → 规范检查 → 草稿包 + 推理链。
    """
    warnings: list[str] = []

    # 1. 标题裁剪（按中文字符计）
    title = inp.title.strip()
    if _char_length(title) > _TITLE_MAX:
        title = _truncate(title, _TITLE_MAX)
        warnings.append(f"标题已裁剪至 {_TITLE_MAX} 字（原 {_char_length(inp.title)} 字）")

    # 2. 正文裁剪
    body = inp.body.strip()
    body_len = _char_length(body)
    if body_len > _BODY_MAX:
        body = _truncate(body, _BODY_MAX)
        warnings.append(f"正文已裁剪至 {_BODY_MAX} 字（原 {body_len} 字）")

    # 3. 标签规范化（去空格、补 #、去重）
    tags = _normalize_tags(inp.tags, _TAGS_MAX)
    if len(inp.tags) > _TAGS_MAX:
        warnings.append(f"标签已裁剪至 {_TAGS_MAX} 个（原 {len(inp.tags)} 个）")

    # 4. 图片裁剪
    images = inp.image_urls[:_IMAGES_MAX]
    if len(inp.image_urls) > _IMAGES_MAX:
        warnings.append(f"图片已裁剪至 {_IMAGES_MAX} 张（原 {len(inp.image_urls)} 张）")

    # 5. 平台规范检查
    if _char_length(title) < 5:
        warnings.append("标题过短（<5 字），可能影响推荐")
    if body_len < 20:
        warnings.append("正文过短（<20 字），建议补充内容")
    if not tags:
        warnings.append("未添加标签，建议 3-6 个标签提升曝光")
    if not images:
        warnings.append("未添加配图，小红书笔记建议至少 1 张图")

    # 6. 构造 JSON
    content_dict = {
        "platform": "xhs",
        "title": title,
        "body": body,
        "tags": tags,
        "images": images,
        "topic": inp.topic,
    }
    content_json = json.dumps(content_dict, ensure_ascii=False, indent=2)

    within_limits = (
        _char_length(title) <= _TITLE_MAX
        and _char_length(body) <= _BODY_MAX
        and len(tags) <= _TAGS_MAX
        and len(images) <= _IMAGES_MAX
    )

    chain = build_chain(
        conclusion=f"小红书草稿包生成完成：标题 {_char_length(title)} 字、正文 {_char_length(body)} 字、{len(tags)} 标签、{len(images)} 图",
        causal_analysis=f"裁剪 + 标签规范化 + 规范检查（{len(warnings)} 条提醒）",
        risk_note="草稿包为辅助工具，正式发布前请人工终审内容合规性与图片版权。",
    )
    return SkillOutput(
        data=XhsDraftResult(
            title=title, body=body, tags=tags, images=images,
            warnings=warnings, content_json=content_json,
            char_count=_char_length(body), within_limits=within_limits,
        ),
        reasoning=[chain],
        confidence=0.95,
        sample_size=1,
    )


def _char_length(text: str) -> int:
    """计算字符长度（中文按 1 字计，与小红书计数一致）。"""
    return len(text)


def _truncate(text: str, max_chars: int) -> str:
    """按字符数截断（不拆分 emoji / 多字节字符）。"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _normalize_tags(tags: list[str], max_count: int) -> list[str]:
    """标签规范化：去空格、补 #、去重、限数量。"""
    seen: set[str] = set()
    result: list[str] = []
    for raw in tags:
        t = raw.strip().lstrip("#").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        result.append(f"#{t}")
        if len(result) >= max_count:
            break
    return result
