"""content_publish_check 技能：发布前合规检查。

在内容发布前，做最终合规校验：平台规则审计 + 标题/正文长度检查 + 图片数量检查。
纯计算类（规则扫描始终执行，LLM 复核可选 — 失败不回滚）。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import ContentPlatform, build_chain
from flowmind.skills._content_rules import AuditFinding, audit_rules

_VERSION = "0.1.0"

# 各平台发布限制
PLATFORM_LIMITS: dict[str, dict] = {
    "xhs": {"title_max": 20, "body_max": 1000, "images_max": 18, "tags_max": 10},
    "wechat": {"title_max": 64, "body_max": 200000, "images_max": 50, "tags_max": 0},
    "douyin": {"title_max": 55, "body_max": 1000, "images_max": 35, "tags_max": 10},
}


class PublishCheckInput(BaseModel):
    """发布前合规检查入参。"""
    platform: ContentPlatform = Field(description="目标平台：xhs / wechat / douyin")
    title: str = Field(max_length=300, description="标题")
    body: str = Field(max_length=200000, description="正文")
    tags: list[str] = Field(default_factory=list, max_length=30, description="标签")
    image_count: int = Field(default=0, ge=0, description="配图数量")


class PublishCheckResult(BaseModel):
    """发布前合规检查业务载荷。"""
    platform: str
    can_publish: bool                    # 无 error 级问题 = 可发布
    rule_findings: list[AuditFinding]    # 规则扫描结果
    limit_warnings: list[str]            # 长度/数量超限提醒
    title_length: int
    body_length: int
    image_count: int


@skill(id="content_publish_check", name="发布前合规检查", version=_VERSION)
def content_publish_check(inp: PublishCheckInput) -> SkillOutput[PublishCheckResult]:
    """发布前最终合规检查：平台规则审计 + 长度/数量限制检查。

    数据流：规则扫描 → 长度检查 → 合并问题 → can_publish 判定 + 推理链。
    纯计算类，无外部依赖（规则扫描不走 LLM）。
    """
    findings = audit_rules(inp.platform, inp.title, inp.body, inp.tags)

    limits = PLATFORM_LIMITS.get(inp.platform, PLATFORM_LIMITS["xhs"])
    limit_warnings: list[str] = []

    title_len = len(inp.title)
    body_len = len(inp.body)

    if title_len > limits["title_max"]:
        limit_warnings.append(f"标题 {title_len} 字 > {limits['title_max']} 字限制")
    if body_len > limits["body_max"]:
        limit_warnings.append(f"正文 {body_len} 字 > {limits['body_max']} 字限制")
    if inp.image_count > limits["images_max"]:
        limit_warnings.append(f"图片 {inp.image_count} 张 > {limits['images_max']} 张限制")
    if inp.tags and limits["tags_max"] and len(inp.tags) > limits["tags_max"]:
        limit_warnings.append(f"标签 {len(inp.tags)} 个 > {limits['tags_max']} 个限制")

    errors = [f for f in findings if f.severity == "error"]
    can_publish = not errors and not limit_warnings

    chain = build_chain(
        conclusion=(
            f"{inp.platform} 发布检查{'通过' if can_publish else '未通过'}："
            f"{len(errors)} 条 error + {len(limit_warnings)} 条超限"
        ),
        causal_analysis=(
            f"规则扫描 {len(findings)} 条 + 长度检查 "
            f"（标题 {title_len}/{limits['title_max']}、正文 {body_len}/{limits['body_max']}、"
            f"图 {inp.image_count}/{limits['images_max']}）"
        ),
        risk_note="error 级 finding 或超限需修改后再发布；warning 级建议复核但可发布。",
    )
    return SkillOutput(
        data=PublishCheckResult(
            platform=inp.platform, can_publish=can_publish,
            rule_findings=findings, limit_warnings=limit_warnings,
            title_length=title_len, body_length=body_len,
            image_count=inp.image_count,
        ),
        reasoning=[chain],
        confidence=0.9 if can_publish else 0.7,
        sample_size=len(findings),
    )
