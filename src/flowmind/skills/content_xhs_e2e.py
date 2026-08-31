"""content_xhs_e2e 技能：小红书端到端内容生成。

串联流程：热点 → 选题 → 文案 → 配图 → 合规检查 → 草稿包。
每个子步骤调用对应 skill 的 invoke()，最终输出小红书草稿包。
失败契约：HTTP 依赖类 — r.ok=True + metrics.degraded + data.failure_category。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.skill import invoke, skill

_VERSION = "0.1.0"


class XhsE2EInput(BaseModel):
    """小红书端到端入参。"""
    subject: str = Field(min_length=1, max_length=200, description="产品 / 主题")
    use_hot_topics: bool = Field(default=True, description="是否参考热点选题")
    angle: str | None = Field(default=None, max_length=100, description="选题角度（可选）")
    tone: str | None = Field(default=None, max_length=200, description="语气 / 风格补充")
    keywords: list[str] | None = Field(default=None, max_length=10, description="需融入的关键词")
    image_count: int = Field(default=3, ge=1, le=4, description="配图数量（1-4）")


class XhsStepResult(BaseModel):
    """单步结果。"""
    step: str
    ok: bool
    summary: str = ""
    degraded: bool = False


class XhsE2EResult(BaseModel):
    """小红书端到端业务载荷。"""
    subject: str
    steps: list[XhsStepResult]
    title: str = ""
    body: str = ""
    tags: list[str] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)
    draft_json: str = ""
    hot_topic_used: str | None = None
    # 降级时填充
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


@skill(id="content_xhs_e2e", name="小红书端到端内容生成", version=_VERSION)
def content_xhs_e2e(inp: XhsE2EInput) -> SkillOutput[XhsE2EResult]:
    """小红书端到端：热点 → 选题 → 文案 → 配图 → 合规检查 → 草稿包。

    数据流：串联 6 个子技能，任一步失败即停止并返回 degraded。
    """
    steps: list[XhsStepResult] = []
    result = XhsE2EResult(subject=inp.subject, steps=steps)

    # Step 1: 热点（可选）
    hot_title = None
    if inp.use_hot_topics:
        try:
            r = invoke("content_hot_topics", {"platform": "xhs", "limit": 5})
            if r.ok and r.data.topics:
                hot_title = r.data.topics[0].word
                result.hot_topic_used = hot_title
                steps.append(XhsStepResult(step="hot_topics", ok=True, summary=hot_title))
            else:
                steps.append(XhsStepResult(step="hot_topics", ok=True, summary="热点跳过", degraded=True))
        except Exception as exc:
            steps.append(XhsStepResult(step="hot_topics", ok=True, summary=f"热点跳过：{exc}", degraded=True))

    # Step 2: 选题（结合热点）
    idea_subject = f"{inp.subject}（参考热点：{hot_title}）" if hot_title else inp.subject
    try:
        r = invoke("content_idea_design", {"platform": "xhs", "subject": idea_subject, "count": 1})
        if not r.ok or not r.data.ideas:
            return _xhs_e2e_failed(result, "idea_design", "选题生成失败", r)
        idea = r.data.ideas[0]
        result.title = idea.title
        steps.append(XhsStepResult(step="idea_design", ok=True, summary=idea.title))
    except Exception as exc:
        return _xhs_e2e_failed(result, "idea_design", f"选题异常：{exc}")

    # Step 3: 文案
    try:
        r = invoke("content_copywrite", {
            "platform": "xhs",
            "subject": inp.subject,
            "angle": inp.angle or idea.angle,
            "tone": inp.tone,
            "keywords": inp.keywords,
        })
        if not r.ok:
            return _xhs_e2e_failed(result, "copywrite", "文案生成失败", r)
        result.body = r.data.body
        result.tags = r.data.tags
        if not result.title:
            result.title = r.data.title
        steps.append(XhsStepResult(step="copywrite", ok=True, summary=f"正文 {len(r.data.body)} 字"))
    except Exception as exc:
        return _xhs_e2e_failed(result, "copywrite", f"文案异常：{exc}")

    # Step 4: 配图
    try:
        r = invoke("content_image_gen", {
            "platform": "xhs",
            "prompt": f"{inp.subject}，{result.title}，小红书种草配图",
            "count": inp.image_count,
        })
        if r.ok and r.data.images:
            result.image_urls = [img.url for img in r.data.images]
            steps.append(XhsStepResult(step="image_gen", ok=True, summary=f"{len(result.image_urls)} 张"))
        else:
            steps.append(XhsStepResult(step="image_gen", ok=True, summary="配图跳过", degraded=True))
    except Exception as exc:
        steps.append(XhsStepResult(step="image_gen", ok=True, summary=f"配图跳过：{exc}", degraded=True))

    # Step 5: 合规检查
    try:
        r = invoke("content_audit", {
            "platform": "xhs",
            "title": result.title,
            "body": result.body,
            "tags": result.tags,
        })
        steps.append(XhsStepResult(step="audit", ok=True, summary=f"{len(r.data.findings)} 条 finding"))
    except Exception as exc:
        steps.append(XhsStepResult(step="audit", ok=True, summary=f"审计跳过：{exc}", degraded=True))

    # Step 6: 生成草稿包
    try:
        r = invoke("content_xhs_draft", {
            "title": result.title,
            "body": result.body,
            "tags": result.tags,
            "image_urls": result.image_urls,
            "topic": inp.subject,
        })
        if r.ok:
            result.draft_json = r.data.content_json
            steps.append(XhsStepResult(step="draft", ok=True, summary="草稿包已生成"))
        else:
            return _xhs_e2e_failed(result, "draft", "草稿包生成失败", r)
    except Exception as exc:
        return _xhs_e2e_failed(result, "draft", f"草稿包异常：{exc}")

    chain = ReasoningChain(
        conclusion=f"小红书「{result.title}」内容包生成完成（{len(steps)} 步）",
        evidence=[],
        causal_analysis=" → ".join(s.step for s in steps),
        risk_note="草稿包为辅助工具，请复制到小红书 App 或第三方工具发布。",
    )
    return SkillOutput(data=result, reasoning=[chain], confidence=0.9, sample_size=len(steps))


def _xhs_e2e_failed(result: XhsE2EResult, step: str, warning: str, r=None) -> SkillOutput[XhsE2EResult]:
    """构造端到端失败返回。"""
    result.steps.append(XhsStepResult(step=step, ok=False, summary=warning))
    result.warning = warning
    if r is not None:
        result.failure_category = getattr(r.metrics, 'degraded', False) and "environment" or "internal"
        result.retriable = getattr(r.data, 'retriable', False) if r.data else False
    chain = ReasoningChain(
        conclusion=f"小红书端到端中断于 {step}：{warning[:100]}",
        evidence=[], causal_analysis=" → ".join(s.step for s in result.steps),
        risk_note="请检查对应子步骤的输入与环境后重试。",
    )
    return SkillOutput(
        data=result, reasoning=[chain], confidence=0.0,
        sample_size=len(result.steps), degraded=True,
        degradation_reason=result.failure_category or "unknown",
    )
