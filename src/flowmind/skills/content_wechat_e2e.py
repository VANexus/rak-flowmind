"""content_wechat_e2e 技能：微信公众号端到端发布。

串联流程：选题 → 文案（结构化 Markdown）→ 配图 → 合规检查 → 排版（Markdown→内联样式 HTML）
→ 发布/群发（草稿 / 定时）。

每个子步骤调用对应 skill 的 invoke()，最终输出发布结果。
失败契约：HTTP 依赖类 — r.ok=True + metrics.degraded + data.failure_category。
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal

from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.skill import invoke, skill

_VERSION = "0.2.0"

_CHANNELS = Literal["publish", "mass"]


class WechatE2EInput(BaseModel):
    """微信公众号端到端入参。"""

    subject: str = Field(min_length=1, max_length=200, description="产品 / 主题")
    angle: str | None = Field(default=None, max_length=100, description="选题角度（可选）")
    tone: str | None = Field(default=None, max_length=200, description="语气 / 风格补充")
    keywords: list[str] | None = Field(default=None, max_length=10, description="需融入的关键词")
    auto_publish: bool = Field(default=True, description="是否自动发布（False=只存草稿）")
    channel: _CHANNELS = Field(default="publish", description="publish=发布；mass=群发")
    theme: str = Field(default="default", description="排版主题：default / grace / simple")
    publish_time: int | None = Field(default=None, description="定时发布时间（Unix 秒，需账号开通定时权限）")
    # 账号 override（前端账号管理自填），不传回落环境变量
    app_id: str | None = Field(default=None, description="AppID override")
    app_secret: str | None = Field(default=None, description="AppSecret override")


class StepResult(BaseModel):
    """单步结果。"""
    step: str
    ok: bool
    summary: str = ""
    degraded: bool = False


class WechatE2EResult(BaseModel):
    """微信公众号端到端业务载荷。"""
    subject: str
    steps: list[StepResult]
    title: str = ""
    body_markdown: str = ""
    body_html: str = ""
    tags: list[str] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)
    media_id: str = ""
    publish_id: str | None = None
    msg_id: str | None = None
    status: str = "pending"            # "published" | "mass_sent" | "drafted" | "failed"
    # 降级时填充
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


@skill(id="content_wechat_e2e", name="微信公众号端到端发布", version=_VERSION, category="内容创作")
def content_wechat_e2e(inp: WechatE2EInput) -> SkillOutput[WechatE2EResult]:
    """微信公众号端到端：选题 → 文案 → 配图 → 合规检查 → 排版 → 发布/群发。

    数据流：串联 6 个子技能，任一步失败即停止并返回 degraded。
    """
    steps: list[StepResult] = []
    result = WechatE2EResult(subject=inp.subject, steps=steps)

    # Step 1: 选题
    try:
        r = invoke("content_idea_design", {"platform": "wechat", "subject": inp.subject, "count": 1})
        if not r.ok or not r.data.ideas:
            return _e2e_failed(result, "idea_design", "选题生成失败", r)
        idea = r.data.ideas[0]
        result.title = idea.title
        steps.append(StepResult(step="idea_design", ok=True, summary=idea.title))
    except Exception as exc:
        return _e2e_failed(result, "idea_design", f"选题异常：{exc}")

    # Step 2: 文案（结构化 Markdown 正文）
    try:
        r = invoke("content_copywrite", {
            "platform": "wechat",
            "subject": inp.subject,
            "angle": inp.angle or idea.angle,
            "tone": inp.tone,
            "keywords": inp.keywords,
        })
        if not r.ok:
            return _e2e_failed(result, "copywrite", "文案生成失败", r)
        result.body_markdown = r.data.body
        result.tags = r.data.tags
        if not result.title:
            result.title = r.data.title
        steps.append(StepResult(step="copywrite", ok=True, summary=f"正文 {len(r.data.body)} 字"))
    except Exception as exc:
        return _e2e_failed(result, "copywrite", f"文案异常：{exc}")

    # Step 3: 配图
    try:
        r = invoke("content_image_gen", {
            "platform": "wechat",
            "prompt": f"{inp.subject}，{result.title}，公众号封面图",
            "count": 1,
        })
        if r.ok and r.data.images:
            result.image_urls = [img.url for img in r.data.images]
            steps.append(StepResult(step="image_gen", ok=True, summary=f"{len(result.image_urls)} 张"))
        else:
            steps.append(StepResult(step="image_gen", ok=True, summary="配图跳过（失败但不阻断）", degraded=True))
    except Exception as exc:
        steps.append(StepResult(step="image_gen", ok=True, summary=f"配图跳过：{exc}", degraded=True))

    # Step 4: 合规检查
    try:
        r = invoke("content_publish_check", {
            "platform": "wechat",
            "title": result.title,
            "body": result.body_markdown,
            "tags": result.tags,
            "image_count": len(result.image_urls),
        })
        if r.ok and not r.data.can_publish:
            return _e2e_failed(result, "publish_check", "合规检查未通过", r)
        steps.append(StepResult(step="publish_check", ok=True, summary="通过"))
    except Exception as exc:
        return _e2e_failed(result, "publish_check", f"合规检查异常：{exc}")

    # Step 5: 排版（Markdown → 公众号内联样式 HTML）
    try:
        r = invoke("content_typeset", {"markdown": result.body_markdown, "theme": inp.theme})
        if r.ok and r.data.html:
            result.body_html = r.data.html
            steps.append(StepResult(
                step="typeset", ok=True,
                summary=f"主题「{r.data.theme_label}」，{r.data.stats.get('chars', 0)} 字",
            ))
        else:
            return _e2e_failed(result, "typeset", r.error.message if r.error else "排版失败", r)
    except Exception as exc:
        return _e2e_failed(result, "typeset", f"排版异常：{exc}")

    # Step 6: 发布 / 群发
    try:
        r = invoke("content_wechat_publish", {
            "title": result.title,
            "content": result.body_html,
            "thumb_image_url": result.image_urls[0] if result.image_urls else "https://flowmind.local/mock/cover.jpg",
            "summary": result.tags[:3] and "、".join(result.tags[:3]) or None,
            "publish": inp.auto_publish,
            "channel": inp.channel,
            "publish_time": inp.publish_time,
            "app_id": inp.app_id,
            "app_secret": inp.app_secret,
        })
        if r.ok and not r.metrics.degraded:
            result.media_id = r.data.media_id
            result.publish_id = r.data.publish_id
            result.msg_id = r.data.msg_id
            result.status = r.data.status
            steps.append(StepResult(step="publish", ok=True, summary=result.status))
        else:
            return _e2e_failed(result, "publish", r.data.warning or "发布失败", r)
    except Exception as exc:
        return _e2e_failed(result, "publish", f"发布异常：{exc}")

    chain = ReasoningChain(
        conclusion=f"微信公众号「{result.title}」{result.status}（{len(steps)} 步完成）",
        evidence=[], causal_analysis=" → ".join(s.step for s in steps),
        risk_note="发布/群发为异步过程，请在公众号后台确认最终状态。",
    )
    return SkillOutput(data=result, reasoning=[chain], confidence=0.9, sample_size=len(steps))


def _e2e_failed(result: WechatE2EResult, step: str, warning: str, r=None) -> SkillOutput[WechatE2EResult]:
    """构造端到端失败返回。"""
    result.steps.append(StepResult(step=step, ok=False, summary=warning))
    result.status = "failed"
    result.warning = warning
    if r is not None:
        result.failure_category = getattr(r.metrics, 'degraded', False) and "environment" or "internal"
        result.retriable = getattr(r.data, 'retriable', False) if r.data else False
    chain = ReasoningChain(
        conclusion=f"微信端到端中断于 {step}：{warning[:100]}",
        evidence=[], causal_analysis=" → ".join(s.step for s in result.steps),
        risk_note="请检查对应子步骤的输入与环境后重试。",
    )
    return SkillOutput(
        data=result, reasoning=[chain], confidence=0.0,
        sample_size=len(result.steps), degraded=True,
        degradation_reason=result.failure_category or "unknown",
    )
