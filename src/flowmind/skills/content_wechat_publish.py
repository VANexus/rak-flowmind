"""content_wechat_publish 技能：微信公众号发布（草稿箱 + 发布接口）。

流程：获取 access_token → 上传封面图 → 创建草稿 → 发布。
入参：标题 + HTML 正文 + 封面图 URL + 可选摘要/作者。
出参：草稿 media_id + 发布 publish_id + 状态。

失败契约：HTTP 依赖类 — r.ok=True + metrics.degraded + data.failure_category。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.skill import skill
from flowmind.skills._secrets import get_api_key
from flowmind.skills._wechat_client import (
    WechatAPIError,
    add_draft,
    free_publish,
    get_access_token,
    upload_thumb_image,
)

_VERSION = "0.1.0"


class WechatPublishInput(BaseModel):
    """微信公众号发布入参。"""
    title: str = Field(min_length=1, max_length=64, description="文章标题（≤64 字节）")
    content: str = Field(min_length=1, max_length=200000, description="HTML 正文（含 <img> 标签）")
    thumb_image_url: str = Field(min_length=1, description="封面图 URL（http/https）")
    summary: str | None = Field(default=None, max_length=120, description="摘要（≤120 字）")
    author: str | None = Field(default=None, max_length=8, description="作者（≤8 字）")
    publish: bool = Field(default=True, description="是否立即发布（False=只存草稿）")


class WechatPublishResult(BaseModel):
    """微信公众号发布业务载荷。"""
    status: str                        # "published" | "drafted"
    media_id: str                      # 草稿 media_id
    publish_id: str | None = None      # 发布 ID（发布时）
    steps_completed: list[str] = Field(default_factory=list)
    # 降级时填充
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


@skill(id="content_wechat_publish", name="微信公众号发布", version=_VERSION)
def content_wechat_publish(inp: WechatPublishInput) -> SkillOutput[WechatPublishResult]:
    """微信公众号端到存草稿 + 发布：token → 封面上传 → 草稿 → 发布。

    数据流：读 AppID/Secret → access_token → 上传封面图 → 创建草稿 → （可选）发布。
    失败走 degraded SkillOutput（HTTP 依赖类）。
    """
    cfg = load_config().wechat_publish
    steps: list[str] = []

    # 1. 读凭证
    app_id = get_api_key(cfg.app_id_env)
    app_secret = get_api_key(cfg.app_secret_env)
    if not app_id or not app_secret:
        return _degraded(
            "未设置微信公众号凭证。请设置环境变量 "
            f"{cfg.app_id_env} / {cfg.app_secret_env}。",
            category="environment",
            retriable=False,
            steps=["read_credentials"],
        )

    try:
        # 2. 获取 access_token
        token = get_access_token(
            app_id=app_id, app_secret=app_secret,
            api_base=cfg.api_base, timeout_s=cfg.timeout_s,
        )
        steps.append("get_access_token")

        # 3. 上传封面图
        thumb_media_id = upload_thumb_image(
            access_token=token, image_url=inp.thumb_image_url,
            api_base=cfg.api_base, timeout_s=cfg.timeout_s,
        )
        steps.append("upload_thumb")

        # 4. 创建草稿
        media_id = add_draft(
            access_token=token,
            title=inp.title,
            content=inp.content,
            thumb_media_id=thumb_media_id,
            summary=inp.summary,
            author=inp.author,
            api_base=cfg.api_base,
            timeout_s=cfg.timeout_s,
        )
        steps.append("add_draft")

        # 5. 发布（可选）
        publish_id = None
        if inp.publish:
            publish_id = free_publish(
                access_token=token, media_id=media_id,
                api_base=cfg.api_base, timeout_s=cfg.timeout_s,
            )
            steps.append("free_publish")

    except WechatAPIError as exc:
        return _degraded(
            f"微信 API 失败（步骤 {steps[-1] if steps else 'unknown'}）：{exc}",
            category=exc.category,
            retriable=exc.retriable,
            steps=steps,
            media_id=media_id if "media_id" in dir() else "",
        )

    status = "published" if inp.publish and publish_id else "drafted"
    chain = ReasoningChain(
        conclusion=f"微信公众号文章「{inp.title}」{status}（{len(steps)} 步完成）",
        evidence=[],
        causal_analysis=" → ".join(steps),
        risk_note="发布后请在微信公众平台后台确认文章状态；草稿需手动提交审核。",
    )
    return SkillOutput(
        data=WechatPublishResult(
            status=status, media_id=media_id, publish_id=publish_id,
            steps_completed=steps,
        ),
        reasoning=[chain],
        confidence=0.95,
        sample_size=1,
    )


def _degraded(
    warning: str, category: str, retriable: bool,
    steps: list[str], media_id: str = "",
) -> SkillOutput[WechatPublishResult]:
    """构造降级返回。"""
    from flowmind.contracts import Evidence
    chain = ReasoningChain(
        conclusion=f"微信发布降级：{warning[:100]}",
        evidence=[Evidence(metric="已完成步骤", value=len(steps), threshold=None, comparison="count")],
        causal_analysis=" → ".join(steps) if steps else "未开始",
        risk_note="请检查网络 / 凭证 / 图片 URL 后重试。",
    )
    return SkillOutput(
        data=WechatPublishResult(
            status="failed", media_id=media_id,
            steps_completed=steps,
            failure_category=category,
            retriable=retriable,
            warning=warning[:500],
        ),
        reasoning=[chain],
        confidence=0.0,
        sample_size=0,
        degraded=True,
        degradation_reason=category,
    )
