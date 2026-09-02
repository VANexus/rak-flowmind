"""content_wechat_publish 技能：微信公众号发布 / 群发。

流程：凭证解析 → access_token → 上传封面图 → 正文图转存（uploadimg）→ 创建草稿 →
      发布（freepublish/submit，可定时）或 群发（message/mass/sendall）。

入参：标题 + HTML 正文 + 封面图 URL + 可选摘要/作者/定时/渠道/账号 override。
出参：草稿 media_id + publish_id（发布）或 msg_id（群发）+ 状态。

失败契约：HTTP 依赖类 — r.ok=True + metrics.degraded + data.failure_category。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import Evidence, ReasoningChain, SkillOutput
from flowmind.skill import skill
from flowmind.skills._secrets import get_api_key
from flowmind.skills._wechat_client import (
    WechatAPIError,
    add_draft,
    free_publish,
    get_access_token,
    mass_send,
    upload_content_images,
    upload_thumb_image,
)

_VERSION = "0.2.0"

_CHANNELS = Literal["publish", "mass"]


class WechatPublishInput(BaseModel):
    """微信公众号发布/群发入参。"""

    title: str = Field(min_length=1, max_length=64, description="文章标题（≤64 字节）")
    content: str = Field(min_length=1, max_length=200000, description="HTML 正文（含 <img> 标签）")
    thumb_image_url: str = Field(min_length=1, description="封面图 URL（http/https）")
    summary: str | None = Field(default=None, max_length=120, description="摘要（≤120 字）")
    author: str | None = Field(default=None, max_length=8, description="作者（≤8 字）")
    publish: bool = Field(default=True, description="是否立即发布/群发（False=只存草稿）")
    channel: _CHANNELS = Field(default="publish", description="publish=发布到图文消息；mass=群发（推送给全部粉丝）")
    publish_time: int | None = Field(
        default=None, description="定时发布时间（Unix 秒）。仅 channel=publish 时有效，"
                                  "且公众号需开通『定时发布』权限；否则用 FlowMind 定时调度兜底。"
    )
    # 账号 override：产品模式下由前端传入（从 DB 解密），不传则回落环境变量凭证
    app_id: str | None = Field(default=None, description="AppID override（来自账号管理，优先于环境变量）")
    app_secret: str | None = Field(default=None, description="AppSecret override（来自账号管理，优先于环境变量）")
    skip_body_image_upload: bool = Field(default=False, description="跳过正文图转存（调试用）")


class WechatPublishResult(BaseModel):
    """微信公众号发布业务载荷。"""

    status: str                        # "published" | "mass_sent" | "drafted"
    media_id: str                      # 草稿 media_id
    publish_id: str | None = None      # 发布 ID（channel=publish 时）
    msg_id: str | None = None          # 群发 ID（channel=mass 时）
    publish_time: int | None = None
    body_images: list[dict] = Field(default_factory=list)   # 正文图转存记录
    steps_completed: list[str] = Field(default_factory=list)
    # 降级时填充
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


@skill(id="content_wechat_publish", name="微信公众号发布/群发", version=_VERSION, category="内容创作")
def content_wechat_publish(inp: WechatPublishInput) -> SkillOutput[WechatPublishResult]:
    """微信公众号发布/群发：token → 封面上传 → 正文图转存 → 草稿 → 发布/群发。

    数据流：读凭证（override 优先）→ access_token → 上传封面图 → 正文图 uploadimg
    转存 → 创建草稿 →（可选）freepublish/submit 或 mass/sendall。
    失败走 degraded SkillOutput（HTTP 依赖类）。
    """
    cfg = load_config().wechat_publish
    steps: list[str] = []

    # 1. 读凭证：显式传入优先（前端账号管理自填），否则回落环境变量
    app_id = inp.app_id or get_api_key(cfg.app_id_env)
    app_secret = inp.app_secret or get_api_key(cfg.app_secret_env)
    if not app_id or not app_secret:
        return _degraded(
            "未设置微信公众号凭证。请在前端「公众号账号」中添加并测试，"
            f"或设置环境变量 {cfg.app_id_env} / {cfg.app_secret_env}。",
            category="environment", retriable=False, steps=["read_credentials"],
        )

    media_id = ""
    body_images: list[dict] = []
    publish_id: str | None = None
    msg_id: str | None = None

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

        # 4. 正文图转存为公众号图片（mmbiz URL）
        content = inp.content
        if not inp.skip_body_image_upload:
            content, body_images = upload_content_images(
                access_token=token, content=inp.content,
                api_base=cfg.api_base, timeout_s=cfg.timeout_s,
            )
            steps.append("upload_content_images")

        # 5. 创建草稿
        media_id = add_draft(
            access_token=token,
            title=inp.title,
            content=content,
            thumb_media_id=thumb_media_id,
            summary=inp.summary,
            author=inp.author,
            api_base=cfg.api_base,
            timeout_s=cfg.timeout_s,
        )
        steps.append("add_draft")

        # 6. 发布 / 群发（可选）
        if inp.publish:
            if inp.channel == "mass":
                msg_id = mass_send(
                    access_token=token, media_id=media_id,
                    clientmsgid=None, api_base=cfg.api_base, timeout_s=cfg.timeout_s,
                )
                steps.append("mass_send")
            else:
                publish_id = free_publish(
                    access_token=token, media_id=media_id,
                    publish_time=inp.publish_time,
                    api_base=cfg.api_base, timeout_s=cfg.timeout_s,
                )
                steps.append("free_publish")

    except WechatAPIError as exc:
        return _degraded(
            f"微信 API 失败（步骤 {steps[-1] if steps else 'unknown'}）：{exc}",
            category=exc.category,
            retriable=exc.retriable,
            steps=steps,
            media_id=media_id,
            body_images=body_images,
        )

    if inp.channel == "mass" and inp.publish:
        status = "mass_sent"
    elif inp.publish and publish_id:
        status = "published"
    else:
        status = "drafted"

    risk = "群发将推送给全部粉丝，提交后异步执行；草稿 media_id 在群发后失效。"
    if inp.channel == "publish":
        risk = "发布后请在微信公众平台后台确认文章状态；发布接口仅认证账号可用。"
        if inp.publish_time:
            risk += " 已请求定时发布（需账号开通『定时发布』权限，失败请改用 FlowMind 定时调度）。"

    chain = ReasoningChain(
        conclusion=f"微信公众号文章「{inp.title}」{status}（{len(steps)} 步完成）",
        evidence=[Evidence(metric="已完成步骤", value=len(steps), threshold=None, comparison="count")],
        causal_analysis=" → ".join(steps),
        risk_note=risk,
    )
    return SkillOutput(
        data=WechatPublishResult(
            status=status, media_id=media_id, publish_id=publish_id, msg_id=msg_id,
            publish_time=inp.publish_time, body_images=body_images,
            steps_completed=steps,
        ),
        reasoning=[chain],
        confidence=0.95,
        sample_size=1,
    )


def _degraded(
    warning: str, category: str, retriable: bool,
    steps: list[str], media_id: str = "", body_images: list | None = None,
) -> SkillOutput[WechatPublishResult]:
    """构造降级返回。"""
    from flowmind.contracts import Evidence
    chain = ReasoningChain(
        conclusion=f"微信发布降级：{warning[:100]}",
        evidence=[Evidence(metric="已完成步骤", value=len(steps), threshold=None, comparison="count")],
        causal_analysis=" → ".join(steps) if steps else "未开始",
        risk_note="请检查网络 / 凭证 / 图片 URL / 账号认证状态后重试。",
    )
    return SkillOutput(
        data=WechatPublishResult(
            status="failed", media_id=media_id,
            body_images=body_images or [],
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
