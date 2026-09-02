"""content_wechat_publish_status 技能：查询发布 / 群发状态（前端历史轮询用）。

按 publish_id（freepublish/get）或 msg_id（message/mass/get）查询，并归一化为
status_text。凭证显式传入优先，否则回落环境变量。

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
    get_access_token,
    get_mass_status,
    get_publish_status,
)

_VERSION = "0.1.0"

# freepublish/get publish_status 归一化
_PUBLISH_STATUS_TEXT = {
    0: "发布成功",
    1: "发布中",
    2: "原草稿审核失败",
    3: "发布成功（需审核）",
}
# message/mass/get msg_status 归一化
_MASS_STATUS_TEXT = {
    0: "群发成功",
    1: "群发中",
    2: "群发失败",
    3: "被封禁",
    4: "触发频控",
    5: "审核中",
}


class WechatPublishStatusInput(BaseModel):
    """状态查询入参。publish_id 与 msg_id 二选一。"""

    publish_id: str | None = Field(default=None, description="发布 ID（freepublish/submit 返回）")
    msg_id: str | None = Field(default=None, description="群发 ID（mass/sendall 返回）")
    app_id: str | None = Field(default=None, description="AppID override")
    app_secret: str | None = Field(default=None, description="AppSecret override")


class WechatPublishStatusResult(BaseModel):
    """状态查询业务载荷。"""

    kind: str                                  # "publish" | "mass"
    status_text: str
    status_code: int | None = None
    article_url: str | None = None
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


@skill(id="content_wechat_publish_status", name="公众号发布状态查询", version=_VERSION, category="内容创作")
def content_wechat_publish_status(inp: WechatPublishStatusInput) -> SkillOutput[WechatPublishStatusResult]:
    """查询发布/群发状态。publish_id → freepublish/get；msg_id → mass/get。"""
    cfg = load_config().wechat_publish

    app_id = inp.app_id or get_api_key(cfg.app_id_env)
    app_secret = inp.app_secret or get_api_key(cfg.app_secret_env)
    if not app_id or not app_secret:
        return _degraded("缺少公众号凭证", category="environment", retriable=False)
    if not inp.publish_id and not inp.msg_id:
        return _degraded("publish_id 与 msg_id 至少传一个", category="environment", retriable=False)

    try:
        token = get_access_token(
            app_id=app_id, app_secret=app_secret,
            api_base=cfg.api_base, timeout_s=cfg.timeout_s,
        )
        if inp.publish_id:
            raw = get_publish_status(
                access_token=token, publish_id=inp.publish_id,
                api_base=cfg.api_base, timeout_s=cfg.timeout_s,
            )
            code = raw.get("publish_status")
            return SkillOutput(
                data=WechatPublishStatusResult(
                    kind="publish",
                    status_text=_PUBLISH_STATUS_TEXT.get(code, f"未知状态({code})"),
                    status_code=code,
                    article_url=_first_article_url(raw),
                ),
                reasoning=[ReasoningChain(
                    conclusion=f"发布状态：{_PUBLISH_STATUS_TEXT.get(code, code)}",
                    evidence=[], causal_analysis="freepublish/get",
                    risk_note="发布为异步过程，可能需要数分钟。",
                )],
                confidence=0.95, sample_size=1,
            )
        raw = get_mass_status(
            access_token=token, msg_id=inp.msg_id,
            api_base=cfg.api_base, timeout_s=cfg.timeout_s,
        )
        code = raw.get("msg_status")
        return SkillOutput(
            data=WechatPublishStatusResult(
                kind="mass",
                status_text=_MASS_STATUS_TEXT.get(code, f"未知状态({code})"),
                status_code=code,
            ),
            reasoning=[ReasoningChain(
                conclusion=f"群发状态：{_MASS_STATUS_TEXT.get(code, code)}",
                evidence=[], causal_analysis="message/mass/get",
                risk_note="群发为异步执行；如遇『审核中』请到公众号后台查看。",
            )],
            confidence=0.95, sample_size=1,
        )
    except WechatAPIError as exc:
        return _degraded(f"查询失败：{exc}", category=exc.category, retriable=exc.retriable)


def _first_article_url(raw: dict) -> str | None:
    try:
        items = raw.get("article_detail", {}).get("item", [])
        for it in items:
            if it.get("article_url"):
                return it["article_url"]
    except Exception:
        pass
    return None


def _degraded(warning: str, category: str, retriable: bool) -> SkillOutput[WechatPublishStatusResult]:
    chain = ReasoningChain(
        conclusion=f"状态查询降级：{warning[:100]}",
        evidence=[], causal_analysis="status_query",
        risk_note="请检查凭证与网络后重试。",
    )
    return SkillOutput(
        data=WechatPublishStatusResult(
            kind="unknown", status_text="查询失败",
            failure_category=category, retriable=retriable, warning=warning[:500],
        ),
        reasoning=[chain],
        confidence=0.0, sample_size=0,
        degraded=True, degradation_reason=category,
    )
