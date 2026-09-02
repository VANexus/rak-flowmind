"""content_wechat_account_test 技能：公众号连接测试（前端「测试连接」按钮）。

流程：用 AppID/Secret（显式传入优先，否则读环境变量）请求 access_token，
并尽力获取公众号基本信息（昵称/头像/二维码）。用于用户自填凭证后即时校验。

失败契约：HTTP 依赖类 — r.ok=True + metrics.degraded + data.failure_category。
"""
from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import Evidence, ReasoningChain, SkillOutput
from flowmind.skill import skill
from flowmind.skills._secrets import get_api_key
from flowmind.skills._wechat_client import WechatAPIError, get_access_token

_VERSION = "0.1.0"


class WechatAccountTestInput(BaseModel):
    """公众号连接测试入参。凭证优先用显式传入，否则回落环境变量。"""

    app_id: str | None = Field(default=None, description="AppID（公众号后台『基本配置』）")
    app_secret: str | None = Field(default=None, description="AppSecret（公众号后台『基本配置』）")


class WechatAccountTestResult(BaseModel):
    """连接测试业务载荷。"""

    ok: bool
    app_id_masked: str                     # 掩码展示用
    nickname: str | None = None
    account_info: dict | None = None
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


@skill(id="content_wechat_account_test", name="公众号连接测试", version=_VERSION, category="内容创作")
def content_wechat_account_test(inp: WechatAccountTestInput) -> SkillOutput[WechatAccountTestResult]:
    """公众号凭证连通性测试：token → （尽力）账号信息。

    数据流：读凭证 → access_token → getaccountbasicinfo（尽力）。
    """
    cfg = load_config().wechat_publish

    app_id = inp.app_id or get_api_key(cfg.app_id_env)
    app_secret = inp.app_secret or get_api_key(cfg.app_secret_env)
    masked = _mask(app_id or "")
    if not app_id or not app_secret:
        return _degraded(
            f"缺少公众号凭证。请填写 AppID / AppSecret，或在环境变量设置 {cfg.app_id_env} / {cfg.app_secret_env}。",
            category="environment", retriable=False, masked=masked,
        )

    account_info: dict | None = None
    try:
        token = get_access_token(
            app_id=app_id, app_secret=app_secret,
            api_base=cfg.api_base, timeout_s=cfg.timeout_s,
        )
        # 尽力获取账号基本信息（需认证账号；失败不阻断）
        try:
            url = f"{cfg.api_base.rstrip('/')}/account/getaccountbasicinfo"
            with httpx.Client(timeout=cfg.timeout_s) as c:
                resp = c.get(url, params={"access_token": token})
            data = resp.json()
            if isinstance(data, dict) and data.get("errcode", 0) == 0:
                account_info = {
                    "nickname": data.get("nickname"),
                    "head_img": data.get("head_img"),
                    "qrcode_url": data.get("qrcode_url"),
                    "service_type": data.get("service_type_info"),
                    "verify_type": data.get("verify_type_info"),
                }
        except Exception:
            account_info = None
    except WechatAPIError as exc:
        return _degraded(
            f"连接失败：{exc}",
            category=exc.category, retriable=exc.retriable, masked=masked,
        )

    chain = ReasoningChain(
        conclusion=f"公众号连接成功（{masked}）",
        evidence=[Evidence(metric="access_token", value=1, threshold=0, comparison="count")],
        causal_analysis="get_access_token → getaccountbasicinfo",
        risk_note="若需发布，请确认账号已通过认证（非认证/个人主体无法调用发布接口）。",
    )
    return SkillOutput(
        data=WechatAccountTestResult(
            ok=True, app_id_masked=masked, nickname=(account_info or {}).get("nickname"),
            account_info=account_info,
        ),
        reasoning=[chain],
        confidence=0.98,
        sample_size=1,
    )


def _mask(app_id: str) -> str:
    """掩码展示 AppID：保留前 6 位与后 4 位。"""
    if len(app_id) <= 10:
        return app_id[:2] + "****"
    return f"{app_id[:6]}****{app_id[-4:]}"


def _degraded(warning: str, category: str, retriable: bool, masked: str) -> SkillOutput[WechatAccountTestResult]:
    chain = ReasoningChain(
        conclusion=f"公众号连接失败：{warning[:100]}",
        evidence=[], causal_analysis="get_access_token",
        risk_note="请检查 AppID/AppSecret 是否正确、IP 是否已加入公众号后台白名单。",
    )
    return SkillOutput(
        data=WechatAccountTestResult(
            ok=False, app_id_masked=masked,
            failure_category=category, retriable=retriable, warning=warning[:500],
        ),
        reasoning=[chain],
        confidence=0.0,
        sample_size=0,
        degraded=True,
        degradation_reason=category,
    )
