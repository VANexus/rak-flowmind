"""b2b_channel_login 技能：内嵌登录 —— 从用户浏览器（CDP 直连）捕获平台会话。

流程：渠道授权页点「站内登录」→ 在**用户自己的浏览器**新开平台登录页正常登录
（验证码/滑块/2FA 由平台原生 UI 处理，真实指纹零风控）→ 前端轮询本技能 →
CDP 读取浏览器内该平台 cookies → 出现 ``sessionid`` 即捕获成功 →
调用方（Next.js）AES-256-GCM 加密落保险库。

无账密传输、无无头浏览器、无两轮流、完全无状态。会话只从用户浏览器读出
一次即交调用方加密，本技能不落任何存储。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import build_chain

_VERSION = "0.4.0"

# cookies(urls) 按域名过滤用；登录判定 cookie 两平台同名 sessionid
_PLATFORM_URLS = {
    "tiktok": "https://www.tiktok.com/",
    "instagram": "https://www.instagram.com/",
}
_LOGIN_COOKIE = "sessionid"


class ChannelLoginInput(BaseModel):
    """内嵌登录捕获入参：无凭据，只读用户浏览器已有会话。"""

    platform: Literal["tiktok", "instagram"] = Field(description="目标平台")
    cdp_url: str = Field(default="", description="用户浏览器 CDP 地址")


class ChannelLoginPlan(BaseModel):
    """捕获结果。status=pending 表示浏览器暂无登录会话，前端继续轮询。"""

    platform: str
    ok: bool
    cookie: str = ""
    message: str = ""
    status: Literal["ok", "pending", "error"] = "error"


@skill(id="b2b_channel_login", name="渠道登录会话捕获", version=_VERSION)
def b2b_channel_login(inp: ChannelLoginInput) -> SkillOutput[ChannelLoginPlan]:
    """CDP 直连用户浏览器读取平台 cookies：有 sessionid 即捕获，无则 pending 等待登录。"""
    try:
        from flowmind.skills._cdp_browser import user_browser

        with user_browser(inp.cdp_url, connect_timeout_s=5.0) as (_pw, ctx):
            cookies = ctx.cookies(_PLATFORM_URLS[inp.platform])
    except Exception as exc:  # noqa: BLE001
        return _result(inp.platform, status="error", message=f"浏览器直连失败：{exc}")

    cookie_str = "; ".join(
        f"{c['name']}={c['value']}"
        for c in cookies
        if c.get("name") and c.get("value")
    )
    has_session = any(c.get("name") == _LOGIN_COOKIE and c.get("value") for c in cookies)

    if not has_session:
        return _result(
            inp.platform, status="pending",
            message="浏览器里还没有该平台的登录会话——请在打开的登录页完成登录（验证码/滑块按平台提示操作），登录成功后这里会自动捕获。",
        )
    return _result(
        inp.platform, status="ok", ok=True, cookie=cookie_str,
        message="登录成功，会话已捕获",
    )


def _result(
    platform: str,
    *,
    status: Literal["ok", "pending", "error"],
    ok: bool = False,
    cookie: str = "",
    message: str = "",
) -> SkillOutput[ChannelLoginPlan]:
    if status == "ok":
        conclusion = f"{platform} 登录会话捕获成功"
    elif status == "pending":
        conclusion = f"{platform} 暂无登录会话，等待用户在浏览器完成登录"
    else:
        conclusion = f"{platform} 会话捕获失败：{message}"
    chain = build_chain(
        conclusion=conclusion,
        causal_analysis="CDP 直连用户浏览器 → context.cookies(platform_url) → 检查 sessionid → 全量 cookie 返回调用方加密",
        risk_note="会话只从用户浏览器读出一次，由调用方 AES-256-GCM 加密入库；本技能无状态、不落任何存储。",
    )
    return SkillOutput(
        data=ChannelLoginPlan(platform=platform, ok=ok, cookie=cookie, message=message, status=status),
        reasoning=[chain],
        confidence=0.95 if status == "ok" else (0.6 if status == "pending" else 0.3),
        sample_size=1,
        degraded=status != "ok",
        degradation_reason=None if status == "ok" else message,
    )
