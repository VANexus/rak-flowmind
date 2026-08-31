"""b2b_channel_login 技能：站内登录第三方平台，捕获会话 cookie。

流程：点「登录」→ 本机弹出有头 Chromium 打开平台登录页 → 用户在窗口里手动登录
（含验证码/2FA）→ 脚本轮询检测登录态 cookie → 提取全部 cookie 序列化返回 →
调用方（Next.js）加密落库（ai_config）。之后趋势 adapter 注入会话即可解锁
登录态功能（TikTok 全量榜单、IG 话题数据）。

登录态检测 cookie：TikTok/Instagram 均以 ``sessionid`` 为准。
注意：有头浏览器在 MCP 进程所在机器弹出（当前为本地部署，等价于"自己的电脑"）；
远程部署形态走 Worker 托管登录（见 browser_worker_saas_design.md M3）。
"""
from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import build_chain

_VERSION = "0.2.0"

_LOGIN_URLS = {
    "tiktok": "https://www.tiktok.com/login/phone-or-email/email",
    "instagram": "https://www.instagram.com/accounts/login/",
}
# 登录成功判定 cookie（TikTok / Instagram 均为 sessionid）
_LOGIN_COOKIE = "sessionid"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

_STEALTH_INIT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
window.chrome = {runtime: {}};
"""


class ChannelLoginInput(BaseModel):
    """渠道登录入参。"""
    platform: Literal["tiktok", "instagram"] = Field(description="目标平台")
    timeout_s: int = Field(default=240, ge=30, le=600, description="等待用户完成登录的时限（秒）")


class ChannelLoginPlan(BaseModel):
    """渠道登录结果载荷。ok=False 时 cookie 为空并给出 message。"""
    platform: str
    ok: bool
    cookie: str = ""
    message: str = ""


@skill(id="b2b_channel_login", name="渠道站内登录", version=_VERSION)
def b2b_channel_login(inp: ChannelLoginInput) -> SkillOutput[ChannelLoginPlan]:
    """弹出本机有头浏览器，用户手动登录平台后捕获会话 cookie（绝不保存密码）。"""
    url = _LOGIN_URLS[inp.platform]
    cookie_str = ""
    ok = False
    message = ""

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        return _result(inp.platform, False, "", f"未安装 playwright：{exc}", _VERSION)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False, args=["--disable-blink-features=AutomationControlled"])
            try:
                ctx = browser.new_context(locale="en-US", viewport={"width": 1280, "height": 860}, user_agent=_UA)
                ctx.add_init_script(_STEALTH_INIT)
                page = ctx.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=45_000)

                deadline = time.monotonic() + inp.timeout_s
                while time.monotonic() < deadline:
                    cookies = ctx.cookies()
                    if any(c.get("name") == _LOGIN_COOKIE and c.get("value") for c in cookies):
                        pairs = [f"{c['name']}={c['value']}" for c in cookies if c.get("name") and c.get("value")]
                        cookie_str = "; ".join(pairs)
                        ok = True
                        message = "登录成功，会话已捕获"
                        break
                    time.sleep(3)
                else:
                    message = f"等待登录超时（{inp.timeout_s}s），未捕获到会话"
                try:
                    ctx.close()
                except Exception:  # noqa: BLE001
                    pass
            finally:
                browser.close()
    except Exception as exc:  # noqa: BLE001
        return _result(inp.platform, False, "", f"登录窗口异常：{type(exc).__name__}: {exc}", _VERSION)

    return _result(inp.platform, ok, cookie_str, message, _VERSION)


def _result(platform: str, ok: bool, cookie: str, message: str, version: str) -> SkillOutput[ChannelLoginPlan]:
    chain = build_chain(
        conclusion=f"{platform} 渠道登录{'成功' if ok else '未完成'}：{message}",
        causal_analysis=f"有头 Chromium 打开登录页 → 轮询 sessionid cookie → 提取 {len(cookie)} 字符会话",
        risk_note="只捕获 cookie 会话，绝不收集账号密码；会话由调用方加密保存。",
    )
    return SkillOutput(
        data=ChannelLoginPlan(platform=platform, ok=ok, cookie=cookie if ok else "", message=message),
        reasoning=[chain],
        confidence=0.95 if ok else 0.3,
        sample_size=1,
        degraded=not ok,
        degradation_reason=None if ok else message,
    )
