"""b2b_channel_verify 技能：渠道会话探活（只读，绝不触发写操作）。

对给定平台会话 cookie 发一次轻量官方端点请求，判定会话是否仍有效：
- TikTok：``GET /passport/account/info/v2/``（需 sessionid，返回当前账号信息）
- Instagram：``GET /api/v1/users/web_profile_info/?username=instagram``（需 sessionid）

用途：保险库「校验会话」按钮、每日探活告警（browser_worker_saas_design.md M2）。
"""
from __future__ import annotations

from typing import Literal

import httpx
from pydantic import BaseModel, Field

from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import build_chain

_VERSION = "0.1.0"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

_PROBES = {
    "tiktok": {
        "url": "https://www.tiktok.com/passport/account/info/v2/",
        "params": {"aid": "1988"},
        "headers": {},
    },
    "instagram": {
        "url": "https://www.instagram.com/api/v1/users/web_profile_info/",
        "params": {"username": "instagram"},
        "headers": {"X-IG-App-ID": "936619743392459", "Accept": "application/json"},
    },
}


def _cookie_value(cookie: str, name: str) -> str:
    for part in cookie.split(";"):
        k, _, v = part.strip().partition("=")
        if k == name:
            return v
    return ""


class ChannelVerifyInput(BaseModel):
    """会话探活入参。"""
    platform: Literal["tiktok", "instagram"] = Field(description="目标平台")
    cookie: str = Field(description="会话 cookie 串（\"k=v; k2=v2\"），须含 sessionid")


class ChannelVerifyPlan(BaseModel):
    """探活结果。status: active | expired | risk_control。"""
    platform: str
    ok: bool            # 探活请求本身成功执行（得到明确结论）
    status: str         # active → 会话有效；expired → 失效需重登；risk_control → 风控/验证码
    account: str = ""   # 探活拿到的账号标识（昵称/用户名），尽量回填
    message: str = ""


@skill(id="b2b_channel_verify", name="渠道会话探活", version=_VERSION)
def b2b_channel_verify(inp: ChannelVerifyInput) -> SkillOutput[ChannelVerifyPlan]:
    """对平台会话做一次只读探活，判定 active / expired / risk_control。"""
    cookie = (inp.cookie or "").strip()
    if "sessionid=" not in cookie:
        return _result(inp.platform, True, "expired", "", "会话缺少 sessionid，视为失效")

    probe = _PROBES[inp.platform]
    headers = {
        "User-Agent": _UA,
        "Cookie": cookie,
        "Referer": f"https://www.{inp.platform}.com/",
        **probe["headers"],
    }
    try:
        resp = httpx.get(
            probe["url"],
            params=probe["params"],
            headers=headers,
            timeout=15.0,
            follow_redirects=False,
        )
    except Exception as exc:  # noqa: BLE001
        return _result(inp.platform, False, "risk_control", "", f"探活请求异常：{type(exc).__name__}: {exc}")

    if resp.status_code == 200:
        account = _extract_account(inp.platform, resp)
        return _result(inp.platform, True, "active", account, f"会话有效（{resp.status_code}）")

    if resp.status_code in (401, 403):
        return _result(inp.platform, True, "expired", "", f"会话已失效（HTTP {resp.status_code}），请重新登录")

    if resp.status_code in (429,):
        return _result(inp.platform, True, "risk_control", "", "触发限流（HTTP 429），稍后再试")

    if 300 <= resp.status_code < 400:
        return _result(inp.platform, True, "expired", "", f"被重定向到登录页（HTTP {resp.status_code}），会话失效")

    return _result(inp.platform, True, "risk_control", "", f"HTTP {resp.status_code}，疑似风控，请人工确认")


def _extract_account(platform: str, resp: httpx.Response) -> str:
    """尽量从探活响应里抠出账号标识，用于展示「这是哪个号」。"""
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return ""
    if platform == "tiktok":
        d = data.get("data") or {}
        user = d.get("user") or d or {}
        return str(user.get("nickname") or user.get("unique_id") or "")
    if platform == "instagram":
        user = (data.get("data") or {}).get("user") or {}
        return str(user.get("username") or user.get("full_name") or "")
    return ""


def _result(platform: str, ok: bool, status: str, account: str, message: str) -> SkillOutput[ChannelVerifyPlan]:
    chain = build_chain(
        conclusion=f"{platform} 会话探活：{status}{'（' + account + '）' if account else ''} — {message}",
        causal_analysis=f"只读请求平台官方账号端点 → 按 HTTP 状态分类 active/expired/risk_control，不发任何写请求",
        risk_note="探活为只读操作；会话 cookie 仅用于本次请求头，不落日志。",
    )
    return SkillOutput(
        data=ChannelVerifyPlan(platform=platform, ok=ok, status=status, account=account, message=message),
        reasoning=[chain],
        confidence=0.9 if ok else 0.4,
        sample_size=1,
        degraded=not ok,
        degradation_reason=None if ok else message,
    )
