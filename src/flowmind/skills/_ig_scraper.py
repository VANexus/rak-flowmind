"""Instagram 自托管 adapter（登录会话 + web topsearch，零第三方 API）。

数据源：``GET https://www.instagram.com/web/search/topsearch/?context=blended&query=...``
（IG 网页版话题搜索）。实测要求：

- **必须带登录会话**（未登录直接 302 → 登录页），会话由站内「渠道授权」登录捕获；
- 请求头需 ``x-csrftoken``（取自 cookie ``csrftoken``）+ ``x-requested-with: XMLHttpRequest``；
- 返回 ``hashtags[]``：``{name, media_count}``，按 media_count 排序即话题热力榜。

输出行结构与 TikTok adapter 一致：``[{word, heat, delta, rank, industry, source}]``，
source 固定 ``ig_scraper``；IG 无公开热度曲线，delta 为 None。
"""
from __future__ import annotations

import httpx

from ._trend_adapters import TrendAdapter, TrendError

TOPSEARCH_URL = "https://www.instagram.com/web/search/topsearch/"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _cookie_value(cookie: str, name: str) -> str:
    for part in cookie.split(";"):
        k, _, v = part.strip().partition("=")
        if k == name:
            return v
    return ""


def parse_topsearch(payload: dict, *, limit: int) -> list[dict]:
    """解析 topsearch 响应 → 统一趋势行（纯函数，便于单测）。"""
    raw = payload.get("hashtags")
    items: list = []
    if isinstance(raw, list):
        items = raw
    else:
        # 兼容 dict/嵌套变体
        for v in (payload.get("data") or {}).values() if isinstance(payload.get("data"), dict) else []:
            if isinstance(v, list):
                items = v
                break

    out: list[dict] = []
    for it in items:
        if len(out) >= limit:
            break
        if not isinstance(it, dict):
            continue
        # 兼容两种条目：{"name","media_count"} 与 {"position","hashtag":{...}}
        node = it.get("hashtag") if isinstance(it.get("hashtag"), dict) else it
        name = str(node.get("name") or "").strip().lstrip("#")
        if not name:
            continue
        mc = node.get("media_count")
        heat = int(mc) if isinstance(mc, (int, float)) and not isinstance(mc, bool) else 0
        out.append({
            "word": name,
            "heat": heat,
            "delta": None,
            "rank": len(out) + 1,
            "industry": "通用",
            "source": "ig_scraper",
        })
    if not out:
        raise TrendError("Instagram 话题搜索结果为空", category="unknown", retriable=False)
    out.sort(key=lambda r: -r["heat"])
    for i, r in enumerate(out):
        r["rank"] = i + 1
    return out


class InstagramSelfHostAdapter(TrendAdapter):
    """Instagram 话题搜索（CDP 直连用户浏览器 / 登录会话直连 web 端点）。"""

    name = "ig_scraper"

    def __init__(
        self,
        *,
        session_cookie: str = "",
        timeout_s: float = 30.0,
        proxy: str | None = None,
        cdp_url: str = "",
    ):
        # 会话（保险库/设置兜底路径）；cdp_url 存在时优先走用户浏览器
        self.session_cookie = (session_cookie or "").strip()
        self.timeout_s = timeout_s
        self.proxy = (proxy or "").strip() or None
        self.cdp_url = (cdp_url or "").strip()

    def fetch(self, platform: str, *, industry_id: int | None = None, limit: int = 20, keyword: str | None = None) -> list[dict]:
        if platform != "instagram":
            raise TrendError(f"InstagramSelfHostAdapter 不支持平台 {platform}", category="unknown", retriable=False)
        kw = (keyword or "").strip().lstrip("#")
        if not kw:
            raise TrendError(
                "Instagram 话题搜索需要提供关键词（keyword）；网页端无匿名全站趋势榜。",
                category="environment", retriable=False,
            )
        if self.cdp_url:
            return self._fetch_via_user_browser(kw, limit=limit)
        return self._fetch_via_cookie(kw, limit=limit)

    # ── 主路径：CDP 直连用户浏览器（真实指纹 + 浏览器登录态） ──────────

    def _fetch_via_user_browser(self, kw: str, *, limit: int) -> list[dict]:
        from flowmind.skills._cdp_browser import open_page, user_browser

        with user_browser(self.cdp_url) as (_pw, ctx):
            page = open_page(ctx, url=f"https://www.instagram.com/explore/tags/{kw}/")
            try:
                if "/accounts/login" in (page.url or ""):
                    raise TrendError(
                        "Instagram 未登录：请在你的浏览器里登录 instagram.com 后重试。",
                        category="environment", retriable=False,
                    )
                # 同源 fetch：自动携带全部 cookies（HttpOnly 也带）+ 真实 TLS 指纹
                try:
                    payload = page.evaluate(
                        """async (kw) => {
                            const r = await fetch(
                                '/web/search/topsearch/?context=blended&query=' + encodeURIComponent(kw),
                                { headers: { 'X-Requested-With': 'XMLHttpRequest', 'X-IG-App-ID': '936619743392459' } },
                            );
                            if (!r.ok) return { __status: r.status };
                            return await r.json();
                        }""",
                        kw,
                    )
                except Exception as exc:  # noqa: BLE001
                    raise TrendError(
                        f"Instagram 页面内请求失败：{type(exc).__name__}: {exc}",
                        category="environment", retriable=True,
                    ) from exc
            finally:
                try:
                    page.close()
                except Exception:  # noqa: BLE001
                    pass

        if not isinstance(payload, dict):
            raise TrendError("Instagram 响应非 JSON", category="unknown", retriable=True)
        status = payload.get("__status")
        if status in (401, 403, 302):
            raise TrendError(
                "Instagram 拒绝访问：请在你的浏览器里登录 instagram.com 后重试。",
                category="environment", retriable=False,
            )
        return parse_topsearch(payload, limit=limit)

    # ── 兜底路径：保险库/设置会话 cookie 直连 ─────────────────────────

    def _fetch_via_cookie(self, kw: str, *, limit: int) -> list[dict]:
        if not self.session_cookie or "sessionid=" not in self.session_cookie:
            raise TrendError(
                "Instagram 未登录：请先连接你的浏览器（渠道授权），或在「设置 → B 端运营」粘贴会话。",
                category="environment", retriable=False,
            )

        headers = {
            "User-Agent": _UA,
            "Accept": "application/json",
            "X-Requested-With": "XMLHttpRequest",
            "X-CSRFToken": _cookie_value(self.session_cookie, "csrftoken"),
            "X-IG-App-ID": "936619743392459",
            "Referer": f"https://www.instagram.com/explore/tags/{kw}/",
        }
        try:
            resp = httpx.get(
                TOPSEARCH_URL,
                params={"context": "blended", "query": kw},
                headers={**headers, "Cookie": self.session_cookie},
                timeout=self.timeout_s,
                proxy=self.proxy,
                follow_redirects=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise TrendError(
                f"Instagram 搜索请求失败：{type(exc).__name__}: {exc}",
                category="environment", retriable=True,
            ) from exc

        if resp.status_code in (301, 302, 303, 307):
            raise TrendError(
                "Instagram 会话已失效（被重定向到登录页），请重新在「渠道授权」登录。",
                category="environment", retriable=False,
            )
        if resp.status_code == 401 or resp.status_code == 403:
            raise TrendError(
                f"Instagram 拒绝访问（HTTP {resp.status_code}）：会话失效或风控，请重新登录。",
                category="environment", retriable=False,
            )
        if resp.status_code != 200:
            raise TrendError(
                f"Instagram 搜索 HTTP {resp.status_code}",
                category="environment", retriable=True,
            )
        try:
            payload = resp.json()
        except ValueError as exc:
            raise TrendError("Instagram 响应非 JSON", category="unknown", retriable=True) from exc
        return parse_topsearch(payload, limit=limit)
