"""TikTok Creative Center 自托管抓取 adapter（零第三方 API，真实数据）。

数据源：``POST https://ads.tiktok.com/CreativeOne/KnowledgeAPI/GetHashtagList``
（Creative Center 热门话题榜后端）。经实测：

- **httpx 直连即可**（仅需 Chrome UA + Origin/Referer），无需浏览器；
- 匿名会话返回 **Top 3**（totalCount=3，官方匿名限额）；
- 带 TikTok 登录会话 cookie（M2 账号保险库）可解锁全量 20 条/页；
- 响应字段：``items[{hashtagName, vv, publishCnt, popularityCurve[{timestamp,value}], rankIndex}]``，
  ``popularityCurve`` 为 7 天热度曲线，可计算真实涨幅。

主路径失败（如 Akamai 开始拦截 httpx）自动降级 Playwright 无头浏览器
拦截页面 XHR（``_fetch_via_browser``）。输出行结构：
``[{word, heat, delta, rank, industry, source}]``，source 固定 ``cc_scraper``。
"""
from __future__ import annotations

import httpx

from ._trend_adapters import TrendAdapter, TrendError

DEFAULT_CC_PAGE_URL = "https://ads.tiktok.com/business/creativecenter/hashtag/popular/pc/en"
HASHTAG_LIST_URL = "https://ads.tiktok.com/CreativeOne/KnowledgeAPI/GetHashtagList"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
_BASE_HEADERS = {
    "User-Agent": _UA,
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Origin": "https://ads.tiktok.com",
    "Referer": DEFAULT_CC_PAGE_URL,
}

# 反自动化检测初始化（仅浏览器降级路径使用；无头 UA + webdriver 标记会被 Akamai 秒杀）
_STEALTH_INIT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
window.chrome = {runtime: {}};
"""


def curve_delta(curve: list | None) -> int | None:
    """7 天热度曲线 → 整段涨幅百分比（首尾对比）；无效曲线返回 None。"""
    if not isinstance(curve, list) or len(curve) < 2:
        return None
    vals: list[float] = []
    for pt in curve:
        if not isinstance(pt, dict):
            continue
        v = pt.get("value")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append(float(v))
    if len(vals) < 2:
        return None
    first = vals[0]
    last = vals[-1]
    if first <= 0:
        return None
    return round((last - first) / first * 100)


def parse_cc_hashtag_items(items: list, *, limit: int) -> list[dict]:
    """解析 GetHashtagList.items → 统一趋势行（纯函数，便于单测）。"""
    out: list[dict] = []
    for it in items:  # 先过滤后截断
        if len(out) >= limit:
            break
        if not isinstance(it, dict):
            continue
        word = str(it.get("hashtagName") or it.get("name") or "").strip().lstrip("#")
        if not word:
            continue
        vv = it.get("vv")
        heat = int(vv) if isinstance(vv, (int, float)) and not isinstance(vv, bool) else 0
        rank = it.get("rankIndex")
        out.append({
            "word": word,
            "heat": heat,
            "delta": curve_delta(it.get("popularityCurve")),
            "rank": int(rank) if isinstance(rank, (int, float)) and not isinstance(rank, bool) else len(out) + 1,
            "industry": "通用",
            "source": "cc_scraper",
        })
    if not out:
        raise TrendError("Creative Center 榜单解析为空（响应结构可能已变化）", category="unknown", retriable=False)
    return out


def parse_cc_payload(payload: dict, *, limit: int) -> list[dict]:
    """浏览器 XHR 兜底路径的旧版解析（creative_radar_api hashtag/list 响应）。"""
    data = payload.get("data") if isinstance(payload, dict) else None
    items: list | None = None
    if isinstance(data, dict):
        for key in ("list", "hashtag_list", "hashtags", "items"):
            cand = data.get(key)
            if isinstance(cand, list):
                items = cand
                break
    elif isinstance(data, list):
        items = data
    if items is None and isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list) and all(isinstance(x, dict) for x in v[:3]):
                items = v
                break
    if items is None:
        raise TrendError("Creative Center 响应缺少榜单列表", category="unknown", retriable=False)

    out: list[dict] = []
    for it in items:
        if len(out) >= limit:
            break
        if not isinstance(it, dict):
            continue
        word = _word_of(it)
        if not word:
            continue
        stat = it.get("statistic") or it.get("stats") or {}
        heat = _int_of(stat.get("vv") if isinstance(stat, dict) else None)
        if heat == 0:
            heat = _int_of(stat.get("publish_cnt") if isinstance(stat, dict) else None)
        if heat == 0:
            heat = _int_of(it.get("vv", it.get("heat", it.get("publishCnt"))))
        delta = _rise_rate(stat if isinstance(stat, dict) else it)
        out.append({
            "word": word,
            "heat": heat,
            "delta": delta,
            "rank": len(out) + 1,
            "industry": "通用",
            "source": "cc_scraper",
        })
    if not out:
        raise TrendError("Creative Center 榜单解析为空（页面结构可能已变化）", category="unknown", retriable=False)
    return out


def _word_of(it: dict) -> str:
    for parent_key in ("hashtag", "tag", "keyword"):
        parent = it.get(parent_key)
        if isinstance(parent, dict):
            for k in ("name", "hashtag_name", "title", "word"):
                v = parent.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip().lstrip("#")
    for k in ("hashtag_name", "name", "hashtagName", "title", "word"):
        v = it.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lstrip("#")
    return ""


def _int_of(v) -> int:
    if isinstance(v, bool) or v is None:
        return 0
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        digits = "".join(ch for ch in v if ch.isdigit())
        return int(digits) if digits else 0
    return 0


def _rise_rate(obj: dict) -> int | None:
    for k in ("vv_rise_rate", "rise_rate", "growth_rate", "riseRate"):
        v = obj.get(k)
        if v is None or isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            return int(v * 100) if abs(v) < 10 else int(v)
        if isinstance(v, str):
            s = v.replace("%", "").strip()
            neg = s.startswith("-")
            digits = "".join(ch for ch in s if ch.isdigit() or ch == ".")
            if digits:
                try:
                    n = float(digits)
                    n = n * 100 if abs(n) < 10 else n
                    return -int(n) if neg else int(n)
                except ValueError:
                    continue
    return None


class TikTokCreativeScraperAdapter(TrendAdapter):
    """Creative Center 热门话题榜（httpx 直连主路径 + Playwright 浏览器降级）。"""

    name = "cc_scraper"

    def __init__(
        self,
        *,
        page_url: str = DEFAULT_CC_PAGE_URL,
        country: str = "US",
        period_days: int = 7,
        timeout_s: float = 90.0,
        headless: bool = True,
        proxy: str | None = None,
        session_cookie: str | None = None,
        cdp_url: str = "",
    ):
        self.page_url = page_url
        self.country = country
        self.period_days = period_days
        self.timeout_s = timeout_s
        self.headless = headless
        # 海外出口代理（可选）：http://host:port / socks5://host:port
        self.proxy = (proxy or "").strip() or None
        # TikTok 登录会话（保险库/设置兜底注入），可解锁全量榜单；空 = 匿名 Top3
        self.session_cookie = (session_cookie or "").strip() or None
        # CDP 直连用户浏览器：提供时优先走用户浏览器（真实指纹 + 浏览器登录态）
        self.cdp_url = (cdp_url or "").strip()

    def fetch(self, platform: str, *, industry_id: int | None = None, limit: int = 20, keyword: str | None = None) -> list[dict]:
        if platform != "tiktok":
            raise TrendError(f"TikTokCreativeScraperAdapter 不支持平台 {platform}", category="unknown", retriable=False)
        if self.cdp_url:
            rows = self._fetch_via_cdp_browser(limit=limit)
            if rows:
                return rows
            raise TrendError(
                "CDP 直连未拦截到榜单响应（检查浏览器是否登录 TikTok / 网络是否可达）。",
                category="environment", retriable=True,
            )
        try:
            return self._fetch_via_http(limit=limit)
        except TrendError as http_exc:
            if http_exc.category != "environment":
                raise
            rows = self._fetch_via_browser(limit=limit)  # httpx 被风控时降级
            if rows:
                return rows
            raise http_exc from None

    # ── CDP 直连：用户自己的浏览器（真实指纹 + 浏览器登录态） ─────────

    def _fetch_via_cdp_browser(self, *, limit: int) -> list[dict]:
        captured: list[dict] = []

        def _on_response(resp):  # Playwright 内部回调
            try:
                if "creative_radar_api" not in resp.url and "KnowledgeAPI" not in resp.url:
                    return
                if "hashtag" not in resp.url:
                    return
                body = resp.json()
                if isinstance(body, dict):
                    captured.append(body)
            except Exception:  # noqa: BLE001
                pass

        from flowmind.skills._cdp_browser import open_page, user_browser

        try:
            with user_browser(self.cdp_url, connect_timeout_s=10.0) as (_pw, ctx):
                page = open_page(ctx, url=self.page_url, timeout_ms=int(self.timeout_s * 1000))
                try:
                    page.on("response", _on_response)
                    page.wait_for_load_state("networkidle", timeout=self.timeout_s * 1000)
                    page.wait_for_timeout(2000)
                finally:
                    try:
                        page.close()
                    except Exception:  # noqa: BLE001
                        pass
        except TrendError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TrendError(
                f"CDP 直连抓取失败：{type(exc).__name__}: {exc}",
                category="environment", retriable=True,
            ) from exc

        best: list[dict] | None = None
        for payload in captured:
            try:
                if isinstance(payload.get("items"), list):
                    rows = parse_cc_hashtag_items(payload["items"], limit=limit)
                else:
                    rows = parse_cc_payload(payload, limit=limit)
                if best is None or len(rows) > len(best):
                    best = rows
            except TrendError:
                continue
        return best or []

    # ── 主路径：httpx 直连 GetHashtagList ──────────────────────────────

    def _fetch_via_http(self, *, limit: int) -> list[dict]:
        body: dict = {
            "timeRange": self.period_days,
            "countryCode": self.country,
            "page": 1,
            "limit": max(limit, 20),
        }
        headers = dict(_BASE_HEADERS)
        if self.session_cookie:
            headers["Cookie"] = self.session_cookie
        try:
            resp = httpx.post(
                HASHTAG_LIST_URL,
                json=body,
                headers=headers,
                timeout=min(self.timeout_s, 30.0),
                proxy=self.proxy,
            )
        except Exception as exc:  # noqa: BLE001
            raise TrendError(
                f"Creative Center 直连失败：{type(exc).__name__}: {exc}",
                category="environment", retriable=True,
            ) from exc
        if resp.status_code == 403:
            raise TrendError(
                "Creative Center 403（Akamai 风控），尝试浏览器降级",
                category="environment", retriable=True,
            )
        if resp.status_code != 200:
            raise TrendError(
                f"Creative Center HTTP {resp.status_code}",
                category="environment", retriable=True,
            )
        try:
            payload = resp.json()
        except ValueError as exc:
            raise TrendError("Creative Center 响应非 JSON", category="unknown", retriable=True) from exc

        base = payload.get("BaseResp") or {}
        if isinstance(base, dict) and base.get("StatusCode") not in (0, None):
            raise TrendError(
                f"Creative Center 业务错误：StatusCode={base.get('StatusCode')} {base.get('StatusMessage', '')}",
                category="environment", retriable=True,
            )
        items = payload.get("items")
        if not isinstance(items, list):
            raise TrendError("Creative Center 响应缺少 items", category="unknown", retriable=True)
        return parse_cc_hashtag_items(items, limit=limit)

    # ── 降级路径：Playwright 拦截页面 XHR ──────────────────────────────

    def _fetch_via_browser(self, *, limit: int) -> list[dict]:
        captured: list[dict] = []

        def _on_response(resp):  # Playwright 内部线程回调
            try:
                if "creative_radar_api" not in resp.url and "KnowledgeAPI" not in resp.url:
                    return
                if "hashtag" not in resp.url:
                    return
                body = resp.json()
                if isinstance(body, dict):
                    captured.append(body)
            except Exception:  # noqa: BLE001
                pass

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise TrendError(
                "浏览器降级需要 playwright（uv sync 后执行 `playwright install chromium`）。",
                category="environment", retriable=False,
            ) from exc

        try:
            with sync_playwright() as p:
                launch_kwargs: dict = {"headless": self.headless}
                if self.proxy:
                    launch_kwargs["proxy"] = {"server": self.proxy}
                browser = p.chromium.launch(**launch_kwargs)
                try:
                    ctx = browser.new_context(
                        locale="en-US",
                        viewport={"width": 1440, "height": 900},
                        user_agent=_UA,
                    )
                    ctx.add_init_script(_STEALTH_INIT)
                    page = ctx.new_page()
                    page.on("response", _on_response)
                    page.goto(self.page_url, wait_until="domcontentloaded", timeout=self.timeout_s * 1000)
                    page.wait_for_load_state("networkidle", timeout=self.timeout_s * 1000)
                    page.wait_for_timeout(2000)
                finally:
                    browser.close()
        except TrendError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TrendError(
                f"Creative Center 浏览器抓取失败：{type(exc).__name__}: {exc}",
                category="environment", retriable=True,
            ) from exc

        if not captured:
            raise TrendError(
                "浏览器降级未拦截到榜单响应（可能被验证码拦截或页面结构变化）。",
                category="environment", retriable=True,
            )

        best: list[dict] | None = None
        for payload in captured:
            # GetHashtagList 形状优先，其次旧版 radar 形状
            try:
                if isinstance(payload.get("items"), list):
                    rows = parse_cc_hashtag_items(payload["items"], limit=limit)
                else:
                    rows = parse_cc_payload(payload, limit=limit)
                if best is None or len(rows) > len(best):
                    best = rows
            except TrendError:
                continue
        return best or []
