"""用户浏览器 CDP 直连（browser_worker_saas_design.md M3「托管登录」的替代实现）。

不新开 Playwright 浏览器、不串流——直接 ``connect_over_cdp`` 连**用户自己的浏览器**
（启动时带 ``--remote-debugging-port=9222``）。收益：

- 真实浏览器指纹 + 用户已有登录态 → 平台风控零感知；
- 会话永不离开用户浏览器 → 无需保管、无需探活续命；
- 登录 = 用户在自己浏览器里正常登录平台，一次登录所有登录态功能全解锁。

``browser.close()`` 对 connect_over_cdp 只断开连接、**不会关闭用户浏览器**。
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from ._trend_adapters import TrendError


@contextmanager
def user_browser(cdp_url: str, *, connect_timeout_s: float = 8.0) -> Iterator[tuple]:
    """连接用户浏览器，yield (playwright 实例, 默认 context)。

    默认 context 即用户日常浏览上下文（全部 cookies / 登录态 / 扩展）。
    连接失败（浏览器未带调试端口启动）→ TrendError environment。
    """
    url = (cdp_url or "").strip()
    if not url:
        raise TrendError("未配置浏览器 CDP 地址", category="environment", retriable=False)
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # noqa: BLE001
        raise TrendError(
            "CDP 直连需要 playwright（uv sync 后执行 `playwright install chromium`）。",
            category="environment", retriable=False,
        ) from exc

    pw = sync_playwright().start()
    try:
        try:
            browser = pw.chromium.connect_over_cdp(url, timeout=connect_timeout_s * 1000)
        except Exception as exc:  # noqa: BLE001
            raise TrendError(
                "无法连接你的浏览器（CDP 调试端口未开）。"
                "请完全退出浏览器后，用以下命令重启：Chrome: chrome.exe --remote-debugging-port=9222；"
                "Edge: msedge.exe --remote-debugging-port=9222（也可在「渠道授权」页复制命令）。",
                category="environment", retriable=False,
            ) from exc
        try:
            ctx = browser.contexts[0] if browser.contexts else browser.new_context()
            yield pw, ctx
        finally:
            # 仅断开 CDP 连接，用户浏览器保持运行
            browser.close()
    finally:
        pw.stop()


def open_page(ctx, *, url: str = "about:blank", timeout_ms: int = 45_000):
    """在默认 context 开一个工作 tab（用完由调用方关闭），返回 page。"""
    page = ctx.new_page()
    if url and url != "about:blank":
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    return page
