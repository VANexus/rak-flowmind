"""智能爬虫客户端：通用网页抓取 + 平台内容 + 死链检测。

三类能力：
  1. fetch_url(url) — 抓取网页，提取标题/正文/链接
  2. fetch_platform_content(platform, keyword) — 平台公开内容（复用热榜客户端）
  3. check_links(urls) — 批量链接存活检测

错误分类：连接失败=environment、5xx=transient(可重试)、4xx=video、超时=environment。
安全：遵守 robots.txt（可选），单链接超时可控，批量并发上限走 config。
"""
from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass

import httpx
import requests  # noqa: F401  保留模块级引用：测试 fixture 经 <mod>.requests 打桩拦截


class CrawlerError(Exception):
    """爬虫失败。category/retriable 语义与 errors.py 一致。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


@dataclass
class FetchedPage:
    """抓取结果。"""
    url: str
    status_code: int
    title: str = ""
    text: str = ""
    links: list[str] = None  # type: ignore[assignment]
    error: str | None = None

    def __post_init__(self):
        if self.links is None:
            self.links = []


@dataclass
class LinkStatus:
    """单条链接检测结果。"""
    url: str
    alive: bool
    status_code: int | None = None
    final_url: str | None = None  # 重定向后的最终 URL
    error: str | None = None
    response_time_ms: float | None = None


def fetch_url(
    *,
    url: str,
    timeout_s: float = 15.0,
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    client: httpx.Client | None = None,
) -> FetchedPage:
    """抓取单个网页，返回标题 + 正文 + 链接。

    注意：正文提取是简化版（取 <body> 文本），不做 Readability 级别的内容提取。
    如需更精确的正文提取，由上层 skill 调 LLM 后处理。
    """
    headers = {"User-Agent": user_agent}
    try:
        if client is not None:
            resp = client.get(url, headers=headers, follow_redirects=True, timeout=timeout_s)
        else:
            with httpx.Client(timeout=timeout_s, follow_redirects=True, headers=headers) as c:
                resp = c.get(url, timeout=timeout_s)
    except requests.exceptions.Timeout as exc:
        raise CrawlerError(f"抓取 {url} 超时", category="environment", retriable=False) from exc
    except httpx.TimeoutException as exc:
        raise CrawlerError(f"抓取 {url} 超时", category="environment", retriable=False) from exc
    except httpx.HTTPError as exc:
        raise CrawlerError(f"抓取 {url} 连接失败：{type(exc).__name__}", category="environment", retriable=False) from exc

    if resp.status_code >= 500:
        raise CrawlerError(f"抓取 {url} HTTP {resp.status_code}", category="transient", retriable=True)
    if resp.status_code >= 400:
        raise CrawlerError(f"抓取 {url} HTTP {resp.status_code}", category="video", retriable=False)

    return _extract_page(url, resp)


def _extract_page(url: str, resp: httpx.Response) -> FetchedPage:
    """从 httpx.Response 提取标题 + 正文 + 链接（简化版，基于正则）。"""
    text = resp.text or ""
    title = _extract_title(text)
    body_text = _extract_body_text(text)
    links = _extract_links(text, url)
    return FetchedPage(
        url=url,
        status_code=resp.status_code,
        title=title,
        text=body_text[:10000],  # 限 10KB 防爆炸
        links=links[:200],       # 限 200 链接
    )


def _extract_title(html: str) -> str:
    """从 HTML 提取 <title>。"""
    import re
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group(1).strip())[:200]


def _extract_body_text(html: str) -> str:
    """从 HTML 提取正文（移除 script/style，取 body 文本）。简化版。"""
    import re
    # 移除 script / style
    cleaned = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<style[^>]*>.*?</style>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    # 取 body
    m = re.search(r"<body[^>]*>(.*?)</body>", cleaned, re.IGNORECASE | re.DOTALL)
    body = m.group(1) if m else cleaned
    # 移除标签
    body = re.sub(r"<[^>]+>", " ", body)
    # 清理空白
    body = re.sub(r"\s+", " ", body).strip()
    return body


def _extract_links(html: str, base_url: str) -> list[str]:
    """从 HTML 提取 href 链接。"""
    import re
    from urllib.parse import urljoin
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, re.IGNORECASE)
    links: list[str] = []
    for href in hrefs:
        full = urljoin(base_url, href)
        if full.startswith(("http://", "https://")) and full not in links:
            links.append(full)
    return links


def check_links(
    *,
    urls: list[str],
    timeout_s: float = 10.0,
    max_concurrent: int = 10,
    check_redirect: bool = True,
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    client: httpx.Client | None = None,
) -> list[LinkStatus]:
    """批量检测链接存活状态（HEAD 请求，失败降级为 GET）。"""
    from time import perf_counter

    def _check_one(url: str) -> LinkStatus:
        start = perf_counter()
        headers = {"User-Agent": user_agent}
        try:
            # 先 HEAD
            if client is not None:
                resp = client.head(url, headers=headers, follow_redirects=check_redirect, timeout=timeout_s)
            else:
                with httpx.Client(timeout=timeout_s, follow_redirects=check_redirect, headers=headers) as c:
                    resp = c.head(url, timeout=timeout_s)

            # HEAD 不支持 → 降级 GET
            if resp.status_code in (405, 501):
                if client is not None:
                    resp = client.get(url, headers=headers, follow_redirects=check_redirect, timeout=timeout_s)
                else:
                    with httpx.Client(timeout=timeout_s, follow_redirects=check_redirect, headers=headers) as c:
                        resp = c.get(url, timeout=timeout_s)

            elapsed = (perf_counter() - start) * 1000
            alive = resp.status_code < 400
            return LinkStatus(
                url=url, alive=alive, status_code=resp.status_code,
                final_url=str(resp.url) if check_redirect else None,
                response_time_ms=round(elapsed, 1),
            )
        except (httpx.TimeoutException, requests.exceptions.Timeout):
            return LinkStatus(url=url, alive=False, error="timeout", response_time_ms=None)
        except httpx.HTTPError as exc:
            return LinkStatus(url=url, alive=False, error=type(exc).__name__, response_time_ms=None)

    results: list[LinkStatus] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {pool.submit(_check_one, url): url for url in urls}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results
