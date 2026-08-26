"""热点聚合客户端：抓取公开热榜 API（DailyHotApi 协议）并解析为统一结构。

平台映射（小红书/公众号无公开热榜 → 代理平台）在 config 的 hot_topic_endpoints 里，
本模块只做"按 endpoint 抓取 + 解析"，不关心平台语义。

解析兼容字段变体：word 取 title/name，heat 取 hotValue/rank 或从 hot 字符串解析，
url 取 url/mobilUrl，source 取顶层 name/title。

错误分类：连接失败=environment、5xx=transient(可重试)、4xx=video、结构异常=unknown。
"""
from __future__ import annotations

import re
from typing import Any

import httpx
import requests  # noqa: F401  保留模块级引用：测试 fixture 经 <mod>.requests 打桩拦截

_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([万亿w亿k])?")


def _parse_heat(raw: Any) -> int:
    """把热度解析为 int。支持 hotValue 数字、'1234.5万' 字符串、rank 数字。"""
    if raw is None:
        return 0
    if isinstance(raw, (int, float)):
        return int(raw)
    s = str(raw).strip()
    if not s:
        return 0
    m = _WEIGHT_RE.match(s)
    if not m:
        return 0
    base = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit in ("万", "w"):
        base *= 10000
    elif unit in ("亿", "e"):
        base *= 100_000_000
    elif unit == "k":
        base *= 1000
    return int(base)


def fetch_hot_topics(
    *,
    api_base: str,
    endpoint: str,
    limit: int = 20,
    timeout_s: float = 10.0,
    client: httpx.Client | None = None,
) -> list[dict]:
    """GET {api_base}/{endpoint}，返回标准化热榜 [{word,heat,delta,url,source}]。

    delta 恒为 None（聚合 API 不提供趋势字段，由消费方决定展示）。
    """
    url = f"{api_base.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        if client is not None:
            resp = client.get(url)
        else:
            with httpx.Client(timeout=timeout_s) as c:
                resp = c.get(url)
    except requests.exceptions.Timeout as exc:
        raise HotTopicError("热点抓取超时", category="environment", retriable=False) from exc
    except httpx.TimeoutException as exc:
        raise HotTopicError("热点抓取超时", category="environment", retriable=False) from exc
    except httpx.HTTPError as exc:
        raise HotTopicError(f"热点抓取连接失败：{type(exc).__name__}", category="environment", retriable=False) from exc

    if resp.status_code >= 500:
        raise HotTopicError(f"热点 HTTP {resp.status_code}", category="transient", retriable=True)
    if resp.status_code >= 400:
        raise HotTopicError(f"热点 HTTP {resp.status_code}", category="video", retriable=False)

    try:
        data = resp.json()
    except ValueError as exc:
        raise HotTopicError("热点接口不是合法 JSON", category="unknown", retriable=False) from exc

    return _parse_payload(data, source=endpoint, limit=limit)


def _parse_payload(payload: dict, *, source: str, limit: int) -> list[dict]:
    """解析 DailyHotApi 协议：data 是列表，顶层 name/title 是榜单名。"""
    items = payload.get("data")
    if not isinstance(items, list):
        raise HotTopicError("热点接口缺少 data 列表", category="unknown", retriable=False)
    source_name = str(payload.get("name") or payload.get("title") or source)
    out: list[dict] = []
    for it in items[:limit]:
        if not isinstance(it, dict):
            continue
        word = str(it.get("title") or it.get("name") or "").strip()
        if not word:
            continue
        heat = _parse_heat(it.get("hotValue", it.get("hot", it.get("rank"))))
        url = str(it.get("url") or it.get("mobilUrl") or "")
        out.append({
            "word": word,
            "heat": heat,
            "delta": None,
            "url": url,
            "source": source_name,
        })
    return out


class HotTopicError(Exception):
    """热点抓取失败。category/retriable 语义与 errors.py 一致。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable
