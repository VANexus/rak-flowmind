"""可插拔关键词趋势数据源 adapter（全部自托管真实数据，零第三方趋势 API）。

- ``TikTokCreativeScraperAdapter``：Creative Center 热门话题榜
  （POST /CreativeOne/KnowledgeAPI/GetHashtagList，httpx 直连 + 浏览器降级），
  匿名 Top3，注入登录会话解锁全量 20 条/页。
- ``InstagramSelfHostAdapter``：IG 网页版话题搜索（web/search/topsearch），
  必须带站内「渠道授权」登录捕获的会话。
- ``AlibabaHotSellTrendAdapter``：调阿里国际站 TOP alibaba.product.list 拉在线商品，
  对商品标题/关键词做词频统计得到「热销词」榜单。

统一输出结构：``[{word, heat, delta, rank, industry, source}]``。
后续接入新数据源时，只需新增一个 ``TrendAdapter`` 子类，不改上层 ``b2b_keyword_trends`` 技能。
"""
from __future__ import annotations

import re
from collections import Counter


class TrendError(Exception):
    """趋势抓取失败。category/retriable 语义与 errors.py 一致。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


class TrendAdapter:
    """趋势数据源基类。子类实现 ``fetch``。"""

    name: str = "base"

    def fetch(
        self,
        platform: str,
        *,
        industry_id: int | None = None,
        limit: int = 20,
        keyword: str | None = None,
    ) -> list[dict]:
        raise NotImplementedError


# 热销词统计时的英文停用词（商品标题常见无信息量词）
_STOPWORDS = frozenset({
    "for", "with", "and", "the", "hot", "new", "sale", "wholesale", "free",
    "shipping", "factory", "supply", "oem", "odm", "custom", "2023", "2024", "2025", "2026",
})
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]{2,}")


class AlibabaHotSellTrendAdapter(TrendAdapter):
    """阿里国际站「热销词」统计后端（TOP 协议，复用 AlibabaClient）。

    call(alibaba.product.list) 拉店铺在线商品 → 对商品标题 + 关键词做词频统计 →
    输出热销词榜单（heat=出现次数）。需要已完成开放平台授权（AppKey/Secret/Session）。
    """

    name = "alibaba_hot_sell"

    def __init__(self, *, alibaba_cfg, client=None):
        self.alibaba_cfg = alibaba_cfg
        self._client = client

    def fetch(self, platform: str, *, industry_id: int | None = None, limit: int = 20, keyword: str | None = None) -> list[dict]:
        if platform != "alibaba":
            raise TrendError(f"AlibabaHotSellTrendAdapter 不支持平台 {platform}", category="unknown", retriable=False)

        client = self._client
        if client is None:
            from flowmind.skills._alibaba_client import new_client_from_config

            client = new_client_from_config(self.alibaba_cfg)

        if not client.app_key or not client.app_secret or not client.session:
            raise TrendError(
                "阿里国际站未授权（缺少 ALIBABA_APP_KEY/APP_SECRET/SESSION）；"
                "请在「设置 → B 端运营」完成开放平台授权后重试。",
                category="environment", retriable=False,
            )

        try:
            resp = client.call("alibaba.product.list", {"pageNo": 1, "pageSize": 100})
        except Exception as exc:  # AlibabaAPIError 及传输层错误统一映射为 TrendError
            category = getattr(exc, "category", "environment")
            retriable = bool(getattr(exc, "retriable", False))
            raise TrendError(str(exc), category=category, retriable=retriable) from exc

        return self._parse(resp, limit=limit)

    def _parse(self, resp: dict, *, limit: int) -> list[dict]:
        items = _extract_list(resp)
        if items is None:
            raise TrendError("阿里商品接口缺少商品列表", category="unknown", retriable=False)

        freq: Counter[str] = Counter()
        for it in items:
            if not isinstance(it, dict):
                continue
            subject = str(it.get("subject") or it.get("name") or "")
            kws = it.get("keywords") or it.get("keyword") or []
            if isinstance(kws, str):
                kws = [x.strip() for x in kws.split(",") if x.strip()]
            # 标题词计 1 次、显式关键词拆词后每个词计 2 次（权重更高）
            for w in {m.group(0).lower() for m in _WORD_RE.finditer(subject)} - _STOPWORDS:
                freq[w] += 1
            for k in kws:
                if not isinstance(k, str):
                    continue
                for m in _WORD_RE.finditer(k.lower()):
                    w = m.group(0)
                    if w not in _STOPWORDS:
                        freq[w] += 2

        ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:limit]
        return [
            {"word": w, "heat": c, "delta": None, "rank": i + 1, "industry": "通用", "source": "alibaba_hot_sell"}
            for i, (w, c) in enumerate(ranked)
        ]


def _extract_list(payload) -> list | None:
    """宽容提取接口返回里的列表（第三方返回结构不稳定）。"""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None
    for k in ("items", "list", "hashtags", "tags", "products", "product_list", "result", "records"):
        v = payload.get(k)
        if isinstance(v, list):
            return v
    data = payload.get("data")
    if isinstance(data, (dict, list)):
        return _extract_list(data)
    return None


def resolve_adapter(platform: str, cfg, *, alibaba_cfg=None, session_cookie: str = "") -> TrendAdapter:
    """按平台路由到自托管 adapter：tiktok→Creative Center 抓取；instagram→IG 会话直连；alibaba→TOP 热销词统计。

    ``session_cookie`` 为对应平台的登录会话（站内「渠道授权」捕获），
    TikTok 用于解锁全量榜单，IG 必需。
    """
    if platform == "tiktok":
        from flowmind.skills._cc_scraper import TikTokCreativeScraperAdapter

        return TikTokCreativeScraperAdapter(
            page_url=cfg.cc_scrape_page_url,
            country=cfg.default_country,
            period_days=cfg.cc_scrape_period_days,
            timeout_s=cfg.cc_scrape_timeout_s,
            headless=cfg.cc_scrape_headless,
            proxy=getattr(cfg, "cc_scrape_proxy", "") or None,
            session_cookie=session_cookie,
        )
    if platform == "instagram":
        from flowmind.skills._ig_scraper import InstagramSelfHostAdapter

        return InstagramSelfHostAdapter(
            session_cookie=session_cookie,
            timeout_s=getattr(cfg, "trend_timeout_s", 30.0) or 30.0,
        )
    if platform == "alibaba":
        if alibaba_cfg is None:
            from flowmind.config import load_config

            alibaba_cfg = load_config().alibaba
        return AlibabaHotSellTrendAdapter(alibaba_cfg=alibaba_cfg)
    raise TrendError(
        f"平台 {platform} 不受支持。可用平台：tiktok / instagram / alibaba。",
        category="environment",
        retriable=False,
    )
