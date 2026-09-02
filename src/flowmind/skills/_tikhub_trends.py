"""TikTok 趋势 adapter（TikHub 第三方 API，主数据源）。

数据端点 ``POST /api/v1/tiktok/ads/get_trends_hashtag_list`` 返回的就是 TikTok
Creative Center Trends 同构数据（TikHub 在服务端代抓），字段与旧自建
``_cc_scraper.GetHashtagList`` 完全一致，故解析复用 ``parse_cc_hashtag_items``，
仅把 source 标记为 ``tikhub``。

相比旧自建路径的收益（已真机验证）：
- 无需 cookie / 浏览器 / CDP，匿名即返回全量榜单（totalCount=100，旧路径匿名仅 Top3）；
- 无 Akamai 风控、无 Playwright 依赖，单次 HTTP 往返；
- 支持 industry_id 一级行业过滤（旧直连接口未用上）。

输出统一行结构 ``[{word, heat, delta, rank, industry, source}]``，
上层 b2b_keyword_trends / b2b_daily_digest 零改动。
"""
from __future__ import annotations

from ._cc_scraper import parse_cc_hashtag_items
from ._trend_adapters import TrendAdapter, TrendError

# 单页拉取条数（TikHub 文档默认 20；保守不调大，分页聚合到 limit）
PAGE_SIZE = 20


class TikTokTikHubTrendAdapter(TrendAdapter):
    """TikTok 热门话题榜（TikHub Analytics API）。"""

    name = "tikhub"

    def __init__(
        self,
        *,
        country: str = "US",
        period_days: int = 7,
        timeout_s: float = 30.0,
        max_pages: int = 5,
        session_cookie: str | None = None,
        client=None,
    ):
        self.country = country or "US"
        self.period_days = period_days
        self.timeout_s = timeout_s
        self.max_pages = max(1, int(max_pages))
        # 可选：站内渠道授权捕获的 TikTok 登录态，透传给 TikHub 解锁更多数据
        self.session_cookie = (session_cookie or "").strip() or None
        self._client = client  # 注入便于单测；缺省时按 config 组装

    def fetch(self, platform: str, *, industry_id: int | None = None, limit: int = 20,
              keyword: str | None = None) -> list[dict]:
        if platform != "tiktok":
            raise TrendError(f"TikTokTikHubTrendAdapter 不支持平台 {platform}",
                             category="unknown", retriable=False)
        limit = max(1, int(limit))

        client = self._client
        if client is None:
            from flowmind.config import get_config
            from flowmind.skills._tikhub_client import new_client_from_config

            client = new_client_from_config(get_config().keyword_trend)
        if not getattr(client, "api_key", ""):
            raise TrendError(
                "TikTok 趋势走 TikHub 但未配置 AI_TRENDS_API_KEY；"
                "请在 .env 填写 TikHub API Key（或将 keyword_trend.tiktok_trend_source "
                "切回 cc_scraper 走旧自建路径）。",
                category="environment", retriable=False,
            )

        rows: list[dict] = []
        page = 1
        while len(rows) < limit and page <= self.max_pages:
            data = self._call_page(
                client, page=page, need=min(PAGE_SIZE, limit - len(rows)),
                industry_id=industry_id,
            )
            items = data.get("items")
            if not isinstance(items, list) or not items:
                break
            page_rows = parse_cc_hashtag_items(items, limit=PAGE_SIZE)
            for row in page_rows:
                row["source"] = "tikhub"
            rows.extend(page_rows)

            pagination = data.get("pagination") if isinstance(data.get("pagination"), dict) else {}
            if not pagination.get("hasMore"):
                break
            page += 1

        if not rows:
            raise TrendError(
                "TikHub 热门标签榜返回为空（可尝试移除行业筛选或更换国家/周期）。",
                category="unknown", retriable=False,
            )
        # 全局 rank 连续化：优先保留接口 rankIndex，缺失时按累计序号补齐
        for i, row in enumerate(rows[:limit], start=1):
            if not isinstance(row.get("rank"), int) or row["rank"] <= 0:
                row["rank"] = i
        return rows[:limit]

    def _call_page(self, client, *, page: int, need: int, industry_id: int | None) -> dict:
        try:
            return client.trending_hashtags(
                time_range=self.period_days,
                country_code=self.country,
                page=page,
                limit=max(need, PAGE_SIZE),
                industry_id=industry_id,
                cookie=self.session_cookie,
            )
        except TrendError:
            raise
        except Exception as exc:  # TikHubError：category/retriable 透传
            category = getattr(exc, "category", "environment")
            retriable = bool(getattr(exc, "retriable", False))
            raise TrendError(f"TikHub 趋势拉取失败：{exc}",
                             category=category, retriable=retriable) from exc
