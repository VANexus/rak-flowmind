"""Instagram 趋势 adapter（TikHub 第三方 API，主数据源）。

数据端点 ``GET /api/v1/instagram/v2/search_hashtags?keyword=...``：
IG 网页端没有匿名可用的全站话题榜单，趋势以「关键词 → 话题搜索」形式提供
（关键词由上层按品类词池轮换注入）。已真机验证（2026-09）：
- 无需登录 cookie / 浏览器 / CDP，单次 HTTP 往返即返回全量匹配话题；
- items 字段：``{id, name, media_count, profile_pic_url, allow_following}``。

输出统一行结构 ``[{word, heat, delta, rank, industry, source}]``：
heat=media_count（该话题下的帖子总量，量级与 TikTok vv 不同口径，前端展示一致），
delta 无对应口径恒为 None，rank 按 media_count 降序。
上层 b2b_keyword_trends / b2b_daily_digest 零改动。
"""
from __future__ import annotations

from ._trend_adapters import TrendAdapter, TrendError


class InstagramTikHubTrendAdapter(TrendAdapter):
    """Instagram 话题搜索（TikHub IG V2 API，关键词驱动）。"""

    name = "tikhub-instagram"

    def __init__(self, *, timeout_s: float = 30.0, client=None):
        self.timeout_s = timeout_s
        self._client = client  # 注入便于单测；缺省时按 config 组装

    def fetch(self, platform: str, *, industry_id: int | None = None, limit: int = 20,
              keyword: str | None = None) -> list[dict]:
        if platform != "instagram":
            raise TrendError(f"InstagramTikHubTrendAdapter 不支持平台 {platform}",
                             category="unknown", retriable=False)
        kw = (keyword or "").strip().lstrip("#")
        if not kw:
            raise TrendError(
                "Instagram 话题趋势需要关键词（IG 无匿名全站榜单）；请输入关键词后重试。",
                category="unknown", retriable=False,
            )
        limit = max(1, int(limit))

        client = self._client
        if client is None:
            from flowmind.config import get_config
            from flowmind.skills._tikhub_client import new_client_from_config

            client = new_client_from_config(get_config().keyword_trend)
        if not getattr(client, "api_key", ""):
            raise TrendError(
                "Instagram 趋势走 TikHub 但未配置 TIKHUB_API_KEY；请在 .env 填写 TikHub API Key。",
                category="environment", retriable=False,
            )

        try:
            data = client.instagram_search_hashtags(keyword=kw)
        except Exception as exc:  # TikHubError：category/retriable 透传
            category = getattr(exc, "category", "environment")
            retriable = bool(getattr(exc, "retriable", False))
            raise TrendError(f"TikHub IG 话题搜索失败：{exc}",
                             category=category, retriable=retriable) from exc

        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise TrendError("TikHub IG 话题搜索返回结构异常（缺少 items）",
                             category="unknown", retriable=False)
        rows = [
            {
                "word": str(it.get("name") or "").strip(),
                "heat": int(it.get("media_count") or 0),
                "delta": None,
                "rank": 0,
                "industry": "通用",
                "source": self.name,
            }
            for it in items
            if isinstance(it, dict) and str(it.get("name") or "").strip()
        ]
        rows.sort(key=lambda r: (-r["heat"], r["word"]))
        if not rows:
            raise TrendError(
                f"TikHub IG 话题搜索「{kw}」返回为空（可更换关键词重试）。",
                category="unknown", retriable=False,
            )
        for i, row in enumerate(rows[:limit], start=1):
            row["rank"] = i
        return rows[:limit]
