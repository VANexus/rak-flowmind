"""趋势 adapter 测试：resolve_adapter 路由 + AlibabaHotSellTrendAdapter。

TikTok/IG 自托管抓取的解析逻辑见 test_cc_scraper.py / test_ig_scraper.py；
本文件聚焦路由与阿里热销词统计。
"""
from __future__ import annotations

import pytest

from flowmind.skills._trend_adapters import (
    AlibabaHotSellTrendAdapter,
    TrendError,
    resolve_adapter,
)


class _Cfg:
    """resolve_adapter 所需的最小配置桩。"""

    default_country = "US"
    cc_scrape_page_url = "https://ads.tiktok.com/creative/creativeCenter/trends/hashtag"
    cc_scrape_period_days = 7
    cc_scrape_timeout_s = 90.0
    cc_scrape_headless = True
    cc_scrape_proxy = ""
    trend_timeout_s = 15.0


# =====================================================================
# 1. resolve_adapter 路由
# =====================================================================


def test_resolve_tiktok_routes_to_cc_scraper():
    from flowmind.skills._cc_scraper import TikTokCreativeScraperAdapter

    adapter = resolve_adapter("tiktok", _Cfg())
    assert isinstance(adapter, TikTokCreativeScraperAdapter)
    assert adapter.name == "cc_scraper"


def test_resolve_tiktok_passes_session_cookie():
    adapter = resolve_adapter("tiktok", _Cfg(), session_cookie="sessionid=abc; ttid=web")
    assert adapter.session_cookie == "sessionid=abc; ttid=web"


def test_resolve_instagram_routes_to_ig_scraper():
    from flowmind.skills._ig_scraper import InstagramSelfHostAdapter

    adapter = resolve_adapter("instagram", _Cfg(), session_cookie="sessionid=xyz")
    assert isinstance(adapter, InstagramSelfHostAdapter)
    assert adapter.name == "ig_scraper"
    assert adapter.session_cookie == "sessionid=xyz"


def test_resolve_unknown_platform_raises():
    with pytest.raises(TrendError) as ei:
        resolve_adapter("pinterest", _Cfg())
    assert ei.value.category == "environment"


# =====================================================================
# 2. AlibabaHotSellTrendAdapter（TOP alibaba.product.list 热销词统计）
# =====================================================================

class _FakeAlibabaClient:
    """打桩 AlibabaClient：记录 call 参数，返回构造的 resp。"""

    def __init__(self, *, authorized=True, resp=None, exc=None):
        self.app_key = "k" if authorized else ""
        self.app_secret = "s" if authorized else ""
        self.session = "sess" if authorized else None
        self.resp = resp
        self.exc = exc
        self.called = None

    def call(self, method, biz_params):
        self.called = (method, biz_params)
        if self.exc:
            raise self.exc
        return self.resp


def test_alibaba_unauthorized_raises_environment():
    adapter = AlibabaHotSellTrendAdapter(alibaba_cfg=object(), client=_FakeAlibabaClient(authorized=False))
    with pytest.raises(TrendError) as ei:
        adapter.fetch("alibaba")
    assert ei.value.category == "environment"
    assert "授权" in str(ei.value)


def test_alibaba_word_freq_ranking():
    resp = {"result": [
        {"subject": "Stainless Steel Water Bottle Wholesale", "keywords": ["water bottle", "steel"]},
        {"subject": "Insulated Water Bottle 1L", "keywords": []},
        {"subject": "Glass Water Bottle With Bamboo Lid", "keywords": "glass bottle, bamboo"},
        "not-a-dict",  # 非法项跳过
    ]}
    client = _FakeAlibabaClient(resp=resp)
    adapter = AlibabaHotSellTrendAdapter(alibaba_cfg=object(), client=client)
    out = adapter.fetch("alibaba", limit=5)

    assert client.called[0] == "alibaba.product.list"
    words = [r["word"] for r in out]
    assert "water" in words and "bottle" in words
    # 停用词不进榜
    assert "wholesale" not in words and "with" not in words
    # rank 连续、source 正确
    assert [r["rank"] for r in out] == [1, 2, 3, 4, 5]
    assert all(r["source"] == "alibaba_hot_sell" for r in out)


def test_alibaba_api_error_mapped_to_trend_error():
    from flowmind.skills._alibaba_client import AlibabaAPIError

    exc = AlibabaAPIError("接口错误 401", category="video", retriable=False)
    client = _FakeAlibabaClient(exc=exc)
    adapter = AlibabaHotSellTrendAdapter(alibaba_cfg=object(), client=client)
    with pytest.raises(TrendError) as ei:
        adapter.fetch("alibaba")
    assert ei.value.category == "video"
    assert ei.value.retriable is False


def test_alibaba_missing_product_list_raises_unknown():
    client = _FakeAlibabaClient(resp={"ok": True})
    adapter = AlibabaHotSellTrendAdapter(alibaba_cfg=object(), client=client)
    with pytest.raises(TrendError) as ei:
        adapter.fetch("alibaba")
    assert ei.value.category == "unknown"
