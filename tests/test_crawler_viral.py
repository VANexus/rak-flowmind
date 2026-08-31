"""crawler_viral 技能测试：通过 invoke() 走信封层。

覆盖：正常热榜 / 降级（API 不可达回退种子数据）/ 多平台 / 无效平台。
HTTP 依赖类：成功 ok=True，降级 ok=True + degraded=True。
"""
from __future__ import annotations

from unittest.mock import patch

import flowmind.skills.crawler_viral as mod
from flowmind.skill import invoke
from flowmind.skills._hot_topics_client import HotTopicError


def _mock_topics(platform: str, count: int = 5):
    return [
        {"word": f"{platform} 热点 {i}", "heat": 1000 - i * 100, "url": f"https://{platform}.com/{i}"}
        for i in range(count)
    ]


def test_viral_success():
    with patch.object(mod, "fetch_hot_topics", return_value=_mock_topics("xhs", 5)):
        r = invoke("crawler_viral", {"platform": "xhs", "limit": 5})
    assert r.ok is True
    d = r.data
    assert d.platform == "xhs"
    assert len(d.items) == 5
    assert r.metrics.degraded is False


def test_viral_degrades_to_seed_on_api_failure():
    err = HotTopicError("API 503", category="transient", retriable=True)
    with patch.object(mod, "fetch_hot_topics", side_effect=err):
        r = invoke("crawler_viral", {"platform": "xhs", "limit": 5})
    assert r.ok is True
    d = r.data
    assert r.metrics.degraded is True
    assert d.failure_category == "transient"
    # 种子数据应仍有内容
    assert len(d.items) > 0


def test_viral_wechat_platform():
    with patch.object(mod, "fetch_hot_topics", return_value=_mock_topics("wechat", 3)):
        r = invoke("crawler_viral", {"platform": "wechat", "limit": 3})
    assert r.ok is True
    assert r.data.platform == "wechat"


def test_viral_respects_limit():
    with patch.object(mod, "fetch_hot_topics", return_value=_mock_topics("xhs", 20)):
        r = invoke("crawler_viral", {"platform": "xhs", "limit": 3})
    assert r.ok is True
    assert len(r.data.items) <= 3


def test_viral_invalid_platform():
    r = invoke("crawler_viral", {"platform": "tiktok", "limit": 5})
    assert r.ok is False
    assert r.error.code == "VALIDATION"
