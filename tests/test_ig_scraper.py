"""Instagram 自托管 adapter 单测（parse_topsearch 纯函数 + 会话校验）。"""
from __future__ import annotations

import pytest

from flowmind.skills._ig_scraper import InstagramSelfHostAdapter, parse_topsearch
from flowmind.skills._trend_adapters import TrendError


def _payload() -> dict:
    """IG web topsearch 已知响应结构（hashtags 数组，dict/两段式条目兼容）。"""
    return {
        "users": [],
        "hashtags": [
            {"name": "skincare", "media_count": 120_000_000},
            {"position": 2, "hashtag": {"name": "glassskin", "media_count": 890_000}},
            {"name": "cleantok", "media_count": 5_600_000},
            {"name": "", "media_count": 1},  # 无名 → 跳过
            "junk",
        ],
    }


def test_parse_topsearch_sorted_by_heat() -> None:
    rows = parse_topsearch(_payload(), limit=20)
    assert [r["word"] for r in rows] == ["skincare", "cleantok", "glassskin"]
    assert rows[0]["heat"] == 120_000_000
    assert rows[1]["heat"] == 5_600_000
    assert rows[2]["heat"] == 890_000  # 两段式 hashtag 嵌套也解析
    # 按 heat 降序后 rank 重排
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert all(r["source"] == "ig_scraper" for r in rows)
    assert all(r["delta"] is None for r in rows)


def test_parse_topsearch_limit() -> None:
    rows = parse_topsearch(_payload(), limit=2)
    assert len(rows) == 2


def test_parse_topsearch_empty_raises() -> None:
    with pytest.raises(TrendError):
        parse_topsearch({"hashtags": []}, limit=10)
    with pytest.raises(TrendError):
        parse_topsearch({}, limit=10)


# ── 会话与入参校验 ──────────────────────────────────────────────────


def test_fetch_without_session_raises() -> None:
    adapter = InstagramSelfHostAdapter(session_cookie="")
    with pytest.raises(TrendError) as ei:
        adapter.fetch("instagram", keyword="skincare")
    assert ei.value.category == "environment"
    assert "渠道授权" in str(ei.value)


def test_fetch_session_without_sessionid_raises() -> None:
    adapter = InstagramSelfHostAdapter(session_cookie="csrftoken=abc; other=1")
    with pytest.raises(TrendError) as ei:
        adapter.fetch("instagram", keyword="skincare")
    assert ei.value.category == "environment"


def test_fetch_without_keyword_raises() -> None:
    adapter = InstagramSelfHostAdapter(session_cookie="sessionid=abc; csrftoken=t")
    with pytest.raises(TrendError) as ei:
        adapter.fetch("instagram")
    assert "关键词" in str(ei.value)


def test_fetch_wrong_platform_raises() -> None:
    adapter = InstagramSelfHostAdapter(session_cookie="sessionid=abc")
    with pytest.raises(TrendError):
        adapter.fetch("tiktok", keyword="x")
