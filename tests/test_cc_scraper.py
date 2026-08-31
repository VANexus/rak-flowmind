"""TikTok Creative Center 自托管抓取 adapter 单测（纯函数）。"""
from __future__ import annotations

import pytest

from flowmind.skills._cc_scraper import curve_delta, parse_cc_hashtag_items, parse_cc_payload
from flowmind.skills._trend_adapters import TrendError


def _payload_canonical() -> dict:
    """creative_radar_api 已知响应结构。"""
    return {
        "code": 0,
        "message": "OK",
        "data": {
            "list": [
                {
                    "hashtag": {"id": "1", "name": "skincare routine"},
                    "statistic": {"rank": 1, "vv": 1200000, "publish_cnt": 5300, "vv_rise_rate": 0.35},
                },
                {
                    "hashtag": {"id": "2", "name": "#glassskin"},
                    "statistic": {"rank": 2, "vv": 980000, "publish_cnt": 4100, "vv_rise_rate": "-12%"},
                },
                {"hashtag": {}, "statistic": {}},  # 无名条目 → 跳过
            ],
        },
    }


def test_parse_canonical_shape() -> None:
    rows = parse_cc_payload(_payload_canonical(), limit=20)
    assert [r["word"] for r in rows] == ["skincare routine", "glassskin"]
    assert rows[0]["heat"] == 1200000
    assert rows[0]["delta"] == 35  # 0.35 → 35%
    assert rows[1]["delta"] == -12  # "-12%" → -12
    assert rows[0]["source"] == "cc_scraper"
    assert rows[0]["rank"] == 1


def test_parse_limit() -> None:
    payload = _payload_canonical()
    payload["data"]["list"] = payload["data"]["list"] * 10
    rows = parse_cc_payload(payload, limit=5)
    assert len(rows) == 5


def test_parse_flat_items_fallback() -> None:
    payload = {
        "data": {
            "items": [
                {"hashtagName": "cleantok", "vv": "5,600"},
                {"name": "tiktokmademebuyit", "heat": 77000},
            ],
        },
    }
    rows = parse_cc_payload(payload, limit=10)
    assert [r["word"] for r in rows] == ["cleantok", "tiktokmademebuyit"]
    assert rows[0]["heat"] == 5600
    assert rows[1]["heat"] == 77000


def test_parse_missing_list_raises() -> None:
    with pytest.raises(TrendError):
        parse_cc_payload({"code": 0, "data": {}}, limit=10)


def test_parse_empty_rows_raises() -> None:
    with pytest.raises(TrendError):
        parse_cc_payload({"code": 0, "data": {"list": [1, "x", None]}}, limit=10)


# ── GetHashtagList 形状（httpx 主路径）───────────────────────────────


def _hashtag_items() -> list:
    """CreativeOne/KnowledgeAPI/GetHashtagList 实测响应结构。"""
    return [
        {
            "hashtagID": 2534,
            "hashtagName": "dollyparton",
            "industryIDs": [23000000000],
            "popularityCurve": [{"timestamp": 1787529600, "value": 100}, {"timestamp": 1787616000, "value": 160}],
            "publishCnt": 346223,
            "rankIndex": 1,
            "vv": 1937783877,
        },
        {
            "hashtagName": "#dolly",
            "popularityCurve": [{"timestamp": 1787529600, "value": 0}, {"timestamp": 1787616000, "value": 500}],
            "publishCnt": 97704,
            "rankIndex": 2,
            "vv": 364989556,
        },
        {"hashtagName": "", "vv": 1},  # 无名 → 跳过
        "junk",
    ]


def test_parse_hashtag_items() -> None:
    rows = parse_cc_hashtag_items(_hashtag_items(), limit=20)
    assert [r["word"] for r in rows] == ["dollyparton", "dolly"]
    assert rows[0]["heat"] == 1937783877
    assert rows[0]["delta"] == 60  # 100 → 160
    assert rows[0]["rank"] == 1
    assert rows[1]["rank"] == 2  # rankIndex 保留
    assert rows[1]["delta"] is None  # 曲线首值为 0 → 无法计算
    assert all(r["source"] == "cc_scraper" for r in rows)


def test_parse_hashtag_items_limit() -> None:
    rows = parse_cc_hashtag_items(_hashtag_items() * 5, limit=4)
    assert len(rows) == 4  # 无效条目不占槽位


def test_parse_hashtag_items_empty_raises() -> None:
    with pytest.raises(TrendError):
        parse_cc_hashtag_items([], limit=10)


# ── curve_delta ─────────────────────────────────────────────────────


def test_curve_delta_variants() -> None:
    assert curve_delta([{"value": 10}, {"value": 12}, {"value": 15}]) == 50
    assert curve_delta([{"value": 10}, {"value": 5}]) == -50
    assert curve_delta([{"value": 0}, {"value": 100}]) is None
    assert curve_delta([{"value": 10}]) is None
    assert curve_delta(None) is None
    assert curve_delta([{"value": 10}, "junk", {"value": 20}]) == 100  # 非法点跳过
