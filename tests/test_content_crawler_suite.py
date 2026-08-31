"""content_crawler_suite 技能测试：通过 invoke() 走信封层。

覆盖：三源聚合 / 单源失败不阻断 / 全失败降级 / 空 URL 跳过死链。
HTTP 依赖类：成功 ok=True，降级 ok=True + degraded=True。
"""
from __future__ import annotations

from unittest.mock import patch

import flowmind.skills.content_crawler_suite as mod
from flowmind.contracts import ReliabilityMetrics, SkillOutput, SkillResult, new_trace
from flowmind.skill import invoke
from flowmind.skills import crawler_sentiment, crawler_viral, crawler_deadlink


def _wrap(skill_id: str, output: SkillOutput) -> SkillResult:
    """将 SkillOutput 包装为 invoke() 返回的 SkillResult 信封。"""
    return SkillResult(
        ok=True,
        skill=skill_id,
        version="0.1.0",
        trace=new_trace(),
        data=output.data,
        reasoning=output.reasoning,
        metrics=ReliabilityMetrics(
            latency_ms=100.0,
            confidence=output.confidence,
            sample_size=output.sample_size,
            degraded=output.degraded,
            degradation_reason=output.degradation_reason,
        ),
    )


def _make_sentiment_result(keyword):
    return _wrap("crawler_sentiment", SkillOutput(
        data=crawler_sentiment.SentimentResult(
            keyword=keyword,
            platforms_queried=["weibo"],
            total_mentions=2,
            items=[
                crawler_sentiment.SentimentItem(platform="weibo", title="t1", url="https://w/1"),
                crawler_sentiment.SentimentItem(platform="weibo", title="t2", url="https://w/2"),
            ],
        ),
        reasoning=[], confidence=0.85, sample_size=2,
    ))


def _make_viral_result(platform):
    return _wrap("crawler_viral", SkillOutput(
        data=crawler_viral.ViralResult(
            platform=platform,
            items=[
                crawler_viral.ViralItem(title="热点1", heat=1000),
                crawler_viral.ViralItem(title="热点2", heat=500),
            ],
            source="test",
            total_available=2,
        ),
        reasoning=[], confidence=0.85, sample_size=2,
    ))


def _make_deadlink_result():
    return _wrap("crawler_deadlink", SkillOutput(
        data=crawler_deadlink.DeadLinkResult(
            total=2, alive=1, dead=1,
            links=[
                crawler_deadlink.LinkCheckResult(url="https://a.com", alive=True, status_code=200),
                crawler_deadlink.LinkCheckResult(url="https://b.com", alive=False, status_code=404),
            ],
        ),
        reasoning=[], confidence=0.9, sample_size=2,
    ))


def test_suite_aggregates_all_sources():
    """三源全部成功 → 聚合结果。"""

    def mock_invoke(skill_id, args):
        if skill_id == "crawler_sentiment":
            return _make_sentiment_result(args["keyword"])
        if skill_id == "crawler_viral":
            return _make_viral_result(args["platform"])
        if skill_id == "crawler_deadlink":
            return _make_deadlink_result()
        raise ValueError(f"未知 skill: {skill_id}")

    with patch.object(mod, "invoke", side_effect=mock_invoke):
        r = invoke("content_crawler_suite", {
            "keyword": "保温杯",
            "platform": "xhs",
            "urls": ["https://a.com", "https://b.com"],
            "limit_per_source": 10,
        })
    assert r.ok is True
    d = r.data
    assert d.keyword == "保温杯"
    assert len(d.sources) == 3
    assert len(d.sentiment_items) == 2
    assert len(d.viral_items) == 2
    assert len(d.dead_link_results) == 2
    assert r.metrics.degraded is False


def test_suite_partial_failure_degrades():
    """单源失败 → degraded 但不断阻断。"""

    def mock_invoke(skill_id, args):
        if skill_id == "crawler_sentiment":
            return _make_sentiment_result(args["keyword"])
        if skill_id == "crawler_viral":
            raise RuntimeError("热榜 API 不可达")
        if skill_id == "crawler_deadlink":
            return _make_deadlink_result()
        raise ValueError(f"未知 skill: {skill_id}")

    with patch.object(mod, "invoke", side_effect=mock_invoke):
        r = invoke("content_crawler_suite", {
            "keyword": "测试",
            "platform": "xhs",
            "urls": ["https://a.com"],
        })
    assert r.ok is True
    d = r.data
    # viral 失败 → degraded
    assert r.metrics.degraded is True
    assert any(s.source == "viral" and not s.ok for s in d.sources)


def test_suite_all_failure_degrades():
    """全源失败 → degraded + failure_category。"""

    def mock_invoke(skill_id, args):
        raise RuntimeError(f"{skill_id} 模拟失败")

    with patch.object(mod, "invoke", side_effect=mock_invoke):
        r = invoke("content_crawler_suite", {
            "keyword": "测试",
            "platform": "xhs",
            "urls": ["https://a.com"],
        })
    assert r.ok is True
    d = r.data
    assert r.metrics.degraded is True
    assert d.failure_category == "environment"
    assert d.retriable is True


def test_suite_empty_urls_skips_deadlink():
    """无 URL 输入 → 死链源标记 ok=True count=0。"""

    def mock_invoke(skill_id, args):
        if skill_id == "crawler_sentiment":
            return _make_sentiment_result(args["keyword"])
        if skill_id == "crawler_viral":
            return _make_viral_result(args["platform"])
        raise ValueError(f"不应调用 {skill_id}")

    with patch.object(mod, "invoke", side_effect=mock_invoke):
        r = invoke("content_crawler_suite", {
            "keyword": "测试",
            "platform": "xhs",
            "urls": [],
        })
    assert r.ok is True
    d = r.data
    deadlink_source = next(s for s in d.sources if s.source == "deadlink")
    assert deadlink_source.ok is True
    assert deadlink_source.count == 0
