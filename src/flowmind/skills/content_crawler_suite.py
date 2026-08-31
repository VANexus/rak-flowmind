"""content_crawler_suite 技能：智能爬虫套件（舆情 + 爆款 + 死链 三源聚合）。

一次调用，同时跑舆情收集 + 爆款收集 + 死链检测，输出聚合结果。
失败契约：HTTP 依赖类 — r.ok=True + metrics.degraded + data.failure_category。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.skill import invoke, skill

_VERSION = "0.1.0"


class CrawlerSuiteInput(BaseModel):
    """爬虫套件入参。"""
    keyword: str = Field(min_length=1, max_length=100, description="监测关键词（舆情用）")
    platform: str = Field(default="xhs", description="爆款收集目标平台：xhs / wechat / douyin")
    urls: list[str] = Field(default_factory=list, max_length=50, description="待检测链接（死链检测用）")
    limit_per_source: int = Field(default=10, ge=1, le=30, description="每源返回条数上限")


class SourceResult(BaseModel):
    """单源结果摘要。"""
    source: str
    ok: bool
    count: int = 0
    degraded: bool = False
    error: str | None = None


class CrawlerSuiteResult(BaseModel):
    """爬虫套件业务载荷。"""
    keyword: str
    platform: str
    sources: list[SourceResult]
    sentiment_items: list[dict] = Field(default_factory=list)
    viral_items: list[dict] = Field(default_factory=list)
    dead_link_results: list[dict] = Field(default_factory=list)
    # 降级时填充
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


@skill(id="content_crawler_suite", name="智能爬虫套件", version=_VERSION)
def content_crawler_suite(inp: CrawlerSuiteInput) -> SkillOutput[CrawlerSuiteResult]:
    """智能爬虫套件：舆情 + 爆款 + 死链 三源聚合。

    数据流：并行调用 3 个子技能 → 聚合结果 → CrawlerSuiteResult。
    单源失败不阻断整体（标记 degraded），全失败才整体 degraded。
    """
    sources: list[SourceResult] = []
    sentiment_items: list[dict] = []
    viral_items: list[dict] = []
    dead_link_results: list[dict] = []

    # 1. 舆情收集
    try:
        r = invoke("crawler_sentiment", {
            "keyword": inp.keyword,
            "platforms": ["weibo", "toutiao"],
            "limit": inp.limit_per_source,
        })
        if r.ok:
            sentiment_items = [item.model_dump() for item in (r.data.items or [])]
            sources.append(SourceResult(
                source="sentiment", ok=True,
                count=len(sentiment_items), degraded=r.metrics.degraded,
            ))
        else:
            sources.append(SourceResult(source="sentiment", ok=False, error="调用失败"))
    except Exception as exc:
        sources.append(SourceResult(source="sentiment", ok=False, error=str(exc)[:100]))

    # 2. 爆款收集
    try:
        r = invoke("crawler_viral", {
            "platform": inp.platform,
            "limit": inp.limit_per_source,
        })
        if r.ok:
            viral_items = [item.model_dump() for item in (r.data.items or [])]
            sources.append(SourceResult(
                source="viral", ok=True,
                count=len(viral_items), degraded=r.metrics.degraded,
            ))
        else:
            sources.append(SourceResult(source="viral", ok=False, error="调用失败"))
    except Exception as exc:
        sources.append(SourceResult(source="viral", ok=False, error=str(exc)[:100]))

    # 3. 死链检测
    if inp.urls:
        try:
            r = invoke("crawler_deadlink", {
                "urls": inp.urls,
            })
            if r.ok:
                dead_link_results = [link.model_dump() for link in (r.data.links or [])]
                sources.append(SourceResult(
                    source="deadlink", ok=True,
                    count=len(dead_link_results),
                ))
            else:
                sources.append(SourceResult(source="deadlink", ok=False, error="调用失败"))
        except Exception as exc:
            sources.append(SourceResult(source="deadlink", ok=False, error=str(exc)[:100]))
    else:
        sources.append(SourceResult(source="deadlink", ok=True, count=0, error="无 URL 输入"))

    # 聚合
    failed = [s for s in sources if not s.ok]
    total_items = len(sentiment_items) + len(viral_items) + len(dead_link_results)

    chain = ReasoningChain(
        conclusion=f"爬虫套件完成：{len(sources)} 源，{total_items} 条结果（{len(failed)} 源失败）",
        evidence=[],
        causal_analysis=" + ".join(f"{s.source}({s.count})" for s in sources),
        risk_note="公开数据源受反爬限制，建议结合平台官方 API 获取更完整数据。",
    )

    if len(failed) == len(sources):
        # 全失败
        return SkillOutput(
            data=CrawlerSuiteResult(
                keyword=inp.keyword, platform=inp.platform,
                sources=sources,
                sentiment_items=sentiment_items,
                viral_items=viral_items,
                dead_link_results=dead_link_results,
                failure_category="environment",
                retriable=True,
                warning=f"所有数据源失败：{'; '.join(s.error or '' for s in failed)[:200]}",
            ),
            reasoning=[chain],
            confidence=0.0,
            sample_size=0,
            degraded=True,
            degradation_reason="environment",
        )

    return SkillOutput(
        data=CrawlerSuiteResult(
            keyword=inp.keyword, platform=inp.platform,
            sources=sources,
            sentiment_items=sentiment_items,
            viral_items=viral_items,
            dead_link_results=dead_link_results,
        ),
        reasoning=[chain],
        confidence=0.75 if failed else 0.85,
        sample_size=total_items,
        degraded=bool(failed),
        degradation_reason="environment" if failed else None,
    )
