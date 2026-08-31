"""crawler_sentiment 技能：舆情收集。

给定关键词，从多平台公开内容中收集提及，输出结构化舆情数据。
数据源：微博搜索 / 头条搜索 / 知乎公开内容（通过通用爬虫抓取公开页面）。
失败契约：HTTP 依赖类 — r.ok=True + metrics.degraded + data.failure_category。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.skill import skill
from flowmind.skills._crawler_client import CrawlerError, fetch_url

_VERSION = "0.1.0"


class SentimentInput(BaseModel):
    """舆情收集入参。"""
    keyword: str = Field(min_length=1, max_length=100, description="监测关键词")
    platforms: list[str] = Field(
        default=["weibo", "toutiao"],
        description="平台列表：weibo / toutiao / zhihu",
    )
    limit: int = Field(default=10, ge=1, le=50, description="每平台返回条数上限")


class SentimentItem(BaseModel):
    """单条舆情。"""
    platform: str
    title: str
    url: str
    snippet: str = ""          # 内容摘要
    source: str = ""           # 来源页面
    error: str | None = None   # 抓取失败时的错误


class SentimentResult(BaseModel):
    """舆情收集业务载荷。"""
    keyword: str
    platforms_queried: list[str]
    total_mentions: int
    items: list[SentimentItem]
    # 降级时填充
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


# 平台公开搜索 URL 模板（无需 API key）
_SEARCH_TEMPLATES: dict[str, str] = {
    "weibo": "https://s.weibo.com/weibo?q={keyword}&typeall=1&suball=1&timescope=custom:2024-01-01-0:2024-12-31-0&Refer=g",
    "toutiao": "https://so.toutiao.com/search?keyword={keyword}&pd=information",
    "zhihu": "https://www.zhihu.com/search?type=content&q={keyword}",
}


@skill(id="crawler_sentiment", name="舆情收集", version=_VERSION)
def crawler_sentiment(inp: SentimentInput) -> SkillOutput[SentimentResult]:
    """给定关键词，从多平台公开页面收集舆情提及。

    数据流：平台→搜索 URL → 抓取公开页面 → 解析标题/链接 → 结构化舆情。
    失败走 degraded SkillOutput（HTTP 依赖类）。
    """
    cfg = load_config().crawler
    items: list[SentimentItem] = []
    errors: list[str] = []
    error_categories: list[str] = []

    for platform in inp.platforms:
        template = _SEARCH_TEMPLATES.get(platform)
        if not template:
            items.append(SentimentItem(
                platform=platform, title="", url="",
                error=f"不支持的平台：{platform}（支持：{list(_SEARCH_TEMPLATES.keys())}）",
            ))
            continue

        url = template.format(keyword=_url_encode(inp.keyword))
        try:
            page = fetch_url(url=url, timeout_s=cfg.timeout_s, user_agent=cfg.user_agent)
            # 从页面提取链接作为舆情条目
            for link_url in page.links[:inp.limit]:
                items.append(SentimentItem(
                    platform=platform,
                    title=page.title or inp.keyword,
                    url=link_url,
                    snippet=page.text[:200] if page.text else "",
                    source=page.url,
                ))
        except CrawlerError as exc:
            errors.append(f"{platform}: {exc}")
            error_categories.append(exc.category)
            items.append(SentimentItem(
                platform=platform, title="", url="", error=str(exc),
            ))

    # 限流
    items = items[:inp.limit * len(inp.platforms)]

    has_errors = bool(errors)
    chain = ReasoningChain(
        conclusion=f"舆情收集「{inp.keyword}」：{len(items)} 条提及（{len(inp.platforms)} 平台）",
        evidence=[],
        causal_analysis=f"搜索模板 → 抓取公开页面 → 提取链接（{len(errors)} 平台失败）",
        risk_note="公开页面抓取受反爬限制，数据可能不完整；建议结合平台官方 API。",
    )

    # 取第一个错误类别作为代表（全部失败时所有错误同类别概率高）
    primary_category = error_categories[0] if error_categories else "environment"
    all_failed = has_errors and all(item.error for item in items)
    if all_failed:
        # 全部失败 → degraded，类别从 CrawlerError 传播
        return SkillOutput(
            data=SentimentResult(
                keyword=inp.keyword,
                platforms_queried=inp.platforms,
                total_mentions=0,
                items=[],
                failure_category=primary_category,
                retriable=True,
                warning=f"所有平台抓取失败：{'; '.join(errors)[:300]}",
            ),
            reasoning=[chain],
            confidence=0.0,
            sample_size=0,
            degraded=True,
            degradation_reason=primary_category,
        )

    return SkillOutput(
        data=SentimentResult(
            keyword=inp.keyword,
            platforms_queried=inp.platforms,
            total_mentions=len(items),
            items=items,
            failure_category=primary_category if has_errors else None,
            warning=f"部分平台失败：{'; '.join(errors)[:200]}" if errors else None,
        ),
        reasoning=[chain],
        confidence=0.7 if has_errors else 0.85,
        sample_size=len(items),
        degraded=has_errors,
        degradation_reason=primary_category if has_errors else None,
    )


def _url_encode(text: str) -> str:
    """URL 编码。"""
    from urllib.parse import quote
    return quote(text)
