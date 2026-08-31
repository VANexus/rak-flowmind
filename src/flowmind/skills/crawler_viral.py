"""crawler_viral 技能：爆款内容收集。

抓取各平台公开热榜 / 热门内容，筛选高热度条目。
数据源：复用 _hot_topics_client（聚合 API）+ 平台公开页面抓取。
失败契约：HTTP 依赖类 — r.ok=True + metrics.degraded + data.failure_category（种子兜底）。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import ContentPlatform, build_chain
from flowmind.skills._hot_topics_client import HotTopicError, fetch_hot_topics

_VERSION = "0.1.0"


class ViralInput(BaseModel):
    """爆款收集入参。"""
    platform: ContentPlatform = Field(description="目标平台：xhs / wechat / douyin")
    category: str | None = Field(default=None, max_length=50, description="分类筛选关键词")
    min_heat: int = Field(default=0, ge=0, description="最低热度阈值（0=不限）")
    limit: int = Field(default=20, ge=1, le=50, description="返回条数上限")


class ViralItem(BaseModel):
    """单条爆款内容。"""
    title: str
    heat: int = 0
    delta: int | None = None
    url: str = ""
    source: str = ""
    category: str | None = None


class ViralResult(BaseModel):
    """爆款收集业务载荷。"""
    platform: str
    source: str                         # 实际数据源
    items: list[ViralItem]
    total_available: int                # 抓取到的总量（筛选前）
    # 降级时填充
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


# 兜底种子（API 不可达时降级展示，明确标记 degraded）
_SEED_VIRAL: dict[str, list[dict]] = {
    "xhs": [
        {"title": "通勤好物清单", "heat": 9200, "url": "", "source": "seed"},
        {"title": "办公室神器推荐", "heat": 8700, "url": "", "source": "seed"},
        {"title": "夏日降温好物", "heat": 8100, "url": "", "source": "seed"},
        {"title": "极简生活指南", "heat": 6800, "url": "", "source": "seed"},
    ],
    "wechat": [
        {"title": "品牌内容方法论", "heat": 7800, "url": "", "source": "seed"},
        {"title": "产品运营复盘", "heat": 7100, "url": "", "source": "seed"},
        {"title": "内容营销趋势", "heat": 6600, "url": "", "source": "seed"},
    ],
    "douyin": [
        {"title": "好物分享合集", "heat": 8800, "url": "", "source": "seed"},
        {"title": "通勤穿搭", "heat": 7600, "url": "", "source": "seed"},
        {"title": "夏日好物推荐", "heat": 7000, "url": "", "source": "seed"},
    ],
}


@skill(id="crawler_viral", name="爆款内容收集", version=_VERSION)
def crawler_viral(inp: ViralInput) -> SkillOutput[ViralResult]:
    """抓取平台公开热榜 / 热门内容，筛选高热度条目。

    数据流：平台→端点映射 → 聚合 API 抓取 → 热度筛选 → ViralResult；
    失败 → 种子兜底 + degraded SkillOutput。
    """
    cfg = load_config().content
    endpoint = cfg.hot_topic_endpoints.get(inp.platform, "weibo")
    limit = inp.limit

    try:
        raw = fetch_hot_topics(
            api_base=cfg.hot_topic_api_base,
            endpoint=endpoint,
            limit=limit * 2,  # 多抓一些用于筛选
            timeout_s=cfg.hot_topic_timeout_s,
        )
    except HotTopicError as exc:
        # 种子兜底
        seeds = _SEED_VIRAL.get(inp.platform, [])
        items = [ViralItem(**t) for t in seeds[:limit]]
        chain = build_chain(
            conclusion=f"{inp.platform} 爆款收集降级：{len(items)} 条种子数据",
            causal_analysis=f"热榜 API 不可达（{exc.category}）→ 种子兜底",
            risk_note="种子数据非实时热点，仅供演示。",
        )
        return SkillOutput(
            data=ViralResult(
                platform=inp.platform, source=f"seed({endpoint})",
                items=items, total_available=len(items),
                failure_category=exc.category, retriable=exc.retriable,
                warning="热榜 API 不可达，已用种子数据降级展示",
            ),
            reasoning=[chain],
            confidence=0.0,
            sample_size=len(items),
            degraded=True,
            degradation_reason=exc.category,
        )

    # 筛选
    items: list[ViralItem] = []
    for t in raw:
        heat = t.get("heat", 0)
        if heat < inp.min_heat:
            continue
        title = t.get("word", "").strip()
        if not title:
            continue
        items.append(ViralItem(
            title=title[:100],
            heat=heat,
            delta=t.get("delta"),
            url=t.get("url", ""),
            source=t.get("source", endpoint),
        ))
        if len(items) >= limit:
            break

    # 分类关键词筛选
    if inp.category:
        items = [i for i in items if inp.category.lower() in i.title.lower() or inp.category.lower() in (i.source.lower())]
        items = items[:limit]

    chain = build_chain(
        conclusion=f"{inp.platform} 爆款收集：{len(items)} 条（源 /{endpoint}）",
        causal_analysis=f"聚合 API 抓取 {len(raw)} 条 → 热度≥{inp.min_heat} 筛选 → {len(items)} 条",
        risk_note="公开热榜为聚合数据，热度随时间变化；平台算法推荐内容更精准。",
    )
    return SkillOutput(
        data=ViralResult(
            platform=inp.platform, source=endpoint,
            items=items, total_available=len(raw),
        ),
        reasoning=[chain],
        confidence=0.85,
        sample_size=len(items),
    )
