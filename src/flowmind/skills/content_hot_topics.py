"""content_hot_topics 技能：抓取公开热榜（聚合 API，DailyHotApi 协议）。

平台 → 榜单端点映射走 config.hot_topic_endpoints（小红书/公众号无公开热榜 → 代理平台）。

错误契约：HTTP/解析失败返回 **degraded SkillOutput**（ok=True + degraded=True + 种子兜底），
绝不静默返回假数据——data 里带 failure_category / retriable / warning 供消费方决策。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import ContentPlatform, build_chain
from flowmind.skills._hot_topics_client import HotTopicError, fetch_hot_topics

_VERSION = "0.1.0"

# 兜底种子：真实 API 不可达时用于降级展示（明确标记 degraded，不冒充真实热点）
_SEED_TOPICS: dict[str, list[dict]] = {
    "xhs": [
        {"word": "通勤好物", "heat": 920, "delta": 12, "url": "", "source": "seed"},
        {"word": "办公室神器", "heat": 870, "delta": 8, "url": "", "source": "seed"},
        {"word": "夏日降温", "heat": 810, "delta": 21, "url": "", "source": "seed"},
        {"word": "极简生活", "heat": 680, "delta": 5, "url": "", "source": "seed"},
    ],
    "wechat": [
        {"word": "品牌内容", "heat": 780, "delta": 6, "url": "", "source": "seed"},
        {"word": "产品方法论", "heat": 710, "delta": 4, "url": "", "source": "seed"},
        {"word": "内容营销", "heat": 660, "delta": 2, "url": "", "source": "seed"},
    ],
    "douyin": [
        {"word": "好物分享", "heat": 880, "delta": 15, "url": "", "source": "seed"},
        {"word": "通勤", "heat": 760, "delta": 7, "url": "", "source": "seed"},
        {"word": "夏日好物", "heat": 700, "delta": 11, "url": "", "source": "seed"},
    ],
}


class ContentHotInput(BaseModel):
    """热点雷达入参。"""
    platform: ContentPlatform = Field(description="目标平台：xhs / wechat / douyin")
    limit: int | None = Field(default=None, ge=1, le=50, description="返回条数上限")


class HotTopic(BaseModel):
    """单条热点。delta=None 表示无趋势数据。"""
    word: str
    heat: int
    delta: int | None = None
    url: str = ""
    source: str


class ContentHotPlan(BaseModel):
    """热点雷达业务载荷。"""
    platform: str
    source: str        # 实际榜单源（端点名 / 种子）
    endpoint: str      # 配置的端点
    degraded: bool     # 是否降级（种子兜底）
    topics: list[HotTopic]
    # 降级时的结构化失败信息（degraded=True 时填充）
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


@skill(id="content_hot_topics", name="平台热点雷达", version=_VERSION)
def content_hot_topics(inp: ContentHotInput) -> SkillOutput[ContentHotPlan]:
    """抓取指定平台的公开热榜；聚合 API 不可达时返回降级结果（种子兜底，degraded=True）。

    数据流：平台→端点映射 → 聚合 API 抓取 → 解析 → ContentHotPlan + 推理链；
    失败 → 种子兜底 + degraded SkillOutput（绝不静默）。
    """
    cfg = load_config().content
    endpoint = cfg.hot_topic_endpoints.get(inp.platform, "weibo")
    limit = inp.limit or cfg.hot_topic_limit

    try:
        raw = fetch_hot_topics(
            api_base=cfg.hot_topic_api_base,
            endpoint=endpoint,
            limit=limit,
            timeout_s=cfg.hot_topic_timeout_s,
        )
    except HotTopicError as exc:
        seeds = _SEED_TOPICS.get(inp.platform, [])
        plan = ContentHotPlan(
            platform=inp.platform,
            source=f"seed({endpoint})",
            endpoint=endpoint,
            degraded=True,
            topics=[HotTopic(**t) for t in seeds[:limit]],
            failure_category=exc.category,
            retriable=exc.retriable,
            warning=f"热榜 API 不可达（{exc.category}），已用种子数据降级展示",
        )
        chain = build_chain(
            conclusion=f"{inp.platform} 热点抓取降级：使用 {len(seeds)} 条种子数据",
            causal_analysis=f"GET /{endpoint} → {type(exc).__name__}（{exc.category}）",
            risk_note="种子数据非实时热点，仅供演示；恢复后可刷新。",
        )
        return SkillOutput(
            data=plan, reasoning=[chain], confidence=0.0,
            sample_size=len(seeds), degraded=True, degradation_reason=exc.category,
        )

    topics = [HotTopic(**t) for t in raw]
    chain = build_chain(
        conclusion=f"{inp.platform} 热点抓取成功：{len(topics)} 条（源 /{endpoint}）",
        causal_analysis=f"GET /{endpoint} 返回 {len(raw)} 条原始数据，保留 {len(topics)} 条",
        risk_note="公开热榜为聚合数据，热度随时间变化；趋势 delta 视数据源而定。",
    )
    return SkillOutput(
        data=ContentHotPlan(
            platform=inp.platform, source=endpoint, endpoint=endpoint,
            degraded=False, topics=topics,
        ),
        reasoning=[chain],
        confidence=0.9,
        sample_size=len(topics),
    )
