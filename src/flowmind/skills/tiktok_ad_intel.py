"""tiktok_ad_intel 技能：TikTok 广告创意情报（真实 Creative Center 广告库，绝不 mock）。

数据全部来自 TikHub Ads 系列（服务端代抓 Creative Center Top Ads）：
- search_ads：按关键词/行业/目标搜索真实在投广告（素材、文案、CTR、点赞、时长、视频地址）；
- filters：广告筛选项字典（258 个行业、7 类营销目标、语言、创意形式、周期）；
- locations：Creative Center 支持的 73 个国家/地区；
- hashtag_detail：单个热门话题的受众年龄/国家画像、长周期热度曲线、代表视频。

源不可用时统一返回 degraded 空态（failure_category/retriable/warning），绝不返回假数据。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import build_chain
from flowmind.skills._intel_common import fail_fields, intel_client
from flowmind.skills._tikhub_intel_parse import (
    parse_ad_filters,
    parse_ad_materials,
    parse_ad_pagination,
    parse_hashtag_detail,
    parse_locations,
)

_VERSION = "0.1.0"
AdIntelAction = Literal["search_ads", "filters", "locations", "hashtag_detail"]


class AdMaterial(BaseModel):
    id: str
    rank: int = 0
    title: str = ""
    brand: str = ""
    ctr: float | None = None
    likes: int | None = None
    cost: int | None = None
    objective: str = ""
    industry_key: str = ""
    is_search: bool = False
    duration_s: float | None = None
    cover_url: str = ""
    video_url: str = ""
    width: int | None = None
    height: int | None = None


class AdIntelInput(BaseModel):
    action: AdIntelAction = Field(description="search_ads=搜广告库 / filters=筛选项字典 / locations=国家列表 / hashtag_detail=话题画像")
    keyword: str | None = Field(default=None, description="search_ads 必填：搜索关键词，如 skincare")
    hashtag_id: str | None = Field(default=None, description="hashtag_detail 必填：话题 ID（标签榜返回的 hashtagID）")
    period: int = Field(default=180, description="search_ads 时间窗（天），可选 7/30/180")
    objective: int | None = Field(default=None, description="营销目标：1流量2应用安装3转化4视频浏览5触达6潜在客户")
    industry: str | None = Field(default=None, description="行业 ID（filters 返回的 industry.id）")
    country_code: str = Field(default="US", description="国家代码")
    page: int = Field(default=1, ge=1)
    limit: int = Field(default=20, ge=1, le=50, description="期望条数；TikHub 单页硬上限 20，超出自动钳到 20")
    order_by: str = Field(default="for_you", description="for_you 推荐 / likes 按点赞")
    time_range: int = Field(default=30, description="hashtag_detail 周期 7/30/90")


class AdIntelPlan(BaseModel):
    action: str
    source: str = "tikhub"
    degraded: bool = False
    materials: list[AdMaterial] = []
    pagination: dict = Field(default_factory=dict)
    filters: dict = Field(default_factory=dict)
    locations: list[dict] = []
    hashtag_detail: dict = Field(default_factory=dict)
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


@skill(id="tiktok_ad_intel", name="TikTok广告创意情报", version=_VERSION)
def tiktok_ad_intel(inp: AdIntelInput) -> SkillOutput[AdIntelPlan]:
    """查询 TikTok 真实在投广告创意库 / 筛选项字典 / 国家列表 / 话题画像；
    失败走 degraded 空态，绝不返回 mock 数据。"""
    degraded = False
    warning: str | None = None
    failure_category: str | None = None
    retriable = False
    materials: list[dict] = []
    pagination: dict = {}
    filters: dict = {}
    locations: list[dict] = []
    detail: dict = {}
    result_n = 0

    try:
        client = intel_client()
        if inp.action == "search_ads":
            kw = (inp.keyword or "").strip()
            if not kw:
                raise ValueError("search_ads 需要关键词 keyword")
            # TikHub Creative Center 单页最多返回 20 条，传 21+ 会直接返回空，这里钳制保护调用方
            safe_limit = min(max(int(inp.limit), 1), 20)
            raw = client.ads_search(
                keyword=kw, period=inp.period, page=inp.page, limit=safe_limit,
                country_code=inp.country_code, order_by=inp.order_by,
                objective=inp.objective, industry=inp.industry,
            )
            materials = parse_ad_materials(raw)
            pagination = parse_ad_pagination(raw)
            pagination["page_size"] = len(materials)
            result_n = len(materials)
        elif inp.action == "filters":
            filters = parse_ad_filters(client.ads_filters())
            result_n = sum(len(v) for v in filters.values())
        elif inp.action == "locations":
            locations = parse_locations(client.ads_locations())
            result_n = len(locations)
        elif inp.action == "hashtag_detail":
            if not (inp.hashtag_id or "").strip():
                raise ValueError("hashtag_detail 需要 hashtag_id")
            detail = parse_hashtag_detail(client.ads_hashtag_detail(
                hashtag_id=inp.hashtag_id.strip(), time_range=inp.time_range,
                country_code=inp.country_code,
            ))
            result_n = len(detail.get("curve", []))
    except Exception as exc:  # TikHubError + 参数错误统一降级
        degraded = True
        fb = fail_fields(exc)
        failure_category, retriable, warning = fb["failure_category"], fb["retriable"], fb["warning"]

    chain = build_chain(
        conclusion=f"广告情报 {inp.action} {'降级' if degraded else '成功'}：{result_n} 项（源 tikhub）",
        causal_analysis=f"TikHub Ads 系列 action={inp.action} → 解析 {result_n} 项",
        risk_note="广告库随投放实时变化；degraded 空态代表数据源当前不可达，修复后可重试。",
    )
    return SkillOutput(
        data=AdIntelPlan(
            action=inp.action, degraded=degraded,
            materials=[AdMaterial(**m) for m in materials],
            pagination=pagination, filters=filters, locations=locations,
            hashtag_detail=detail,
            failure_category=failure_category, retriable=retriable, warning=warning,
        ),
        reasoning=[chain], confidence=0.0 if degraded else 0.9,
        sample_size=result_n, degraded=degraded, degradation_reason=failure_category,
    )
