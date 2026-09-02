"""b2b_keyword_trends 技能：行业关键词趋势榜单（真实数据源，绝不 mock）。

数据源按平台路由：tiktok→TikHub API 代抓 Creative Center（默认，无需登录会话；
可切回 cc_scraper 自建路径）；instagram→TikHub IG 话题搜索（默认，关键词必填，
无需登录会话；可切回 self_host 会话直连）；alibaba→TOP 热销词统计。
错误契约：抓取失败走 degraded SkillOutput（keywords=[] + failure_category/retriable/warning）。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import build_chain
from flowmind.skills._trend_adapters import TrendError, resolve_adapter

_VERSION = "0.4.0"

TrendPlatform = Literal["tiktok", "instagram", "alibaba"]


class KeywordTrendInput(BaseModel):
    """关键词趋势榜单入参。"""
    platform: TrendPlatform = Field(description="数据源平台：tiktok / instagram / alibaba")
    industry_id: int | None = Field(default=None, description="行业一级 ID（TikTok 走 TikHub 时支持行业过滤；CC 一级行业 ID 如 22000000000=服装配饰）")
    keyword: str | None = Field(default=None, description="搜索关键词（instagram 必填，走 TikHub 话题搜索；alibaba 可用于过滤商品池）")
    session_cookie: str | None = Field(default=None, description="平台登录会话（可选）：TikTok 透传给 TikHub 解锁更多数据；仅 self_host 回退路径必需")
    browser_debug_url: str | None = Field(
        default=None,
        description="用户浏览器 CDP 地址（如 http://127.0.0.1:9222）。提供时优先直连用户浏览器抓取——真实指纹 + 浏览器登录态",
    )
    limit: int | None = Field(default=None, ge=1, le=50, description="返回条数上限")


class TrendKeyword(BaseModel):
    """单条趋势关键词。delta=None 表示无涨幅数据。"""
    word: str
    heat: int
    delta: int | None = None
    rank: int = 0
    industry: str = "通用"
    source: str = ""


class KeywordTrendPlan(BaseModel):
    """关键词趋势榜单业务载荷。"""
    platform: str
    source: str
    degraded: bool
    keywords: list[TrendKeyword]
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None
    # TikHub 响应缓存元信息（_tikhub_cache）：{mode: local|local_fallback|speculative|live, hit, age_s}
    cache: dict | None = None


@skill(id="b2b_keyword_trends", name="行业关键词趋势榜单", version=_VERSION)
def b2b_keyword_trends(inp: KeywordTrendInput) -> SkillOutput[KeywordTrendPlan]:
    """抓取指定平台/行业的趋势关键词榜单；源不可用时返回 degraded 空数据（绝不返回假 mock）。

    数据流：平台 → adapter 路由 → 真实抓取（TikHub/IG 会话回退/TOP）→ 解析 →
    KeywordTrendPlan + 推理链；失败（不支持平台 / fetch 抛错 / 缺凭证）→
    degraded=True + keywords=[] + 结构化 failure_category/retriable/warning，
    具体修复动作（配 AI_TRENDS_API_KEY / 渠道授权登录回退）由 adapter 错误消息给出。
    """
    cfg = load_config().keyword_trend
    limit = inp.limit or 20

    degraded = False
    source_label = "unresolved"
    failure_category: str | None = None
    retriable = False
    warning: str | None = None
    raw: list[dict] = []

    adapter_name_for_chain: str = source_label
    try:
        adapter = resolve_adapter(
            inp.platform, cfg,
            alibaba_cfg=load_config().alibaba,
            session_cookie=(inp.session_cookie or "").strip(),
            cdp_url=(inp.browser_debug_url or "").strip(),
        )
        source_label = adapter.name
        adapter_name_for_chain = adapter.name
        raw = adapter.fetch(
            inp.platform,
            industry_id=inp.industry_id,
            limit=limit,
            keyword=inp.keyword,
        )
    except TrendError as exc:
        degraded = True
        raw = []
        source_label = f"degraded({source_label})"
        adapter_name_for_chain = source_label
        failure_category = exc.category
        retriable = exc.retriable
        # 修复引导：TikTok/Instagram 主路径都走 TikHub（配 API Key），self_host 回退才需要渠道登录
        warning = f"趋势抓取不可用（{exc.category}）：{str(exc).strip()} 请按上述原因修复配置/网络后重试。"

    keywords = [
        TrendKeyword(
            word=str(it.get("word") or "").strip(),
            heat=int(it.get("heat") or 0),
            delta=it.get("delta"),
            rank=int(it.get("rank") or i + 1),
            industry=str(it.get("industry") or "通用"),
            source=str(it.get("source") or source_label),
        )
        for i, it in enumerate(raw)
        if str(it.get("word") or "").strip()
    ]

    # TikHub 缓存元信息（仅 tikhub 主路径有值；alibaba/self_host 为 None）
    from flowmind.skills._tikhub_client import get_last_cache_meta

    cache_meta = get_last_cache_meta()

    chain = build_chain(
        conclusion=(
            f"{inp.platform} 关键词趋势抓取{'降级' if degraded else '成功'}："
            f"{len(keywords)} 条（源 {source_label}）"
        ),
        causal_analysis=f"平台 {inp.platform} → adapter {adapter_name_for_chain} → 保留 {len(keywords)} 条",
        risk_note="趋势数据随时间变化；degraded 空态代表数据源当前不可达（已标注原因类别），修复凭证/网络后可一键刷新。",
    )
    return SkillOutput(
        data=KeywordTrendPlan(
            platform=inp.platform, source=source_label, degraded=degraded,
            keywords=keywords, failure_category=failure_category, retriable=retriable, warning=warning,
            cache=cache_meta,
        ),
        reasoning=[chain],
        confidence=0.0 if degraded else 0.9,
        sample_size=len(keywords),
        degraded=degraded,
        degradation_reason=failure_category,
    )