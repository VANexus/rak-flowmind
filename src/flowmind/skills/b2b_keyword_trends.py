"""b2b_keyword_trends 技能：行业关键词趋势榜单（全部自托管数据源）。

数据源按平台路由：tiktok→Creative Center 抓取（登录会话解锁全量）；
instagram→IG 会话直连（必填关键词）；alibaba→TOP 热销词统计。
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

_VERSION = "0.2.0"

TrendPlatform = Literal["tiktok", "instagram", "alibaba"]


class KeywordTrendInput(BaseModel):
    """关键词趋势榜单入参。"""
    platform: TrendPlatform = Field(description="数据源平台：tiktok / instagram / alibaba")
    industry_id: int | None = Field(default=None, description="行业一级 ID（保留参数，当前自托管源未启用行业过滤）")
    keyword: str | None = Field(default=None, description="搜索关键词（instagram 必填；alibaba 可用于过滤商品池）")
    session_cookie: str | None = Field(default=None, description="平台登录会话（站内渠道授权捕获）：TikTok 解锁全量榜单；IG 必需")
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


@skill(id="b2b_keyword_trends", name="行业关键词趋势榜单", version=_VERSION)
def b2b_keyword_trends(inp: KeywordTrendInput) -> SkillOutput[KeywordTrendPlan]:
    """抓取指定平台/行业的趋势关键词榜单；源不可用时返回 degraded 空数据（绝不返回假 mock）。

    数据流：平台 → adapter 路由 → 自托管真实抓取 → 解析 → KeywordTrendPlan + 推理链；
    失败（不支持平台 / fetch 抛错 / 缺登录会话）→ degraded=True + keywords=[] +
    结构化 failure_category/retriable/warning，前端展示空态 + 跳转到
    「设置 → B 端运营」渠道授权的 CTA。
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
        warning = (
            f"趋势抓取不可用（{exc.category}）：{str(exc).strip()}"
            "。请在「设置 → B 端运营」完成对应平台的渠道授权登录后重试。"
        )

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

    chain = build_chain(
        conclusion=(
            f"{inp.platform} 关键词趋势抓取{'降级' if degraded else '成功'}："
            f"{len(keywords)} 条（源 {source_label}）"
        ),
        causal_analysis=f"平台 {inp.platform} → adapter {adapter_name_for_chain} → 保留 {len(keywords)} 条",
        risk_note="趋势数据随时间变化；degraded 空态等待渠道授权登录后可一键刷新。",
    )
    return SkillOutput(
        data=KeywordTrendPlan(
            platform=inp.platform, source=source_label, degraded=degraded,
            keywords=keywords, failure_category=failure_category, retriable=retriable, warning=warning,
        ),
        reasoning=[chain],
        confidence=0.0 if degraded else 0.9,
        sample_size=len(keywords),
        degraded=degraded,
        degradation_reason=failure_category,
    )