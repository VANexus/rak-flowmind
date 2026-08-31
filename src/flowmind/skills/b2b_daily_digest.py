"""b2b_daily_digest 技能：每日趋势摘要编排（三平台榜单 → 长尾词 → 飞书/企微推送）。

串行调 b2b_keyword_trends（tiktok / instagram / alibaba，limit 10）→
b2b_longtail_keywords（以各平台热词为种子）→ 组装 markdown 摘要 →
按开关调 b2b_push_feishu / b2b_push_wecom。各环节失败均结构化降级
（榜单 degraded 照常进摘要并标注，推送失败返回 ok=False），绝不静默成功。
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from flowmind.contracts import SkillOutput
from flowmind.skill import invoke, skill
from flowmind.skills._content_common import build_chain

_VERSION = "0.1.0"

_PLATFORM_LABELS = {"tiktok": "TikTok", "instagram": "Instagram", "alibaba": "阿里国际站热销词"}


class DigestInput(BaseModel):
    """每日摘要入参。"""
    limit: int = Field(default=10, ge=1, le=20, description="每平台榜单条数上限")
    industry: str | None = Field(default=None, max_length=100, description="长尾词行业；缺省取各平台榜首行业")
    push_feishu: bool = Field(default=True, description="是否推送飞书")
    push_wecom: bool = Field(default=False, description="是否推送企微")
    feishu_webhook_url: str | None = Field(default=None, max_length=500, description="飞书 webhook（缺省读 env）")
    wecom_webhook_url: str | None = Field(default=None, max_length=500, description="企微 webhook（缺省读 env）")


class DigestKeyword(BaseModel):
    """摘要中的单条热词。"""
    word: str
    heat: int
    rank: int


class DigestSection(BaseModel):
    """单平台榜单摘要。degraded=True 表示数据源不可达（无真实数据，绝不造假）。"""
    platform: str
    label: str
    source: str = ""
    degraded: bool = False
    failure_category: str | None = None
    keywords: list[DigestKeyword] = Field(default_factory=list)


class DigestPushResult(BaseModel):
    """单渠道推送结果。"""
    channel: str  # feishu / wecom
    ok: bool
    latency_ms: float = 0.0
    error: str | None = None


class DigestPlan(BaseModel):
    """每日摘要业务载荷。"""
    date: str
    sections: list[DigestSection]
    longtail_words: list[str] = Field(default_factory=list)
    longtail_error: str | None = None
    markdown: str
    pushes: list[DigestPushResult] = Field(default_factory=list)


def _render_markdown(date: str, sections: list[DigestSection],
                     longtail_words: list[str], longtail_error: str | None) -> str:
    """组装摘要 markdown（纯文本列表，飞书卡片与企微 markdown 通用）。"""
    lines = [f"## B端关键词趋势日报 · {date}"]
    for sec in sections:
        if sec.degraded:
            lines.append(f"\n**{sec.label}**：数据源不可达（{sec.failure_category or 'unknown'}），今日无榜单")
        else:
            lines.append(f"\n**{sec.label}**（{sec.source}）")
            for kw in sec.keywords:
                lines.append(f"{kw.rank}. {kw.word}（热度 {kw.heat}）")
    if longtail_words:
        lines.append("\n**长尾词推荐**")
        lines.append("、".join(longtail_words))
    if longtail_error:
        lines.append(f"\n长尾词生成失败：{longtail_error}")
    return "\n".join(lines)


@skill(id="b2b_daily_digest", name="B端每日趋势摘要", version=_VERSION)
def b2b_daily_digest(inp: DigestInput) -> SkillOutput[DigestPlan]:
    """编排每日趋势摘要：三平台榜单 → 长尾词 → 组装 markdown → 按开关推送。

    数据流：串行 invoke 三平台 b2b_keyword_trends → b2b_longtail_keywords →
    _render_markdown → 按开关 invoke b2b_push_feishu / b2b_push_wecom。
    榜单 degraded 照常进摘要并标注原因；长尾词失败不阻塞推送；
    推送失败结构化返回，绝不静默成功。
    """
    date = datetime.now().strftime("%Y-%m-%d")

    # ── 1. 三平台趋势榜单 ──
    sections: list[DigestSection] = []
    seed_words: list[str] = []
    for platform in ("tiktok", "instagram", "alibaba"):
        res = invoke("b2b_keyword_trends", {"platform": platform, "limit": inp.limit})
        if not res.ok:
            sections.append(DigestSection(
                platform=platform, label=_PLATFORM_LABELS[platform], degraded=True,
                failure_category=res.error.code if res.error else "unknown",
            ))
            continue
        data = res.data
        kws = [
            DigestKeyword(word=k.word, heat=k.heat, rank=k.rank)
            for k in getattr(data, "keywords", [])
        ]
        sections.append(DigestSection(
            platform=platform, label=_PLATFORM_LABELS[platform],
            source=getattr(data, "source", ""), degraded=bool(getattr(data, "degraded", False)),
            failure_category=getattr(data, "failure_category", None), keywords=kws,
        ))
        if not getattr(data, "degraded", False):
            seed_words.extend(k.word for k in kws)

    # ── 2. 长尾词（失败不阻塞推送）──
    longtail_words: list[str] = []
    longtail_error: str | None = None
    if seed_words:
        industry = inp.industry or next(
            (s.keywords[0].word for s in sections if not s.degraded and s.keywords), "通用",
        )
        lt = invoke("b2b_longtail_keywords", {
            "industry": industry, "seed_keywords": seed_words[:10], "limit": 10,
        })
        if lt.ok:
            longtail_words = [k.word for k in getattr(lt.data, "keywords", [])][:10]
        else:
            longtail_error = lt.error.message if lt.error else "未知错误"

    markdown = _render_markdown(date, sections, longtail_words, longtail_error)

    # ── 3. 按开关推送 ──
    pushes: list[DigestPushResult] = []
    if inp.push_feishu:
        args: dict = {"title": f"B端关键词趋势日报 {date}", "markdown": markdown}
        if inp.feishu_webhook_url:
            args["webhook_url"] = inp.feishu_webhook_url
        r = invoke("b2b_push_feishu", args)
        pushes.append(DigestPushResult(
            channel="feishu", ok=bool(getattr(r.data, "ok", False)),
            latency_ms=getattr(r.data, "latency_ms", 0.0), error=getattr(r.data, "error", None),
        ))
    if inp.push_wecom:
        args = {"title": f"B端关键词趋势日报 {date}", "markdown": markdown}
        if inp.wecom_webhook_url:
            args["webhook_url"] = inp.wecom_webhook_url
        r = invoke("b2b_push_wecom", args)
        pushes.append(DigestPushResult(
            channel="wecom", ok=bool(getattr(r.data, "ok", False)),
            latency_ms=getattr(r.data, "latency_ms", 0.0), error=getattr(r.data, "error", None),
        ))

    ok_pushes = sum(1 for p in pushes if p.ok)
    chain = build_chain(
        conclusion=f"每日摘要编排完成：{sum(1 for s in sections if not s.degraded)}/3 平台真实榜单，"
                   f"推送 {ok_pushes}/{len(pushes)} 成功",
        causal_analysis="b2b_keyword_trends ×3 → b2b_longtail_keywords → markdown → b2b_push_feishu/wecom",
        risk_note="榜单 degraded 代表数据源不可达（已标注），推送失败结构化返回；全部数据为真实来源，无任何演示数据。",
    )
    return SkillOutput(
        data=DigestPlan(
            date=date, sections=sections, longtail_words=longtail_words,
            longtail_error=longtail_error, markdown=markdown, pushes=pushes,
        ),
        reasoning=[chain],
        confidence=0.9 if not pushes or ok_pushes == len(pushes) else 0.6,
        sample_size=sum(len(s.keywords) for s in sections),
        degraded=any(s.degraded for s in sections),
        degradation_reason=next((s.failure_category for s in sections if s.degraded), None),
    )
