"""content_web_fetch 技能：通用网页抓取。

给定 URL，抓取网页内容，提取标题 + 正文 + 链接。
供上层技能（舆情 / 爆款 / 文案参考）复用。
失败契约：HTTP 依赖类 — r.ok=True + metrics.degraded + data.failure_category。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.skill import skill
from flowmind.skills._crawler_client import CrawlerError, fetch_url

_VERSION = "0.1.0"


class WebFetchInput(BaseModel):
    """通用网页抓取入参。"""
    url: str = Field(min_length=1, description="目标 URL（http/https）")
    max_text_length: int = Field(default=10000, ge=100, le=50000, description="正文最大字符数")


class WebFetchResult(BaseModel):
    """网页抓取业务载荷。"""
    url: str
    final_url: str                       # 重定向后的最终 URL
    status_code: int
    title: str
    text: str                            # 正文
    links: list[str]                     # 页面内链接
    # 降级时填充
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


@skill(id="content_web_fetch", name="通用网页抓取", version=_VERSION)
def content_web_fetch(inp: WebFetchInput) -> SkillOutput[WebFetchResult]:
    """抓取单个网页，提取标题 + 正文 + 链接。

    数据流：URL → HTTP GET → 解析 HTML → 提取标题/正文/链接 → WebFetchResult。
    失败走 degraded SkillOutput（HTTP 依赖类）。
    """
    cfg = load_config().crawler

    try:
        page = fetch_url(url=inp.url, timeout_s=cfg.timeout_s, user_agent=cfg.user_agent)
    except CrawlerError as exc:
        return SkillOutput(
            data=WebFetchResult(
                url=inp.url, final_url="", status_code=0,
                title="", text="", links=[],
                failure_category=exc.category, retriable=exc.retriable,
                warning=f"抓取失败：{exc}",
            ),
            reasoning=[ReasoningChain(
                conclusion=f"网页抓取降级：{exc}",
                evidence=[], causal_analysis=f"GET {inp.url} → {type(exc).__name__}",
                risk_note="请检查 URL 是否可达、是否被反爬限制。",
            )],
            confidence=0.0, sample_size=0,
            degraded=True, degradation_reason=exc.category,
        )

    text = page.text[:inp.max_text_length]
    chain = ReasoningChain(
        conclusion=f"网页抓取成功：{page.title[:50]}（{len(text)} 字正文，{len(page.links)} 链接）",
        evidence=[],
        causal_analysis=f"GET {inp.url} → HTTP {page.status_code} → 提取标题/正文/链接",
        risk_note="正文提取为简化版（正则），复杂页面建议用 Readability 算法或 LLM 后处理。",
    )
    return SkillOutput(
        data=WebFetchResult(
            url=inp.url, final_url=page.url,
            status_code=page.status_code, title=page.title,
            text=text, links=page.links[:200],
        ),
        reasoning=[chain],
        confidence=0.85,
        sample_size=1,
    )
