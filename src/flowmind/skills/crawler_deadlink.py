"""crawler_deadlink 技能：僵尸链接检测。

批量检测 URL 存活状态（HEAD 请求，失败降级为 GET），输出存活/死亡统计。
失败契约：HTTP 依赖类 — r.ok=True + metrics.degraded + data.failure_category。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.skill import skill
from flowmind.skills._crawler_client import CrawlerError, check_links

_VERSION = "0.1.0"


class DeadLinkInput(BaseModel):
    """僵尸链接检测入参。"""
    urls: list[str] = Field(default_factory=list, max_length=200, description="待检测链接列表（≤200）")
    timeout_s: float | None = Field(default=None, ge=1, le=60, description="单链接超时（秒）")
    check_redirect: bool = Field(default=True, description="是否跟踪重定向")


class LinkCheckResult(BaseModel):
    """单条链接检测结果。"""
    url: str
    alive: bool
    status_code: int | None = None
    final_url: str | None = None
    error: str | None = None
    response_time_ms: float | None = None


class DeadLinkResult(BaseModel):
    """僵尸链接检测业务载荷。"""
    total: int
    alive: int
    dead: int
    links: list[LinkCheckResult]
    # 降级时填充
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


@skill(id="crawler_deadlink", name="僵尸链接检测", version=_VERSION)
def crawler_deadlink(inp: DeadLinkInput) -> SkillOutput[DeadLinkResult]:
    """批量检测 URL 存活状态，输出存活/死亡统计。

    数据流：URL 列表 → 并发 HEAD 请求（失败降级 GET）→ 统计 → DeadLinkResult。
    注意：即使部分链接检测失败，也返回成功检测的结果（不整体 degraded）。
    仅当全部失败或并发异常时走 degraded。
    """
    cfg = load_config().crawler
    timeout = inp.timeout_s or cfg.timeout_s

    # 空列表快速返回
    if not inp.urls:
        chain = ReasoningChain(
            conclusion="死链检测跳过：无输入链接",
            evidence=[], causal_analysis="空 URL 列表 → 直接返回",
            risk_note="提供 ≥1 个 URL 以执行检测。",
        )
        return SkillOutput(
            data=DeadLinkResult(total=0, alive=0, dead=0, links=[]),
            reasoning=[chain], confidence=0.9, sample_size=0,
        )

    # 限流
    urls = inp.urls[:cfg.max_links_per_check]

    try:
        results = check_links(
            urls=urls,
            timeout_s=timeout,
            max_concurrent=cfg.max_concurrent,
            check_redirect=inp.check_redirect,
            user_agent=cfg.user_agent,
        )
    except CrawlerError as exc:
        # 并发异常 → degraded（单链接失败已在 check_links 内部处理为 alive=False）
        chain = ReasoningChain(
            conclusion=f"死链检测降级：{exc}",
            evidence=[], causal_analysis=f"并发 HEAD 请求异常 → {type(exc).__name__}",
            risk_note="请检查网络后重试；部分链接可能有反爬限制。",
        )
        return SkillOutput(
            data=DeadLinkResult(
                total=len(urls), alive=0, dead=0, links=[],
                failure_category=exc.category, retriable=exc.retriable,
                warning=f"死链检测异常：{exc}",
            ),
            reasoning=[chain], confidence=0.0, sample_size=0,
            degraded=True, degradation_reason=exc.category,
        )

    alive_count = sum(1 for r in results if r.alive)
    dead_count = sum(1 for r in results if not r.alive)

    links = [
        LinkCheckResult(
            url=r.url, alive=r.alive, status_code=r.status_code,
            final_url=r.final_url, error=r.error,
            response_time_ms=r.response_time_ms,
        )
        for r in results
    ]

    chain = ReasoningChain(
        conclusion=f"死链检测完成：{len(urls)} 条链接，{alive_count} 存活 / {dead_count} 死亡",
        evidence=[],
        causal_analysis=f"并发 HEAD 请求（超时 {timeout}s，并发 {cfg.max_concurrent}）→ 统计",
        risk_note="HEAD 请求可能被服务器拒绝（405），已自动降级为 GET；部分链接可能有反爬限制。",
    )
    return SkillOutput(
        data=DeadLinkResult(
            total=len(urls), alive=alive_count, dead=dead_count, links=links,
        ),
        reasoning=[chain],
        confidence=0.9,
        sample_size=len(urls),
    )
