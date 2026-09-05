"""content_hot_boards 技能：热榜引擎多榜抓取（小红书种草流水线 · 选题洞察数据源）。

与 content_hot_topics 的关系：hot_topics 是单平台单榜（雷达），hot_boards 是
「多种热榜」的专门处理——一次抓取多个榜型（综合/垂类/话题/灵感），每个榜独立
归一化输出，单榜失败只降级该榜、不阻断其他榜，全失败才 degraded。

数据流：榜型 → config.hot_board_endpoints 端点映射 → 逐榜 fetch_hot_topics
（DailyHotApi 协议）→ 归一化条目 {board, title, heat, rank, source, fetchedAt}。
聚合去重/过滤打分/选题包装由前端热榜引擎（score 可复算）完成，本技能只出
「干净的原始多榜」，不编造热度。

错误契约：单榜 HTTP/解析失败 → 该榜 degraded=True + failure_category/warning；
全部榜失败 → SkillOutput degraded=True（空 boards，不返回假数据）。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import ContentPlatform, build_chain
from flowmind.skills._hot_topics_client import HotTopicError, fetch_hot_topics

_VERSION = "0.1.0"

BoardType = Literal["general", "vertical", "topic", "inspiration"]
BOARD_ORDER: list[BoardType] = ["general", "vertical", "topic", "inspiration"]


class ContentHotBoardsInput(BaseModel):
    """热榜引擎入参。"""
    platform: ContentPlatform = Field(description="目标平台：xhs / wechat / douyin")
    boards: list[BoardType] | None = Field(
        default=None, description="要抓取的榜型子集；缺省=全部（general/vertical/topic/inspiration）"
    )
    limit: int | None = Field(default=None, ge=1, le=50, description="每个榜返回条数上限")


class HotBoardItem(BaseModel):
    """单榜单条归一化热点。"""
    board: BoardType
    title: str
    heat: int
    rank: int          # 榜内序号（1-based，用于前端 rankScore）
    source: str        # 榜单源（端点名）
    url: str = ""
    fetchedAt: str     # 抓取时间（ISO8601）


class HotBoard(BaseModel):
    """单个榜型结果。"""
    id: BoardType
    label: str
    endpoint: str
    source: str
    degraded: bool = False
    topics: list[HotBoardItem] = Field(default_factory=list)
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


class ContentHotBoardsPlan(BaseModel):
    """热榜引擎业务载荷：多榜 + 全局抓取时间。"""
    platform: str
    boards: list[HotBoard] = Field(default_factory=list)
    fetchedAt: str
    degraded: bool = False       # 全局是否降级（= 全部榜失败）
    failure_category: str | None = None
    warning: str | None = None


@skill(id="content_hot_boards", name="热榜引擎·多榜抓取", version=_VERSION)
def content_hot_boards(inp: ContentHotBoardsInput) -> SkillOutput[ContentHotBoardsPlan]:
    """抓取多个真实热榜（综合/垂类/话题/灵感），逐榜归一化输出。

    单榜失败只降级该榜；全部失败才 degraded（空数据，绝不返回假 mock）。
    """
    cfg = load_config().content
    endpoints: dict[str, str] = cfg.hot_board_endpoints
    labels: dict[str, str] = cfg.hot_board_labels
    limit = inp.limit or cfg.hot_topic_limit
    boards_wanted = inp.boards or list(BOARD_ORDER)

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_boards: list[HotBoard] = []
    ok_count = 0
    failed: list[str] = []

    for board_id in boards_wanted:
        endpoint = endpoints.get(board_id, "thepaper")
        label = labels.get(board_id, board_id)
        try:
            raw = fetch_hot_topics(
                api_base=cfg.hot_topic_api_base,
                endpoint=endpoint,
                limit=limit,
                timeout_s=cfg.hot_topic_timeout_s,
            )
        except HotTopicError as exc:
            failed.append(board_id)
            out_boards.append(HotBoard(
                id=board_id, label=label, endpoint=endpoint, source=f"unavailable({endpoint})",
                degraded=True, topics=[],
                failure_category=exc.category, retriable=exc.retriable,
                warning=f"{label}不可达（{exc.category}），请检查热榜源或稍后重试",
            ))
            continue

        items = [
            HotBoardItem(
                board=board_id, title=t["word"], heat=t["heat"],
                rank=idx + 1, source=t["source"], url=t["url"], fetchedAt=fetched_at,
            )
            for idx, t in enumerate(raw)
        ]
        ok_count += 1
        out_boards.append(HotBoard(
            id=board_id, label=label, endpoint=endpoint,
            source=endpoint, degraded=False, topics=items,
        ))

    all_failed = ok_count == 0 and bool(boards_wanted)
    plan = ContentHotBoardsPlan(
        platform=inp.platform, boards=out_boards, fetchedAt=fetched_at,
        degraded=all_failed,
        failure_category="all-boards-unavailable" if all_failed else None,
        warning="全部热榜源不可达，请检查聚合 API 配置" if all_failed else None,
    )

    if all_failed:
        chain = build_chain(
            conclusion=f"{inp.platform} 热榜引擎无数据返回（{len(failed)} 榜全部失败）",
            causal_analysis=" → ".join(f"{b}:unavailable" for b in failed) or "no boards",
            risk_note="不做种子兜底，宁可返回空也不返回假热点；恢复后可重试。",
        )
        return SkillOutput(
            data=plan, reasoning=[chain], confidence=0.0,
            sample_size=0, degraded=True, degradation_reason="all-boards-unavailable",
        )

    chain = build_chain(
        conclusion=f"{inp.platform} 热榜引擎抓取完成：{ok_count}/{len(boards_wanted)} 榜可用"
                   + (f"，{len(failed)} 榜降级" if failed else ""),
        causal_analysis=f"逐榜 GET 抓取：可用 {[f'{b}.{endpoints.get(b)}' for b in boards_wanted if b not in failed]}"
                        + (f"；失败 {failed}" if failed else ""),
        risk_note="公开聚合热榜为代理源，热度随时间变化；聚合去重/打分由前端热榜引擎完成。",
    )
    return SkillOutput(
        data=plan, reasoning=[chain], confidence=0.9, sample_size=sum(len(b.topics) for b in out_boards),
    )
