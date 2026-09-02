"""tiktok_content_intel 技能：内容/达人/音乐情报 + Instagram 话题帖子（真实数据，绝不 mock）。

TikTok（App V3/Web 系列）：
- trending_words：站内每日趋势搜索词；
- video_search：关键词搜爆款视频（播放/点赞/评论/分享/收藏/达人/无水印地址）；
- music_chart：热门音乐榜；
- creator_insights：创作者搜索洞察（热门选题 + 7 日趋势序列 + 行业分层）；
- creator_profile：达人资料（粉丝/作品/认证），可选附注册国家。

Instagram（V2）：
- ig_hashtag_posts：话题下真实帖子（赞/评/播放/作者/缩略图/视频地址）。

源不可用时统一 degraded 空态，绝不返回假数据。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import build_chain
from flowmind.skills._intel_common import fail_fields, intel_client
from flowmind.skills._tikhub_intel_parse import (
    parse_creator_insights,
    parse_ig_hashtag_posts,
    parse_music_chart,
    parse_trending_searchwords,
    parse_user_profile,
    parse_video_search,
)

_VERSION = "0.1.0"
ContentIntelAction = Literal[
    "trending_words", "video_search", "music_chart",
    "creator_insights", "creator_profile", "ig_hashtag_posts",
]


class VideoItem(BaseModel):
    aweme_id: str
    desc: str = ""
    create_time: int | None = None
    duration_s: float | None = None
    play: int | None = None
    likes: int | None = None
    comments: int | None = None
    shares: int | None = None
    collects: int | None = None
    author_id: str = ""
    author: str = ""
    author_handle: str = ""
    author_followers: int | None = None
    cover_url: str = ""
    video_url: str = ""
    music_title: str = ""


class MusicItem(BaseModel):
    rank: int
    music_id: str
    title: str = ""
    author: str = ""
    duration_s: int | None = None
    user_count: int | None = None
    trend: int | None = None
    cover_url: str = ""
    artists: list[str] = []


class InsightItem(BaseModel):
    query_id: str
    query: str
    popularity: int | None = None
    popularity_v2: int | None = None
    video_num: int | None = None
    trend_seq: list[int] = []
    category_l1: str = ""
    category_l2: str = ""
    business_types: list[str] = []


class IgPost(BaseModel):
    media_id: str
    code: str
    caption: str = ""
    hashtags: list[str] = []
    likes: int | None = None
    comments: int | None = None
    plays: int | None = None
    is_video: bool = False
    media_type: int | None = None
    thumbnail: str = ""
    video_url: str = ""
    taken_at: int | None = None
    username: str = ""
    user_fullname: str = ""
    verified: bool = False


class ContentIntelInput(BaseModel):
    action: ContentIntelAction
    keyword: str | None = Field(default=None, description="video_search/ig_hashtag_posts 必填")
    unique_id: str | None = Field(default=None, description="creator_profile 必填：达人 handle")
    with_country: bool = Field(default=False, description="creator_profile 是否额外查询注册国家（多一次调用）")
    limit: int = Field(default=20, ge=1, le=50)
    region: str = Field(default="US")
    feed_type: str = Field(default="top", description="IG：top/recent")


class ContentIntelPlan(BaseModel):
    action: str
    source: str = "tikhub"
    degraded: bool = False
    trending_words: list[dict] = []
    videos: list[VideoItem] = []
    music: list[MusicItem] = []
    insights: list[InsightItem] = []
    profile: dict = Field(default_factory=dict)
    ig_posts: list[IgPost] = []
    ig_pagination_token: str = ""
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


@skill(id="tiktok_content_intel", name="TikTok内容达人情报", version=_VERSION)
def tiktok_content_intel(inp: ContentIntelInput) -> SkillOutput[ContentIntelPlan]:
    """TikTok 内容/达人/音乐情报与 IG 话题帖子；失败 degraded 空态，绝不 mock。"""
    degraded = False
    warning = failure_category = None
    retriable = False
    trending_words: list[dict] = []
    videos: list[dict] = []
    music: list[dict] = []
    insights: list[dict] = []
    profile: dict = {}
    ig_posts: list[dict] = []
    ig_token = ""
    result_n = 0

    try:
        client = intel_client()
        if inp.action == "trending_words":
            trending_words = parse_trending_searchwords(client.web_trending_searchwords())
            result_n = len(trending_words)
        elif inp.action == "video_search":
            kw = (inp.keyword or "").strip()
            if not kw:
                raise ValueError("video_search 需要关键词 keyword")
            videos = parse_video_search(client.app_video_search(
                keyword=kw, count=min(inp.limit, 30), region=inp.region))
            result_n = len(videos)
        elif inp.action == "music_chart":
            music = parse_music_chart(client.app_music_chart(count=inp.limit))
            result_n = len(music)
        elif inp.action == "creator_insights":
            insights = parse_creator_insights(client.app_creator_insights(limit=inp.limit))
            result_n = len(insights)
        elif inp.action == "creator_profile":
            handle = (inp.unique_id or "").strip().lstrip("@")
            if not handle:
                raise ValueError("creator_profile 需要 unique_id")
            profile = parse_user_profile(client.app_user_profile(unique_id=handle))
            if inp.with_country and profile:
                try:
                    country = client.app_user_country(username=handle)
                    profile["country"] = str(country.get("country") or "")
                except Exception:
                    profile["country"] = ""
            result_n = 1 if profile else 0
        elif inp.action == "ig_hashtag_posts":
            kw = (inp.keyword or "").strip().lstrip("#")
            if not kw:
                raise ValueError("ig_hashtag_posts 需要关键词 keyword")
            parsed = parse_ig_hashtag_posts(client.instagram_hashtag_posts(
                keyword=kw, feed_type=inp.feed_type))
            ig_posts = parsed["posts"][: inp.limit]
            ig_token = parsed["pagination_token"]
            result_n = len(ig_posts)
    except Exception as exc:
        degraded = True
        fb = fail_fields(exc)
        failure_category, retriable, warning = fb["failure_category"], fb["retriable"], fb["warning"]

    chain = build_chain(
        conclusion=f"内容情报 {inp.action} {'降级' if degraded else '成功'}：{result_n} 项（源 tikhub）",
        causal_analysis=f"TikHub AppV3/Web/IG action={inp.action} → 解析 {result_n} 项",
        risk_note="内容/达人数据实时变化；degraded 空态代表数据源当前不可达，修复后可重试。",
    )
    return SkillOutput(
        data=ContentIntelPlan(
            action=inp.action, degraded=degraded,
            trending_words=trending_words,
            videos=[VideoItem(**v) for v in videos],
            music=[MusicItem(**m) for m in music],
            insights=[InsightItem(**i) for i in insights],
            profile=profile,
            ig_posts=[IgPost(**p) for p in ig_posts], ig_pagination_token=ig_token,
            failure_category=failure_category, retriable=retriable, warning=warning,
        ),
        reasoning=[chain], confidence=0.0 if degraded else 0.9,
        sample_size=result_n, degraded=degraded, degradation_reason=failure_category,
    )
