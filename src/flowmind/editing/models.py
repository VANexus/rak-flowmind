"""editing 领域模型：素材/文案/脚本/分镜/时间线与项目状态。

全部 pydantic v2 模型：既做 LLM JSON 输出的校验边界，也是前端二次编辑
的数据契约（中间产物 schema，见 docs/agent-video-editing.md §5）。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

# ── 阶段名（顺序即执行顺序；审批可插在 copy/script/storyboard/pre_render）──

STAGE_UNDERSTAND = "understand"
STAGE_COPY = "copy"
STAGE_SCRIPT = "script"
STAGE_STORYBOARD = "storyboard"
STAGE_RENDER = "render"
STAGE_ORDER: list[str] = [
    STAGE_UNDERSTAND, STAGE_COPY, STAGE_SCRIPT, STAGE_STORYBOARD, STAGE_RENDER,
]

# ── 项目状态机 ──

PSTATE_RUNNING = "running"
PSTATE_AWAITING_REVIEW = "awaiting_review"
PSTATE_SUCCEEDED = "succeeded"
PSTATE_FAILED = "failed"
PSTATE_CANCELLED = "cancelled"
PSTATE_TERMINAL = frozenset(
    {PSTATE_SUCCEEDED, PSTATE_FAILED, PSTATE_CANCELLED})

STAGE_PENDING = "pending"
STAGE_DONE = "done"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── understand 产物 ──

class ShotInfo(BaseModel):
    """视频镜头切分结果（ffmpeg scdet）。"""
    index: int
    start_s: float
    end_s: float
    score: float = 0.0          # 场景变化分数


class AsrSegment(BaseModel):
    """语音转写句段（v1 预留：asr_enabled 开启后由云 ASR 填充）。"""
    begin_s: float
    end_s: float
    text: str


class ContentNote(BaseModel):
    """视觉理解笔记：抽帧时刻的画面描述与烧录字幕文本（v1.1）。

    分镜 Agent 靠它"看见"素材——没有它就是盲剪（实测教训）。
    """
    t_s: float                      # 抽帧时刻（秒）
    scene: str                      # 画面内容一句话（主体/场景/动作）
    subtitle: str = ""              # 画面中烧录的字幕/文字（OCR 级读出）
    importance: int = 3             # 信息/情绪重要度 1~5


class AssetMeta(BaseModel):
    """单素材理解产物。"""
    id: str
    kind: Literal["video", "image", "audio"]
    path: str
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = False
    shots: list[ShotInfo] = Field(default_factory=list)
    asr: list[AsrSegment] = Field(default_factory=list)
    content_notes: list[ContentNote] = Field(default_factory=list)  # v1.1
    vision_degraded: str | None = None   # vision 失败原因（非空=盲剪降级，显式）
    tags: list[str] = Field(default_factory=list)
    summary: str = ""


class UnderstandReport(BaseModel):
    """understand 阶段产物。"""
    assets: list[AssetMeta] = Field(default_factory=list)
    total_duration_s: float = 0.0
    summary: str = ""           # 供 LLM 创意阶段消费的素材总览


# ── copy 产物（LLM 生成）──

class CopySection(BaseModel):
    purpose: str
    text: str


class CopyDoc(BaseModel):
    """文案产物：标题/钩子/正文/CTA/完整口播稿。"""
    title: str
    hook: str
    body_sections: list[CopySection] = Field(default_factory=list)
    cta: str = ""
    full_narration: str = ""
    style_notes: str = ""


# ── script 产物（LLM 生成）──

class ScriptSection(BaseModel):
    """结构化脚本段落：一段 ≈ 一个叙事单元，对应分镜里的一组镜头。"""
    id: str
    purpose: str                       # 段落职能 hook/body/cta...
    narration: str                     # 口播/旁白文本
    visual_brief: str                  # 画面需求描述（分镜 Agent 的输入）
    duration_s: float = Field(ge=0.5, le=60.0)
    subtitle_text: str = ""            # 字幕（空则取 narration）


class ScriptDoc(BaseModel):
    sections: list[ScriptSection] = Field(default_factory=list)
    total_duration_s: float = 0.0


# ── storyboard 产物（LLM 生成 + 素材引用校验）──

# xfade 支持且观感适合短视频的转场白名单（none=硬切/1帧 eps）
TRANSITIONS: frozenset[str] = frozenset({
    "none", "fade", "zoomin", "slideleft", "slideright",
    "circleopen", "dissolve", "fadeblack", "radial",
})


def clamp_transition(v: object) -> str:
    """转场白名单兜底（未知值回 none 硬切，不抛错）。"""
    return v if isinstance(v, str) and v in TRANSITIONS else "none"


class TitleCard(BaseModel):
    """开场/片尾设计卡：深色底 + 动画大字（v1.3 设计层）。"""
    main: str                       # 主标题（大字，钩子/CTA，建议 ≤14 字）
    sub: str = ""                   # 副标题（小字，可选）
    duration_s: float = Field(default=2.0, ge=1.0, le=4.0)


class StoryShot(BaseModel):
    """单镜头：引用素材区间/图片 + 呈现方式 + 转场 + 设计参数（v1.3）。"""
    id: str
    section_id: str
    source_asset: str                  # AssetMeta.id
    source_in: float = 0.0             # 视频裁剪起点（秒）
    duration_s: float = Field(ge=0.3, le=60.0)  # 输出时长（含定格）
    mode: Literal["video_clip", "image_kenburns"] = "video_clip"
    pan: Literal["none", "in", "out", "left", "right"] = "none"   # 图片 Ken Burns
    transition: str = "fade"           # TRANSITIONS 白名单（normalize 兜底）
    speed: float = Field(default=1.0, ge=0.25, le=4.0)    # v1.1 变速（>1 快进；与提示词 0.25~4 一致）
    freeze_s: float = Field(default=0.0, ge=0.0, le=3.0)  # v1.1 尾帧定格秒数（与提示词 0~3 一致）
    subtitle_text: str = ""
    subtitle_emphasis: list[str] = Field(default_factory=list)  # v1.3 字幕关键词高亮
    badge: str = ""                    # v1.3 右上角角标（如 01/08；空=无）
    zoom_start: float = Field(default=1.0, ge=0.9, le=1.2)  # v1.3 缓推缓拉
    zoom_end: float = Field(default=1.0, ge=0.9, le=1.2)
    hook: bool = False                 # v1.3 钩子镜：字幕居中放大（开场抓人）
    note: str = ""                     # 分镜理由（前端展示 + 审批参考）


class StoryboardDoc(BaseModel):
    shots: list[StoryShot] = Field(default_factory=list)
    title_card: TitleCard | None = None    # v1.3 开场设计卡
    end_card: TitleCard | None = None      # v1.3 片尾 CTA 卡
    bgm_note: str = ""                 # BGM 意向（bgm_path 存在时实际混音）
    total_duration_s: float = 0.0


# ── timeline（render 输入；前端二次编辑的最终时间线）──

class TimelineShot(StoryShot):
    """绝对时间轴镜头（timeline.json）：加全局起点与字幕区间。

    mode 额外允许 "card"（v1.3 设计卡：渲染深色底，文字走 ASS 动画；
    该模式只由 pipeline 从 title_card/end_card 合成，LLM 分镜不允许出）。
    """
    mode: Literal["video_clip", "image_kenburns", "card"] = "video_clip"  # type: ignore[assignment]
    global_start_s: float = 0.0
    global_end_s: float = 0.0
    source_path: str = ""              # 素材绝对路径（render 直接消费）


class TimelineDoc(BaseModel):
    width: int = 1920
    height: int = 1080
    fps: int = 30
    duration_s: float = 0.0
    voiceover_path: str | None = None
    bgm_path: str | None = None        # v1.3 BGM 底乐（含淡入淡出/闪避）
    title_card: TitleCard | None = None
    end_card: TitleCard | None = None
    shots: list[TimelineShot] = Field(default_factory=list)
    srt_path: str | None = None


# ── 项目记录（store 持久化的聚合根）──

class ProjectRecord(BaseModel):
    """剪辑项目记录：状态机 + 阶段进度 + 审批配置 + 输入回放。

    与任务引擎解耦：project_id 是聚合根，task_ids[] 记录历次驱动任务
    （每次审批续跑/revise 都是一次新的引擎任务）。
    """
    project_id: str
    state: str = PSTATE_RUNNING
    stage: str = STAGE_UNDERSTAND
    # 各阶段状态：pending / done（approve 后 done）——awaiting 由 state 表达
    stages: dict[str, str] = Field(
        default_factory=lambda: {s: STAGE_PENDING for s in STAGE_ORDER})
    review_stages: list[str] = Field(default_factory=list)
    # 审批反馈：stage → feedback 历史（revise 时追加，进该阶段 LLM 提示词）
    review_feedback: dict[str, list[str]] = Field(default_factory=dict)
    # 已审批放行的阶段（approve 后置位；续跑时跳过其审批暂停）
    approved: list[str] = Field(default_factory=list)
    input: dict = Field(default_factory=dict)      # EditProjectInput dump
    task_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    storage: str = "file"                          # pg | file
    tenant_id: str | None = None
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)

    def touch(self) -> None:
        self.updated_at = _now()


# ── 素材库条目（edit_assets）──

class AssetRecord(BaseModel):
    """素材库登记条目（全局索引，租户字段隔离）。"""
    id: str
    kind: str
    path: str
    tags: list[str] = Field(default_factory=list)
    project_id: str | None = None
    tenant_id: str | None = None
    created_at: str = Field(default_factory=_now)
