"""localize_submit 技能：批量提交视频本地化异步任务（本地任务引擎）。

每个视频创建一个独立 task（TaskManager.submit → PG 落库 queued → GPU 串行
执行 localize_video 流水线），立即返回 task_ids 列表，不阻塞等待；任务进度
经 localize_status 查询 / MQTT ``mcp-base-gpu/tasks/{id}/events`` 实时推送。

入参字段与 localize_video 的 LocalizeVideoInput 对齐（逐任务同参提交），
output_path 不开放（每个任务的产物路径由流水线按输入推导）。

队列背压语义（错误码决策记录）：
- 一个都没受理（pending 已满）→ TaskQueueFull 向上抛，invoke() 兜底为
  ok=False / error.code=INTERNAL / message 带「稍后重试」——即 429 背压语义。
  errors.py 现有 ErrorCode 只有 NOT_FOUND/VALIDATION/INTERNAL：队列满既非
  入参错误（VALIDATION）也非资源缺失（NOT_FOUND），选 INTERNAL + 可读
  message 是最小错误映射；HTTP 层（阶段 4）可按异常类型补映射 429。
- 中途满（已受理 k>0 后满）→ degraded=True 的 partial success：已受理
  task_ids 保留在 report 里，failure_category=transient（可稍后重提剩余）。
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from flowmind.config import load_config
from flowmind.contracts import Evidence, ReasoningChain, SkillOutput
from flowmind.errors import is_retriable
from flowmind.rules import Rule, evaluate_rules
from flowmind.skill import skill
from flowmind.tasks import TaskQueueFull
from flowmind.tasks.manager import get_task_manager

_VERSION = "0.2.0"

TASK_SKILL_ID = "localize_video"  # 每个视频对应一个 localize_video 任务


# ── 入参 ──

class SubmitInput(BaseModel):
    """批量提交入参（字段语义与 LocalizeVideoInput 一致，作用到每个视频）。"""

    videos: list[str] = Field(
        ..., min_length=1, description="视频文件路径或 URL 列表，每条创建一个独立任务"
    )
    target_lang: str | None = Field(
        default=None, description="目标语言；None=任务执行时读 config 默认",
    )
    source_lang: str | None = Field(
        default=None, description="源语言；None=任务执行时读 config 默认",
    )
    voice_id: str | None = Field(
        default=None,
        description="配音音色（预设音色名或百炼复刻音色 ID）；None=任务执行时读 config",
    )
    keep_background_audio: bool = Field(
        default=True, description="原声 -12dB 保留为背景（与 localize_video 默认一致）",
    )
    tts_backend: str | None = Field(
        default=None, description="配音后端 auto/local/cloud；None=任务执行时读 config",
    )
    voice_ref_audio: str | None = Field(
        default=None, description="本地克隆参考音频路径/URL（缺省克隆原片人声）",
    )
    voice_ref_text: str | None = Field(
        default=None, description="参考音频转写（缺省由流水线补转写）",
    )
    erase_backend: str | None = Field(
        default=None, description="字幕擦除后端 auto/local/delogo；None=任务执行时读 config",
    )

    @field_validator("videos")
    @classmethod
    def _no_empty_strings(cls, v: list[str]) -> list[str]:
        """剔除空字符串；若全部为空则报错。"""
        cleaned = [p for p in v if isinstance(p, str) and p.strip()]
        if not cleaned:
            raise ValueError("videos 不能全为空字符串")
        return cleaned

    @model_validator(mode="after")
    def _ext_preflight(self) -> "SubmitInput":
        """扩展名预检：全部本地路径被拒时提前报 VALIDATION（不占用队列槽位）。

        只要存在任何合法路径即放行；「部分被拒」的分桶在技能体内做，
        被拒路径进 report 供 Agent 感知。
        """
        cfg = load_config().localizer
        accepted, _ = _split_paths(self.videos, cfg.allowed_extensions)
        if not accepted:
            raise ValueError(
                f"全部视频因扩展名被拒（允许：{cfg.allowed_extensions}）；"
                f"被拒：{self.videos}"
            )
        return self


# ── 出参 ──

class SubmitReport(BaseModel):
    """批量提交业务载荷。"""
    task_ids: list[str]           # 受理成功的任务 id（与 accepted 一致）
    accepted: int                 # 实际受理数
    rejected_count: int           # 扩展名预检被拒数
    rejected_paths: list[str]
    skill_id: str = TASK_SKILL_ID  # 每个任务执行的技能
    # ── partial（队列中途满）时的降级信息（正常成功时为默认值） ──
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


# ── 预检：扩展名分桶 ──

def _split_paths(
    videos: list[str], allowed_exts: list[str]
) -> tuple[list[str], list[str]]:
    """返回 (accepted, rejected)。URL 总是 accepted；本地路径按扩展名筛。"""
    allowed = {e.lower() for e in allowed_exts}
    accepted: list[str] = []
    rejected: list[str] = []
    for p in videos:
        if p.startswith(("http://", "https://")):
            accepted.append(p)
            continue
        ext = Path(p).suffix.lower()
        if ext in allowed:
            accepted.append(p)
        else:
            rejected.append(p)
    return accepted, rejected


def _task_args(video: str, inp: SubmitInput) -> dict:
    """单个视频的任务入参（None 字段不落 args_json，保持存储干净）。"""
    args: dict = {
        "video_path": video,
        "keep_background_audio": inp.keep_background_audio,
    }
    for key in ("target_lang", "source_lang", "voice_id", "tts_backend",
                "voice_ref_audio", "voice_ref_text", "erase_backend"):
        val = getattr(inp, key)
        if val is not None:
            args[key] = val
    return args


# ── 规则 ──

def _rules(cfg) -> list[Rule]:
    """提交侧告警规则：批量超额（占用队列槽位）+ 部分视频被拒。"""
    return [
        Rule(
            id="LOC-W01",
            name="批量超额",
            expression=f"受理数 > {cfg.max_videos_per_batch}",
            predicate=lambda m: m["n_videos"] > cfg.max_videos_per_batch,
            evidence=lambda m: [Evidence(
                metric="受理视频数",
                value=m["n_videos"],
                threshold=cfg.max_videos_per_batch,
                comparison=">",
            )],
        ),
        Rule(
            id="LOC-W02",
            name="部分视频被拒",
            expression="存在非允许扩展名的视频",
            predicate=lambda m: m["rejected_count"] > 0,
            evidence=lambda m: [Evidence(
                metric="拒绝数",
                value=m["rejected_count"],
                threshold=0,
                comparison=">",
            )],
        ),
    ]


def _build_chain(hits: list, evidence: list, accepted: int,
                 rejected: int, task_count: int) -> ReasoningChain:
    """四段式推理链：提交结论 → 规则 → 证据 → 因果与风险。"""
    rule_names = "、".join(h.name for h in hits) if hits else "（无）"
    conclusion = (
        f"已受理 {accepted} 个本地化任务（扩展名预检拒绝 {rejected} 个），"
        f"任务 id 已返回，异步执行中。"
    )
    if hits:
        risk_note = f"命中规则：{rule_names}；建议复核后再扩量提交。"
    else:
        risk_note = "参数在通用默认阈值内，任务已进入队列排队执行。"
    causal_analysis = (
        f"逐视频调 TaskManager.submit 创建 {task_count} 个 localize_video 任务"
        f"（PG 落库 queued → GPU 串行执行）；扩展名预检基于 "
        f"allowed_extensions={load_config().localizer.allowed_extensions}。"
    )
    return ReasoningChain(
        conclusion=conclusion,
        triggered_rules=hits,
        evidence=evidence,
        causal_analysis=causal_analysis,
        risk_note=risk_note,
    )


# ── 入口 ──

@skill(id="localize_submit", name="批量提交视频本地化任务", version=_VERSION)
def localize_submit(inp: SubmitInput) -> SkillOutput[SubmitReport]:
    """逐视频创建本地化异步任务，立即返回 task_ids（不等待执行完成）。

    数据流：videos → 扩展名预检分桶 → 逐条 TaskManager.submit（PG queued）
    → SubmitReport + ReasoningChain → 框架套 SkillResult 信封。
    队列背压：全部未受理 → TaskQueueFull 上抛（ok=False 429 语义）；
    中途满 → degraded partial success（已受理 task_ids 保留）。
    """
    cfg = load_config().localizer
    accepted, rejected = _split_paths(inp.videos, cfg.allowed_extensions)

    metrics = {"n_videos": len(accepted), "rejected_count": len(rejected)}
    hits, evidence = evaluate_rules(_rules(cfg), metrics)

    manager = get_task_manager()
    task_ids: list[str] = []
    for video in accepted:
        try:
            task_ids.append(manager.submit(TASK_SKILL_ID, _task_args(video, inp)))
        except TaskQueueFull:
            if not task_ids:
                raise  # 全部未受理 → 429 背压语义（见模块 docstring 决策记录）
            return _partial_output(
                inp, task_ids, rejected, hits, evidence,
                warning=f"队列已满，仅受理前 {len(task_ids)}/{len(accepted)} 个视频；"
                        f"其余可稍后重提",
            )

    report = SubmitReport(
        task_ids=task_ids,
        accepted=len(task_ids),
        rejected_count=len(rejected),
        rejected_paths=rejected,
    )
    return SkillOutput(
        data=report,
        reasoning=[_build_chain(hits, evidence, len(task_ids), len(rejected), len(task_ids))],
        confidence=1.0,
        sample_size=len(inp.videos),
    )


def _partial_output(
    inp: SubmitInput,
    task_ids: list[str],
    rejected: list[str],
    hits: list,
    evidence: list,
    warning: str,
) -> SkillOutput[SubmitReport]:
    """队列中途满的 partial success：已受理 task_ids 保留，transient 可重试。"""
    report = SubmitReport(
        task_ids=task_ids,
        accepted=len(task_ids),
        rejected_count=len(rejected),
        rejected_paths=rejected,
        failure_category="transient",
        retriable=is_retriable("transient"),
        warning=warning,
    )
    chain = ReasoningChain(
        conclusion=(
            f"批量提交部分成功：{len(task_ids)} 个任务已受理，队列满未受理剩余视频"
        ),
        triggered_rules=hits,
        evidence=evidence,
        causal_analysis=(
            f"提交第 {len(task_ids) + 1} 个视频时 TaskQueueFull（max_pending_tasks 上限）"
        ),
        risk_note=(
            "已受理任务不受影响继续执行；transient 类可稍后用 localize_submit "
            "重提未受理的视频。"
        ),
    )
    return SkillOutput(
        data=report, reasoning=[chain], confidence=0.0,
        sample_size=len(inp.videos),
        degraded=True, degradation_reason="queue_full_partial",
    )
