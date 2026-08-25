"""localize_video 技能:单条视频本地化(in-process 同步)。

跟既有 5 个 localize_* HTTP skill 不一样 — 不调 HTTP,直接在同进程内 import
并跑 video_localization_engine.VLE 的 VideoLocalizationPipeline。
- 单条视频 (单进程同步),不是 batch / task_id 模型
- 输入是 video_path (本地路径) + target_lang, 输出是 mp4 路径
- 异常在技能体内 catch + 分类后以 degraded SkillOutput 返回

阈值/配置项 (backend 切换 / mask dilation / TTS / translator) 全部走
LocalizeVideoConfig,不带默认值进函数体。
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from flowmind.config import LocalizeVideoConfig, load_config
from flowmind.contracts import Evidence, ReasoningChain, SkillOutput
from flowmind.errors import _classify_exception, is_retriable
from flowmind.rules import Rule, evaluate_rules
from flowmind.skill import skill

_VERSION = "0.1.0"


# ── 入参 ──

class LocalizeVideoInput(BaseModel):
    """单条视频本地化入参 (in-process 同步)。"""
    video_path: str = Field(
        ..., min_length=1,
        description="本地视频文件路径 (mp4)",
    )
    target_lang: str | None = Field(
        default=None,
        description="目标语言代码;None=读 cfg.target_lang_default",
    )
    source_lang: str | None = Field(
        default=None,
        description="源语言代码;None=读 cfg.source_lang_default",
    )
    enable_tts: bool | None = Field(
        default=None,
        description="是否生成 TTS;None=读 cfg.tts_default",
    )
    # 可选 backend 覆盖 (走 config,技能体内取出后传给 VLE PipelineConfig)
    translator_backend: str | None = Field(default=None)
    renderer_backend: str | None = Field(default=None)
    tts_backend: str | None = Field(default=None)
    inpaint_backend: str | None = Field(default=None)
    mask_dilation_y: int | None = Field(
        default=None, ge=0,
        description="mask y 方向扩张像素;None=读 cfg.mask_dilation_y",
    )
    output_path: str | None = Field(
        default=None,
        description="输出 mp4 路径;None=VLE 自动写 {stem}_{target_lang}.mp4",
    )

    @field_validator("video_path")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("video_path 不能为空")
        return v


# ── 出参 ──

class LocalizeVideoReport(BaseModel):
    """单条本地化业务载荷。"""
    video_path: str
    output_path: str | None      # 成功时 mp4 路径;失败时 None
    target_lang: str             # 生效的目标语言
    source_lang: str             # 生效的源语言
    enable_tts: bool             # 生效的 TTS 开关
    instances: int               # 字幕 instance 数
    frames: int                  # 处理的帧数
    degraded: bool = False       # 是否降级
    failure_category: str | None = None  # "environment" / "video" / "transient" / "unknown"
    retriable: bool = False
    warning: str | None = None


# ── 规则 / 档位 ──

def _rules() -> list[Rule]:
    """基于 in-process 任务的简易规则。"""
    return [
        Rule(
            id="LVS-01",
            name="目标语言非默认",
            expression="target_lang != cfg.target_lang_default",
            predicate=lambda m: bool(m.get("target_override", False)),
            evidence=lambda m: [Evidence(
                metric="目标语言",
                value=m.get("target_lang", ""),
                threshold="default=en",
                comparison="≠",
            )],
        ),
        Rule(
            id="LVS-02",
            name="in-process 单条",
            expression="单条视频同步本地化",
            predicate=lambda m: True,
            evidence=lambda m: [Evidence(
                metric="任务粒度",
                value="single-video-sync",
                comparison="==",
            )],
        ),
    ]


def _build_chain(
    metrics: dict, hits: list, evidence: list[Evidence], report: LocalizeVideoReport,
    cfg: LocalizeVideoConfig,
) -> ReasoningChain:
    """组装四段式因果推理链。"""
    rule_names = "、".join(h.name for h in hits) if hits else "（无）"
    conclusion = (
        f"in-process 本地化 {report.video_path} → "
        f"{report.output_path or '（失败，见 warning）'}；"
        f"识别 {report.instances} 个字幕,处理 {report.frames} 帧。"
    )
    risk_note = (
        f"命中规则：{rule_names}；输出位于 {report.output_path}。"
        if hits and report.output_path
        else f"参数在通用默认阈值内；输出位于 {report.output_path}。"
        if report.output_path
        else f"任务未完成，请看 warning（{report.failure_category or 'unknown'}）。"
    )
    causal_analysis = (
        f"VLE PipelineConfig 直接 in-process 跑 L1-L7；"
        f"translator/renderer/tts/inpaint backend 由 cfg 选定，"
        f"末选 = {metrics.get('effective_backends', '默认')}。"
    )
    return ReasoningChain(
        conclusion=conclusion,
        triggered_rules=hits,
        evidence=evidence,
        causal_analysis=causal_analysis,
        risk_note=risk_note,
    )


def _failure_output(
    inp: LocalizeVideoInput, exc: Exception, category: str, cfg: LocalizeVideoConfig,
    eff_target: str, eff_source: str, eff_tts: bool,
) -> SkillOutput[LocalizeVideoReport]:
    """统一失败返回:degraded SkillOutput。
    注意:warning 不放完整 exc 消息(避免泄漏),只放 category + 异常类型名。
    """
    report = LocalizeVideoReport(
        video_path=inp.video_path,
        output_path=None,
        target_lang=eff_target,
        source_lang=eff_source,
        enable_tts=eff_tts,
        instances=0, frames=0,
        degraded=True,
        failure_category=category,
        retriable=is_retriable(category),
        warning=f"VLE 调用失败（{category}）:{type(exc).__name__}",
    )
    chain = ReasoningChain(
        conclusion=f"视频本地化失败（{category}）",
        triggered_rules=[],
        evidence=[],
        causal_analysis=f"VideoLocalizationPipeline.run_localize() → {type(exc).__name__}",
        risk_note=(
            f"{'可重试' if is_retriable(category) else '需修环境/视频本身'}。"
        ),
    )
    return SkillOutput(
        data=report, reasoning=[chain], confidence=0.0,
        sample_size=1, degraded=True, degradation_reason=category,
    )


# ── 入口 ──

@skill(id="localize_video", name="单条视频本地化（in-process 同步）", version=_VERSION)
def localize_video(inp: LocalizeVideoInput) -> SkillOutput[LocalizeVideoReport]:
    """in-process 单条视频本地化:跑 VLE L1-L7,返回 mp4 路径 + 四段式推理链。

    数据流:
      VideoLocalizationPipeline(video_path, config=PipelineConfig(...))
        .run_localize(target_locale=..., output_path=...)
      → 写 mp4 + artifacts,返回 LocalizedTrack (带 rendered_video_path)
      → 套 SkillOutput 信封

    backend 选择 / mask_dilation_y 等所有可调项走 cfg;None 字段回落 cfg 默认。
    """
    cfg: LocalizeVideoConfig = load_config().localize_video

    # 解析生效值
    eff_target = inp.target_lang or cfg.target_lang_default
    eff_source = inp.source_lang or cfg.source_lang_default
    eff_tts = inp.enable_tts if inp.enable_tts is not None else cfg.tts_default
    eff_translator = inp.translator_backend or cfg.translator_backend_default
    eff_renderer = inp.renderer_backend or cfg.renderer_backend_default
    eff_tts_backend = inp.tts_backend or cfg.tts_backend_default
    eff_inpaint = inp.inpaint_backend or cfg.inpaint_backend_default
    eff_mask_y = inp.mask_dilation_y if inp.mask_dilation_y is not None else cfg.mask_dilation_y_default
    eff_x = cfg.mask_dilation_x_default  # 一律走 cfg

    # 文件预检 (cheap,不走 VLE)
    if not Path(inp.video_path).exists():
        return _failure_output(
            inp, FileNotFoundError(inp.video_path), "video", cfg,
            eff_target, eff_source, eff_tts,
        )
    if cfg.allowed_extensions:
        ext = Path(inp.video_path).suffix.lower()
        if ext and ext not in {e.lower() for e in cfg.allowed_extensions}:
            return _failure_output(
                inp, ValueError(f"扩展名 {ext} 不在 {cfg.allowed_extensions}"),
                "video", cfg, eff_target, eff_source, eff_tts,
            )

    # 文件预检过了才 import VLE (orchestrator 顶层 import 会拉 cv2, 别为不存在的文件付出)
    from video_localization_engine.orchestrator import (
        PipelineConfig as VLEPipelineConfig,
        VideoLocalizationPipeline,
    )

    vle_cfg = VLEPipelineConfig(
        target_locale=eff_target,
        translator_backend=eff_translator,
        renderer_backend=eff_renderer,
        tts_backend=eff_tts_backend,
        inpaint_backend=eff_inpaint,
        mask_dilation_px=eff_x,
        mask_dilation_y=eff_mask_y,
    )

    # 跑 VLE in-process
    try:
        pipe = VideoLocalizationPipeline(inp.video_path, config=vle_cfg)
    except Exception as exc:
        return _failure_output(
            inp, exc, _classify_exception(exc), cfg,
            eff_target, eff_source, eff_tts,
        )

    try:
        track = pipe.run_localize(target_locale=eff_target, output_path=inp.output_path)
    except Exception as exc:
        return _failure_output(
            inp, exc, _classify_exception(exc), cfg,
            eff_target, eff_source, eff_tts,
        )
    finally:
        try:
            pipe.close()
        except Exception:
            pass

    # 计算 instance / frame (拿 pipe 内部状态)
    instances_n = len(getattr(pipe, "_all_instances", []) or [])
    report = LocalizeVideoReport(
        video_path=inp.video_path,
        output_path=track.rendered_video_path,
        target_lang=eff_target,
        source_lang=eff_source,
        enable_tts=eff_tts,
        instances=instances_n,
        frames=pipe.meta.frame_count if pipe.meta else 0,
    )

    metrics = {
        "target_lang": eff_target,
        "target_override": inp.target_lang is not None,
        "effective_backends": f"translator={eff_translator}/renderer={eff_renderer}/tts={eff_tts_backend}/inpaint={eff_inpaint}",
    }
    hits, evidence = evaluate_rules(_rules(), metrics)
    chain = _build_chain(metrics, hits, evidence, report, cfg)

    return SkillOutput(
        data=report, reasoning=[chain],
        confidence=1.0, sample_size=1,
    )
