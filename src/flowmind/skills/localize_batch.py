"""localize_batch 技能：批量视频本地化编排。

对一组视频做预检（确定性，不调 HTTP）+ 调 video-localizer HTTP API 提交批量任务。
包上 FlowMind 信封：四段式推理链 / trace_id / 结构化错误 / config 化阈值。
HTTP 层统一走 VLClient（vl_client.py），本文件不直接发请求。

支持的源语言/目标语言通过 `LocalizerConfig.supported_*_langs` 配置；
阈值类（批量上限 / 成本分界 / TTS 默认 / 允许扩展名 / 服务地址）同样走 config，
不带默认值硬编码进函数体。
"""
from __future__ import annotations

from pathlib import Path

import requests  # noqa: F401  保留模块级引用：测试 fixture 经 <mod>.requests 打桩拦截 VLClient
from pydantic import BaseModel, Field, field_validator, model_validator

from flowmind.config import LocalizerConfig, load_config
from flowmind.contracts import Evidence, ReasoningChain, SkillOutput
from flowmind.errors import is_retriable
from flowmind.rules import Rule, evaluate_rules
from flowmind.skill import skill
from flowmind.vl_client import VLAPIError, VLClient

_VERSION = "0.1.0"


class _ChunkFailedError(Exception):
    """分批提交过程中某批失败时抛出，携带已成功的 batch_ids 与原始异常。

    不再依赖 invoke() 透传 details（contracts.py / skill.py 不变量）——技能体内
    catch 后分类并以 degraded SkillOutput 返回，把 successful_batch_ids / category
    等放到 LocalizerReport 的字段里。
    """
    def __init__(self, message: str, *, successful_batch_ids: list[str], failed_chunk_index: int, cause: Exception):
        super().__init__(f"{message}：{cause}")
        self.successful_batch_ids = successful_batch_ids
        self.failed_chunk_index = failed_chunk_index
        self.cause = cause


# ── 入参 ──

class LocalizerInput(BaseModel):
    """批量本地化技能入参（v0.3 OCR 定位 + 擦除 + 重绘方案）。

    **极简调用**：Agent 只需要传 `video_paths`，其他字段全走 config 默认。
    显式传任何字段都会覆盖默认。详细策略参见 OPENCLAW_OPERATOR.md §3.1。

    v0.3 重大变更：
    - 字幕处理唯一受支持策略是 `ocr_erase_redraw`（OCR 定位 bbox → 擦除原字幕 → 用目标语言重绘）
    - 老的 `delogo` / `inpaint` / `overlay` / `auto` 全部弃用
    - 业务理由：车企出海营销——目标观众不读中文，保留原字幕=视觉噪音
    """
    video_paths: list[str] = Field(
        ..., min_length=1, description="视频文件路径或 URL 列表，至少一条"
    )
    # 语言类：默认 None → 走 cfg.source_lang_default / target_lang_default
    target_lang: str | None = Field(
        default=None,
        description="目标语言代码；None=读 cfg.target_lang_default（用户偏好，可在 flowmind.config.toml 覆盖）",
    )
    source_lang: str | None = Field(
        default=None,
        description="源语言代码；None=读 cfg.source_lang_default",
    )
    # None 表示「未指定」，由 skill 读取 cfg.tts_default 兜底；显式 True/False 覆盖默认。
    enable_tts: bool | None = Field(default=None, description="是否生成 TTS 配音；None=读 cfg.tts_default")
    chat_id: str | None = Field(default=None, description="飞书通知 chat_id（可选）")
    remove_subtitles: bool | None = Field(
        default=None,
        description="是否擦除硬烧中文字幕；None = 读 cfg.remove_subtitles_default（默认 True）",
    )
    remove_subtitles_strategy: str | None = Field(
        default=None,
        description="字幕处理策略：ocr_erase_redraw=OCR 定位+擦除+重绘（v0.3 唯一支持）；None = 读 cfg.remove_subtitles_strategy_default",
    )

    @field_validator("video_paths")
    @classmethod
    def _no_empty_strings(cls, v: list[str]) -> list[str]:
        """剔除空字符串；若全部为空则报错。"""
        cleaned = [p for p in v if isinstance(p, str) and p.strip()]
        if not cleaned:
            raise ValueError("video_paths 不能全为空字符串")
        return cleaned

    @model_validator(mode="after")
    def _validate(self) -> "LocalizerInput":
        """跨字段校验：语言支持 + 字幕策略白名单 + 扩展名预检。

        放 model_validator(mode="after") 是因为校验需要 load_config()，无法在
        field_validator 阶段做（彼时 model 还未绑定 config 上下文）。
        把"全部扩展名被拒"也放在这里，让它走 VALIDATION 而不是 INTERNAL。
        """
        cfg = load_config().localizer

        # target_lang / source_lang：None 走 cfg 默认；空字符串也走默认
        eff_target = self.target_lang or cfg.target_lang_default
        eff_source = self.source_lang or cfg.source_lang_default

        if eff_target not in cfg.supported_target_langs:
            raise ValueError(
                f"不支持的目标语言：{eff_target}；"
                f"支持：{cfg.supported_target_langs}"
            )
        if eff_source not in cfg.supported_source_langs:
            raise ValueError(
                f"不支持的源语言：{eff_source}；"
                f"支持：{cfg.supported_source_langs}"
            )

        # 字幕处理策略白名单（v0.3：只支持 ocr_erase_redraw；None 走 cfg 默认，不在此校验）
        if self.remove_subtitles_strategy is not None:
            allowed_strategies = ("ocr_erase_redraw",)
            if self.remove_subtitles_strategy not in allowed_strategies:
                raise ValueError(
                    f"不支持的字幕处理策略：{self.remove_subtitles_strategy}；"
                    f"v0.3 仅支持：{list(allowed_strategies)}"
                )

        # 扩展名预检：只要存在任何合法路径即放行；全被拒则报错。
        # 真正的「提交/拒绝分桶」在技能体内做。
        accepted, _ = _split_paths(self.video_paths, cfg.allowed_extensions)
        if not accepted:
            raise ValueError(
                f"全部视频因扩展名被拒（允许：{cfg.allowed_extensions}）；"
                f"被拒：{self.video_paths}"
            )

        return self


# ── 出参 ──

class LocalizerReport(BaseModel):
    """批量本地化技能业务载荷。"""
    # batch_id: 第一批的批号（兼容旧字段）；batch_ids: 实际所有批号（拆 N 批就有 N 个）
    batch_id: str
    batch_ids: list[str]
    batch_count: int              # 拆了几批（1 表示没拆）
    job_ids: list[str]
    total: int
    submitted_count: int          # 通过预检的视频数
    rejected_count: int           # 被扩展名筛掉的视频数
    rejected_paths: list[str]
    cost_band: str                # "低" / "中" / "高"
    time_band: str                # "低" / "中" / "高"（与 cost 共用阈值启发式）
    tts_recommended: bool         # 是否开启 TTS（直接反映入参 enable_tts）
    batch_size_warning: bool      # 提交数 > max_videos_per_batch
    api_message: str              # video-localizer 合并后的提示语
    remove_subtitles: bool        # 实际传给 VL 的去字幕开关
    remove_subtitles_strategy: str # 实际传给 VL 的字幕消除策略
    # ── v0.3 失败模式（degraded=True 时填充；正常成功时为默认值） ──
    failure_category: str | None = None    # "environment" / "video" / "transient" / "unknown"
    retriable: bool = False
    successful_batch_ids: list[str] = []  # 中途失败的 partial success 信息
    failed_chunk_index: int | None = None
    warning: str | None = None            # 人类可读的失败原因


# ── 预检：扩展名拒绝 ──

def _split_paths(
    video_paths: list[str], allowed_exts: list[str]
) -> tuple[list[str], list[str]]:
    """返回 (accepted, rejected)。URL 总是 accepted；本地路径按扩展名筛。"""
    allowed = {e.lower() for e in allowed_exts}
    accepted: list[str] = []
    rejected: list[str] = []
    for p in video_paths:
        if p.startswith(("http://", "https://")):
            accepted.append(p)
            continue
        ext = Path(p).suffix.lower()
        if ext in allowed:
            accepted.append(p)
        else:
            rejected.append(p)
    return accepted, rejected


# ── 规则 / 档位 ──

def _rules(cfg: LocalizerConfig) -> list[Rule]:
    """基于配置阈值构造规则集。命中规则会进入 reasoning_chain.triggered_rules。"""
    return [
        Rule(
            id="LOC-W01",
            name="批量超额",
            expression=f"提交数 > {cfg.max_videos_per_batch}",
            predicate=lambda m: m["n_videos"] > cfg.max_videos_per_batch,
            evidence=lambda m: [Evidence(
                metric="提交视频数",
                value=m["n_videos"],
                threshold=cfg.max_videos_per_batch,
                comparison=">",
            )],
        ),
        Rule(
            id="LOC-W02",
            name="部分扩展名被拒",
            expression="存在非允许扩展名的视频",
            predicate=lambda m: m["rejected_count"] > 0,
            evidence=lambda m: [Evidence(
                metric="拒绝数",
                value=m["rejected_count"],
                threshold=0,
                comparison=">",
            )],
        ),
        Rule(
            id="LOC-W03",
            name="高成本批",
            expression=f"提交数 ≥ {cfg.cost_high_min}",
            predicate=lambda m: m["n_videos"] >= cfg.cost_high_min,
            evidence=lambda m: [Evidence(
                metric="提交视频数",
                value=m["n_videos"],
                threshold=cfg.cost_high_min,
                comparison="≥",
            )],
        ),
    ]


def _band(n: int, low_max: int, high_min: int) -> str:
    """按视频数量分档：<=低阈值=低；>=高阈值=高；之间=中。"""
    if n <= low_max:
        return "低"
    if n >= high_min:
        return "高"
    return "中"


def _health_failure_report(
    inp: "LocalizerInput", cfg: LocalizerConfig, health_url: str,
    exc: Exception, category: str,
) -> SkillOutput[LocalizerReport]:
    """健康检查失败的统一返回——构造 degraded LocalizerReport + SkillOutput。

    注意：故意不把 `health_url` 和 `exc` 的完整消息写进 report / reasoning——这些
    字段对 Agent 可见，会泄漏内部 host / 凭证。保留 category + 异常类型供 Agent
    做下一步决策，详细信息只写日志。
    """
    report = LocalizerReport(
        batch_id="", batch_ids=[], batch_count=0, job_ids=[],
        total=len(inp.video_paths), submitted_count=0, rejected_count=0,
        rejected_paths=[], cost_band="低", time_band="低",
        tts_recommended=False, batch_size_warning=False,
        api_message="video-localizer 健康检查失败",
        remove_subtitles=False, remove_subtitles_strategy="ocr_erase_redraw",
        failure_category=category,
        retriable=is_retriable(category),
        warning=f"VL 不通（{category}），未提交任何任务",
    )
    chain = ReasoningChain(
        conclusion=f"健康检查失败，未提交（{category}）",
        triggered_rules=[],
        evidence=[],
        causal_analysis=f"探测 video-localizer 健康端点 → {type(exc).__name__}",
        risk_note="先查 video-localizer 服务是否运行；environment 类错误别重试。",
    )
    return SkillOutput(
        data=report, reasoning=[chain], confidence=0.0,
        sample_size=len(inp.video_paths),
        degraded=True, degradation_reason=category,
    )


def _chunk_failure_output(
    inp: "LocalizerInput",
    cfg: LocalizerConfig,
    accepted: list[str],
    rejected: list[str],
    batch_ids: list[str],
    all_job_ids: list[str],
    chunks: list[list[str]],
    idx: int,
    hits: list,
    evidence: list[Evidence],
    cost_band: str,
    time_band: str,
    effective_tts: bool,
    effective_remove_subtitles: bool,
    effective_strategy: str,
    batch_size_warning: bool,
    exc: Exception,
    category: str,
) -> SkillOutput[LocalizerReport]:
    """分批提交过程中某批失败的统一返回——partial success 信息在 report 里。"""
    partial_batch_ids = list(batch_ids)
    partial_job_ids = list(all_job_ids)
    submitted_before = sum(len(c) for c in chunks[:idx])
    report = LocalizerReport(
        batch_id=partial_batch_ids[0] if partial_batch_ids else "",
        batch_ids=partial_batch_ids,
        batch_count=len(chunks),
        job_ids=partial_job_ids,
        total=len(accepted),
        submitted_count=submitted_before,
        rejected_count=len(rejected),
        rejected_paths=rejected,
        cost_band=cost_band,
        time_band=time_band,
        tts_recommended=effective_tts,
        batch_size_warning=batch_size_warning,
        api_message=f"第 {idx + 1}/{len(chunks)} 批失败（{category}）",
        remove_subtitles=effective_remove_subtitles,
        remove_subtitles_strategy=effective_strategy,
        failure_category=category,
        retriable=is_retriable(category),
        successful_batch_ids=partial_batch_ids,
        failed_chunk_index=idx,
        warning=f"已成功 {len(partial_batch_ids)}/{len(chunks)} 批；第 {idx + 1} 批 {category} 失败",
    )
    chain = ReasoningChain(
        conclusion=(
            f"批量提交部分失败：{len(partial_batch_ids)}/{len(chunks)} 批成功"
            + (f"，拆 {len(chunks)} 批提交" if len(chunks) > 1 else "")
            + f"（{category}）"
        ),
        triggered_rules=hits,
        evidence=evidence,
        causal_analysis=f"提交视频本地化批任务第 {idx + 1} 批 → {type(exc).__name__}",
        risk_note=(
            f"已提交任务可继续轮询；{'可重试' if is_retriable(category) else '需修环境/输入'}后再补齐。"
        ),
    )
    return SkillOutput(
        data=report, reasoning=[chain], confidence=0.0,
        sample_size=len(inp.video_paths),
        degraded=True, degradation_reason=category,
    )


# ── 四段式推理链 ──

def _build_chain(
    metrics: dict,
    hits: list,
    evidence: list[Evidence],
    cfg: LocalizerConfig,
    submitted: int,
    rejected: int,
    cost_band: str,
    tts: bool,
    batch_count: int = 1,
) -> ReasoningChain:
    """组装四段式因果推理链。

    四个文本字段全部非空：conclusion（结论）/ causal_analysis（因果推理）/
    risk_note（风险提示）+ triggered_rules + evidence 自动来自 evaluate_rules。
    """
    rule_names = "、".join(h.name for h in hits) if hits else "（无）"
    chunk_note = f"，拆 {batch_count} 批提交" if batch_count > 1 else ""
    conclusion = (
        f"批量提交 {submitted} 个视频（拒绝 {rejected} 个）{chunk_note}，"
        f"成本档位「{cost_band}」，{'建议' if tts else '不'}开 TTS。"
    )
    if hits:
        risk_note = f"命中规则：{rule_names}；建议人工复核后再确认提交。"
    else:
        risk_note = "参数在通用默认阈值内，可直接提交。"
    causal_analysis = (
        f"基于 {submitted} 条合法路径，参考 "
        f"max_videos_per_batch={cfg.max_videos_per_batch}、"
        f"cost_low_max={cfg.cost_low_max}、"
        f"cost_high_min={cfg.cost_high_min} 等阈值求值得出。"
    )
    return ReasoningChain(
        conclusion=conclusion,
        triggered_rules=hits,
        evidence=evidence,
        causal_analysis=causal_analysis,
        risk_note=risk_note,
    )


# ── 入口 ──

@skill(id="localize_batch", name="批量视频本地化编排", version=_VERSION)
def localize_batch(inp: LocalizerInput) -> SkillOutput[LocalizerReport]:
    """预检一组视频，调 video-localizer 提交批量本地化任务，返回结构化结果与四段式推理链。

    数据流：input → 预检（扩展名 / 批量上限 / 成本档位 / TTS）→ HTTP POST /batch
    → LocalizerReport + ReasoningChain → 框架套 SkillResult 信封。
    """
    cfg = load_config().localizer

    # fail-fast：先 GET /health 探活。VL 挂了立刻返回 degraded SkillOutput（不调 POST）。
    # 注意：不能 raise（SkillError 不带 category 字段——契约层不变量），
    # 而是在技能体内 catch VLAPIError（已带分类）后直接返回 degraded SkillOutput。
    client = VLClient(cfg)
    try:
        client.health_check()
    except VLAPIError as exc:
        return _health_failure_report(inp, cfg, str(client.base_url), exc, exc.category)

    # 预检（扩展名分桶）；model_validator 已保证 accepted 非空，此处不再兜底。
    accepted, rejected = _split_paths(inp.video_paths, cfg.allowed_extensions)

    # 规则求值
    metrics = {"n_videos": len(accepted), "rejected_count": len(rejected)}
    hits, evidence = evaluate_rules(_rules(cfg), metrics)

    cost_band = _band(len(accepted), cfg.cost_low_max, cfg.cost_high_min)
    time_band = _band(len(accepted), cfg.cost_low_max, cfg.cost_high_min)
    batch_size_warning = any(h.rule_id == "LOC-W01" for h in hits)

    # 解析实际生效的 target/source/remove_subtitles/strategy/enable_tts
    # None → 走 config 默认（用户偏好，写在 flowmind.config.toml）
    effective_target_lang = inp.target_lang or cfg.target_lang_default
    effective_source_lang = inp.source_lang or cfg.source_lang_default
    effective_remove_subtitles = (
        inp.remove_subtitles
        if inp.remove_subtitles is not None
        else cfg.remove_subtitles_default
    )
    effective_strategy = (
        inp.remove_subtitles_strategy
        if inp.remove_subtitles_strategy is not None
        else cfg.remove_subtitles_strategy_default
    )
    effective_tts = (
        inp.enable_tts if inp.enable_tts is not None else cfg.tts_default
    )

    # HTTP 提交：自动分批。中途失败 → degraded SkillOutput（partial success 在 data 里）。
    max_per_batch = max(1, cfg.max_videos_per_batch)
    chunks = [accepted[i:i + max_per_batch] for i in range(0, len(accepted), max_per_batch)]
    batch_ids: list[str] = []
    all_job_ids: list[str] = []
    api_messages: list[str] = []

    for idx, chunk in enumerate(chunks):
        payload = {
            "video_paths": chunk,
            "target_lang": effective_target_lang,
            "source_lang": effective_source_lang,
            "enable_tts": effective_tts,
            "remove_subtitles": effective_remove_subtitles,
            "remove_subtitles_strategy": effective_strategy,
        }
        try:
            body = client.post("/batch", payload)
        except VLAPIError as exc:
            # VLAPIError 已带分类（environment/transient/video），partial success 照旧透出
            return _chunk_failure_output(
                inp, cfg, accepted, rejected, batch_ids, all_job_ids,
                chunks, idx, hits, evidence, cost_band, time_band,
                effective_tts, effective_remove_subtitles, effective_strategy,
                batch_size_warning, exc, exc.category,
            )

        batch_ids.append(str(body.get("batch_id", "")))
        all_job_ids.extend(list(body.get("job_ids", [])))
        api_messages.append(str(body.get("message", "")))

    report = LocalizerReport(
        batch_id=batch_ids[0] if batch_ids else "",
        batch_ids=batch_ids,
        batch_count=len(chunks),
        job_ids=all_job_ids,
        total=len(accepted),
        submitted_count=len(accepted),
        rejected_count=len(rejected),
        rejected_paths=rejected,
        cost_band=cost_band,
        time_band=time_band,
        tts_recommended=effective_tts,
        batch_size_warning=batch_size_warning,
        api_message=" | ".join(api_messages),
        remove_subtitles=effective_remove_subtitles,
        remove_subtitles_strategy=effective_strategy,
    )
    chain = _build_chain(
        metrics, hits, evidence, cfg,
        submitted=len(accepted),
        rejected=len(rejected),
        cost_band=cost_band,
        tts=effective_tts,
        batch_count=len(chunks),
    )
    return SkillOutput(
        data=report,
        reasoning=[chain],
        confidence=1.0,
        sample_size=len(inp.video_paths),
    )