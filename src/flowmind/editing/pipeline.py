"""剪辑流水线编排器：阶段执行 + 审批闸门 + 进度/取消 + 产物落盘。

运行位置：TaskManager worker 内（edit_pipeline 技能调用本模块）——进度经
TaskContext.progress_cb、取消经 TaskContext.cancel_check 协作生效。

阶段顺序与审批闸门（设计文档 §4）：

    understand → copy →[copy 闸]→ script →[script 闸]→ storyboard
                →[storyboard 闸]→ [pre_render 闸]→ render → succeeded

闸门语义：阶段产物落库后，若该阶段闸在 project.review_stages 且未 approve，
项目置 awaiting_review 并**正常结束本次任务**（不占 GPU worker）；
pre_render 闸在进入渲染前停靠（前置检查，续跑也能命中）。edit_review 的
approve/revise 驱动新的续跑任务；失败/取消的项目经 edit_retry 断点续跑。
revise 把该阶段打回 pending 并把反馈注入该阶段 LLM payload 的 _feedback 键。

错误语义：阶段失败 → 项目 state=failed + error 落库 → 抛
EditingStageError（category 与 errors.py 一致），由技能层转 degraded 信封。
取消：阶段边界检测 → 项目 state=cancelled 落库 → 抛 tasks.CancelledError。
"""
from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path

from flowmind.editing import llm as edit_llm
from flowmind.editing import media
from flowmind.editing.models import (
    PSTATE_AWAITING_REVIEW,
    PSTATE_CANCELLED,
    PSTATE_FAILED,
    PSTATE_RUNNING,
    PSTATE_SUCCEEDED,
    STAGE_COPY,
    STAGE_DONE,
    STAGE_ORDER,
    STAGE_RENDER,
    STAGE_SCRIPT,
    STAGE_STORYBOARD,
    STAGE_UNDERSTAND,
    AssetMeta,
    AssetRecord,
    ContentNote,
    CopyDoc,
    ProjectRecord,
    ScriptDoc,
    StoryboardDoc,
    TimelineDoc,
    TimelineShot,
    UnderstandReport,
    clamp_transition,
)
from flowmind.editing.store import get_project_store
from flowmind.tasks import CancelledError, current_task_context

logger = logging.getLogger(__name__)

# 阶段 → 进度区间（0~100）
_STAGE_PROGRESS: dict[str, tuple[float, float]] = {
    STAGE_UNDERSTAND: (0.0, 15.0),
    STAGE_COPY: (15.0, 35.0),
    STAGE_SCRIPT: (35.0, 55.0),
    STAGE_STORYBOARD: (55.0, 75.0),
    STAGE_RENDER: (75.0, 100.0),
}


class EditingStageError(RuntimeError):
    """阶段执行失败。category/retriable 语义与 errors.py 一致。"""

    def __init__(self, message: str, category: str = "unknown",
                 retriable: bool = False, stage: str = "") -> None:
        super().__init__(message)
        self.category = category
        self.retriable = retriable
        self.stage = stage


def project_dir(project_id: str) -> Path:
    """项目产物目录：data_dir/editing/<project_id>/。"""
    from flowmind.config import load_config
    root = Path(load_config().localizer.data_dir).expanduser().resolve()
    return root / "editing" / project_id


def _ctx_progress(stage: str, pct: float, event: str, message: str = "") -> None:
    """进度上报（事件语义 JSON 串进 message，见设计文档 §6）。"""
    ctx = current_task_context()
    if ctx is None:
        return
    payload = json.dumps(
        {"stage": stage, "event": event, "detail": message},
        ensure_ascii=False)
    ctx.progress_cb(stage, pct, payload)


def _ctx_cancel(stage: str) -> None:
    ctx = current_task_context()
    if ctx is not None and ctx.cancel_check():
        raise CancelledError(f"剪辑项目在阶段 {stage} 边界被取消")


def run_pipeline(project_id: str) -> ProjectRecord:
    """执行/续跑一个剪辑项目直到：全部完成 / 审批闸 / 失败 / 取消。"""
    store = get_project_store()
    project = store.get_project(project_id)
    if project is None:
        raise EditingStageError(
            f"项目不存在：{project_id}", category="unknown")

    try:
        for stage in STAGE_ORDER:
            if project.stages.get(stage) == STAGE_DONE:
                continue
            _ctx_cancel(stage)
            # pre_render 闸前置检查：进入渲染前停靠（续跑时前置阶段已 done
            # 被跳过，只有前置检查才能让 pre_render 闸在续跑路径上命中）
            if stage == STAGE_RENDER and _pending_gate(project, "pre_render"):
                gate = "pre_render"
                project.state = PSTATE_AWAITING_REVIEW
                project.stage = gate
                store.save_project(project)
                _ctx_progress(STAGE_RENDER, _STAGE_PROGRESS[STAGE_RENDER][0],
                              "awaiting_review", gate)
                logger.info("项目 %s 进入审批等待：%s", project_id, gate)
                return project
            project.state = PSTATE_RUNNING
            project.stage = stage
            store.save_project(project)

            handler = _STAGE_HANDLERS[stage]
            handler(project, store)

            project.stages[stage] = STAGE_DONE
            store.save_project(project)

            # 审批闸：storyboard 完成后可停 storyboard 与 pre_render 两道
            gate = _pending_gate(project, stage)
            if gate is not None:
                project.state = PSTATE_AWAITING_REVIEW
                project.stage = gate
                store.save_project(project)
                _ctx_progress(stage, _STAGE_PROGRESS[stage][1],
                              "awaiting_review", gate)
                logger.info("项目 %s 进入审批等待：%s", project_id, gate)
                return project

        project.state = PSTATE_SUCCEEDED
        project.stage = STAGE_ORDER[-1]
        store.save_project(project)
        _ctx_progress(STAGE_RENDER, 100.0, "done", "成片产出")
        return project
    except CancelledError:
        project.state = PSTATE_CANCELLED
        store.save_project(project)
        raise
    except edit_llm.EditingLLMError as exc:
        _fail(project, store, str(exc), stage=project.stage)
        raise EditingStageError(
            str(exc), category=exc.category, retriable=exc.retriable,
            stage=project.stage) from exc
    except media.MediaError as exc:
        _fail(project, store, str(exc), stage=project.stage)
        raise EditingStageError(
            str(exc), category=exc.category, retriable=exc.retriable,
            stage=project.stage) from exc
    except EditingStageError as exc:
        _fail(project, store, str(exc), stage=exc.stage or project.stage)
        raise
    except Exception as exc:  # noqa: BLE001 未预期异常兜底：项目失败不静默
        logger.exception("剪辑项目 %s 阶段 %s 未预期异常",
                         project_id, project.stage)
        _fail(project, store, f"{type(exc).__name__}: {exc}",
              stage=project.stage)
        raise EditingStageError(
            f"{type(exc).__name__}: {exc}", category="unknown",
            stage=project.stage) from exc


def _fail(project: ProjectRecord, store, message: str, *, stage: str) -> None:
    project.state = PSTATE_FAILED
    project.error = f"[{stage}] {message}"
    store.save_project(project)


def _pending_gate(project: ProjectRecord, gate: str) -> str | None:
    """审批闸：该闸在 project.review_stages 且未 approve → 返回闸名（停靠）。

    copy/script/storyboard 闸在对应阶段完成后检查；pre_render 闸由
    run_pipeline 在进入渲染前前置检查（保证续跑路径也能命中）。
    """
    if gate in project.review_stages and gate not in project.approved:
        return gate
    return None


# ── 阶段实现 ──

def _stage_understand(project: ProjectRecord, store) -> None:
    cfg_data = project.input
    paths: list[str] = cfg_data.get("asset_paths") or []
    if not paths:
        raise EditingStageError("项目没有任何素材路径", category="video",
                                stage=STAGE_UNDERSTAND)
    assets: list[AssetMeta] = []
    for i, path in enumerate(paths):
        _ctx_cancel(STAGE_UNDERSTAND)
        _ctx_progress(STAGE_UNDERSTAND,
                      _STAGE_PROGRESS[STAGE_UNDERSTAND][0]
                      + (i / len(paths)) * 10, "understanding", path)
        try:
            info = media.probe_summary(path)
        except media.MediaError as exc:
            raise EditingStageError(
                f"素材不可读：{path}（{exc}）", category=exc.category,
                stage=STAGE_UNDERSTAND) from exc
        kind = info["kind"]
        shots = []
        if kind == "video":
            try:
                shots = media.scene_shots(
                    path, threshold=cfg_data.get("scene_threshold", 0.3),
                    max_shots=cfg_data.get("max_shots_per_video", 12))
            except media.MediaError:
                logger.warning("场景切分失败，退化为整段单镜头：%s",
                               path, exc_info=True)
                shots = []
        asset = AssetMeta(
            id=f"a{i + 1}", kind=kind, path=str(Path(path).resolve()),
            duration_s=round(info["duration_s"], 3),
            width=info["width"], height=info["height"], fps=info["fps"],
            has_audio=info["has_audio"], shots=shots,
            tags=[kind, _duration_bucket(info["duration_s"])],
            summary=f"{kind} 素材 {info['width']}x{info['height']} "
                    f"{info['duration_s']:.1f}s "
                    f"{'shots=' + str(len(shots)) if shots else ''}".strip(),
        )
        # v1.1 视觉理解：抽帧 → MiniMax vision 看图 + 读画面字幕。
        # transient 失败=显式盲剪降级（vision_degraded 记原因，绝不静默）；
        # environment/配额类失败=快速失败——同一 key 的 copy/script/storyboard
        # 必然同样失败，盲剪只会白白烧完整个流水线（10-28 实测教训）
        cfg = _cfg().editing
        if kind == "video" and cfg.vision_enabled and info["duration_s"] > 1:
            n = min(cfg.understand_frame_samples, cfg.vision_max_frames)
            times = [info["duration_s"] * (i + 0.5) / n for i in range(n)]
            frames_dir = project_dir(project.project_id) / "frames" / asset.id
            try:
                frames = media.extract_frames(
                    path, times, frames_dir, width=cfg.vision_frame_width)
                _ctx_progress(STAGE_UNDERSTAND, 8.0, "vision",
                              f"{asset.id} 看图 {len(frames)} 帧")
                notes = edit_llm.describe_frames(
                    [(t, str(p)) for t, p in frames],
                    api_base=cfg.llm_api_base,
                    api_key=_api_key(cfg.llm_api_key_env) or "",
                    model=cfg.llm_vision_model or cfg.llm_model,
                    timeout_s=cfg.llm_timeout_s,
                    max_tokens=cfg.llm_max_tokens, mock=cfg.mock,
                    max_attempts=cfg.llm_max_attempts,
                    protocol=cfg.llm_protocol)
                try:
                    asset.content_notes = [
                        ContentNote(
                            t_s=round(float(n0.get("t_s", times[j])), 2),
                            scene=str(n0.get("scene", ""))[:200],
                            subtitle=str(n0.get("subtitle", ""))[:200],
                            importance=int(n0.get("importance", 3)))
                        for j, n0 in enumerate(notes)]
                except (TypeError, ValueError) as exc:
                    # 笔记字段脏数据：降级盲剪而非整项目失败（describe_frames
                    # 已做数值清洗，这里是最后一道防线）
                    raise edit_llm.EditingLLMError(
                        f"vision 笔记字段不可解析：{exc}",
                        category="transient") from exc
            except edit_llm.EditingLLMError as exc:
                if exc.category == "environment":
                    raise EditingStageError(
                        f"视觉理解环境/配额失败（快速失败，不做盲剪）：{exc}",
                        category="environment",
                        stage=STAGE_UNDERSTAND) from exc
                asset.vision_degraded = f"{type(exc).__name__}: {exc}"
                logger.warning("素材 %s 视觉理解降级（盲剪模式）：%s",
                               asset.id, exc)
                _ctx_progress(STAGE_UNDERSTAND, 8.0, "vision_degraded",
                              str(exc)[:200])
            except media.MediaError as exc:
                asset.vision_degraded = f"{type(exc).__name__}: {exc}"
                logger.warning("素材 %s 视觉理解降级（盲剪模式）：%s",
                               asset.id, exc)
                _ctx_progress(STAGE_UNDERSTAND, 8.0, "vision_degraded",
                              str(exc)[:200])
        assets.append(asset)
        # 素材库自动登记（edit_assets 可见）
        store.save_asset(AssetRecord(
            id=f"{project.project_id}:{asset.id}", kind=kind, path=asset.path,
            tags=asset.tags + [project.project_id],
            project_id=project.project_id, tenant_id=project.tenant_id))

    total = sum(a.duration_s for a in assets)
    noted = sum(1 for a in assets if a.content_notes)
    report = UnderstandReport(
        assets=assets, total_duration_s=round(total, 3),
        summary=f"共 {len(assets)} 个素材（"
                + "、".join(
                    f"{a.kind}:{a.id}"
                    + (f"[{len(a.content_notes)}个内容点]" if a.content_notes
                       else "")
                    for a in assets)
                + f"），素材总时长 {total:.1f}s"
                + (f"；{noted} 个视频已完成视觉理解" if noted else ""))
    _write_artifact(project.project_id, "understand.json", report)
    _ctx_progress(STAGE_UNDERSTAND, 15.0, "stage_done", report.summary)


def _duration_bucket(duration_s: float) -> str:
    if duration_s <= 10:
        return "short"
    if duration_s <= 60:
        return "medium"
    return "long"


def _load_understand(project: ProjectRecord) -> UnderstandReport:
    data = _read_artifact(project.project_id, "understand.json")
    if data is None:
        raise EditingStageError(
            "缺少 understand.json（阶段顺序被破坏）", stage=STAGE_COPY)
    return UnderstandReport.model_validate(data)


def _feedback_payload(project: ProjectRecord, stage: str) -> dict:
    return {"_feedback": project.review_feedback.get(stage, [])}


def _content_points(report: UnderstandReport, cap: int = 40) -> list[dict]:
    """视觉内容点清单（喂给 copy/script 的事实弹药库）。

    此前 copy/script 只看 asset_summary 一行摘要=闭眼写文案，实测产出
    "第一点直击要害"式空洞排比（用户反馈"字幕文案很怪"）。cap 防多素材
    载荷膨胀（storyboard 目录另有 _cap_catalog 预算守卫）。
    """
    points: list[dict] = []
    for a in report.assets:
        for c in a.content_notes:
            points.append({"asset": a.id, "t_s": c.t_s,
                           "scene": c.scene, "subtitle": c.subtitle})
    return points[:cap]


def _stage_copy(project: ProjectRecord, store) -> None:
    cfg = _cfg().editing
    report = _load_understand(project)
    payload = {
        "theme": project.input.get("theme", ""),
        "style": project.input.get("style") or "",
        "target_duration_s": project.input.get("target_duration_s") or 0,
        "asset_summary": report.summary,
        "content_points": _content_points(report),
        "extra_brief": project.input.get("extra_brief") or "",
        **_feedback_payload(project, STAGE_COPY),
    }
    _ctx_progress(STAGE_COPY, 20.0, "generating", "文案生成中")
    doc = edit_llm.generate_json(
        STAGE_COPY, payload, schema=CopyDoc,
        api_base=cfg.llm_api_base,
        api_key=_api_key(cfg.llm_api_key_env) or "",
        model=cfg.llm_model, timeout_s=cfg.llm_timeout_s,
        max_tokens=cfg.llm_max_tokens, mock=cfg.mock,
        max_attempts=cfg.llm_max_attempts, protocol=cfg.llm_protocol)
    _write_artifact(project.project_id, "copy.json", doc)
    _ctx_progress(STAGE_COPY, 35.0, "stage_done", f"标题：{doc.title}")


def _stage_script(project: ProjectRecord, store) -> None:
    cfg = _cfg().editing
    report = _load_understand(project)
    copy_raw = _read_artifact(project.project_id, "copy.json")
    if copy_raw is None:
        raise EditingStageError("缺少 copy.json", stage=STAGE_SCRIPT)
    copy_doc = CopyDoc.model_validate(copy_raw)
    budget = float(project.input.get("target_duration_s")
                   or min(cfg.max_duration_s, max(15.0, report.total_duration_s)))
    budget = min(budget, cfg.max_duration_s)
    payload = {
        "theme": project.input.get("theme", ""),
        "copy": copy_doc.model_dump(),
        "duration_budget_s": round(budget, 1),
        "content_points": _content_points(report),
        "extra_brief": project.input.get("extra_brief") or "",
        **_feedback_payload(project, STAGE_SCRIPT),
    }
    _ctx_progress(STAGE_SCRIPT, 40.0, "generating", "脚本生成中")
    doc = edit_llm.generate_json(
        STAGE_SCRIPT, payload, schema=ScriptDoc,
        api_base=cfg.llm_api_base,
        api_key=_api_key(cfg.llm_api_key_env) or "",
        model=cfg.llm_model, timeout_s=cfg.llm_timeout_s,
        max_tokens=cfg.llm_max_tokens, mock=cfg.mock,
        max_attempts=cfg.llm_max_attempts, protocol=cfg.llm_protocol)
    if not doc.sections:
        raise EditingStageError("脚本为空（LLM 产出 0 段）", category="unknown",
                                stage=STAGE_SCRIPT)
    doc.total_duration_s = round(sum(s.duration_s for s in doc.sections), 3)
    _write_artifact(project.project_id, "script.json", doc)
    _ctx_progress(STAGE_SCRIPT, 55.0, "stage_done",
                  f"{len(doc.sections)} 段 / {doc.total_duration_s}s")


def _stage_storyboard(project: ProjectRecord, store) -> None:
    cfg = _cfg().editing
    report = _load_understand(project)
    script_raw = _read_artifact(project.project_id, "script.json")
    if script_raw is None:
        raise EditingStageError("缺少 script.json", stage=STAGE_STORYBOARD)
    script = ScriptDoc.model_validate(script_raw)
    catalog = _cap_catalog([{
        "id": a.id, "kind": a.kind, "duration_s": a.duration_s,
        "width": a.width, "height": a.height,
        "shots": [s.model_dump() for s in a.shots] if a.kind == "video" else [],
        "content_notes": [c.model_dump() for c in a.content_notes],
        "summary": a.summary,
    } for a in report.assets])
    payload = {
        "sections": [s.model_dump() for s in script.sections],
        "duration_budget_s": script.total_duration_s,
        "assets": catalog,
        "extra_brief": project.input.get("extra_brief") or "",
        **_feedback_payload(project, STAGE_STORYBOARD),
    }
    _ctx_progress(STAGE_STORYBOARD, 60.0, "generating", "分镜生成中")
    doc = edit_llm.generate_json(
        STAGE_STORYBOARD, payload, schema=StoryboardDoc,
        api_base=cfg.llm_api_base,
        api_key=_api_key(cfg.llm_api_key_env) or "",
        model=cfg.llm_model, timeout_s=cfg.llm_timeout_s,
        max_tokens=cfg.llm_max_tokens, mock=cfg.mock,
        max_attempts=cfg.llm_max_attempts, protocol=cfg.llm_protocol)
    _validate_storyboard(doc, report.assets)
    # section_id 漂移（模型编了脚本里不存在的段落）置空 + 告警，不整段打回；
    # 总时长显著偏离预算也只告警（渲染层会按素材实际时长钳制）
    valid_sections = {s.id for s in script.sections}
    for shot in doc.shots:
        if shot.section_id and shot.section_id not in valid_sections:
            logger.warning("镜头 %s 的 section_id=%s 不在脚本段落中，置空",
                           shot.id, shot.section_id)
            shot.section_id = ""
    doc.total_duration_s = round(sum(s.duration_s for s in doc.shots), 3)
    budget = script.total_duration_s or float(
        project.input.get("target_duration_s") or 0)
    if budget > 0 and doc.total_duration_s > 0:
        deviation = abs(doc.total_duration_s - budget) / budget
        if deviation > 0.3:
            _ctx_progress(
                STAGE_STORYBOARD, 75.0, "budget_drift",
                f"分镜总时长 {doc.total_duration_s}s 偏离预算 {budget}s "
                f"超过 30%（{deviation:.0%}）")
    _write_artifact(project.project_id, "storyboard.json", doc)
    _ctx_progress(STAGE_STORYBOARD, 75.0, "stage_done",
                  f"{len(doc.shots)} 镜头 / {doc.total_duration_s}s")


def _cap_catalog(catalog: list[dict], *,
                 budget_chars: int = 150_000) -> list[dict]:
    """storyboard 载荷预算守卫：素材目录超限时逐档裁剪（不静默丢素材）。

    上限 20 素材 × 12 镜头 × 8 笔记全量内联可能打爆模型上下文。裁剪梯度：
    先截文本 + 笔记按重要度留 top5 + 镜头留前 10；仍超限再压到 top3/前 8。
    素材条目本身不删（分镜仍可引用全部素材，只是"看"得粗一点）。
    """
    if len(json.dumps(catalog, ensure_ascii=False)) <= budget_chars:
        return catalog
    for cap_notes, text_len, cap_shots in ((5, 120, 10), (3, 80, 8)):
        catalog = [{
            **a,
            "content_notes": [
                {**c, "scene": str(c.get("scene", ""))[:text_len],
                 "subtitle": str(c.get("subtitle", ""))[:text_len]}
                for c in sorted(a.get("content_notes") or [],
                                key=lambda c: c.get("importance", 0),
                                reverse=True)[:cap_notes]],
            "shots": (a.get("shots") or [])[:cap_shots],
            "summary": str(a.get("summary", ""))[:160],
        } for a in catalog]
        size = len(json.dumps(catalog, ensure_ascii=False))
        if size <= budget_chars:
            logger.info("素材目录裁剪后 %d 字符（预算 %d）", size, budget_chars)
            return catalog
    logger.warning("素材目录按最小档裁剪后仍 %d 字符（预算 %d），原样传入",
                   size, budget_chars)
    return catalog


def _validate_storyboard(doc: StoryboardDoc, assets: list[AssetMeta]) -> None:
    """LLM 分镜引用校验：素材存在、视频区间合法、镜头非空。"""
    if not doc.shots:
        raise EditingStageError("分镜为空（LLM 产出 0 镜头）",
                                stage=STAGE_STORYBOARD)
    by_id = {a.id: a for a in assets}
    for shot in doc.shots:
        if shot.mode == "card":
            raise EditingStageError(
                f"镜头 {shot.id} 使用了保留模式 card（设计卡只能由"
                f" title_card/end_card 字段产生）", stage=STAGE_STORYBOARD)
        asset = by_id.get(shot.source_asset)
        if asset is None:
            raise EditingStageError(
                f"镜头 {shot.id} 引用了不存在的素材：{shot.source_asset}",
                stage=STAGE_STORYBOARD)
        if shot.mode == "video_clip" and asset.kind != "video":
            raise EditingStageError(
                f"镜头 {shot.id} 对非视频素材使用了 video_clip：{asset.id}",
                stage=STAGE_STORYBOARD)
        if shot.mode == "image_kenburns" and asset.kind != "image":
            raise EditingStageError(
                f"镜头 {shot.id} 对非图片素材使用了 image_kenburns：{asset.id}",
                stage=STAGE_STORYBOARD)
        if asset.duration_s > 0 and shot.source_in >= asset.duration_s:
            raise EditingStageError(
                f"镜头 {shot.id} 的 source_in={shot.source_in} 超出素材时长 "
                f"{asset.duration_s}s", stage=STAGE_STORYBOARD)
    # v1.1 叙事顺序校验：source_in 大幅回跳视为乱序（允许 ≤3 次回跳做
    # 开头呼应/书签式剪辑；超过即打回重排——乱序是盲剪垃圾成片的主因）
    inversions = sum(
        1 for prev, cur in zip(doc.shots, doc.shots[1:])
        if cur.source_in < prev.source_in - 5.0)
    budget = max(3, len(doc.shots) // 4)
    if inversions > budget:
        raise EditingStageError(
            f"分镜叙事乱序：{inversions} 处大幅回跳（容许 {budget} 次）。"
            f"请按 content_notes 时间顺序重排镜头（revise 可带反馈重跑）",
            stage=STAGE_STORYBOARD)


def _stage_render(project: ProjectRecord, store) -> None:
    cfg = _cfg().editing
    pdir = project_dir(project.project_id)
    work = pdir / "work"
    report = _load_understand(project)
    sb_raw = _read_artifact(project.project_id, "storyboard.json")
    if sb_raw is None:
        raise EditingStageError("缺少 storyboard.json", stage=STAGE_RENDER)
    storyboard = StoryboardDoc.model_validate(sb_raw)
    by_id = {a.id: a for a in report.assets}

    # ── 时间线：绝对时间轴 + 素材路径解析 ──
    # 钳制：LLM 给的裁剪区间可能超出素材尾部（ffmpeg -t 越界产出比预期短
    # 的段 → xfade offset 越界报错），时间线以素材实际可取时长为准
    width, height, fps = _resolve_format(project, report)
    shots: list[TimelineShot] = []
    # v1.3 设计卡 → 合成时间线镜头（深色底由 media 渲染，文字走 ASS 动画）
    if storyboard.title_card is not None:
        shots.append(TimelineShot(
            id="card_title", section_id="", source_asset="", source_in=0.0,
            duration_s=storyboard.title_card.duration_s, mode="card",
            pan="none", transition="none", subtitle_text="",
            note="开场设计卡"))
    for shot in storyboard.shots:
        asset = by_id[shot.source_asset]
        duration = shot.duration_s
        speed = shot.speed or 1.0
        freeze = shot.freeze_s or 0.0
        # freeze 先对齐帧网格（tpad 按整帧补），保证内容段/定格段都落整帧
        freeze = round(freeze * fps) / fps
        if asset.duration_s > 0:
            remain = asset.duration_s - shot.source_in
            if remain <= 0:
                raise EditingStageError(
                    f"镜头 {shot.id} 起点 {shot.source_in}s 已超出素材 "
                    f"{asset.id} 时长 {asset.duration_s}s",
                    stage=STAGE_RENDER)
            # v1.1 变速源窗口 = (duration − freeze) × speed ≤ remain
            duration = max(0.3, min(duration, freeze + remain / speed))
        # 时长量化到帧网格：段长按整帧落地（fps=30 时 1 帧 = 33ms），时间线
        # /转场折算/实际段长必须同一网格，否则 offset 恒等式失效截掉尾镜
        # （离线渲染回归实测）。量化值保留 6 位小数避免浮点表示噪声
        duration = round(duration * fps) / fps
        shot_data = shot.model_dump()
        shot_data.update(duration_s=round(duration, 6), freeze_s=freeze,
                         transition=clamp_transition(shot.transition),
                         source_path=asset.path)
        shots.append(TimelineShot(**shot_data))
    if storyboard.end_card is not None:
        shots.append(TimelineShot(
            id="card_end", section_id="", source_asset="", source_in=0.0,
            duration_s=storyboard.end_card.duration_s, mode="card",
            pan="none", transition="fade", subtitle_text="",
            note="片尾设计卡"))
    # 全局时间折算 xfade 重叠（转场是重叠耗时：成片总时长 = Σd − Σstep，
    # SRT/timeline 必须按折算后的时间轴对齐成片，否则字幕逐镜漂移）
    starts = media.effective_global_starts(
        shots, transition_duration_s=cfg.transition_duration_s, fps=fps)
    for shot, start in zip(shots, starts):
        shot.global_start_s = start
        shot.global_end_s = round(start + shot.duration_s, 3)
    total = starts[-1] + shots[-1].duration_s if shots else 0.0
    timeline = TimelineDoc(
        width=width, height=height, fps=fps,
        duration_s=round(total, 3), shots=shots,
        title_card=storyboard.title_card, end_card=storyboard.end_card)

    # ── BGM 底乐（v1.3：input.bgm_path 本地音频；缺失告警不阻断）──
    bgm_raw = project.input.get("bgm_path")
    if bgm_raw:
        bgm_file = Path(bgm_raw)
        if bgm_file.exists():
            timeline.bgm_path = str(bgm_file)
        else:
            logger.warning("BGM 文件不存在，成片将无底乐：%s", bgm_raw)
            _ctx_progress(STAGE_RENDER, 80.0, "bgm_missing", str(bgm_raw))

    # ── 字幕（subtitle=False 跳过生成与烧录；timeline.srt_path=None 可查）──
    # v1.3：ASS 动态字幕（关键词高亮/弹入/角标/设计卡文字）+ SRT 外挂兼容
    ass_path: Path | None = None
    srt_path: Path | None = None
    if project.input.get("subtitle", True):
        ass_path = pdir / "output.ass"
        ass_path.write_text(media.build_ass(timeline), encoding="utf-8")
        srt_path = pdir / "output.srt"
        srt_path.write_text(media.build_srt(shots), encoding="utf-8")
        timeline.srt_path = str(srt_path)

    # ── 口播（可选；mock 桩直接跳过合成）──
    if project.input.get("voiceover"):
        _ctx_progress(STAGE_RENDER, 80.0, "tts", "口播合成中")
        timeline.voiceover_path = _synthesize_voiceover(project, work)

    _ctx_cancel(STAGE_RENDER)
    _ctx_progress(STAGE_RENDER, 85.0, "rendering",
                  f"{timeline.width}x{timeline.height}@{timeline.fps} "
                  f"{timeline.duration_s}s")
    out = media.render(
        timeline, workdir=work, subtitle_file=ass_path, mock=cfg.mock,
        transition_duration_s=cfg.transition_duration_s,
        subtitle_font=cfg.subtitle_font,
        subtitle_font_size=cfg.subtitle_font_size,
        x264_preset=cfg.x264_preset, video_bitrate=cfg.video_bitrate)
    # 成片归位到项目根（中间件 seg_*/joined 留在 work/）；SaaS 下载语义
    # 与 edit_status 产物清单都以项目根为准
    final = pdir / "output.mp4"
    shutil.move(str(out), str(final))
    if timeline.voiceover_path and Path(timeline.voiceover_path).exists():
        vo_final = pdir / "voiceover.mp3"
        shutil.copyfile(timeline.voiceover_path, vo_final)
        timeline.voiceover_path = str(vo_final)
    _write_artifact(project.project_id, "timeline.json", timeline)
    # v1.1 成片自检帧：均匀抽 N 帧落 preview/——先给人工/Agent 过目，
    # 不再拿盲剪产物直接交付
    try:
        previews = media.extract_preview(
            str(final), pdir / "preview", count=cfg.preview_frames)
        _ctx_progress(STAGE_RENDER, 98.0, "stage_done",
                      f"{final}（自检帧 {len(previews)} 张）")
    except media.MediaError:
        logger.warning("成片自检帧抽取失败（不影响成片）", exc_info=True)
        _ctx_progress(STAGE_RENDER, 98.0, "stage_done", str(final))


def _synthesize_voiceover(project: ProjectRecord, work: Path) -> str:
    """整段口播稿 → wav（_cloud_tts 预设音色）。key 缺失显式报错。"""
    cfg = _cfg().editing
    if cfg.mock:
        return ""   # mock：不合成，渲染落静音轨
    key = _api_key(cfg.tts_key_env)
    if not key:
        raise EditingStageError(
            f"voiceover=True 但缺少 TTS key（环境变量 {cfg.tts_key_env}）",
            category="environment", stage=STAGE_RENDER)
    copy_raw = _read_artifact(project.project_id, "copy.json")
    copy_doc = CopyDoc.model_validate(copy_raw or {})
    narration = copy_doc.full_narration or copy_doc.title
    from flowmind.skills._cloud_tts import TTSError, synthesize_text
    work.mkdir(parents=True, exist_ok=True)
    out = work / "voiceover.mp3"
    try:
        return synthesize_text(
            narration, out_path=str(out), voice_id=cfg.tts_voice,
            api_key=key, target_model=cfg.tts_model)
    except TTSError as exc:
        raise EditingStageError(
            f"口播合成失败：{exc}", category=getattr(exc, "category", "unknown"),
            retriable=getattr(exc, "retriable", False),
            stage=STAGE_RENDER) from exc


def _resolve_format(project: ProjectRecord,
                    report: UnderstandReport) -> tuple[int, int, int]:
    """输出画幅：入参 aspect_ratio → 首个视频素材画幅就近映射 → config 默认。"""
    cfg = _cfg().editing
    ratio = project.input.get("aspect_ratio")
    if not ratio:
        video = next((a for a in report.assets
                      if a.kind == "video" and a.width > 0 and a.height > 0),
                     None)
        if video is not None:
            ratio = _nearest_aspect(video.width / video.height)
    ratio = ratio or cfg.default_aspect_ratio
    presets = {
        "16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080),
    }
    return presets.get(ratio, (1920, 1080)) + (30,)


def _nearest_aspect(ratio: float) -> str:
    """宽高比就近映射到支持的画幅（log 距离对横竖屏对称）。"""
    import math
    targets = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0}
    return min(targets, key=lambda k: abs(math.log(ratio / targets[k])))


# ── 产物 IO / 小工具 ──

def _cfg():
    from flowmind.config import get_config
    return get_config()


def _api_key(env_name: str) -> str | None:
    from flowmind.skills._secrets import get_api_key
    return get_api_key(env_name)


def _write_artifact(project_id: str, name: str, model) -> Path:
    pdir = project_dir(project_id)
    pdir.mkdir(parents=True, exist_ok=True)
    path = pdir / name
    if hasattr(model, "model_dump_json"):
        path.write_text(model.model_dump_json(indent=1), encoding="utf-8")
    else:
        path.write_text(json.dumps(model, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return path


def _read_artifact(project_id: str, name: str) -> dict | None:
    path = project_dir(project_id) / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


_STAGE_HANDLERS = {
    STAGE_UNDERSTAND: _stage_understand,
    STAGE_COPY: _stage_copy,
    STAGE_SCRIPT: _stage_script,
    STAGE_STORYBOARD: _stage_storyboard,
    STAGE_RENDER: _stage_render,
}


def new_project_id() -> str:
    return uuid.uuid4().hex
