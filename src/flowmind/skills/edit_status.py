"""edit_status 技能：剪辑项目状态查询（含审批等待信息与产物清单）。

批量查询；租户过滤与任务引擎一致（启用鉴权而跨租户 → 与不存在同形，
不给枚举探针留口子）。产物清单扫描项目目录，成片未产出时 outputs 为空。
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from flowmind import auth
from flowmind.contracts import Evidence, ReasoningChain, SkillOutput
from flowmind.editing.models import PSTATE_AWAITING_REVIEW, resolve_backend
from flowmind.editing.pipeline import project_dir
from flowmind.editing.store import get_project_store
from flowmind.skill import skill

_VERSION = "0.1.0"

_ARTIFACT_NAMES = ("understand.json", "copy.json", "script.json",
                   "storyboard.json", "timeline.json", "output.srt")
_OUTPUT_NAMES = ("output.mp4", "output.srt", "voiceover.mp3")


class StatusInput(BaseModel):
    project_ids: list[str] = Field(..., min_length=1,
                                   description="项目 id 列表")


class ProjectStatus(BaseModel):
    project_id: str
    found: bool = True
    state: str = ""
    stage: str = ""
    stages: dict[str, str] = Field(default_factory=dict)
    task_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    storage: str = "file"
    # 审批等待详情（state=awaiting_review 时非空）
    review: dict | None = None
    artifacts: list[str] = Field(default_factory=list)
    outputs: dict[str, str] = Field(default_factory=dict)
    preview: list[str] = Field(default_factory=list)   # v1.1 成片自检帧
    watermark_removed: bool = False
    # ── OpenFilm 后端补充字段（其它后端为空/默认值）──
    backend: str = ""
    workspace: str | None = None
    entry_file: str | None = None
    studio_url: str | None = None
    progress: float | None = None
    render: dict | None = None
    feedback: list[str] = Field(default_factory=list)
    next: str = ""


class StatusReport(BaseModel):
    projects: list[ProjectStatus]


def _visible(project) -> bool:
    """跨租户同形不可见（与 TaskManager._visible 语义一致）。"""
    required, tenant = auth.tenant_scope()
    if not required:
        return True
    if project is None or tenant is None:
        return False
    return project.tenant_id == tenant


def _summarize_artifact(pdir: Path, name: str) -> str | None:
    """中间产物的一句话摘要（前端审批卡片直接展示用）。"""
    path = pdir / name
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 摘要失败不影响查询主体
        return f"{name}（存在但不可解析）"
    if name == "copy.json":
        return f"《{data.get('title', '')}》 hook={data.get('hook', '')}"
    if name == "script.json":
        secs = data.get("sections") or []
        return f"{len(secs)} 段，总时长 {data.get('total_duration_s', '?')}s"
    if name == "storyboard.json":
        shots = data.get("shots") or []
        notes = "；".join((s.get("note") or "")[:40] for s in shots[:3])
        return f"{len(shots)} 镜头。选材理由：{notes}"
    if name == "understand.json":
        assets = data.get("assets") or []
        return f"{len(assets)} 素材：{'、'.join(a.get('id', '?') + ':' + a.get('kind', '?') for a in assets)}"
    return path.name


def _openfilm_stages(project, snapshot: dict, render_info: dict
                     ) -> dict[str, str]:
    """OpenFilm 三阶段视图：authoring / pre_render / render。"""
    stages = {
        "authoring": (
            "done" if project.state != "awaiting_agent"
            and snapshot.get("film_exists") else "pending"),
        "pre_render": (
            "done" if ("pre_render" not in project.review_stages
                       or "pre_render" in project.approved) else "pending"),
        "render": "pending",
    }
    if project.state == "running":
        stages["render"] = "running"
    elif project.state == "succeeded":
        stages["render"] = "done"
    elif project.state == "failed":
        stages["render"] = "failed"
    elif project.state == "cancelled":
        stages["render"] = "cancelled"
    elif project.state == "awaiting_review":
        stages["pre_render"] = "pending"
    task_status = str(render_info.get("task_status") or "")
    if task_status in {"queued", "running"}:
        stages["render"] = task_status
    return stages


def _openfilm_summary(snapshot: dict) -> str:
    film = "film.html 已就绪" if snapshot.get("film_exists") \
        else "film.html 缺失"
    return (f"OpenFilm 项目：{film}；{len(snapshot.get('pages') or [])} 个页面、"
            f"{len(snapshot.get('assets') or [])} 个素材")


@skill(id="edit_status", name="剪辑项目状态查询", version=_VERSION)
def edit_status(inp: StatusInput) -> SkillOutput[StatusReport]:
    """查询项目状态：阶段进度 / 审批等待详情 / 产物与成片清单。"""
    store = get_project_store()
    results: list[ProjectStatus] = []
    for pid in inp.project_ids:
        project = store.get_project(pid)
        if project is None or not _visible(project):
            results.append(ProjectStatus(project_id=pid, found=False,
                                         state="not_found", stage="not_found"))
            continue
        pdir = project_dir(pid)
        is_openfilm = (project.backend == "openfilm"
                       or "openfilm" in project.input)
        cybercut_state = None
        openfilm_snapshot: dict | None = None
        render_info: dict | None = None
        progress: float | None = None
        project_backend = resolve_backend(project)

        if is_openfilm:
            from flowmind.editing.openfilm_backend import (
                OpenFilmError,
                artifact_list,
                next_step,
                status_snapshot,
            )
            try:
                openfilm_snapshot = status_snapshot(pid, project)
            except OpenFilmError as exc:
                openfilm_snapshot = {
                    "workspace": "", "entry_file": "",
                    "film_exists": False, "pages": [], "assets": [],
                    "output_path": None, "studio_url": None, "render": {},
                    "feedback": [], "review_stages": [],
                }
                if not project.error:
                    project.error = f"OpenFilm 状态读取失败：{exc}"
            render_info = dict(openfilm_snapshot.get("render") or {})
            task_id = render_info.get("task_id") or (
                project.task_ids[-1] if project.task_ids else None)
            if task_id:
                from flowmind.tasks.manager import get_task_manager
                try:
                    task = get_task_manager().get_task(task_id)
                except Exception as exc:  # noqa: BLE001 任务库不可用时仍返回项目状态
                    task = None
                    render_info["task_lookup_error"] = (
                        f"{type(exc).__name__}: {exc}"[:300])
                if task:
                    render_info.update({
                        "task_id": task_id,
                        "task_status": task.get("status"),
                        "task_stage": task.get("stage"),
                        "task_progress": task.get("progress"),
                        "task_error": task.get("error"),
                    })
                    if task.get("progress") is not None:
                        progress = float(task.get("progress") or 0.0)
            current_state, current_stage = project.state, project.stage
            current_stages = _openfilm_stages(
                project, openfilm_snapshot, render_info)
            task_state = str(render_info.get("task_status") or "")
            # 任务库终态可能先于项目回写（尤其排队中取消）：以终态为准展示，
            # 但项目记录仍保留 worker 最后写入的审计值。
            if task_state in {"cancelled", "failed", "succeeded"} and \
                    current_state == "running":
                current_state = task_state
                current_stages["render"] = (
                    "cancelled" if task_state == "cancelled"
                    else "done" if task_state == "succeeded" else "failed")
            current_error = project.error or render_info.get("task_error")
            review = None
            if current_state == PSTATE_AWAITING_REVIEW:
                review = {
                    "gate": current_stage,
                    "feedback_history": list(
                        project.review_feedback.get(current_stage, [])),
                    "hint": (f"经 edit_review 对闸门 {current_stage} 执行 "
                             "approve / revise（带 feedback）/ reject"),
                    "summary": (_openfilm_summary(openfilm_snapshot)
                                + "；外部 Agent 完成后由 FlowMind 渲染"),
                }
            artifacts = artifact_list(pid)
            outputs = ({"output.mp4": openfilm_snapshot["output_path"]}
                       if openfilm_snapshot.get("output_path") else {})
            project_backend = "openfilm"
        else:
            if project_backend == "cybercut":
                from flowmind.editing.cybercut_backend import live_status
                cybercut_state = live_status(pid)
            current_state, current_stage = project.state, project.stage
            current_stages, current_error = project.stages, project.error
            if cybercut_state:
                from flowmind.editing.cybercut_backend import (
                    review_stage_map,
                    stage_map,
                )
                run = cybercut_state["run"]
                statuses = cybercut_state["statuses"]
                waiting_node = next((key for key, value in statuses.items()
                                     if value == "waiting_approval"), None)
                terminal_map = {"succeeded": "succeeded", "failed": "failed",
                                "cancelled": "cancelled"}
                if run["status"] == "cancelled" and project.state == "failed":
                    terminal_map["cancelled"] = "failed"
                current_state = (PSTATE_AWAITING_REVIEW if waiting_node else
                                 terminal_map.get(run["status"], "running"))
                current_stage = (review_stage_map().get(
                    waiting_node, waiting_node) if waiting_node else "render")
                current_stages = {
                    stage: ("done" if statuses.get(node) == "succeeded"
                            else "pending")
                    for stage, node in stage_map().items()
                }
                current_error = (run.get("failure_message")
                                 or run.get("failure_code"))
            review = None
            if current_state == PSTATE_AWAITING_REVIEW:
                review = {
                    "gate": current_stage,
                    "feedback_history": project.review_feedback.get(
                        current_stage, []),
                    "hint": (f"经 edit_review 对闸门 {current_stage} 执行 "
                             "approve / revise（带 feedback）/ reject"),
                    "summary": next(iter(filter(None, (
                        _summarize_artifact(pdir, "storyboard.json"),
                        _summarize_artifact(pdir, "script.json"),
                        _summarize_artifact(pdir, "copy.json"),
                    ))), None),
                }
            artifacts = [n for n in _ARTIFACT_NAMES if (pdir / n).exists()]
            outputs = {
                n: str(pdir / n) for n in _OUTPUT_NAMES if (pdir / n).exists()}
            if cybercut_state:
                cybercut_artifacts = cybercut_state.get("artifacts", {})
                artifacts = sorted(cybercut_artifacts)
                outputs = {
                    name: cybercut_artifacts[kind]
                    for name, kind in (("output.mp4", "render.mp4"),
                                       ("cover.jpg", "render.cover.jpg"))
                    if kind in cybercut_artifacts
                }

        preview_dir = pdir / "preview"
        preview = ([str(p) for p in sorted(preview_dir.glob("preview_*.jpg"))]
                   if preview_dir.exists() else [])
        results.append(ProjectStatus(
            project_id=pid, state=current_state, stage=current_stage,
            stages=current_stages, task_ids=project.task_ids,
            error=current_error, storage=project.storage,
            review=review, artifacts=artifacts, outputs=outputs,
            preview=preview,
            watermark_removed=bool(
                (project.input.get("cybercut") or {}).get("watermark_removed")
            ),
            backend=project_backend,
            workspace=((openfilm_snapshot or {}).get("workspace") or None),
            entry_file=((openfilm_snapshot or {}).get("entry_file") or None),
            studio_url=((openfilm_snapshot or {}).get("studio_url") or None),
            progress=progress,
            render=render_info,
            feedback=list((openfilm_snapshot or {}).get("feedback") or []),
            next=(next_step(project, openfilm_snapshot, current_state)
                  if is_openfilm and openfilm_snapshot else "")))


    found = [r for r in results if r.found]
    awaiting = [r for r in found if r.state == PSTATE_AWAITING_REVIEW]
    done = [r for r in found if r.state == "succeeded"]
    return SkillOutput(
        data=StatusReport(projects=results),
        reasoning=[ReasoningChain(
            conclusion=(f"查询 {len(results)} 项：命中 {len(found)}，"
                        f"待审批 {len(awaiting)}，完成 {len(done)}"),
            evidence=[Evidence(metric="awaiting_review",
                               value=str(len(awaiting)), comparison="=="),
                      Evidence(metric="succeeded",
                               value=str(len(done)), comparison="==")],
            causal_analysis="项目状态机实时读自 ProjectStore（PG/文件），"
                            "产物清单按项目目录实际文件枚举",
            risk_note="not_found 与跨租户同形（防 id 枚举探测）",
            confidence=1.0)],
        confidence=1.0, sample_size=len(results),
    )
