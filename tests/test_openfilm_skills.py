"""OpenFilm skill-level tests: lifecycle, schemas, status, review, routing."""
from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

import flowmind.skills  # noqa: F401  触发 @skill 注册
import flowmind.skills.edit_openfilm_render as eor
import flowmind.skills.edit_openfilm_studio as eos
import flowmind.skills.edit_pipeline as epipe
import flowmind.skills.edit_project as ep
import flowmind.skills.edit_review as er
import flowmind.skills.edit_status as es
import flowmind.skills.localize_submit as lsub
from flowmind.config import EditingConfig, FlowmindConfig, LocalizerConfig, OpenFilmConfig
from flowmind.editing import openfilm_backend as of
from flowmind.editing.models import (
    PSTATE_AWAITING_AGENT,
    PSTATE_AWAITING_REVIEW,
    PSTATE_RUNNING,
    STAGE_AUTHORING,
    ProjectRecord,
)
from flowmind.skill import registry
from flowmind.skills import FAMILIES
from flowmind.skills.edit_openfilm_render import RenderInput, edit_openfilm_render
from flowmind.skills.edit_openfilm_studio import StudioInput, edit_openfilm_studio
from flowmind.skills.edit_pipeline import PipelineInput, edit_pipeline
from flowmind.skills.edit_project import EditProjectInput, edit_project
from flowmind.skills.edit_review import ReviewInput, edit_review
from flowmind.skills.edit_status import StatusInput, edit_status

PID = "b" * 32
_TEST_TMP_BASE = Path(
    os.environ.get("FLOWMIND_OPENFILM_TEST_TMP", "D:/openfilm-test-tmp"))


@pytest.fixture
def workdir() -> Path:
    _TEST_TMP_BASE.mkdir(parents=True, exist_ok=True)
    path = _TEST_TMP_BASE / uuid.uuid4().hex
    path.mkdir()
    return path


class FakeStore:
    def __init__(self) -> None:
        self.projects: dict[str, ProjectRecord] = {}

    def get_project(self, project_id: str):
        record = self.projects.get(project_id)
        return record.model_copy(deep=True) if record is not None else None

    def save_project(self, record: ProjectRecord) -> None:
        self.projects[record.project_id] = record.model_copy(deep=True)


class FakeManager:
    def __init__(self, task_status: dict | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.task_status = task_status

    def submit(self, skill_id: str, args: dict) -> str:
        self.calls.append((skill_id, args))
        return "task-" + str(len(self.calls))

    def get_task(self, task_id: str):
        if self.task_status is None:
            return None
        return {"task_id": task_id, **self.task_status}


class InlineSuccessManager(FakeManager):
    """模拟 inline 执行：submit 返回前项目已经渲染完成。"""

    def __init__(self, store: FakeStore) -> None:
        super().__init__()
        self.store = store

    def submit(self, skill_id: str, args: dict) -> str:
        self.calls.append((skill_id, args))
        output = of.project_output(PID)
        output.write_bytes(b"inline-mp4")
        project = self.store.get_project(PID)
        project.state = "succeeded"
        project.stage = "render"
        project.stages["render"] = "done"
        project.input["openfilm"]["render"] = {
            "status": "succeeded",
            "task_id": "task-inline",
            "output": str(output),
            "bytes": 10,
            "error": None,
        }
        self.store.save_project(project)
        return "task-inline"

    def cancel(self, task_id: str) -> bool:
        return True


@pytest.fixture
def env(workdir, monkeypatch):
    data_dir = workdir / "data"
    data_dir.mkdir()
    cfg = FlowmindConfig(
        localizer=LocalizerConfig(data_dir=str(data_dir)),
        editing=EditingConfig(
            backend="openfilm",
            openfilm=OpenFilmConfig(cli=str(workdir / "fake-cli.py")),
        ),
    )
    store = FakeStore()
    manager = FakeManager()
    monkeypatch.setattr("flowmind.config.load_config", lambda: cfg)
    monkeypatch.setattr("flowmind.auth.tenant_scope", lambda: (False, None))
    for module in (ep, er, epipe, lsub):
        monkeypatch.setattr(module, "load_config", lambda: cfg)
    monkeypatch.setattr(of, "load_config", lambda: cfg)
    for module in (ep, es, er, eor, epipe):
        monkeypatch.setattr(module, "get_project_store", lambda: store)
    monkeypatch.setattr(of, "get_project_store", lambda: store)
    monkeypatch.setattr(ep, "get_task_manager", lambda: manager)
    monkeypatch.setattr(er, "get_task_manager", lambda: manager)
    monkeypatch.setattr(eor, "get_task_manager", lambda: manager)
    monkeypatch.setattr(
        "flowmind.tasks.manager.get_task_manager", lambda: manager)
    monkeypatch.setattr(of, "_check_ffmpeg", lambda _env: None)
    monkeypatch.setattr(of, "_preflight_render", lambda: None)
    return cfg, store, manager, data_dir


def add_project(store: FakeStore, *, state=PSTATE_AWAITING_AGENT,
                review_stages=None, task_ids=None) -> ProjectRecord:
    project = ProjectRecord(
        project_id=PID, backend="openfilm", state=state, stage=STAGE_AUTHORING,
        stages={"authoring": "pending", "pre_render": "pending",
                "render": "pending"},
        review_stages=list(review_stages if review_stages is not None
                           else ["pre_render"]),
        input={"openfilm": {"render": {"status": "authored_pending"}}},
        task_ids=list(task_ids or []),
    )
    store.save_project(project)
    of.create_workspace(PID, theme="x", review_stages=project.review_stages)
    return project


def test_existing_projects_keep_their_creation_backend():
    from flowmind.editing.models import resolve_backend

    cybercut = ProjectRecord(project_id="c" * 32, backend="cybercut")
    legacy_cybercut = ProjectRecord(
        project_id="d" * 32, input={"cybercut": {}})
    legacy_flowmind = ProjectRecord(project_id="e" * 32)
    legacy_openfilm = ProjectRecord(
        project_id="f" * 32, input={"openfilm": {}})
    openfilm = ProjectRecord(project_id="g" * 32, backend="openfilm")
    assert resolve_backend(cybercut) == "cybercut"
    assert resolve_backend(legacy_cybercut) == "cybercut"
    # 无 backend、无桥接字段的升级前记录：保守回落旧 FlowMind，绝不因全局
    # 切到 openfilm 而误入 OpenFilm。
    assert resolve_backend(legacy_flowmind) == "flowmind"
    assert resolve_backend(legacy_openfilm) == "openfilm"
    assert resolve_backend(openfilm) == "openfilm"


def test_legacy_pipeline_routing_never_falls_into_openfilm(env, monkeypatch):
    cfg, store, manager, data_dir = env
    calls: list[str] = []

    def fake_flowmind(pid):
        calls.append(pid)
        return SimpleNamespace(state="succeeded", stage="render",
                               stages={}, review_stages=[],
                               artifact_dir=str(data_dir))

    def forbidden_openfilm(_pid):
        raise AssertionError("旧版无桥接项目不应路由到 OpenFilm")

    monkeypatch.setattr(epipe, "run_pipeline", fake_flowmind)
    monkeypatch.setattr(of, "run_render", forbidden_openfilm)
    store.save_project(ProjectRecord(
        project_id=PID, state=PSTATE_RUNNING, input={}))
    result = edit_pipeline(PipelineInput(project_id=PID))
    assert result.data.state == "succeeded"
    assert calls == [PID]


def test_legacy_openfilm_bridge_routes_to_openfilm(env, monkeypatch):
    cfg, store, manager, data_dir = env
    calls: list[str] = []
    store.save_project(ProjectRecord(
        project_id=PID, state=PSTATE_RUNNING, input={"openfilm": {}}))

    def fake_openfilm(pid):
        calls.append(pid)
        return SimpleNamespace(state="succeeded", stage="render",
                               stages={}, review_stages=[],
                               artifact_dir=str(data_dir))

    monkeypatch.setattr(of, "run_render", fake_openfilm)
    result = edit_pipeline(PipelineInput(project_id=PID))
    assert result.data.state == "succeeded"
    assert calls == [PID]


def test_legacy_cybercut_bridge_routes_to_cybercut(env, monkeypatch):
    cfg, store, manager, data_dir = env
    import flowmind.editing.cybercut_backend as cb
    calls: list[str] = []
    store.save_project(ProjectRecord(
        project_id=PID, state=PSTATE_RUNNING, input={"cybercut": {}}))

    def fake_cybercut(pid):
        calls.append(pid)
        return SimpleNamespace(state="succeeded", stage="render",
                               stages={}, review_stages=[],
                               artifact_dir=str(data_dir))

    monkeypatch.setattr(cb, "run_pipeline", fake_cybercut)
    result = edit_pipeline(PipelineInput(project_id=PID))
    assert result.data.state == "succeeded"
    assert calls == [PID]


def test_openfilm_skills_are_registered_and_project_only():
    assert "edit_openfilm_studio" in FAMILIES["edit"]
    assert "edit_openfilm_render" in FAMILIES["edit"]
    specs = registry()
    for skill_id in ("edit_openfilm_studio", "edit_openfilm_render"):
        assert skill_id in specs
        assert set(specs[skill_id].input_model.model_fields) == {"project_id"}


def test_edit_project_openfilm_creates_workspace_without_task(env):
    cfg, store, manager, data_dir = env
    result = edit_project(EditProjectInput(theme="launch", style="clean"))
    report = result.data
    assert report.backend == "openfilm"
    assert report.state == PSTATE_AWAITING_AGENT
    assert report.task_id == ""
    assert report.workspace and Path(report.workspace).is_dir()
    assert Path(report.entry_file).name == "film.html"
    assert "pre_render" in report.review_stages
    assert manager.calls == []          # 不引入内置创作 Agent
    saved = store.get_project(report.project_id)
    assert saved.backend == "openfilm"
    assert saved.state == PSTATE_AWAITING_AGENT
    assert saved.task_ids == []
    assert saved.input["openfilm"]["workspace"] == report.workspace


def test_edit_project_openfilm_copies_assets_into_workspace(env):
    cfg, store, manager, data_dir = env
    uploads = data_dir / "uploads"
    uploads.mkdir()
    source = uploads / "clip.mp4"
    source.write_bytes(b"video")
    result = edit_project(EditProjectInput(
        assets=[str(source)], theme="launch"))
    workspace = Path(result.data.workspace)
    assert (workspace / "assets" / "clip.mp4").read_bytes() == b"video"
    assert result.data.assets_accepted == 1


def test_edit_project_openfilm_rejects_cybercut_gate(env):
    with pytest.raises(ValueError, match="pre_render"):
        EditProjectInput(theme="x", review_stages=["storyboard"])


def test_default_cybercut_project_still_submits_edit_pipeline(env):
    cfg, store, manager, data_dir = env
    cfg.editing.backend = "cybercut"
    uploads = data_dir / "uploads"
    uploads.mkdir()
    source = uploads / "clip.mp4"
    source.write_bytes(b"video")
    result = edit_project(EditProjectInput(
        assets=[str(source)], theme="launch"))
    assert manager.calls == [("edit_pipeline",
                              {"project_id": result.data.project_id})]
    assert result.data.task_id
    saved = store.get_project(result.data.project_id)
    assert saved.backend == "cybercut"
    assert result.data.task_id in saved.task_ids


def test_edit_status_openfilm_reports_progress_output_and_review(env):
    cfg, store, manager, data_dir = env
    add_project(store, state=PSTATE_RUNNING, review_stages=["pre_render"],
                task_ids=["task-1"])
    project = store.get_project(PID)
    project.input["openfilm"]["render"] = {
        "status": "running", "task_id": "task-1"}
    store.save_project(project)
    output = of.project_output(PID)
    output.write_bytes(b"real-mp4")
    manager.task_status = {"status": "running", "stage": "render",
                           "progress": 42.5, "error": None}
    (of.project_workspace(PID) / "title.html").write_text(
        "<html></html>", encoding="utf-8")

    result = edit_status(StatusInput(project_ids=[PID]))
    status = result.data.projects[0]
    assert status.backend == "openfilm"
    assert status.workspace == str(of.project_workspace(PID))
    assert status.outputs["output.mp4"] == str(output)
    assert status.progress == 42.5
    assert status.render["task_status"] == "running"
    assert status.stages["render"] == "running"
    assert status.stages["authoring"] == "done"
    assert "openfilm/title.html" in status.artifacts
    assert status.next

    project = store.get_project(PID)
    project.state = PSTATE_AWAITING_REVIEW
    project.stage = "pre_render"
    project.review_feedback["pre_render"] = ["再快一点"]
    store.save_project(project)
    review = edit_status(StatusInput(project_ids=[PID])).data.projects[0].review
    assert review["gate"] == "pre_render"
    assert review["feedback_history"] == ["再快一点"]
    assert "OpenFilm" in review["summary"]

    # 排队中取消：任务终态先于项目回写时，状态查询以任务终态为准
    project = store.get_project(PID)
    project.state = PSTATE_RUNNING
    store.save_project(project)
    manager.task_status = {"status": "cancelled", "stage": "render",
                           "progress": 0.0, "error": None}
    cancelled = edit_status(StatusInput(project_ids=[PID])).data.projects[0]
    assert cancelled.state == "cancelled"
    assert cancelled.stages["render"] == "cancelled"


def test_edit_review_openfilm_approve_revise_reject(env):
    cfg, store, manager, data_dir = env
    add_project(store, state=PSTATE_AWAITING_REVIEW,
                review_stages=["pre_render"])
    project = store.get_project(PID)
    project.stage = "pre_render"
    store.save_project(project)

    approved = edit_review(ReviewInput(
        project_id=PID, stage="pre_render", action="approve"))
    assert approved.data.task_id == "task-1"
    assert manager.calls[0][0] == "edit_pipeline"
    saved = store.get_project(PID)
    assert saved.state == PSTATE_RUNNING
    assert "pre_render" in saved.approved
    assert "pre_render" in saved.task_ids or "task-1" in saved.task_ids

    # revise：不提交渲染，退回外部 Agent
    saved.state = PSTATE_AWAITING_REVIEW
    saved.stage = "pre_render"
    saved.approved = []
    store.save_project(saved)
    revised = edit_review(ReviewInput(
        project_id=PID, stage="pre_render", action="revise",
        feedback="节奏再快一点"))
    assert revised.data.task_id is None
    saved = store.get_project(PID)
    assert saved.state == PSTATE_AWAITING_AGENT
    assert saved.review_feedback["pre_render"] == ["节奏再快一点"]
    assert (of.project_workspace(PID) / of.REVIEW_MARKDOWN_FILENAME).is_file()

    # reject：终止
    saved.state = PSTATE_AWAITING_REVIEW
    saved.stage = "pre_render"
    store.save_project(saved)
    rejected = edit_review(ReviewInput(
        project_id=PID, stage="pre_render", action="reject",
        feedback="方向不对"))
    assert rejected.data.state == "failed"
    assert store.get_project(PID).state == "failed"


def test_edit_openfilm_studio_skill_success_and_failure(env, monkeypatch):
    cfg, store, manager, data_dir = env
    add_project(store)
    monkeypatch.setattr(eos, "open_studio", lambda _pid: {
        "state": PSTATE_AWAITING_AGENT,
        "studio_url": "http://127.0.0.1:4747/open?key=k",
        "workspace": str(of.project_workspace(PID)),
        "cli_duration_s": 1.2,
    })
    result = edit_openfilm_studio(StudioInput(project_id=PID))
    assert result.data.opened is True
    assert result.data.studio_url.startswith("http://127.0.0.1:4747/")

    def boom(_pid):
        raise of.OpenFilmUnavailable("未安装 OpenFilm CLI")

    monkeypatch.setattr(eos, "open_studio", boom)
    failed = edit_openfilm_studio(StudioInput(project_id=PID))
    assert failed.data.opened is False
    assert failed.degraded is True
    assert failed.degradation_reason == "environment"


def test_edit_openfilm_render_skill_gate_and_task(env):
    cfg, store, manager, data_dir = env
    add_project(store, review_stages=["pre_render"])
    # 有审批闸：不提交任务，进入 awaiting_review
    waiting = edit_openfilm_render(RenderInput(project_id=PID))
    assert waiting.data.state == PSTATE_AWAITING_REVIEW
    assert waiting.data.task_id is None
    assert manager.calls == []

    # 无审批闸：提交 edit_pipeline，记录 task_id
    project = store.get_project(PID)
    project.review_stages = []
    project.state = PSTATE_AWAITING_AGENT
    project.stage = STAGE_AUTHORING
    store.save_project(project)
    result = edit_openfilm_render(RenderInput(project_id=PID))
    assert result.data.task_id == "task-1"
    assert result.data.state == "running"
    assert result.data.render_status == "queued"
    assert result.data.output_path is None
    assert manager.calls == [("edit_pipeline", {"project_id": PID})]
    saved = store.get_project(PID)
    assert saved.input["openfilm"]["render"]["task_id"] == "task-1"
    assert "task-1" in saved.task_ids


def test_concurrent_render_requests_submit_only_one_task(env):
    cfg, store, manager, data_dir = env
    add_project(store, review_stages=[])
    barrier = threading.Barrier(6)
    results: list = []
    lock = threading.Lock()

    def attempt() -> None:
        barrier.wait()
        result = edit_openfilm_render(RenderInput(project_id=PID))
        with lock:
            results.append(result)

    threads = [threading.Thread(target=attempt) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(manager.calls) == 1
    assert sum(1 for result in results if not result.degraded) == 1
    assert sum(1 for result in results if result.degraded) == 5
    assert store.get_project(PID).input["openfilm"]["render"]["claim"]["token"]


def test_edit_openfilm_render_reports_inline_success_consistently(env):
    cfg, store, manager, data_dir = env
    add_project(store, review_stages=[])
    manager = InlineSuccessManager(store)
    eor.get_task_manager = lambda: manager

    result = edit_openfilm_render(RenderInput(project_id=PID))
    report = result.data
    assert report.state == "succeeded"
    assert report.render_status == "succeeded"
    assert report.output_path == str(of.project_output(PID))
    assert Path(report.output_path).read_bytes() == b"inline-mp4"
    saved = store.get_project(PID)
    assert saved.state == "succeeded"
    assert saved.input["openfilm"]["render"]["task_id"] == "task-inline"


def test_edit_openfilm_render_environment_error(env, monkeypatch):
    cfg, store, manager, data_dir = env
    add_project(store, review_stages=[])
    monkeypatch.setattr(
        of, "_preflight_render",
        lambda: (_ for _ in ()).throw(
            of.OpenFilmUnavailable("OpenFilm 渲染缺少可执行的 ffmpeg")))
    result = edit_openfilm_render(RenderInput(project_id=PID))
    assert result.degraded is True
    assert result.degradation_reason == "environment"
    assert result.data.task_id is None
    assert manager.calls == []
    assert store.get_project(PID).state == "failed"


def test_edit_pipeline_routes_openfilm_to_render(env, monkeypatch):
    cfg, store, manager, data_dir = env
    add_project(store, review_stages=[])
    called: list[str] = []

    def fake_render(project_id):
        called.append(project_id)
        return SimpleNamespace(
            state="succeeded", stage="render",
            stages={"render": "done"}, review_stages=[],
            artifact_dir=str(data_dir), output_path="output.mp4")

    monkeypatch.setattr(of, "run_render", fake_render)
    result = edit_pipeline(PipelineInput(project_id=PID))
    assert called == [PID]
    assert result.data.state == "succeeded"
    assert result.data.stage == "render"


def test_edit_openfilm_skills_do_not_accept_commands_or_paths():
    for model in (StudioInput, RenderInput):
        schema = model.model_json_schema()
        assert set(schema["properties"]) == {"project_id"}
        assert "path" not in schema["properties"]
        assert "command" not in schema["properties"]
