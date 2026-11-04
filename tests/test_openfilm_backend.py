"""OpenFilm adapter tests: workspace containment, CLI, Studio, render, cancel."""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from flowmind.config import EditingConfig, FlowmindConfig, LocalizerConfig, OpenFilmConfig
from flowmind.editing import openfilm_backend as of
from flowmind.editing.models import (
    PSTATE_AWAITING_AGENT,
    PSTATE_AWAITING_REVIEW,
    PSTATE_RUNNING,
    STAGE_AUTHORING,
    ProjectRecord,
)
from flowmind.tasks import CancelledError, TaskContext, reset_task_context, set_task_context


class FakeStore:
    """ProjectStore test double：读写均深拷贝，暴露并发覆盖问题。"""

    def __init__(self) -> None:
        self.projects: dict[str, ProjectRecord] = {}
        self._lock = threading.Lock()

    def get_project(self, project_id: str):
        with self._lock:
            record = self.projects.get(project_id)
            return record.model_copy(deep=True) if record is not None else None

    def save_project(self, record: ProjectRecord) -> None:
        with self._lock:
            self.projects[record.project_id] = record.model_copy(deep=True)


PID = "a" * 32

#: 不用 pytest 的 tmp_path：本机用户目录含中文，pytest 清理根目录会
#: PermissionError（与 tests/cybercut/conftest.py 同一处置方式）。
_TEST_TMP_BASE = Path(
    os.environ.get("FLOWMIND_OPENFILM_TEST_TMP", "D:/openfilm-test-tmp"))


@pytest.fixture
def workdir() -> Path:
    _TEST_TMP_BASE.mkdir(parents=True, exist_ok=True)
    path = _TEST_TMP_BASE / uuid.uuid4().hex
    path.mkdir()
    return path


def make_project(*, state: str = PSTATE_AWAITING_AGENT,
                 review_stages: list[str] | None = None,
                 task_ids: list[str] | None = None) -> ProjectRecord:
    return ProjectRecord(
        project_id=PID,
        backend="openfilm",
        state=state,
        stage=STAGE_AUTHORING,
        stages={"authoring": "pending", "pre_render": "pending",
                "render": "pending"},
        review_stages=list(review_stages if review_stages is not None
                           else ["pre_render"]),
        input={"openfilm": {"render": {"status": "authored_pending"}},
               "asset_paths": []},
        task_ids=list(task_ids or []),
    )


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
    monkeypatch.setattr("flowmind.config.load_config", lambda: cfg)
    monkeypatch.setattr(of, "load_config", lambda: cfg)
    monkeypatch.setattr(of.auth, "tenant_scope", lambda: (False, None))
    monkeypatch.setattr(of, "_check_ffmpeg", lambda _env: None)
    store = FakeStore()
    monkeypatch.setattr(of, "get_project_store", lambda: store)
    return cfg, store, data_dir


def write_cli(workdir: Path, body: str) -> Path:
    script = workdir / "fake-cli.py"
    script.write_text("import sys, time\nfrom pathlib import Path\n" + body,
                      encoding="utf-8")
    return script


def use_cli(monkeypatch, script: Path) -> None:
    monkeypatch.setattr(
        of, "_resolve_command", lambda: [sys.executable, str(script)])


def test_default_backend_is_still_cybercut_and_openfilm_is_optional():
    cfg = FlowmindConfig()
    assert cfg.editing.backend == "cybercut"
    assert cfg.editing.openfilm.cli == ""
    assert cfg.editing.openfilm.default_review_stages == ["pre_render"]


def test_openfilm_env_overrides(monkeypatch):
    monkeypatch.setenv("FLOWMIND_OPENFILM_CLI", "C:/tools/openfilm.mjs")
    monkeypatch.setenv("FLOWMIND_OPENFILM_NODE", "C:/tools/node.exe")
    monkeypatch.setenv("FLOWMIND_OPENFILM_FFMPEG_BIN_DIR", "C:/ffmpeg/bin")
    monkeypatch.setenv("FLOWMIND_OPENFILM_NO_BROWSER", "0")
    cfg = OpenFilmConfig()
    assert cfg.cli == "C:/tools/openfilm.mjs"
    assert cfg.node == "C:/tools/node.exe"
    assert cfg.ffmpeg_bin_dir == "C:/ffmpeg/bin"
    assert cfg.no_browser is False


def test_workspace_is_inside_flowmind_project_dir(env):
    cfg, store, data_dir = env
    workspace = of.project_workspace(PID)
    output = of.project_output(PID)
    assert workspace == data_dir / "editing" / PID / "openfilm"
    assert output == data_dir / "editing" / PID / "output.mp4"
    assert of._inside(workspace, data_dir)
    assert of._inside(output, data_dir)


def test_project_id_traversal_is_rejected(env):
    with pytest.raises(of.OpenFilmError) as exc:
        of.project_workspace("../../outside-project")
    assert exc.value.category == "validation"
    with pytest.raises(of.OpenFilmError):
        of.project_output("C:/Windows/system32")


def test_create_workspace_scaffolds_film_and_copies_assets(env, workdir):
    cfg, store, data_dir = env
    store.save_project(make_project())
    source_dir = workdir / "uploads"
    source_dir.mkdir()
    source = source_dir / "clip 01.mp4"
    source.write_bytes(b"video")
    created = of.create_workspace(
        PID, theme="launch", style="clean", target_duration_s=20,
        aspect_ratio="16:9", review_stages=["pre_render"],
        asset_sources=[str(source)])
    workspace = Path(created["workspace"])
    assert (workspace / "film.html").is_file()
    assert (workspace / "assets" / "clip 01.mp4").read_bytes() == b"video"
    assert (workspace / "flowmind-project.json").is_file()
    assert (workspace / "AGENTS.md").read_text(encoding="utf-8").find(
        PID) >= 0
    # 再次创建不覆盖外部 Agent 已改写的 film.html
    (workspace / "film.html").write_text("<html>edited</html>",
                                         encoding="utf-8")
    of.create_workspace(PID, theme="launch", review_stages=[])
    assert (workspace / "film.html").read_text(encoding="utf-8") == \
        "<html>edited</html>"


def test_create_workspace_rejects_asset_outside_known_files(env, workdir):
    cfg, store, data_dir = env
    store.save_project(make_project())
    folder = workdir / "assets-dir"
    folder.mkdir()
    with pytest.raises(of.OpenFilmError):
        of.create_workspace(PID, theme="x", asset_sources=[str(folder)])


def test_open_studio_parses_local_url_and_records_it(env, workdir, monkeypatch):
    cfg, store, data_dir = env
    store.save_project(make_project())
    of.create_workspace(PID, theme="x", review_stages=["pre_render"])
    script = write_cli(workdir, """
if sys.argv[1] == "open":
    print("project  " + sys.argv[2])
    print("Studio   http://127.0.0.1:4747/open?key=k&to=abc")
    print("         open this address")
    raise SystemExit(0)
""")
    use_cli(monkeypatch, script)

    result = of.open_studio(PID)
    assert result["studio_url"].startswith("http://127.0.0.1:4747/")
    saved = store.get_project(PID)
    assert saved.input["openfilm"]["studio_url"] == result["studio_url"]


def test_open_studio_rejects_nonlocal_or_mismatched_response(
        env, workdir, monkeypatch):
    cfg, store, data_dir = env
    store.save_project(make_project())
    of.create_workspace(PID, theme="x", review_stages=[])
    script = write_cli(workdir, """
if sys.argv[1] == "open":
    print("project  " + sys.argv[2])
    print("Studio   http://evil.example/open?key=k")
    raise SystemExit(0)
""")
    use_cli(monkeypatch, script)
    with pytest.raises(of.OpenFilmError, match="非本机"):
        of.open_studio(PID)

    script = write_cli(workdir, """
if sys.argv[1] == "open":
    print("project  " + sys.argv[2] + "-other")
    print("Studio   http://127.0.0.1:4747/open?key=k")
    raise SystemExit(0)
""")
    use_cli(monkeypatch, script)
    with pytest.raises(of.OpenFilmError, match="不一致"):
        of.open_studio(PID)


def test_open_studio_requires_existing_film(env):
    cfg, store, data_dir = env
    store.save_project(make_project())
    with pytest.raises(of.OpenFilmError, match="film.html"):
        of.open_studio(PID)


def test_prepare_render_uses_pre_render_gate_without_starting_cli(
        env, monkeypatch):
    cfg, store, data_dir = env
    store.save_project(make_project(review_stages=["pre_render"]))
    of.create_workspace(PID, theme="x", review_stages=["pre_render"])
    monkeypatch.setattr(of, "_resolve_command",
                        lambda: (_ for _ in ()).throw(AssertionError("CLI 不应启动")))
    prepared = of.prepare_render(PID)
    assert prepared["submit"] is False
    assert prepared["state"] == PSTATE_AWAITING_REVIEW
    assert prepared["stage"] == "pre_render"
    saved = store.get_project(PID)
    assert saved.state == PSTATE_AWAITING_REVIEW
    assert saved.input["openfilm"]["render"]["status"] == "awaiting_review"


def test_prepare_render_without_gate_queues(env, monkeypatch):
    cfg, store, data_dir = env
    store.save_project(make_project(review_stages=[]))
    of.create_workspace(PID, theme="x", review_stages=[])
    monkeypatch.setattr(of, "_preflight_render", lambda: None)
    prepared = of.prepare_render(PID)
    assert prepared["submit"] is True
    assert prepared["state"] == PSTATE_RUNNING
    saved = store.get_project(PID)
    assert saved.input["openfilm"]["render"]["status"] == "claiming"
    assert prepared["claim_token"]


def test_prepare_render_is_atomic_under_concurrency(env, monkeypatch):
    cfg, store, data_dir = env
    store.save_project(make_project(review_stages=[]))
    of.create_workspace(PID, theme="x", review_stages=[])
    monkeypatch.setattr(of, "_preflight_render", lambda: None)

    barrier = threading.Barrier(6)
    successes: list[dict] = []
    errors: list[of.OpenFilmError] = []
    lock = threading.Lock()

    def attempt() -> None:
        barrier.wait()
        try:
            prepared = of.prepare_render(PID)
        except of.OpenFilmError as exc:
            with lock:
                errors.append(exc)
            return
        with lock:
            successes.append(prepared)

    threads = [threading.Thread(target=attempt) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(successes) == 1
    assert len(errors) == 5
    assert successes[0]["claim_token"]
    saved = store.get_project(PID)
    assert saved.input["openfilm"]["render"]["claim"]["token"] == \
        successes[0]["claim_token"]
    assert saved.state == PSTATE_RUNNING


def test_render_temp_paths_unique_and_inside_project(env):
    cfg, store, data_dir = env
    store.save_project(make_project())
    paths = [of._render_temp_path(PID) for _ in range(20)]
    assert len(set(paths)) == 20
    assert all(path.parent == of.project_output(PID).parent for path in paths)
    assert all(path.name.startswith(".") for path in paths)


def test_prepare_render_rejects_second_active_render(env, monkeypatch):
    cfg, store, data_dir = env
    project = make_project(state=PSTATE_RUNNING, review_stages=[])
    project.input["openfilm"]["render"] = {"status": "running",
                                          "task_id": "t" * 8}
    store.save_project(project)
    of.create_workspace(PID, theme="x", review_stages=[])
    monkeypatch.setattr(of, "_task_status", lambda _tid: "running")
    with pytest.raises(of.OpenFilmError, match="活跃渲染"):
        of.prepare_render(PID)


def test_prepare_render_recovers_after_queued_task_was_cancelled(
        env, monkeypatch):
    cfg, store, data_dir = env
    project = make_project(state=PSTATE_RUNNING, review_stages=[])
    project.input["openfilm"]["render"] = {"status": "queued",
                                          "task_id": "t" * 8}
    store.save_project(project)
    of.create_workspace(PID, theme="x", review_stages=[])
    monkeypatch.setattr(of, "_task_status", lambda _tid: "cancelled")
    monkeypatch.setattr(of, "_preflight_render", lambda: None)
    prepared = of.prepare_render(PID)
    assert prepared["submit"] is True
    assert store.get_project(PID).input["openfilm"]["render"]["status"] == \
        "claiming"


@pytest.mark.parametrize("body, expected_state, category", [
    ("""
if sys.argv[1] == "render":
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    out.write_bytes(b"fake-mp4")
    for i in range(1, 11):
        print(f"  {i}/10 frames", file=sys.stderr, flush=True)
    raise SystemExit(0)
""", "succeeded", None),
    ("""
if sys.argv[1] == "render":
    print("render boom", file=sys.stderr)
    raise SystemExit(3)
""", "failed", "render"),
    ("""
if sys.argv[1] == "render":
    raise SystemExit(0)
""", "failed", "render"),
])
def test_run_render_success_failure_and_missing_output(
        env, workdir, monkeypatch, body, expected_state, category):
    cfg, store, data_dir = env
    store.save_project(make_project(review_stages=[]))
    of.create_workspace(PID, theme="x", review_stages=[])
    script = write_cli(workdir, body)
    use_cli(monkeypatch, script)
    output = of.project_output(PID)

    if expected_state == "succeeded":
        result = of.run_render(PID)
        assert result.state == "succeeded"
        assert output.read_bytes() == b"fake-mp4"
        saved = store.get_project(PID)
        assert saved.input["openfilm"]["render"]["status"] == "succeeded"
        assert saved.input["openfilm"]["render"]["bytes"] == 8
    else:
        with pytest.raises(of.OpenFilmError) as exc:
            of.run_render(PID)
        assert exc.value.category == category
        assert store.get_project(PID).state == "failed"
        assert not output.exists()
        assert "render" in store.get_project(PID).error.lower() or \
            "未产出" in store.get_project(PID).error


def test_run_render_reports_progress_from_reader_thread(
        env, workdir, monkeypatch):
    cfg, store, data_dir = env
    store.save_project(make_project(review_stages=[]))
    of.create_workspace(PID, theme="x", review_stages=[])
    script = write_cli(workdir, """
if sys.argv[1] == "render":
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    out.write_bytes(b"fake-mp4")
    for i in range(1, 11):
        print(f"  {i}/10 frames", file=sys.stderr, flush=True)
    raise SystemExit(0)
""")
    use_cli(monkeypatch, script)
    events: list[tuple[str, float, str]] = []
    context = TaskContext(
        task_id="t" * 8,
        progress_cb=lambda stage, pct, message="": events.append(
            (stage, pct, message)))
    token = set_task_context(context)
    try:
        of.run_render(PID)
    finally:
        reset_task_context(token)
    assert any("帧" in message for _, _, message in events), events
    assert any(2.0 < pct < 100.0 for _, pct, _ in events), events


def test_run_render_refuses_to_bypass_pre_render_gate(env, monkeypatch):
    cfg, store, data_dir = env
    project = make_project(state=PSTATE_AWAITING_REVIEW,
                           review_stages=["pre_render"])
    project.stage = "pre_render"
    store.save_project(project)
    of.create_workspace(PID, theme="x", review_stages=["pre_render"])
    monkeypatch.setattr(of, "_preflight_render",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("不应进入渲染预检")))
    with pytest.raises(of.OpenFilmError, match="审批"):
        of.run_render(PID)


def _temp_outputs(project_id: str) -> list[Path]:
    return list(of.project_output(project_id).parent.glob(
        ".*.rendering-*.mp4"))


def test_failed_render_removes_temp_and_allows_retry(
        env, workdir, monkeypatch):
    cfg, store, data_dir = env
    store.save_project(make_project(review_stages=[]))
    of.create_workspace(PID, theme="x", review_stages=[])
    fail = write_cli(workdir, """
if sys.argv[1] == "render":
    print("boom", file=sys.stderr)
    raise SystemExit(2)
""")
    use_cli(monkeypatch, fail)
    with pytest.raises(of.OpenFilmError):
        of.run_render(PID)
    assert _temp_outputs(PID) == []
    failed = store.get_project(PID)
    assert failed.state == "failed"
    assert failed.input["openfilm"]["render"]["claim"] is None

    ok = write_cli(workdir, """
if sys.argv[1] == "render":
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    out.write_bytes(b"retry-ok")
    raise SystemExit(0)
""")
    use_cli(monkeypatch, ok)
    prepared = of.prepare_render(PID)
    assert prepared["submit"] is True
    assert prepared["claim_token"]
    of.run_render(PID)
    assert of.project_output(PID).read_bytes() == b"retry-ok"
    assert _temp_outputs(PID) == []
    assert store.get_project(PID).input["openfilm"]["render"]["claim"] is None


def test_run_render_timeout_marks_failed(env, workdir, monkeypatch):
    cfg, store, data_dir = env
    cfg.editing.openfilm.render_timeout_s = 0.6
    store.save_project(make_project(review_stages=[]))
    of.create_workspace(PID, theme="x", review_stages=[])
    script = write_cli(workdir, """
if sys.argv[1] == "render":
    time.sleep(30)
""")
    use_cli(monkeypatch, script)
    started = time.monotonic()
    with pytest.raises(of.OpenFilmError) as exc:
        of.run_render(PID)
    assert time.monotonic() - started < 15
    assert exc.value.category == "timeout"
    assert store.get_project(PID).state == "failed"
    assert "超时" in store.get_project(PID).error


def test_run_render_cancel_marks_cancelled_and_removes_partial(
        env, workdir, monkeypatch):
    cfg, store, data_dir = env
    store.save_project(make_project(review_stages=[]))
    of.create_workspace(PID, theme="x", review_stages=[])
    script = write_cli(workdir, """
if sys.argv[1] == "render":
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    out.write_bytes(b"partial")
    print("  1/10 frames", file=sys.stderr, flush=True)
    time.sleep(30)
""")
    use_cli(monkeypatch, script)
    cancel_at = time.monotonic() + 0.6
    context = TaskContext(task_id="t" * 8,
                          cancel_check=lambda: time.monotonic() > cancel_at)
    token = set_task_context(context)
    try:
        with pytest.raises(CancelledError):
            of.run_render(PID)
    finally:
        reset_task_context(token)
    saved = store.get_project(PID)
    assert saved.state == "cancelled"
    assert saved.input["openfilm"]["render"]["status"] == "cancelled"
    assert saved.input["openfilm"]["render"]["claim"] is None
    assert not of.project_output(PID).exists()
    assert _temp_outputs(PID) == []


def test_check_ffmpeg_reports_missing_binaries(workdir):
    with pytest.raises(of.OpenFilmUnavailable, match="FFmpeg"):
        of._check_ffmpeg({"PATH": str(workdir / "empty-bin")})


def test_run_render_missing_ffmpeg_is_environment_error(env, monkeypatch):
    cfg, store, data_dir = env
    store.save_project(make_project(review_stages=[]))
    of.create_workspace(PID, theme="x", review_stages=[])
    monkeypatch.setattr(
        of, "_preflight_render",
        lambda: (_ for _ in ()).throw(
            of.OpenFilmUnavailable(
                "OpenFilm 渲染缺少可执行的 ffmpeg；请安装 FFmpeg")))
    with pytest.raises(of.OpenFilmUnavailable, match="FFmpeg"):
        of.run_render(PID)
    assert store.get_project(PID).state == "failed"


def test_resolve_command_missing_cli_is_explicit(env):
    cfg, store, data_dir = env
    cfg.editing.openfilm.cli = str(data_dir / "missing-openfilm.mjs")
    with pytest.raises(of.OpenFilmUnavailable, match="找不到 OpenFilm CLI"):
        of._resolve_command()
    report = of.check_ready()
    assert report["available"] is False
    assert report["category"] == "environment"


def test_resolve_command_missing_node_is_explicit(env, workdir):
    cfg, store, data_dir = env
    cli = workdir / "openfilm.mjs"
    cli.write_text("", encoding="utf-8")
    cfg.editing.openfilm.cli = str(cli)
    cfg.editing.openfilm.node = str(workdir / "missing-node.exe")
    with pytest.raises(of.OpenFilmUnavailable, match="Node"):
        of._resolve_command()


def test_mjs_cli_and_native_cli_still_resolve(env, workdir):
    cfg, store, data_dir = env
    node = shutil.which("node")
    mjs = workdir / "openfilm.mjs"
    mjs.write_text("", encoding="utf-8")
    cfg.editing.openfilm.cli = str(mjs)
    if node:
        cfg.editing.openfilm.node = node
        assert of._resolve_command() == [node, str(mjs.resolve())]
    else:
        # 无 Node 时行为仍是显式环境错误，而不是静默改道
        with pytest.raises(of.OpenFilmUnavailable, match="Node"):
            of._resolve_command()

    native = workdir / "openfilm-native.bin"
    native.write_bytes(b"")
    cfg.editing.openfilm.cli = str(native)
    assert of._resolve_command() == [str(native.resolve())]


def _write_npm_cmd_shim(root: Path, js_name: str = "fake-openfilm.mjs") -> Path:
    pkg = root / "node_modules" / "openfilm" / "bin"
    pkg.mkdir(parents=True, exist_ok=True)
    mjs = pkg / js_name
    mjs.write_text("""
const args = process.argv.slice(2);
console.log("project  " + args[1]);
console.log("Studio   http://127.0.0.1:4747/open?key=test");
""", encoding="utf-8")
    shim = root / "openfilm.cmd"
    shim.write_text(
        "@ECHO off\r\n"
        "SET dp0=%~dp0\r\n"
        f"\"%dp0%\\node.exe\" \"%dp0%\\node_modules\\openfilm\\bin\\{js_name}\" %*\r\n",
        encoding="utf-8")
    return shim


@pytest.mark.skipif(os.name != "nt", reason="Windows npm .cmd shim")
def test_windows_npm_cmd_shim_resolves_to_node_mjs(env, workdir, monkeypatch):
    cfg, store, data_dir = env
    node = shutil.which("node")
    if not node:
        pytest.skip("本机无 node，无法验证 .cmd -> node 启动")
    shim = _write_npm_cmd_shim(workdir)
    mjs = (workdir / "node_modules" / "openfilm" / "bin"
           / "fake-openfilm.mjs").resolve()
    cfg.editing.openfilm.cli = str(shim)
    cfg.editing.openfilm.node = node
    assert of._resolve_command() == [node, str(mjs)]

    # PATH 自动发现同样返回 .cmd：也必须解析成 node + mjs
    cfg.editing.openfilm.cli = ""
    monkeypatch.setattr(
        of.shutil, "which",
        lambda name, path=None: str(shim) if name == "openfilm" else None)
    assert of._resolve_command() == [node, str(mjs)]


@pytest.mark.skipif(os.name != "nt", reason="Windows npm .cmd shim")
def test_windows_npm_cmd_shim_real_node_launch(env, workdir):
    cfg, store, data_dir = env
    node = shutil.which("node")
    if not node:
        pytest.skip("本机无 node，无法验证 .cmd 真实启动")
    store.save_project(make_project())
    of.create_workspace(PID, theme="x", review_stages=[])
    shim = _write_npm_cmd_shim(workdir)
    cfg.editing.openfilm.cli = str(shim)
    cfg.editing.openfilm.node = node
    result = of.open_studio(PID)
    assert result["studio_url"] == "http://127.0.0.1:4747/open?key=test"
    assert store.get_project(PID).input["openfilm"]["studio_url"] == \
        result["studio_url"]


def test_apply_review_approve_revise_reject(env):
    cfg, store, data_dir = env
    project = make_project(state=PSTATE_AWAITING_REVIEW)
    project.stage = "pre_render"
    store.save_project(project)
    of.create_workspace(PID, theme="x", review_stages=["pre_render"])

    approved = of.apply_review(PID, "pre_render", "approve", "")
    assert approved["submit_render"] is True
    saved = store.get_project(PID)
    assert saved.state == "running"
    assert "pre_render" in saved.approved

    # 回到等待态再 revise
    saved.state = PSTATE_AWAITING_REVIEW
    saved.stage = "pre_render"
    saved.approved = []
    store.save_project(saved)
    revised = of.apply_review(PID, "pre_render", "revise", "节奏再快一点")
    assert revised["submit_render"] is False
    saved = store.get_project(PID)
    assert saved.state == PSTATE_AWAITING_AGENT
    assert saved.review_feedback["pre_render"] == ["节奏再快一点"]
    review_file = of.project_workspace(PID) / of.REVIEW_MARKDOWN_FILENAME
    assert "节奏再快一点" in review_file.read_text(encoding="utf-8")

    saved.state = PSTATE_AWAITING_REVIEW
    saved.stage = "pre_render"
    store.save_project(saved)
    rejected = of.apply_review(PID, "pre_render", "reject", "方向不对")
    assert rejected["submit_render"] is False
    assert store.get_project(PID).state == "failed"


def test_status_snapshot_and_artifacts_stay_in_workspace(env):
    cfg, store, data_dir = env
    store.save_project(make_project())
    of.create_workspace(PID, theme="x", review_stages=["pre_render"],
                        asset_sources=[])
    workspace = of.project_workspace(PID)
    (workspace / "title.html").write_text("<html></html>", encoding="utf-8")
    (workspace / "assets" / "a.mp4").write_bytes(b"x")
    output = of.project_output(PID)
    output.write_bytes(b"mp4")
    project = store.get_project(PID)
    project.review_feedback["pre_render"] = ["ok"]
    store.save_project(project)

    snapshot = of.status_snapshot(PID, store.get_project(PID))
    assert snapshot["film_exists"] is True
    assert snapshot["pages"] == ["title.html"]
    assert snapshot["assets"] == ["assets/a.mp4"]
    assert snapshot["output_path"] == str(output)
    assert snapshot["feedback"] == ["ok"]
    artifacts = of.artifact_list(PID)
    assert "openfilm/film.html" in artifacts
    assert "openfilm/title.html" in artifacts
    assert "openfilm/assets/a.mp4" in artifacts
    assert all(not item.startswith("..") for item in artifacts)
