"""OpenFilm 后端适配器：FlowMind 管理项目工作区、Studio 与渲染。

产品边界（不可悄悄改变）：

- **FlowMind 是主产品**：project_id、项目状态、成片路径仍归 FlowMind 的
  ProjectStore / edit_status 管理；OpenFilm 不是平行产品。
- **外部编程 Agent 创作内容**：FlowMind 只创建 ``data_dir/editing/<id>/
  openfilm/`` 工作区并写入空白 ``film.html`` + 创作说明，**不内置编程
  Agent，也不替用户生成影片内容**。
- **只按 project_id 操作**：本模块所有对外入口都不接受命令、URL 或任意
  文件路径；CLI 参数由本模块按固定白名单构造，子进程用参数数组 +
  ``shell=False``。
- **项目文件与素材限制在工作区**：工作区固定为
  ``<project_dir>/openfilm``，素材复制到其 ``assets/``；成片固定为
  ``<project_dir>/output.mp4``，渲染输出不采用调用方路径。
- **OpenFilm 可选**：Node/FFmpeg/OpenFilm CLI 缺失时显式返回环境错误，
  绝不静默回落到 CyberCut/旧流水线。

OpenFilm CLI 事实来源（2026-11 阅读 openfilm 0.2.7 源码确认）：

- 项目 = 含 ``film.html`` 的文件夹；``openfilm open <folder>`` 会创建缺失
  的文件夹/空白 film.html 与 ``.film/``，打印 ``project  <path>`` 和
  ``Studio   <url>``，退出码 0/1/2。
- ``openfilm render`` 在项目目录内运行，``--out <file>`` 指定输出；默认
  输出到 ``<project>/<folder>.mp4``，需要 PATH 上的 ffmpeg/ffprobe。
- 首次运行会下载 headless Chromium（约 100 MB）；``OPENFILM_NO_BROWSER=1``
  只阻止自动拉起系统浏览器，Studio 服务照常启动。
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import urlsplit

from flowmind import auth
from flowmind.config import load_config
from flowmind.editing.models import (
    PSTATE_AWAITING_AGENT,
    PSTATE_AWAITING_REVIEW,
    PSTATE_CANCELLED,
    PSTATE_FAILED,
    PSTATE_RUNNING,
    PSTATE_SUCCEEDED,
    STAGE_AUTHORING,
    STAGE_RENDER,
    ProjectRecord,
)
from flowmind.editing.pipeline import project_dir
from flowmind.editing.store import get_project_store
from flowmind.tasks import CancelledError, current_task_context

FILM_FILE = "film.html"
WORKSPACE_DIRNAME = "openfilm"
OUTPUT_FILENAME = "output.mp4"
METADATA_FILENAME = "flowmind-project.json"
AGENT_FILENAME = "AGENTS.md"
REVIEW_FILENAME = "FLOWMIND_REVIEW.json"
REVIEW_MARKDOWN_FILENAME = "FLOWMIND_REVIEW.md"
REVIEW_STAGE = "pre_render"

_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_FRAME_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s+frames", re.IGNORECASE)
_PROJECT_LINE_RE = re.compile(r"^\s*project\s{2,}(.+?)\s*$")
_STUDIO_LINE_RE = re.compile(r"^\s*Studio\s{2,}(\S+)\s*$")
_LOCAL_STUDIO_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_TERMINAL_TASK_STATUSES = frozenset(
    {"succeeded", "failed", "cancelled", "interrupted"})
_ACTIVE_RENDER_STATUSES = frozenset({"claiming", "queued", "running"})
_RENDER_LOCK_NAME = ".openfilm-render.lock"
_RENDER_LOCK_STALE_S = 30.0
_RENDER_CLAIM_STALE_S = 120.0
OPENFILM_EXTRA_HINT = (
    "OpenFilm 项目格式/CLI 手册：https://github.com/openfilm/openfilm/"
    "blob/main/packages/openfilm/MANUAL.md")

_EMPTY_FILM_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=1920, height=1080">
<title>FlowMind OpenFilm Project</title>
</head>
<body>
</body>
</html>
"""


class OpenFilmError(RuntimeError):
    """OpenFilm 后端失败的显式错误（category 进 SkillOutput 降级原因）。"""

    def __init__(self, message: str, category: str = "unknown",
                 retriable: bool = False) -> None:
        super().__init__(message)
        self.category = category
        self.retriable = retriable


class OpenFilmUnavailable(OpenFilmError):
    """Node / OpenFilm CLI / FFmpeg 等运行依赖缺失或不可执行。"""

    def __init__(self, message: str) -> None:
        super().__init__(message, category="environment", retriable=False)


@dataclass
class CliResult:
    """一次 OpenFilm CLI 调用的结果（不含任何调用方输入的命令）。"""
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    cancelled: bool = False
    duration_s: float = 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cfg():
    return load_config().editing.openfilm


def _data_root() -> Path:
    return Path(load_config().localizer.data_dir).expanduser().resolve()


def _norm(path: Path | str) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _inside(path: Path, root: Path) -> bool:
    """路径是否在 root 内（Windows 大小写不敏感；两边都先 resolve）。"""
    path_s, root_s = _norm(path), _norm(root)
    return path_s == root_s or path_s.startswith(root_s + os.sep)


def _assert_inside(path: Path, root: Path, *, what: str) -> None:
    if not _inside(path, root):
        raise OpenFilmError(
            f"{what}越出 FlowMind 项目工作区：{path}", category="validation")


def _assert_project_id(project_id: str) -> str:
    if not isinstance(project_id, str) or not _PROJECT_ID_RE.fullmatch(project_id):
        raise OpenFilmError(
            f"非法项目 id（只允许字母/数字/下划线/连字符）：{project_id!r}",
            category="validation")
    return project_id


def project_workspace(project_id: str) -> Path:
    """项目 OpenFilm 工作区：``<data_dir>/editing/<id>/openfilm``。"""
    _assert_project_id(project_id)
    pdir = project_dir(project_id).resolve()
    _assert_inside(pdir, _data_root(), what="项目目录")
    workspace = (pdir / WORKSPACE_DIRNAME).resolve()
    _assert_inside(workspace, pdir, what="OpenFilm 工作区")
    return workspace


def project_output(project_id: str) -> Path:
    """FlowMind 统一成片路径：``<project_dir>/output.mp4``。"""
    _assert_project_id(project_id)
    pdir = project_dir(project_id).resolve()
    _assert_inside(pdir, _data_root(), what="项目目录")
    output = (pdir / OUTPUT_FILENAME).resolve()
    _assert_inside(output, pdir, what="成片路径")
    return output


def is_openfilm_project(project: ProjectRecord | None) -> bool:
    """项目是否由 OpenFilm 后端创建（新记录看 backend，旧记录看桥接字段）。"""
    if project is None:
        return False
    return project.backend == "openfilm" or "openfilm" in project.input


def _load_project(project_id: str, *, require_openfilm: bool = True
                  ) -> ProjectRecord:
    _assert_project_id(project_id)
    project = get_project_store().get_project(project_id)
    required, tenant = auth.tenant_scope()
    if required and (project is None or tenant is None
                     or project.tenant_id != tenant):
        raise OpenFilmError(f"项目不存在：{project_id}", category="not_found")
    if project is None:
        raise OpenFilmError(f"项目不存在：{project_id}", category="not_found")
    if require_openfilm and not is_openfilm_project(project):
        raise OpenFilmError(
            f"项目 {project_id} 不是 OpenFilm 后端项目（当前 backend="
            f"{project.backend or '未知'}）；请用 editing.backend=openfilm 创建",
            category="validation")
    return project


def _meta(project: ProjectRecord) -> dict[str, Any]:
    meta = project.input.get("openfilm")
    if not isinstance(meta, dict):
        meta = {}
        project.input["openfilm"] = meta
    return meta


def _save(project: ProjectRecord) -> None:
    get_project_store().save_project(project)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _tail(text: str, limit: int = 3000) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return "…" + cleaned[-limit:]


def _safe_asset_name(name: str, used: set[str]) -> str:
    """素材落盘名：保留可读文件名，剔除路径分隔符/Windows 非法字符。"""
    stem = Path(name).stem.strip(" .") or "asset"
    suffix = Path(name).suffix.lower()
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .") or "asset"
    suffix = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", suffix)
    candidate = f"{stem}{suffix}"
    index = 1
    while candidate.casefold() in used:
        candidate = f"{stem}-{index}{suffix}"
        index += 1
    used.add(candidate.casefold())
    return candidate


def _agent_doc(project_id: str, meta: dict[str, Any]) -> str:
    """工作区创作说明（外部编程 Agent 的入口信息，非 FlowMind 运行时）。"""
    assets = meta.get("assets") or []
    asset_lines = "\n".join(f"- `{item['relative']}`" for item in assets) \
        or "- （无；可自行在 `assets/` 放置素材）"
    theme = meta.get("theme") or ""
    style = meta.get("style") or "未指定"
    duration = meta.get("target_duration_s") or "未指定"
    aspect = meta.get("aspect_ratio") or "由 film.html viewport 决定"
    extra = (meta.get("extra_brief") or "").strip()
    extra_block = f"\n## 额外要求\n\n{extra}\n" if extra else ""
    return f"""# OpenFilm 项目（由 FlowMind 管理）

- FlowMind project_id：`{project_id}`
- 入口文件：`film.html`（已生成空白片，直接改写/扩展）
- 素材目录：`assets/`
- 主题：{theme}
- 风格：{style}
- 目标时长：{duration} 秒
- 画幅：{aspect}
{extra_block}
## 外部 Agent 要做什么

只在这个工作区内创作 OpenFilm 项目内容：改写 `film.html`，按 OpenFilm
格式新增页面（`*.html`）与素材（`assets/`）。不要修改工作区之外的
文件，也不要删除/覆盖 `flowmind-project.json` 与 FlowMind 写入的
`FLOWMIND_REVIEW.json`（人工审批反馈）。

已有素材：
{asset_lines}

OpenFilm 项目格式（官方 MANUAL）：
https://github.com/openfilm/openfilm/blob/main/packages/openfilm/MANUAL.md

## 完成后

1. （可选）调用 FlowMind `edit_openfilm_studio`，用返回的 `studio_url`
   在外部浏览器打开 Studio 检查/手动编辑。
2. 调用 FlowMind `edit_openfilm_render`（只传 project_id）。若配置了
   `pre_render` 审批闸，会先进入 `awaiting_review`，由 `edit_review`
   做 approve / revise / reject。
3. 轮询 `edit_status` 查看渲染进度与最终 `output.mp4`。

> FlowMind 不内置编程 Agent 运行时：影片内容必须由外部编程 Agent 写入
> 本工作区。FlowMind 只负责工作区、Studio、渲染、状态与成片。
"""


def create_workspace(
    project_id: str,
    *,
    theme: str,
    style: str | None = None,
    target_duration_s: float | None = None,
    aspect_ratio: str | None = None,
    extra_brief: str | None = None,
    review_stages: list[str] | None = None,
    asset_sources: list[str] | None = None,
) -> dict[str, Any]:
    """创建 OpenFilm 项目工作区并复制初始素材（幂等，不覆盖已有 film.html）。

    返回 ``workspace / entry_file / assets``。所有写入都在 workspace 内；
    素材只从 FlowMind 已校验的本地路径复制，不跟随调用方路径参数。
    """
    _assert_project_id(project_id)
    workspace = project_workspace(project_id)
    workspace.mkdir(parents=True, exist_ok=True)
    assets_dir = workspace / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    _assert_inside(assets_dir.resolve(), workspace, what="素材目录")

    entry = workspace / FILM_FILE
    _assert_inside(entry.resolve(), workspace, what="入口文件")
    if not entry.exists():
        entry.write_text(_EMPTY_FILM_HTML, encoding="utf-8")

    used: set[str] = {
        p.name.casefold() for p in assets_dir.iterdir() if p.is_file()}
    copies: list[dict[str, Any]] = []
    for raw in asset_sources or []:
        try:
            source = Path(raw).resolve(strict=True)
        except OSError as exc:
            raise OpenFilmError(
                f"素材不存在或不可读：{Path(raw).name}",
                category="validation") from exc
        if not source.is_file():
            raise OpenFilmError(
                f"素材不是文件：{source.name}", category="validation")
        name = _safe_asset_name(source.name, used)
        destination = (assets_dir / name).resolve()
        _assert_inside(destination, assets_dir, what="素材落盘路径")
        try:
            shutil.copy2(source, destination)
        except OSError as exc:
            raise OpenFilmError(
                f"素材复制失败：{source.name}: {exc}",
                category="environment") from exc
        copies.append({
            "name": name,
            "relative": f"assets/{name}",
            "source_name": source.name,
            "bytes": destination.stat().st_size,
        })

    meta: dict[str, Any] = {
        "workspace": str(workspace),
        "entry_file": FILM_FILE,
        "theme": theme,
        "style": style,
        "target_duration_s": target_duration_s,
        "aspect_ratio": aspect_ratio,
        "extra_brief": extra_brief,
        "review_stages": list(review_stages or []),
        "assets": copies,
        "created_at": _now(),
        "studio_url": None,
        "render": {"status": "authored_pending"},
    }
    _atomic_write(workspace / METADATA_FILENAME,
                  json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(workspace / AGENT_FILENAME, _agent_doc(project_id, meta))
    return {"workspace": str(workspace), "entry_file": str(entry),
            "assets": copies, "created_at": meta["created_at"]}


# ── CLI 解析与执行（参数数组 + shell=False）──

def _check_node(node: str) -> None:
    try:
        result = subprocess.run(
            [node, "--version"], shell=False, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OpenFilmUnavailable(f"无法执行 Node（{node}）：{exc}") from exc
    version = (result.stdout or result.stderr or "").strip()
    match = re.search(r"v?(\d+)\.(\d+)", version)
    if result.returncode != 0 or not match:
        raise OpenFilmUnavailable(
            f"Node 不可用（{node}，输出：{_tail(version, 200)}）；"
            "OpenFilm 需要 Node >= 22.2")
    if (int(match.group(1)), int(match.group(2))) < (22, 2):
        raise OpenFilmUnavailable(
            f"Node 版本过低（{version}）；OpenFilm 需要 Node >= 22.2")


def _node_binary(cfg, cli_path: str) -> str:
    """为 JS 入口解析 Node：显式配置 → shim 旁的 node.exe → PATH。"""
    node = (cfg.node or "").strip()
    if not node:
        beside = Path(cli_path).parent / "node.exe"
        node = str(beside) if beside.is_file() else (shutil.which("node") or "")
    if not node:
        raise OpenFilmUnavailable(
            "OpenFilm CLI 是 Node 脚本，但找不到 node；请安装 Node >= 22.2 "
            "或配置 FLOWMIND_OPENFILM_NODE")
    _check_node(node)
    return node


def _js_entries_in_shim(text: str) -> list[str]:
    """从 Windows npm/cmd-shim 文本里找 JS 入口（只接受 .mjs/.js）。"""
    patterns = (
        r'"([^"\r\n]+\.(?:mjs|js))"',
        r"'([^'\r\n]+\.(?:mjs|js))'",
        r'(?<![="\'])([^\s"\'<>|]+\.(?:mjs|js))(?=[\s"\'<>|]|$)',
    )
    found: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            candidate = match.group(1).strip()
            if candidate and candidate not in found:
                found.append(candidate)
    return found


def _resolve_windows_shim(cli_path: str, cfg) -> list[str]:
    """Windows npm 的 ``openfilm.cmd`` 不是原生程序：解析其 JS 入口再走 Node。

    ``cmd-shim`` 生成的 shim 形如
    ``"%dp0%\\node_modules\\openfilm\\bin\\openfilm.mjs" %*``。这里只做静态
    解析 + ``node <mjs>`` 启动，不使用 ``cmd.exe``，保持参数数组与
    ``shell=False`` 的安全边界。
    """
    shim = Path(cli_path)
    try:
        text = shim.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise OpenFilmUnavailable(
            f"无法读取 Windows OpenFilm shim（{shim}）：{exc}") from exc
    entries = _js_entries_in_shim(text)
    if not entries:
        raise OpenFilmUnavailable(
            f"Windows OpenFilm shim 未包含可解析的 .mjs/.js 入口：{shim}；"
            "请把 FLOWMIND_OPENFILM_CLI 直接指向 openfilm 包的 "
            "bin/openfilm.mjs")
    base = str(shim.parent)
    resolved: list[str] = []
    for entry in entries:
        # npm shim 的 %dp0% / %~dp0 是 shim 所在目录
        cleaned = re.sub(r"^(?:%dp0%|%~dp0)[\\/]?", "", entry,
                         flags=re.IGNORECASE)
        candidate = Path(cleaned).expanduser()
        if not candidate.is_absolute():
            candidate = Path(base) / candidate
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        if candidate.is_file():
            resolved.append(str(candidate))
    if not resolved:
        raise OpenFilmUnavailable(
            f"Windows OpenFilm shim 指向的 JS 入口不存在：{shim}；"
            "请把 FLOWMIND_OPENFILM_CLI 直接指向 openfilm 包的 "
            "bin/openfilm.mjs")
    # 优先 openfilm 包自身的入口（shim 里可能还有其它辅助脚本）
    preferred = next(
        (path for path in resolved
         if "openfilm" in Path(path).name.lower()
         or "openfilm" in Path(path).parts), resolved[0])
    node = _node_binary(cfg, preferred)
    return [node, preferred]


def _resolve_command() -> list[str]:
    """解析 OpenFilm CLI 启动命令；只读运维配置/PATH，不接受调用方输入。"""
    cfg = _cfg()
    configured = (cfg.cli or "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            cli_path = str(candidate.resolve())
        else:
            resolved = shutil.which(configured)
            if not resolved:
                raise OpenFilmUnavailable(
                    f"找不到 OpenFilm CLI：{configured}；请安装 OpenFilm 并配置 "
                    "FLOWMIND_OPENFILM_CLI（或把 openfilm 加入 PATH）")
            cli_path = resolved
    else:
        cli_path = shutil.which("openfilm") or ""
        if not cli_path:
            raise OpenFilmUnavailable(
                "未安装 OpenFilm CLI（PATH 中无 openfilm）；OpenFilm 是可选能力，"
                "安装 `npm i -g openfilm` 或配置 FLOWMIND_OPENFILM_CLI 后再试")

    suffix = Path(cli_path).suffix.lower()
    if suffix in {".cmd", ".bat"}:
        if os.name != "nt":
            raise OpenFilmUnavailable(
                f"Windows 批处理 shim 只能在 Windows 启动：{cli_path}；"
                "请改配 .mjs 入口")
        return _resolve_windows_shim(cli_path, cfg)
    if suffix in {".mjs", ".js"}:
        return [_node_binary(cfg, cli_path), cli_path]
    return [cli_path]


def _subprocess_env() -> dict[str, str]:
    """子进程环境：仅为本适配器注入 PATH/NO_BROWSER，不改进程全局环境。"""
    cfg = _cfg()
    env = os.environ.copy()
    bin_dir = (cfg.ffmpeg_bin_dir or "").strip() or \
        os.environ.get("FLOWMIND_MEDIA_BIN_DIR", "").strip()
    if bin_dir:
        expanded = str(Path(bin_dir).expanduser())
        env["PATH"] = expanded + os.pathsep + env.get("PATH", "")
    if cfg.no_browser:
        env["OPENFILM_NO_BROWSER"] = "1"
    return env


def _binary_works(path: str) -> bool:
    try:
        result = subprocess.run(
            [path, "-version"], shell=False, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _check_ffmpeg(env: dict[str, str]) -> None:
    missing: list[str] = []
    broken: list[str] = []
    for name in ("ffmpeg", "ffprobe"):
        path = shutil.which(name, path=env.get("PATH"))
        if not path:
            missing.append(name)
        elif not _binary_works(path):
            broken.append(f"{name}（{path}）")
    if missing or broken:
        detail = "、".join(missing + broken)
        raise OpenFilmUnavailable(
            f"OpenFilm 渲染缺少可执行的 {detail}；请安装 FFmpeg/FFprobe，"
            "或配置 FLOWMIND_OPENFILM_FFMPEG_BIN_DIR / FLOWMIND_MEDIA_BIN_DIR "
            "指向其 bin 目录")


def _preflight_render() -> None:
    _resolve_command()
    _check_ffmpeg(_subprocess_env())


def _terminate_tree(proc: subprocess.Popen) -> None:
    """终止整棵子进程树（Windows 用 taskkill /T，POSIX 用进程组）。"""
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                shell=False, capture_output=True, timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (OSError, subprocess.SubprocessError):
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "nt":
                proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _run_cli(
    args: list[str],
    *,
    cwd: Path,
    timeout: float,
    on_line: Callable[[str, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    command: list[str] | None = None,
) -> CliResult:
    """运行 OpenFilm CLI：参数数组、shell=False、可取消、带超时。"""
    cmd = list(command or _resolve_command()) + list(args)
    env = _subprocess_env()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    lock = threading.Lock()

    def _reader(stream, sink: list[str], name: str) -> None:
        try:
            for line in stream:
                with lock:
                    sink.append(line)
                    if len(sink) > 4000:
                        del sink[:1000]
                if on_line is not None:
                    try:
                        on_line(name, line.rstrip("\r\n"))
                    except Exception:  # noqa: BLE001 进度回调失败不影响渲染
                        pass
        finally:
            try:
                stream.close()
            except OSError:
                pass

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env, shell=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=creationflags,
            start_new_session=(os.name != "nt"),
        )
    except OSError as exc:
        raise OpenFilmUnavailable(
            f"启动 OpenFilm CLI 失败（{cmd[0]}）：{exc}") from exc

    threads = [
        threading.Thread(target=_reader,
                         args=(proc.stdout, stdout_lines, "stdout"), daemon=True),
        threading.Thread(target=_reader,
                         args=(proc.stderr, stderr_lines, "stderr"), daemon=True),
    ]
    for thread in threads:
        thread.start()

    started = time.monotonic()
    timed_out = cancelled = False
    try:
        while proc.poll() is None:
            if cancel_check is not None and cancel_check():
                cancelled = True
                _terminate_tree(proc)
                break
            if time.monotonic() - started > timeout:
                timed_out = True
                _terminate_tree(proc)
                break
            time.sleep(0.2)
        if not timed_out and not cancelled:
            try:
                proc.wait(timeout=max(1.0, timeout))
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_tree(proc)
    finally:
        for thread in threads:
            thread.join(timeout=2)
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):
                pass
    with lock:
        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
    return CliResult(
        returncode=int(proc.returncode if proc.returncode is not None else -1),
        stdout=stdout, stderr=stderr, timed_out=timed_out,
        cancelled=cancelled, duration_s=round(time.monotonic() - started, 3),
    )


# ── Studio ──

def _same_path(left: str, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return _norm(left) == _norm(right)


def _local_studio_url(raw: str) -> str:
    parsed = urlsplit(raw)
    if parsed.scheme not in ("http", "https"):
        raise OpenFilmError(
            f"OpenFilm CLI 返回的 Studio 地址不是 HTTP(S)：{raw}",
            category="environment")
    try:
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError as exc:
        raise OpenFilmError(
            f"OpenFilm CLI 返回的 Studio 地址无效：{raw}", category="environment"
        ) from exc
    if host not in _LOCAL_STUDIO_HOSTS or port is None:
        raise OpenFilmError(
            f"OpenFilm CLI 返回非本机 Studio 地址，拒绝写入状态：{raw}",
            category="environment")
    return raw


def _parse_open_output(text: str) -> tuple[str, str]:
    project_path = ""
    studio_url = ""
    for line in text.splitlines():
        project_match = _PROJECT_LINE_RE.match(line)
        if project_match:
            project_path = project_match.group(1).strip()
        studio_match = _STUDIO_LINE_RE.match(line)
        if studio_match:
            studio_url = studio_match.group(1).strip()
    if not project_path or project_path.lower().startswith("none yet"):
        raise OpenFilmError(
            f"OpenFilm 未返回项目路径：{_tail(text, 800)}", category="environment")
    if not studio_url:
        raise OpenFilmError(
            f"OpenFilm 未返回 Studio 地址：{_tail(text, 800)}", category="environment")
    return project_path, _local_studio_url(studio_url)


def open_studio(project_id: str) -> dict[str, Any]:
    """启动/复用 Studio 并打开项目；只按 project_id，返回本机 Studio 地址。"""
    project = _load_project(project_id)
    workspace = project_workspace(project_id)
    entry = workspace / FILM_FILE
    if not entry.is_file():
        raise OpenFilmError(
            f"工作区缺少 {FILM_FILE}：{workspace}；请先由外部 Agent 写入项目",
            category="validation")
    result = _run_cli(
        ["open", str(workspace)], cwd=workspace,
        timeout=float(_cfg().open_timeout_s))
    if result.cancelled:
        raise OpenFilmError("Studio 打开被取消", category="cancelled")
    if result.timed_out:
        raise OpenFilmError(
            f"OpenFilm Studio 打开超时（{_cfg().open_timeout_s:.0f}s）",
            category="timeout", retriable=True)
    if result.returncode != 0:
        category = "validation" if result.returncode == 2 else "environment"
        raise OpenFilmError(
            f"OpenFilm Studio 打开失败（退出码 {result.returncode}）："
            f"{_tail(result.stderr or result.stdout)}", category=category)
    reported_path, studio_url = _parse_open_output(
        f"{result.stdout}\n{result.stderr}")
    if not _same_path(reported_path, workspace):
        raise OpenFilmError(
            f"OpenFilm 打开的项目与 FlowMind 工作区不一致：{reported_path}",
            category="environment")

    project = get_project_store().get_project(project_id) or project
    meta = _meta(project)
    meta["studio_url"] = studio_url
    meta["studio_project_path"] = reported_path
    meta["studio_opened_at"] = _now()
    _save(project)
    return {
        "project_id": project_id,
        "state": project.state,
        "workspace": str(workspace),
        "studio_url": studio_url,
        "cli_duration_s": result.duration_s,
    }


# ── 渲染 ──

def _progress(stage: str, pct: float, event: str, message: str,
              *, context=None) -> None:
    # reader 线程读 stderr 时没有 contextvar；显式传任务上下文，避免进度丢失。
    task_context = context or current_task_context()
    if task_context is None:
        return
    payload = json.dumps(
        {"stage": stage, "event": event, "detail": message},
        ensure_ascii=False)
    task_context.progress_cb(stage, max(0.0, min(100.0, float(pct))), payload)


def _render_line_handler(context=None) -> Callable[[str, str], None]:
    """解析渲染 stderr 的 ``done/total frames``，节流后上报任务进度。"""
    state = {"pct": 0.0, "at": 0.0}

    def handle(_stream: str, line: str) -> None:
        match = _FRAME_RE.search(line)
        if not match:
            return
        done, total = int(match.group(1)), int(match.group(2))
        if total <= 0:
            return
        pct = 2.0 + 93.0 * min(1.0, done / total)
        now = time.monotonic()
        if pct - state["pct"] < 1.0 and now - state["at"] < 2.0:
            return
        state["pct"], state["at"] = pct, now
        _progress("render", pct, "rendering",
                  f"OpenFilm 渲染 {done}/{total} 帧", context=context)

    return handle


def _render_lock_path(project_id: str) -> Path:
    pdir = project_dir(project_id).resolve()
    path = pdir / _RENDER_LOCK_NAME
    _assert_inside(path, pdir, what="渲染锁路径")
    return path


@contextlib.contextmanager
def _render_guard(project_id: str, *, timeout_s: float = 20.0):
    """项目级跨进程渲染锁（O_CREAT|O_EXCL 锁文件，崩溃后按 mtime 过期）。

    只保护“读状态 → 领取/写状态”这一小段；渲染本身不持锁。锁文件位于
    FlowMind 项目目录内，不进入 OpenFilm 工作区。
    """
    path = _render_lock_path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    deadline = time.monotonic() + timeout_s
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except OSError:
                age = 0.0
            if age > _RENDER_LOCK_STALE_S:
                _remove_quiet(path)
                continue
            if time.monotonic() >= deadline:
                raise OpenFilmError(
                    "等待项目渲染领取锁超时（可能有并发请求持锁）",
                    category="transient", retriable=True)
            time.sleep(0.02)
        except OSError as exc:
            raise OpenFilmError(
                f"创建项目渲染领取锁失败：{exc}", category="environment"
            ) from exc
    try:
        os.write(fd, token.encode("ascii"))
        os.close(fd)
        fd = None
        yield token
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        _remove_quiet(path)


def _render_meta(project: ProjectRecord) -> dict[str, Any]:
    meta = _meta(project)
    render = meta.get("render")
    if not isinstance(render, dict):
        render = {}
        meta["render"] = render
    return render


def _claim_active(project: ProjectRecord) -> bool:
    """项目是否已有活跃渲染领取（任务活跃或领取时间未过期）。"""
    render = _render_meta(project)
    claim = render.get("claim")
    if not isinstance(claim, dict) or not claim.get("token"):
        # 旧记录/升级前只写了 status+task_id 的渲染也视为活跃：宁可拒绝
        # 重复提交，也不并发写同一 output.mp4。
        if render.get("status") in _ACTIVE_RENDER_STATUSES:
            status = _task_status(render.get("task_id"))
            if status in {"queued", "running"}:
                return True
            return status is None and project.state == PSTATE_RUNNING
        return False
    status = _task_status(render.get("task_id"))
    if status in _TERMINAL_TASK_STATUSES:
        return False
    if status in {"queued", "running"}:
        return True
    try:
        age = time.time() - float(claim.get("at") or 0.0)
    except (TypeError, ValueError):
        age = _RENDER_CLAIM_STALE_S + 1.0
    return age < _RENDER_CLAIM_STALE_S


def _claim_render(project: ProjectRecord, token: str,
                  *, status: str = "claiming") -> None:
    render = _render_meta(project)
    render.update({
        "status": status,
        "claim": {"token": token, "at": time.time(), "pid": os.getpid()},
        "task_id": None,
        "error": None,
        "category": None,
    })
    project.state = PSTATE_RUNNING
    project.stage = STAGE_RENDER
    project.error = None
    project.stages[STAGE_RENDER] = "pending"


def _release_claim(project: ProjectRecord, token: str | None = None) -> bool:
    render = _render_meta(project)
    claim = render.get("claim")
    if token is not None and (
            not isinstance(claim, dict) or claim.get("token") != token):
        return False
    render["claim"] = None
    return True


def _render_temp_path(project_id: str) -> Path:
    """每次渲染唯一临时文件（UUID，非 PID），成功后原子替换 output.mp4。"""
    output = project_output(project_id)
    temp = output.with_name(f".{output.stem}.rendering-{uuid.uuid4().hex}.mp4")
    _assert_inside(temp, output.parent, what="临时成片路径")
    return temp


def record_render_task(project_id: str, task_id: str,
                       claim_token: str | None = None) -> bool:
    """把渲染任务 id 记入项目；claim_token 不匹配时返回 False（防串任务）。"""
    with _render_guard(project_id):
        project = get_project_store().get_project(project_id)
        if project is None:
            return False
        render = _render_meta(project)
        if claim_token is not None:
            claim = render.get("claim")
            if not isinstance(claim, dict) or \
                    claim.get("token") != claim_token:
                # 快速完成的任务可能在 submit 返回前就落终态并清 claim：
                # 终态下允许补记 task_id 供审计，不覆盖真实渲染状态。
                if render.get("status") not in {
                        "succeeded", "failed", "cancelled", "rejected",
                        "blocked"}:
                    return False
        if task_id not in project.task_ids:
            project.task_ids.append(task_id)
        # worker 可能在 submit 返回前就跑完（极快/失败）：终态不被迟到的
        # “queued” 覆盖，只补 task_id。
        if render.get("status") in {
                "succeeded", "failed", "cancelled", "rejected", "blocked"}:
            render["task_id"] = task_id
        elif render.get("status") == "running":
            render["task_id"] = task_id       # worker 已先推进，别降级成 queued
        else:
            render.update({"task_id": task_id, "status": "queued",
                           "error": None})
        _save(project)
        return True


def release_render_claim(project_id: str, claim_token: str | None = None) -> bool:
    """释放领取（提交失败/放弃时）；token 不匹配不误删别人的领取。"""
    with _render_guard(project_id):
        project = get_project_store().get_project(project_id)
        if project is None:
            return False
        changed = _release_claim(project, claim_token)
        if changed:
            _save(project)
        return changed


def mark_render_submit_failed(project_id: str, message: str,
                              claim_token: str | None = None) -> None:
    """任务提交失败：释放领取并把项目落 failed（不留下 running 假象）。"""
    with _render_guard(project_id):
        project = get_project_store().get_project(project_id)
        if project is None:
            return
        render = _render_meta(project)
        if claim_token is not None:
            claim = render.get("claim")
            if not isinstance(claim, dict) or \
                    claim.get("token") != claim_token:
                return
        render.update({
            "status": "failed", "error": message, "claim": None,
            "finished_at": _now(),
        })
        project.state = PSTATE_FAILED
        project.error = message
        project.stages[STAGE_RENDER] = "failed"
        _save(project)


def current_state(project_id: str) -> str:
    """按当前调用方租户读取项目状态（不可见/不存在返回空串）。"""
    try:
        return _load_project(project_id).state
    except OpenFilmError:
        return ""


def _fail_project(project: ProjectRecord, message: str, *, category: str,
                  status: str = "failed") -> None:
    project.state = PSTATE_FAILED
    project.stage = STAGE_RENDER
    project.error = message
    project.stages[STAGE_RENDER] = "failed"
    render = _render_meta(project)
    render.update({
        "status": status,
        "error": message,
        "category": category,
        "claim": None,
        "finished_at": _now(),
    })
    _save(project)


def _task_status(task_id: str | None) -> str | None:
    """读任务引擎状态（不可用/不可见返回 None，调用方按“仍活跃”保守处理）。"""
    if not task_id:
        return None
    try:
        from flowmind.tasks.manager import get_task_manager
        record = get_task_manager().get_task(task_id)
    except Exception:  # noqa: BLE001 任务库不可用不阻断状态机判断
        return None
    return (record or {}).get("status")


def prepare_render(project_id: str) -> dict[str, Any]:
    """原子领取渲染：同一项目最多一个活跃领取/任务。

    1. 项目级跨进程锁内重读项目 → 检查活跃领取 → 审批闸 → 写入 claim；
    2. 释放锁后做 Node/OpenFilm/FFmpeg 预检；
    3. 预检失败时持锁清 claim 并落 failed。

    返回 ``submit=True`` + ``claim_token`` 时由调用方提交 edit_pipeline；
    返回 ``submit=False`` 表示已进入 ``awaiting_review``（等待 edit_review）。
    """
    _load_project(project_id)             # 先校验项目/租户（不持锁）
    with _render_guard(project_id):
        project = _load_project(project_id)
        workspace = project_workspace(project_id)
        entry = workspace / FILM_FILE
        if not entry.is_file():
            raise OpenFilmError(
                f"工作区缺少 {FILM_FILE}：{workspace}；请先由外部 Agent 写入项目",
                category="validation")
        render = _render_meta(project)
        if _claim_active(project):
            raise OpenFilmError(
                "项目已有活跃渲染领取或渲染任务（同一项目只允许一个），"
                f"task={render.get('task_id') or '领取中'}",
                category="validation")
        gate_pending = (REVIEW_STAGE in project.review_stages
                        and REVIEW_STAGE not in project.approved)
        if gate_pending:
            project.state = PSTATE_AWAITING_REVIEW
            project.stage = REVIEW_STAGE
            project.error = None
            meta = _meta(project)
            meta["authored_at"] = _now()
            render.update({"status": "awaiting_review", "error": None,
                           "task_id": None, "claim": None})
            _save(project)
            return {
                "submit": False,
                "project_id": project_id,
                "state": project.state,
                "stage": project.stage,
                "task_id": None,
                "render_status": "awaiting_review",
                "claim_token": None,
                "workspace": str(workspace),
                "message": ("项目已进入 pre_render 审批等待；经 edit_review "
                            "approve 后开始渲染，revise 则退回外部 Agent 修改"),
            }

        token = uuid.uuid4().hex
        _claim_render(project, token, status="claiming")
        meta = _meta(project)
        meta["authored_at"] = _now()
        _save(project)
        prepared = {
            "submit": True,
            "project_id": project_id,
            "state": project.state,
            "stage": project.stage,
            "task_id": None,
            "render_status": "claiming",
            "claim_token": token,
            "workspace": str(workspace),
            "message": "OpenFilm 渲染已领取，正在做依赖预检",
        }

    try:
        _preflight_render()
    except OpenFilmError as exc:
        with _render_guard(project_id):
            current = get_project_store().get_project(project_id)
            if current is not None:
                render = _render_meta(current)
                claim = render.get("claim")
                if isinstance(claim, dict) and claim.get("token") == token:
                    _fail_project(current, str(exc), category=exc.category,
                                  status="blocked")
        raise
    return prepared


def _remove_quiet(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def run_render(project_id: str) -> SimpleNamespace:
    """在任务引擎 worker 内执行 OpenFilm 渲染并落项目状态/成片。"""
    project = _load_project(project_id)
    workspace = project_workspace(project_id)
    entry = workspace / FILM_FILE
    if not entry.is_file():
        raise OpenFilmError(
            f"工作区缺少 {FILM_FILE}：{workspace}", category="validation")
    output = project_output(project_id)
    # 直接调用内部 edit_pipeline 也不能绕过 pre_render 审批闸；正常路径
    # 由 edit_openfilm_render 在闸未放行时不提交任务，approve 才置 approved。
    if (REVIEW_STAGE in project.review_stages
            and REVIEW_STAGE not in project.approved):
        raise OpenFilmError(
            f"OpenFilm 项目尚未通过 {REVIEW_STAGE} 审批；请先经 edit_review "
            "approve（或创建项目时传 review_stages=[]）",
            category="validation")

    context = current_task_context()
    task_id = context.task_id if context is not None else None
    # 每个渲染的临时文件必须唯一：PID 在同一进程的并发任务间会冲突。
    temp_output = _render_temp_path(project_id)

    # 原子确认/领取：prepare 已领取则沿用其 token；直接调用 edit_pipeline
    # 时在此领取，避免同一项目出现两个活跃渲染。
    with _render_guard(project_id):
        project = _load_project(project_id)
        render = _render_meta(project)
        claim = render.get("claim")
        existing_task = render.get("task_id")
        if existing_task and task_id and existing_task != task_id:
            raise OpenFilmError(
                f"项目已有另一个活跃渲染任务（{existing_task}），拒绝重复执行",
                category="validation")
        if _claim_active(project):
            token = claim.get("token") if isinstance(claim, dict) else None
            if not token:
                token = uuid.uuid4().hex
                _claim_render(project, token, status="running")
                render = _render_meta(project)
        else:
            token = uuid.uuid4().hex
            _claim_render(project, token, status="running")
            render = _render_meta(project)
        render.update({
            "status": "running",
            "task_id": task_id or existing_task,
            "started_at": _now(),
            "output": str(output),
            "error": None,
            "claim": {"token": token, "at": time.time(), "pid": os.getpid()},
        })
        project.state = PSTATE_RUNNING
        project.stage = STAGE_RENDER
        project.error = None
        project.stages[STAGE_RENDER] = "running"
        _save(project)

    try:
        _preflight_render()
    except OpenFilmError as exc:
        with _render_guard(project_id):
            current = get_project_store().get_project(project_id)
            if current is not None:
                current_render = _render_meta(current)
                current_claim = current_render.get("claim")
                if isinstance(current_claim, dict) and \
                        current_claim.get("token") == token:
                    _fail_project(current, str(exc), category=exc.category,
                                  status="blocked")
        raise

    _progress("render", 2.0, "started", "OpenFilm 渲染开始")

    try:
        result = _run_cli(
            ["render", "--fps", str(int(_cfg().render_fps)),
             "--out", str(temp_output)],
            cwd=workspace,
            timeout=float(_cfg().render_timeout_s),
            on_line=_render_line_handler(context),
            cancel_check=(context.cancel_check if context is not None else None),
        )
    except OpenFilmError as exc:
        _remove_quiet(temp_output)
        _fail_project(project, str(exc), category=exc.category)
        raise
    except OSError as exc:
        _remove_quiet(temp_output)
        message = f"OpenFilm 渲染启动失败：{exc}"
        _fail_project(project, message, category="environment")
        raise OpenFilmUnavailable(message) from exc

    if result.cancelled:
        _remove_quiet(temp_output)
        project = get_project_store().get_project(project_id) or project
        project.state = PSTATE_CANCELLED
        project.stage = STAGE_RENDER
        project.stages[STAGE_RENDER] = "cancelled"
        project.error = "OpenFilm 渲染已取消"
        render = _render_meta(project)
        render.update({
            "status": "cancelled",
            "task_id": task_id,
            "cancelled_at": _now(),
            "output": None,
            "claim": None,
        })
        _save(project)
        raise CancelledError("OpenFilm 渲染已取消")

    if result.timed_out:
        _remove_quiet(temp_output)
        message = (f"OpenFilm 渲染超时（{_cfg().render_timeout_s:.0f}s）")
        _fail_project(project, message, category="timeout")
        raise OpenFilmError(message, category="timeout", retriable=True)

    if result.returncode != 0:
        _remove_quiet(temp_output)
        category = "validation" if result.returncode == 2 else "render"
        message = (
            f"OpenFilm 渲染失败（退出码 {result.returncode}）："
            f"{_tail(result.stderr or result.stdout)}")
        _fail_project(project, message, category=category)
        raise OpenFilmError(message, category=category, retriable=False)

    try:
        size = temp_output.stat().st_size
    except OSError:
        size = 0
    if size <= 0:
        _remove_quiet(temp_output)
        message = (
            "OpenFilm 返回成功但未产出非空 output.mp4；"
            f"CLI 输出：{_tail(result.stderr or result.stdout, 1200)}")
        _fail_project(project, message, category="render")
        raise OpenFilmError(message, category="render")
    try:
        os.replace(temp_output, output)
        size = output.stat().st_size
    except OSError as exc:
        _remove_quiet(temp_output)
        message = f"OpenFilm 成片无法落位到 FlowMind 项目目录：{exc}"
        _fail_project(project, message, category="environment")
        raise OpenFilmError(message, category="environment") from exc

    project = get_project_store().get_project(project_id) or project
    project.state = PSTATE_SUCCEEDED
    project.stage = STAGE_RENDER
    project.error = None
    project.stages[STAGE_RENDER] = "done"
    render = _render_meta(project)
    render.update({
        "status": "succeeded",
        "task_id": task_id,
        "started_at": render.get("started_at"),
        "finished_at": _now(),
        "output": str(output),
        "bytes": size,
        "duration_s": result.duration_s,
        "error": None,
        "claim": None,
    })
    _save(project)
    _progress("render", 100.0, "done", f"OpenFilm 成片已产出：{output.name}")
    return SimpleNamespace(
        state=project.state,
        stage=project.stage,
        stages=project.stages,
        review_stages=project.review_stages,
        artifact_dir=str(project_dir(project_id)),
        output_path=str(output),
    )


# ── 审批（复用 edit_review 的 approve/revise/reject 语义）──

def _write_review_note(workspace: Path, project_id: str, action: str,
                       feedback: str) -> None:
    if not workspace.is_dir():
        raise OpenFilmError(
            f"OpenFilm 工作区不存在：{workspace}", category="validation")
    payload = {
        "project_id": project_id,
        "gate": REVIEW_STAGE,
        "action": action,
        "feedback": feedback,
        "recorded_at": _now(),
        "next": ("外部 Agent 修改 film.html 后重新调用 edit_openfilm_render"
                 if action == "revise" else
                 "edit_review approve 后由 FlowMind 开始渲染"),
    }
    _atomic_write(workspace / REVIEW_FILENAME,
                  json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(workspace / REVIEW_MARKDOWN_FILENAME, f"""# FlowMind 审批反馈

- 项目：`{project_id}`
- 闸门：`{REVIEW_STAGE}`
- 动作：`{action}`
- 时间：{payload['recorded_at']}

## 反馈

{feedback or '（无）'}

## 下一步

{payload['next']}
""")


def apply_review(project_id: str, stage: str, action: str,
                 feedback: str | None) -> dict[str, Any]:
    """把 edit_review 动作映射到 OpenFilm 项目生命周期。

    - approve：放行 pre_render，置 running，由调用方提交渲染任务；
    - revise：记录反馈并退回 awaiting_agent（**不渲染**），等外部 Agent 改稿；
    - reject：项目终止为 failed。
    """
    project = _load_project(project_id)
    if stage != REVIEW_STAGE:
        raise OpenFilmError(
            f"OpenFilm 仅支持 {REVIEW_STAGE} 审批闸，收到：{stage}",
            category="validation")
    if project.state != PSTATE_AWAITING_REVIEW or project.stage != REVIEW_STAGE:
        raise OpenFilmError(
            f"项目当前状态 {project.state}/{project.stage} 不在 {REVIEW_STAGE} "
            "审批等待", category="validation")
    text = (feedback or "").strip()
    workspace = project_workspace(project_id)
    _write_review_note(workspace, project_id, action, text)

    if action == "reject":
        project.state = PSTATE_FAILED
        project.error = f"review_rejected[{REVIEW_STAGE}]: {text}"
        project.stages[REVIEW_STAGE] = "rejected"
        render = _render_meta(project)
        render.update({"status": "rejected", "error": project.error,
                       "claim": None, "finished_at": _now()})
        _save(project)
        return {"submit_render": False, "state": project.state,
                "stage": project.stage, "task_id": None,
                "message": "项目已按人工审批终止"}

    if action == "approve":
        if REVIEW_STAGE not in project.approved:
            project.approved.append(REVIEW_STAGE)
        project.state = PSTATE_RUNNING
        project.stage = STAGE_RENDER
        project.error = None
        project.stages[REVIEW_STAGE] = "done"
        project.stages[STAGE_RENDER] = "pending"
        render = _render_meta(project)
        render.update({"status": "approved", "error": None, "claim": None})
        _save(project)
        return {"submit_render": True, "state": project.state,
                "stage": project.stage, "task_id": None,
                "message": "pre_render 已放行，开始渲染"}

    if action == "revise":
        project.review_feedback.setdefault(REVIEW_STAGE, []).append(text)
        project.state = PSTATE_AWAITING_AGENT
        project.stage = STAGE_AUTHORING
        project.error = None
        project.stages[REVIEW_STAGE] = "pending"
        if REVIEW_STAGE in project.approved:
            project.approved.remove(REVIEW_STAGE)
        render = _render_meta(project)
        render.update({"status": "awaiting_agent", "error": None,
                       "claim": None})
        _save(project)
        return {"submit_render": False, "state": project.state,
                "stage": project.stage, "task_id": None,
                "message": ("反馈已记录到工作区 FLOWMIND_REVIEW.md；"
                            "外部 Agent 修改 film.html 后再次调用 "
                            "edit_openfilm_render")}

    raise OpenFilmError(f"不支持的审批动作：{action}", category="validation")


# ── 状态快照（edit_status 用）──

def _relative_inside(workspace: Path, path: Path) -> str | None:
    try:
        resolved = path.resolve()
    except OSError:
        return None
    if not _inside(resolved, workspace):
        return None
    return resolved.relative_to(workspace).as_posix()


def status_snapshot(project_id: str,
                    project: ProjectRecord | None = None) -> dict[str, Any]:
    """OpenFilm 项目的状态补充字段（不发起任何 CLI 调用）。"""
    project = project or _load_project(project_id)
    workspace = project_workspace(project_id)
    meta = _meta(project)
    film = workspace / FILM_FILE
    pages: list[str] = []
    assets: list[str] = []
    if workspace.is_dir():
        for path in sorted(workspace.glob("*.html")):
            relative = _relative_inside(workspace, path)
            if relative and relative != FILM_FILE:
                pages.append(relative)
        assets_dir = workspace / "assets"
        if assets_dir.is_dir():
            for path in sorted(assets_dir.iterdir()):
                relative = _relative_inside(workspace, path)
                if relative and path.is_file():
                    assets.append(relative)
    output = project_output(project_id)
    output_path = str(output) if output.is_file() and output.stat().st_size > 0 \
        else None
    render = meta.get("render") if isinstance(meta.get("render"), dict) else {}
    render_snapshot = dict(render)
    render_snapshot.pop("claim", None)       # 内部领取 token 不暴露给调用方
    return {
        "workspace": str(workspace),
        "entry_file": str(film),
        "film_exists": film.is_file(),
        "pages": pages[:50],
        "assets": assets[:100],
        "output_path": output_path,
        "studio_url": meta.get("studio_url"),
        "render": render_snapshot,
        "feedback": list(project.review_feedback.get(REVIEW_STAGE, [])),
        "review_stages": list(project.review_stages),
    }


def artifact_list(project_id: str) -> list[str]:
    """项目产物相对清单（限制在工作区内，按名称排序）。"""
    workspace = project_workspace(project_id)
    names: list[str] = []
    for name in (FILM_FILE, METADATA_FILENAME, AGENT_FILENAME,
                 REVIEW_FILENAME, REVIEW_MARKDOWN_FILENAME):
        path = workspace / name
        if path.is_file():
            names.append(f"{WORKSPACE_DIRNAME}/{name}")
    for path in sorted(workspace.glob("*.html")):
        relative = _relative_inside(workspace, path)
        if relative and relative != FILM_FILE:
            names.append(f"{WORKSPACE_DIRNAME}/{relative}")
    assets_dir = workspace / "assets"
    if assets_dir.is_dir():
        for path in sorted(assets_dir.iterdir()):
            relative = _relative_inside(workspace, path)
            if relative and path.is_file():
                names.append(f"{WORKSPACE_DIRNAME}/{relative}")
    return names[:200]


def next_step(project: ProjectRecord, snapshot: dict[str, Any],
              state: str | None = None) -> str:
    """给调用方的下一步提示（状态机语义）。"""
    current = state or project.state
    if current == PSTATE_AWAITING_AGENT:
        return ("等待外部编程 Agent 在工作区创作 OpenFilm 项目；完成后调用 "
                "edit_openfilm_render(project_id)")
    if current == PSTATE_AWAITING_REVIEW:
        return ("项目等待 pre_render 审批：edit_review approve 开始渲染，"
                "revise 带反馈退回外部 Agent，reject 终止")
    if current == PSTATE_RUNNING:
        return "渲染进行中：轮询 edit_status；取消用 render task_id + localize_cancel"
    if current == PSTATE_SUCCEEDED:
        return f"成片已产出：{snapshot.get('output_path') or '见 outputs'}"
    if current == PSTATE_CANCELLED:
        return "渲染已取消：可重新调用 edit_openfilm_render 再渲染"
    if current == PSTATE_FAILED:
        return "项目失败：按 error 修复后重新调用 edit_openfilm_render"
    return "可调用 edit_openfilm_studio 打开 Studio，或 edit_openfilm_render 渲染"


def check_ready() -> dict[str, Any]:
    """环境自检（文档/排障用）：不启动 Studio、不写项目。"""
    report: dict[str, Any] = {"available": False, "node": None,
                              "openfilm_cli": None, "ffmpeg": None,
                              "error": None, "category": None}
    try:
        command = _resolve_command()
        report["openfilm_cli"] = command[-1]
        report["node"] = command[0] if command[0] != command[-1] else None
        env = _subprocess_env()
        _check_ffmpeg(env)
        report["ffmpeg"] = shutil.which("ffmpeg", path=env.get("PATH"))
        report["available"] = True
    except OpenFilmError as exc:
        report["error"] = str(exc)
        report["category"] = exc.category
    return report
