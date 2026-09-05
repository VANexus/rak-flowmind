"""localize_download 技能：列出已完成任务的产物文件 + 下载 URL。

读 TaskStore 的 output_paths（任务成功时由 TaskManager 落库），不把二进制
塞进 SkillResult（破坏 JSON 信封）；产物 URL 指向 REST 层的下载端点
``GET /api/v1/tasks/{task_id}/download?file=<name>``（阶段 4 实现），
本地路径同时返回，Agent 可用文件工具直接读。

语义：
- 任务不存在 → degraded + video（资源缺失）
- 任务未到 succeeded → degraded + video（未完成无产物）
- succeeded 但 output_paths 为空 → degraded + warning（空结果任务，
  如无人声视频：任务成功但无产物文件）
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel, Field

from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.errors import is_retriable
from flowmind.skill import skill
from flowmind.tasks import STATUS_SUCCEEDED
from flowmind.tasks.manager import get_task_manager

_VERSION = "0.2.0"

# REST 层产物下载端点（阶段 4 实现；相对路径，随服务同源访问）
_DOWNLOAD_ENDPOINT = "/api/v1/tasks"


# ── 入参 ──

class DownloadInput(BaseModel):
    """download 技能入参。"""
    task_id: str = Field(..., min_length=1, description="已 succeeded 任务的 task_id")


# ── 出参 ──

class DownloadFile(BaseModel):
    """单个产物文件信息。"""
    filename: str
    local_path: str        # 服务端本地路径（Agent 与服务同机时可直接读）
    url: str               # 下载 URL（GET /api/v1/tasks/{task_id}/download?file=<name>）


class DownloadReport(BaseModel):
    """download 技能业务载荷。"""
    task_id: str
    status: str
    files: list[DownloadFile]
    degraded: bool = False     # succeeded 但无产物时为 True（空结果任务）
    warning: str | None = None
    failure_category: str | None = None  # 任务不存在/未完成时为 "video"
    retriable: bool = False


# ── 入口 ──

@skill(id="localize_download", name="获取任务产物清单与下载链接", version=_VERSION)
def localize_download(inp: DownloadInput) -> SkillOutput[DownloadReport]:
    """读任务行的 output_paths，列出产物文件 + 下载 URL。

    任务未完成/不存在 → degraded + video；succeeded 无产物 → degraded +
    warning（空结果任务，非服务故障）。
    """
    manager = get_task_manager()
    rec = manager.get_task(inp.task_id)

    if rec is None:
        return _failure_output(inp.task_id, "unknown", "任务不存在，无法获取产物")
    status = rec.get("status", "unknown")
    if status != STATUS_SUCCEEDED:
        return _failure_output(
            inp.task_id, status, f"任务未完成（status={status}），无产物可下载",
        )

    output_paths = [str(p) for p in (rec.get("output_paths") or []) if p]
    files = [
        DownloadFile(
            filename=Path(p).name,
            local_path=p,
            url=f"{_DOWNLOAD_ENDPOINT}/{inp.task_id}/download?file={quote(Path(p).name)}",
        )
        for p in output_paths
    ]

    degraded = len(files) == 0
    warning = (
        f"任务 {inp.task_id} 状态为 succeeded 但无产物输出"
        f"（空结果任务，如无人声视频）" if degraded else None
    )
    report = DownloadReport(
        task_id=inp.task_id,
        status=status,
        files=files,
        degraded=degraded,
        warning=warning,
    )
    chain = ReasoningChain(
        conclusion=(
            f"任务 {inp.task_id} 产物清单：{len(files)} 个文件"
            + ("（degraded：succeeded 但无产物）" if degraded else "")
        ),
        triggered_rules=[], evidence=[],
        causal_analysis=f"TaskStore output_paths 共 {len(output_paths)} 项",
        risk_note=warning or "按 URL 拉取文件即可；大文件建议用流式下载。",
    )
    return SkillOutput(
        data=report,
        reasoning=[chain],
        confidence=1.0 if not degraded else 0.5,
        sample_size=len(files),
        degraded=degraded,
        degradation_reason=warning,
    )


def _failure_output(task_id: str, status: str, message: str) -> SkillOutput[DownloadReport]:
    """统一的失败返回：degraded + video（资源缺失/状态不对，非临时故障）。"""
    report = DownloadReport(
        task_id=task_id,
        status=status,
        files=[],
        degraded=True,
        warning=message,
        failure_category="video",
        retriable=is_retriable("video"),
    )
    chain = ReasoningChain(
        conclusion=f"获取任务 {task_id} 产物失败（video）",
        triggered_rules=[], evidence=[],
        causal_analysis=message,
        risk_note="video 类通常说明任务不存在或未 succeeded；可先 localize_status 确认。",
    )
    return SkillOutput(
        data=report, reasoning=[chain], confidence=0.0, sample_size=0,
        degraded=True, degradation_reason="video",
    )
