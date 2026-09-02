"""localize_download 技能：列出已完成任务的产物文件 + VL 的 download URL。

不把二进制塞进 SkillResult（破坏 JSON 信封）；让 Agent 按 URL 自行拉取。
小文件路径同时返回本地路径，Agent 可用 Read 等工具直接读。

HTTP 层统一走 VLClient（vl_client.py）。错误分类（environment / video /
transient）经 VLAPIError.category 获得，通过 degraded SkillOutput 返回；
failure_category 在 DownloadReport 字段里。
"""
from __future__ import annotations

import requests  # noqa: F401  保留模块级引用：测试 fixture 经 <mod>.requests 打桩拦截 VLClient
from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.errors import is_retriable
from flowmind.skill import skill
from flowmind.vl_client import VLAPIError, VLClient

_VERSION = "0.1.0"


# ── 入参 ──

class DownloadInput(BaseModel):
    """download 技能入参。"""
    task_id: str = Field(..., min_length=1, description="已 completed 任务的 task_id")


# ── 出参 ──

class DownloadFile(BaseModel):
    """单个产物文件信息。"""
    filename: str
    local_path: str        # VL 端路径
    url: str               # VL 的 download URL（Agent 可自行 GET）


class DownloadReport(BaseModel):
    """download 技能业务载荷。"""
    task_id: str
    status: str
    files: list[DownloadFile]
    degraded: bool = False     # completed 但无产物时为 True（VL 假完成）
    warning: str | None = None
    failure_category: str | None = None  # 仅 degraded=True 且是网络/服务错误时填充
    retriable: bool = False


# ── 入口 ──

@skill(id="localize_download", name="获取任务产物清单与下载链接", version=_VERSION)
def localize_download(inp: DownloadInput) -> SkillOutput[DownloadReport]:
    """调 GET /api/v1/tasks/{task_id} 拉任务详情，列出 outputs 里的文件 + VL 的 download URL。

    任务未完成 → degraded + video（资源状态不对）。
    任务 completed 但 outputs 空 → degraded + warning（VL 假完成信号，让 Agent 警惕）。
    网络错误 → degraded + environment；5xx → degraded + transient。
    """
    cfg = load_config().localizer
    client = VLClient(cfg)
    try:
        body = client.get(f"/tasks/{inp.task_id}")
    except VLAPIError as exc:
        return _failure_output(inp.task_id, exc, exc.category)

    status = body.get("status", "unknown")
    if status != "completed":
        return _failure_output(
            inp.task_id,
            Exception(f"Task {inp.task_id} not completed (status={status})"),
            "video",
        )

    outputs = body.get("outputs") or {}
    base = cfg.api_base.rstrip("/") + cfg.api_prefix
    files = [
        DownloadFile(
            filename=name,
            local_path=str(path),
            url=f"{base}/tasks/{inp.task_id}/download?file={name}",
        )
        for name, path in outputs.items()
    ]

    degraded = len(files) == 0
    warning = (
        f"任务 {inp.task_id} 状态为 completed 但无产物输出，可能是 VL 假完成（ASR 无内容等）"
        if degraded else None
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
            + ("（degraded：completed 但无产物）" if degraded else "")
        ),
        triggered_rules=[],
        evidence=[],
        causal_analysis=f"VL 返回 status={status} + outputs 共 {len(outputs)} 项",
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


def _failure_output(task_id: str, exc: Exception, category: str) -> SkillOutput[DownloadReport]:
    """统一的失败返回：degraded SkillOutput。

    注意：warning 字段不放完整 `str(exc)`（避免泄漏内部 host / 凭证）。仅保留
    category + 异常类型名，Agent 足够据此决策。
    """
    report = DownloadReport(
        task_id=task_id,
        status="unknown",
        files=[],
        degraded=True,
        warning=f"获取任务 {task_id} 失败（{category}）",
        failure_category=category,
        retriable=is_retriable(category),
    )
    chain = ReasoningChain(
        conclusion=f"下载任务 {task_id} 失败（{category}）",
        triggered_rules=[],
        evidence=[],
        causal_analysis=f"查询任务状态端点 → {type(exc).__name__}",
        risk_note=(
            f"{'可重试' if is_retriable(category) else '需查任务是否存在/是否完成'}；"
            f"video 类通常说明任务不存在或未 completed。"
        ),
    )
    return SkillOutput(
        data=report, reasoning=[chain], confidence=0.0, sample_size=0,
        degraded=True, degradation_reason=category,
    )