"""localize_cancel 技能：取消一个正在 queued/running 的 VL 任务。

薄包装 VL `DELETE /api/v1/tasks/{task_id}`；不在此重排或重提。
HTTP 层统一走 VLClient（vl_client.py）。v0.3：错误在技能体内以 degraded
SkillOutput 返回，failure_category 字段告诉 Agent 是 video / transient /
environment 中的哪一类。
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

class CancelInput(BaseModel):
    """cancel 技能入参。"""
    task_id: str = Field(..., min_length=1, description="要取消的 task_id")


# ── 出参 ──

class CancelReport(BaseModel):
    """cancel 技能业务载荷。"""
    task_id: str
    cancelled: bool
    message: str
    failure_category: str | None = None  # "environment" / "video" / "transient" / "unknown"
    retriable: bool = False
    warning: str | None = None


# ── 入口 ──

@skill(id="localize_cancel", name="取消视频本地化任务", version=_VERSION)
def localize_cancel(inp: CancelInput) -> SkillOutput[CancelReport]:
    """调 DELETE /api/v1/tasks/{task_id} 取消任务，返回结构化结果。

    错误分类：
    - 4xx（任务不存在 / 已结束）→ video
    - 5xx → transient（可重试）
    - ConnectionError / Timeout → environment（先查网络）
    """
    cfg = load_config().localizer
    client = VLClient(cfg)

    try:
        body = client.delete(f"/tasks/{inp.task_id}")
    except VLAPIError as exc:
        return _failure_output(inp.task_id, exc, exc.category)

    message = str(body.get("message", ""))

    report = CancelReport(task_id=inp.task_id, cancelled=True, message=message)
    chain = ReasoningChain(
        conclusion=f"已请求取消任务 {inp.task_id}",
        triggered_rules=[],
        evidence=[],
        causal_analysis=f"VL 响应：{message}",
        risk_note="任务已被请求取消；VL 端可能仍需几秒清理。",
    )
    return SkillOutput(
        data=report, reasoning=[chain], confidence=1.0, sample_size=1,
    )


def _failure_output(task_id: str, exc: Exception, category: str) -> SkillOutput[CancelReport]:
    """统一的失败返回：degraded SkillOutput，category 在 report 字段里。

    注意：不写完整 exc 消息到 report / reasoning（避免泄漏内部 host / 凭证）。
    仅保留异常类型名 + category，Agent 足够据此决策。
    """
    report = CancelReport(
        task_id=task_id,
        cancelled=False,
        message=f"VL 调用失败（{category}）",
        failure_category=category,
        retriable=is_retriable(category),
        warning=f"取消失败（{category}）",
    )
    chain = ReasoningChain(
        conclusion=f"取消任务 {task_id} 失败（{category}）",
        triggered_rules=[],
        evidence=[],
        causal_analysis=f"取消任务端点 → {type(exc).__name__}",
        risk_note=(
            f"{'可重试' if is_retriable(category) else '需查环境或任务状态'}；"
            f"transient/environment 通常无需 Agent 介入。"
        ),
    )
    return SkillOutput(
        data=report, reasoning=[chain], confidence=0.0, sample_size=1,
        degraded=True, degradation_reason=category,
    )