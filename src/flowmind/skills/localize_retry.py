"""localize_retry 技能：用同样的入参重新提交一个失败/取消的任务。

内部走两步：GET /tasks/{id} 拿原参数 → POST /tasks（单条）重提。
HTTP 层统一走 VLClient（vl_client.py）。
对 Agent 来说一次调用就行，不用自己拿 source_video 再调 batch。
"""
from __future__ import annotations

import requests  # noqa: F401  保留模块级引用：测试 fixture 经 <mod>.requests 打桩拦截 VLClient

from pydantic import BaseModel, Field

from flowmind.config import LocalizerConfig, load_config
from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.errors import is_retriable
from flowmind.skill import skill
from flowmind.vl_client import VLAPIError, VLClient

_VERSION = "0.1.0"


# ── 入参 ──

class RetryInput(BaseModel):
    """retry 技能入参。"""
    task_id: str = Field(..., min_length=1, description="要重提的原 task_id")


# ── 出参 ──

class RetryReport(BaseModel):
    """retry 技能业务载荷。"""
    original_task_id: str
    new_task_id: str
    original_status: str | None
    source_video: str
    target_lang: str
    enable_tts: bool
    remove_subtitles: bool
    failure_category: str | None = None
    retriable: bool = False
    message: str | None = None    # 失败时的人类可读原因


# ── 入口 ──

@skill(id="localize_retry", name="重提失败任务", version=_VERSION)
def localize_retry(inp: RetryInput) -> SkillOutput[RetryReport]:
    """拿原 task 的入参（source_video/target_lang/enable_tts/remove_subtitles），
    调 POST /tasks 单条重新提交，返回新 task_id。

    失败分类（在 RetryReport.failure_category）：
    - 原 task 不存在（404）→ video
    - 原 task 没 source_video（VL 假完成场景）→ video
    - POST /tasks 5xx → transient（可重试）
    - 连接错 / 超时 → environment（先查网络）
    """
    cfg: LocalizerConfig = load_config().localizer
    client = VLClient(cfg)

    # 1) GET 原 task 拿参数
    try:
        original = client.get(f"/tasks/{inp.task_id}")
    except VLAPIError as exc:
        return _fail_output(inp.task_id, exc, exc.category)
    original_status = original.get("status")

    source_video = original.get("source_video")
    target_lang = original.get("target_language") or "en"
    enable_tts = bool(original.get("enable_tts", False))
    remove_subtitles = bool(original.get("remove_subtitles", True))

    if not source_video:
        return _fail_output(
            inp.task_id,
            Exception(f"Task {inp.task_id} has no source_video, cannot retry"),
            "video",
        )

    # 2) POST /tasks 单条重提
    payload = {
        "video_path": source_video,
        "target_lang": target_lang,
        "source_lang": original.get("source_lang") or "zh",
        "enable_tts": enable_tts,
        "remove_subtitles": remove_subtitles,
    }
    if original.get("chat_id"):
        payload["chat_id"] = original["chat_id"]
    try:
        new_body = client.post("/tasks", payload)
    except VLAPIError as exc:
        return _fail_output(inp.task_id, exc, exc.category)
    new_task_id = new_body.get("task_id") or new_body.get("job_id") or ""

    report = RetryReport(
        original_task_id=inp.task_id,
        new_task_id=new_task_id,
        original_status=original_status,
        source_video=source_video,
        target_lang=target_lang,
        enable_tts=enable_tts,
        remove_subtitles=remove_subtitles,
    )
    chain = ReasoningChain(
        conclusion=f"已重提任务 {inp.task_id} → 新 task {new_task_id}（status={original_status} → queued）",
        triggered_rules=[],
        evidence=[],
        causal_analysis=f"沿用原参数：source_video={source_video} / target_lang={target_lang} / "
                        f"enable_tts={enable_tts} / remove_subtitles={remove_subtitles}",
        risk_note="新任务独立调度；原 task 的失败原因若仍存在会再次失败，看 error 字段定位。",
    )
    return SkillOutput(
        data=report, reasoning=[chain], confidence=1.0, sample_size=1,
    )


def _fail_output(task_id: str, exc: Exception, category: str) -> SkillOutput[RetryReport]:
    """统一的失败返回：degraded SkillOutput，category / message 在 report 字段里。

    注意：message 字段不直接放 `str(exc)`（避免泄漏内部 host / 凭证）；仅保留
    异常类型名 + category，Agent 足够据此决策。
    """
    report = RetryReport(
        original_task_id=task_id,
        new_task_id="",
        original_status=None,
        source_video="",
        target_lang="",
        enable_tts=False,
        remove_subtitles=False,
        failure_category=category,
        retriable=is_retriable(category),
        message=f"重提失败（{category}）：{type(exc).__name__}",
    )
    chain = ReasoningChain(
        conclusion=f"重提任务 {task_id} 失败（{category}）",
        triggered_rules=[],
        evidence=[],
        causal_analysis=f"重提任务端点 → {type(exc).__name__}",
        risk_note=(
            f"{'可重试' if is_retriable(category) else '需查环境或视频资源'}；"
            f"video 类通常需要更换 source_video。"
        ),
    )
    return SkillOutput(
        data=report, reasoning=[chain], confidence=0.0, sample_size=1,
        degraded=True, degradation_reason=category,
    )