"""A2A ↔ FlowMind 类型映射。"""
from __future__ import annotations

from typing import Any


def a2a_task_to_request(
    message: dict,
    metadata: dict | None = None,
) -> dict:
    """从 A2A 用户消息提取 FlowMind 编排请求。

    Args:
        message: A2A Message dict（含 parts）。
        metadata: A2A Task metadata（可能含 skill_group / include_reasoning）。

    Returns:
        {"goal": str, "skill_group": str|None, "include_reasoning": bool}
    """
    parts = message.get("parts", [])
    text_parts = [p.get("text", "") for p in parts if p.get("type") == "text"]
    goal = " ".join(text_parts).strip()

    meta = metadata or {}
    return {
        "goal": goal,
        "skill_group": meta.get("skill_group"),
        "include_reasoning": meta.get("include_reasoning", False),
    }


def result_to_a2a_task(
    task_id: str,
    result: dict,
    include_reasoning: bool,
) -> dict:
    """将编排结果映射回 A2A Task。

    Args:
        task_id: A2A Task ID。
        result: 编排结果（output/reasoning/degraded/error）。
        include_reasoning: 是否附带完整推理链。

    Returns:
        A2A Task dict。
    """
    error = result.get("error")
    degraded = result.get("degraded", False)

    if error:
        state = "failed"
    else:
        state = "completed"

    task: dict[str, Any] = {
        "id": task_id,
        "status": {
            "state": state,
            "degraded": degraded,
        },
    }

    if include_reasoning and result.get("reasoning"):
        task["history"] = result["reasoning"]

    if error:
        task["status"]["message"] = error
    elif result.get("output") is not None:
        task["artifacts"] = [
            {"parts": [{"type": "text", "text": _serialize_output(result["output"])}]}
        ]

    return task


def _serialize_output(output: Any) -> str:
    """将输出序列化为 A2A artifact 文本。"""
    if isinstance(output, str):
        return output
    import json
    return json.dumps(output, ensure_ascii=False, default=str)
