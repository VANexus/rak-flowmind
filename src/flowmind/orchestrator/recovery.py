"""Recovery 节点：错误恢复决策。

职责：
- 根据 SkillResult 决定下一步动作（continue / retry / skip / fail）
- 决策依据：ok 状态 + error.retriable + 剩余重试次数
- 纯函数，无副作用，便于测试
"""
from __future__ import annotations


def decide_recovery(result, retries_left: int) -> dict:
    """根据 SkillResult 决定恢复策略。

    Args:
        result: 上一步的 SkillResult。
        retries_left: 剩余重试次数。

    Returns:
        {"action": "continue"|"retry"|"skip"|"fail", "reason": str}
    """
    if result.ok:
        return {"action": "continue", "reason": "步骤成功"}

    error = result.error
    if error and error.retriable and retries_left > 0:
        return {"action": "retry", "reason": f"可重试错误: {error.message}"}

    if error and error.retriable and retries_left <= 0:
        return {"action": "fail", "reason": f"重试次数耗尽: {error.message}"}

    if error and not error.retriable:
        return {"action": "skip", "reason": f"不可重试错误，跳过: {error.message}"}

    return {"action": "skip", "reason": "未知错误，跳过"}
