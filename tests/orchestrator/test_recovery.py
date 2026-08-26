"""Recovery 节点测试：验证重试/跳过/失败决策逻辑。

决策矩阵：
- ok=True → continue
- ok=False + retriable + retries_left>0 → retry
- ok=False + retriable + retries_left<=0 → fail
- ok=False + not retriable → skip
"""
from __future__ import annotations

from flowmind.contracts import ReliabilityMetrics, SkillError, SkillResult, new_trace


def _make_result(*, ok: bool = True, retriable: bool = False) -> SkillResult:
    return SkillResult(
        ok=ok, skill="x", version="1.0", trace=new_trace(),
        error=(
            None if ok
            else SkillError(code="INTERNAL", message="fail", retriable=retriable)
        ),
        metrics=ReliabilityMetrics(latency_ms=0, confidence=0, sample_size=0),
    )


def test_recovery_retries_transient_error():
    """可重试错误 + 有剩余次数 → retry。"""
    from flowmind.orchestrator.recovery import decide_recovery

    result = _make_result(ok=False, retriable=True)
    decision = decide_recovery(result, retries_left=1)

    assert decision["action"] == "retry"


def test_recovery_skips_non_retriable_error():
    """不可重试错误 → skip。"""
    from flowmind.orchestrator.recovery import decide_recovery

    result = _make_result(ok=False, retriable=False)
    decision = decide_recovery(result, retries_left=1)

    assert decision["action"] == "skip"


def test_recovery_fails_when_no_retries_left():
    """可重试错误 + 无剩余次数 → fail。"""
    from flowmind.orchestrator.recovery import decide_recovery

    result = _make_result(ok=False, retriable=True)
    decision = decide_recovery(result, retries_left=0)

    assert decision["action"] == "fail"


def test_recovery_does_not_retry_success():
    """成功 → continue，不重试。"""
    from flowmind.orchestrator.recovery import decide_recovery

    result = _make_result(ok=True)
    decision = decide_recovery(result, retries_left=1)

    assert decision["action"] == "continue"
