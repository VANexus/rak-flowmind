"""契约层：定义「对任意 Agent 友好」的统一数据结构。

这是整个 SDK 的规格核心——任何返回 SkillResult 的技能天然对 Agent 友好。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class RuleHit(BaseModel):
    """触发规则：四段式推理链的第二段。"""
    rule_id: str
    name: str
    expression: str  # 人类可读的规则表达式
    hit: bool


class Evidence(BaseModel):
    """数据证据：四段式推理链的第三段。"""
    metric: str
    value: float | str
    threshold: float | str | None = None
    comparison: str  # 如 ">"、"=="、"命中区间"
    window: str | None = None  # 如 "近30天"


class ReasoningChain(BaseModel):
    """四段式因果推理链：决策结论 → 触发规则 → 数据证据 → 因果推理与风险提示。"""
    conclusion: str          # 决策结论
    triggered_rules: list[RuleHit] = Field(default_factory=list)  # 触发规则
    evidence: list[Evidence] = Field(default_factory=list)        # 数据证据
    causal_analysis: str     # 因果推理
    risk_note: str           # 风险提示
    confidence: float = 1.0


class ReliabilityMetrics(BaseModel):
    """可靠性指标：供上层 Agent 读取，用于决策与评测。"""
    latency_ms: float
    confidence: float
    sample_size: int
    degraded: bool = False
    degradation_reason: str | None = None


class TraceContext(BaseModel):
    """全链路追踪上下文：trace_id 贯穿每次调用。"""
    trace_id: str
    source: str = "agent"
    target: str = "flowmind"
    timestamp: str  # ISO8601


class SkillError(BaseModel):
    """结构化错误：错误永不静默。"""
    code: str  # 如 "VALIDATION" / "INTERNAL" / "NOT_FOUND"
    message: str
    retriable: bool = False
    details: dict | None = None


class SkillOutput(BaseModel, Generic[T]):
    """技能内部产出：业务数据 + 推理链。由框架套上 SkillResult 信封。"""
    data: T
    reasoning: list[ReasoningChain] = Field(default_factory=list)
    confidence: float = 1.0
    sample_size: int = 0
    degraded: bool = False
    degradation_reason: str | None = None


class SkillResult(BaseModel, Generic[T]):
    """对外统一返回信封：任意 Agent 消费此结构。"""
    ok: bool
    skill: str
    version: str
    trace: TraceContext
    data: T | None = None
    reasoning: list[ReasoningChain] = Field(default_factory=list)
    metrics: ReliabilityMetrics
    error: SkillError | None = None


def new_trace(
    source: str = "agent",
    target: str = "flowmind",
    trace_id: str | None = None,
) -> TraceContext:
    """创建追踪上下文：调用方给了 trace_id 就透传，否则生成。"""
    return TraceContext(
        trace_id=trace_id or str(uuid.uuid4()),
        source=source,
        target=target,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )