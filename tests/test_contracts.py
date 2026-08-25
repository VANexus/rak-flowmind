"""契约层测试：序列化、JSON Schema、泛型载荷、trace 工厂。"""
import json
from pydantic import BaseModel
from flowmind.contracts import (
    RuleHit, Evidence, ReasoningChain, ReliabilityMetrics,
    SkillError, SkillResult, new_trace,
)


class _Payload(BaseModel):
    value: int


def test_new_trace_generates_id_and_timestamp():
    tr = new_trace()
    assert tr.trace_id
    assert tr.source == "agent" and tr.target == "flowmind"
    assert "T" in tr.timestamp  # ISO8601


def test_new_trace_passthrough_id():
    tr = new_trace(trace_id="abc-123")
    assert tr.trace_id == "abc-123"


def test_reasoning_chain_four_parts():
    chain = ReasoningChain(
        conclusion="结论",
        triggered_rules=[RuleHit(rule_id="R1", name="规则", expression="x>1", hit=True)],
        evidence=[Evidence(metric="x", value=2, threshold=1, comparison=">")],
        causal_analysis="因为 x>1",
        risk_note="注意波动",
    )
    assert chain.confidence == 1.0
    assert chain.triggered_rules[0].hit is True


def test_skill_result_json_roundtrip_with_generic_payload():
    result = SkillResult[_Payload](
        ok=True, skill="demo", version="0.1.0", trace=new_trace(),
        data=_Payload(value=7),
        reasoning=[],
        metrics=ReliabilityMetrics(latency_ms=1.2, confidence=1.0, sample_size=1),
    )
    dumped = result.model_dump_json()
    parsed = json.loads(dumped)
    assert parsed["ok"] is True
    assert parsed["data"]["value"] == 7
    assert parsed["metrics"]["sample_size"] == 1


def test_skill_result_error_shape():
    result = SkillResult(
        ok=False, skill="demo", version="0.1.0", trace=new_trace(),
        metrics=ReliabilityMetrics(latency_ms=0.0, confidence=0.0, sample_size=0),
        error=SkillError(code="VALIDATION", message="坏参数"),
    )
    assert result.ok is False
    assert result.error.code == "VALIDATION"


def test_output_schema_generatable():
    schema = SkillResult[_Payload].model_json_schema()
    assert schema["type"] == "object"