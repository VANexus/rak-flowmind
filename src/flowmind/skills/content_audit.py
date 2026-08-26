"""content_audit 技能：平台合规审计 = 规则库确定性扫描 + LLM 复核。

规则扫描（_content_rules.audit_rules）始终执行，命中即返回，可解释；
LLM 复核（可选，cfg.audit_llm_enabled + 有 key 时）补充规则覆盖不到的风险点——
LLM 复核失败不影响规则结果（降级为 llm_reviewed=False），审计主体不失败。

passed 判定：无 error 级 finding 视为通过（warning 级仅提示）。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import (
    ContentPlatform,
    audit_system,
    audit_user,
    build_chain,
)
from flowmind.skills._content_rules import AuditFinding, audit_rules
from flowmind.skills._llm_client import LLMClientError, llm_json
from flowmind.skills._secrets import get_api_key

_VERSION = "0.1.0"


class ContentAuditInput(BaseModel):
    """审计入参。"""
    platform: ContentPlatform = Field(description="目标平台：xhs / wechat / douyin")
    title: str = Field(default="", max_length=300)
    body: str = Field(default="", max_length=5000)
    tags: list[str] = Field(default_factory=list, max_length=20)


class ContentAuditPlan(BaseModel):
    """审计业务载荷。"""
    platform: str
    passed: bool                       # 无 error 级 finding = 通过
    findings: list[AuditFinding]
    llm_reviewed: bool                 # 是否完成了 LLM 复核
    rule_finding_count: int
    llm_finding_count: int


@skill(id="content_audit", name="平台规则审计", version=_VERSION)
def content_audit(inp: ContentAuditInput) -> SkillOutput[ContentAuditPlan]:
    """对文案做平台合规审计：规则扫描（必做）+ LLM 复核（增强）。

    数据流：规则库确定性扫描 → （可选）云 LLM 复核 → 合并 findings → passed 判定 + 推理链。
    """
    cfg = load_config().content
    findings = audit_rules(inp.platform, inp.title, inp.body, inp.tags)
    rule_count = len(findings)
    llm_count = 0
    llm_reviewed = False

    api_key = get_api_key(cfg.llm_api_key_env) if cfg.audit_llm_enabled else None
    if api_key:
        try:
            reply = llm_json(
                prompt=audit_user(inp.platform, inp.title, inp.body, inp.tags),
                system=audit_system(),
                api_key=api_key,
                api_base=cfg.llm_api_base,
                model=cfg.llm_model,
                max_tokens=cfg.llm_max_tokens,
                timeout_s=cfg.llm_timeout_s,
            )
            llm_count = _merge_llm_findings(findings, reply)
            llm_reviewed = True
        except LLMClientError:
            # LLM 复核失败不回滚规则结果（审计主体仍有效）
            llm_reviewed = False

    passed = not any(f.severity == "error" for f in findings)
    chain = build_chain(
        conclusion=(
            f"{inp.platform} 审计{'通过' if passed else '未通过'}："
            f"{len(findings)} 条 finding（规则 {rule_count} 条"
            + (f"，LLM {llm_count} 条" if llm_reviewed else "，LLM 复核未完成") + "）"
        ),
        causal_analysis=(
            f"规则库扫描命中 {rule_count} 条；"
            f"LLM 复核={'完成' if llm_reviewed else '跳过/失败'}"
        ),
        risk_note="审计为辅助工具，正式发布前建议人工终审；error 级需修改后再发。",
    )
    return SkillOutput(
        data=ContentAuditPlan(
            platform=inp.platform, passed=passed, findings=findings,
            llm_reviewed=llm_reviewed, rule_finding_count=rule_count,
            llm_finding_count=llm_count,
        ),
        reasoning=[chain],
        confidence=0.9 if llm_reviewed else 0.8,
        sample_size=len(findings),
    )


def _merge_llm_findings(existing: list[AuditFinding], reply: dict) -> int:
    """把 LLM 复核的 findings 并入（按 message 去重，最多追加 10 条）。"""
    raw = reply.get("findings")
    if not isinstance(raw, list):
        return 0
    known = {f.message for f in existing}
    added = 0
    for it in raw:
        if not isinstance(it, dict) or added >= 10:
            continue
        message = str(it.get("message") or "").strip()
        if not message or message in known:
            continue
        severity = it.get("severity")
        if severity not in ("error", "warning"):
            severity = "warning"
        category = str(it.get("category") or "advert")
        existing.append(AuditFinding(
            category=category,
            severity=severity,  # type: ignore[arg-type]
            message=message[:200],
            suggestion=str(it.get("suggestion") or "").strip()[:300],
            rule_id="llm",
        ))
        known.add(message)
        added += 1
    return added
