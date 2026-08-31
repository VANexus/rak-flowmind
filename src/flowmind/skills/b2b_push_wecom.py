"""b2b_push_wecom 技能：企业微信群机器人推送（markdown 消息）。

webhook URL 优先取入参（供前端「测试推送」即时校验），缺省读环境变量
WECOM_WEBHOOK_URL。失败走 degraded SkillOutput（ok=False + 结构化错误），
绝不静默成功、绝无 mock。
"""
from __future__ import annotations

from time import perf_counter

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import build_chain
from flowmind.skills._push_common import PushError, post_json
from flowmind.skills._secrets import get_api_key

_VERSION = "0.1.0"


class WecomPushInput(BaseModel):
    """企微推送入参。"""
    title: str = Field(min_length=1, max_length=100, description="消息标题")
    markdown: str = Field(min_length=1, max_length=4000, description="markdown 正文（企微上限 4096 字节）")
    webhook_url: str | None = Field(
        default=None, max_length=500,
        description="企微群机器人 webhook；缺省读环境变量 WECOM_WEBHOOK_URL",
    )


class WecomPushPlan(BaseModel):
    """推送结果业务载荷。"""
    ok: bool
    latency_ms: float
    webhook_source: str  # input / env / missing
    error: str | None = None
    retriable: bool = False


@skill(id="b2b_push_wecom", name="企微趋势推送", version=_VERSION)
def b2b_push_wecom(inp: WecomPushInput) -> SkillOutput[WecomPushPlan]:
    """推送趋势摘要到企业微信群（群机器人 markdown 消息）。

    数据流：解析 webhook（入参 > env）→ POST markdown 消息 → 校验 errcode →
    成功返回 {ok, latency_ms}；任何失败返回 ok=False + 结构化错误（不抛裸异常）。
    """
    cfg = load_config().b2b_push
    webhook = (inp.webhook_url or "").strip() or get_api_key(cfg.wecom_webhook_url_env) or ""
    webhook_source = "input" if (inp.webhook_url or "").strip() else ("env" if webhook else "missing")
    if not webhook:
        chain = build_chain(
            conclusion="企微推送跳过：无 webhook",
            causal_analysis=f"入参未传 webhook_url 且环境变量 {cfg.wecom_webhook_url_env} 未设置",
            risk_note="请在「设置 → B 端运营」配置企微群机器人 webhook 后重试。",
        )
        return SkillOutput(
            data=WecomPushPlan(ok=False, latency_ms=0.0, webhook_source="missing",
                               error=f"未设置 {cfg.wecom_webhook_url_env}，请在「设置 → B 端运营」配置企微群机器人 webhook"),
            reasoning=[chain], confidence=0.0, sample_size=0,
            degraded=True, degradation_reason="environment",
        )

    payload = {
        "msgtype": "markdown",
        "markdown": {"content": f"## {inp.title}\n{inp.markdown}"},
    }

    start = perf_counter()
    try:
        body = post_json(webhook, payload, timeout_s=cfg.webhook_timeout_s)
    except PushError as exc:
        latency = (perf_counter() - start) * 1000.0
        chain = build_chain(
            conclusion="企微推送失败",
            causal_analysis=f"POST 企微 webhook → {type(exc).__name__}（{exc.category}）",
            risk_note="推送失败已结构化返回，请检查 webhook 地址与网络后重试。",
        )
        return SkillOutput(
            data=WecomPushPlan(ok=False, latency_ms=latency, webhook_source=webhook_source,
                               error=str(exc), retriable=exc.retriable),
            reasoning=[chain], confidence=0.0, sample_size=0,
            degraded=True, degradation_reason=exc.category,
        )
    latency = (perf_counter() - start) * 1000.0

    # 企微业务错误：HTTP 200 但 errcode != 0（如 webhook 无效 / 频控）
    errcode = body.get("errcode", 0)
    if errcode not in (0, "0"):
        errmsg = body.get("errmsg") or "未知业务错误"
        chain = build_chain(
            conclusion="企微推送被拒",
            causal_analysis=f"webhook 返回 errcode={errcode}：{errmsg}",
            risk_note="常见原因：webhook 地址失效、机器人被移除、频控限流（20 条/分钟）。",
        )
        return SkillOutput(
            data=WecomPushPlan(ok=False, latency_ms=latency, webhook_source=webhook_source,
                               error=f"企微业务错误 errcode={errcode}：{errmsg}"),
            reasoning=[chain], confidence=0.0, sample_size=0,
            degraded=True, degradation_reason="video",
        )

    chain = build_chain(
        conclusion=f"企微 markdown 推送成功（{latency:.0f}ms）",
        causal_analysis=f"POST 企微 webhook（{webhook_source}）→ errcode=0",
        risk_note="消息内容为当日趋势摘要，仅推送到群，不落库。",
    )
    return SkillOutput(
        data=WecomPushPlan(ok=True, latency_ms=latency, webhook_source=webhook_source),
        reasoning=[chain], confidence=0.95, sample_size=1,
    )
