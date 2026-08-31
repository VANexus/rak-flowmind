"""b2b_push_feishu 技能：飞书自定义机器人推送（交互式卡片）。

webhook URL 优先取入参（供前端「测试推送」即时校验），缺省读环境变量
FEISHU_WEBHOOK_URL。失败走 degraded SkillOutput（ok=False + 结构化错误），
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


class FeishuPushInput(BaseModel):
    """飞书推送入参。"""
    title: str = Field(min_length=1, max_length=100, description="卡片标题")
    markdown: str = Field(min_length=1, max_length=9000, description="markdown 正文（卡片 element）")
    webhook_url: str | None = Field(
        default=None, max_length=500,
        description="飞书自定义机器人 webhook；缺省读环境变量 FEISHU_WEBHOOK_URL",
    )


class FeishuPushPlan(BaseModel):
    """推送结果业务载荷。"""
    ok: bool
    latency_ms: float
    webhook_source: str  # input / env / missing
    error: str | None = None
    retriable: bool = False


@skill(id="b2b_push_feishu", name="飞书趋势推送", version=_VERSION)
def b2b_push_feishu(inp: FeishuPushInput) -> SkillOutput[FeishuPushPlan]:
    """推送趋势卡片到飞书群（自定义机器人 interactive 卡片）。

    数据流：解析 webhook（入参 > env）→ POST interactive 卡片 → 校验业务 code →
    成功返回 {ok, latency_ms}；任何失败返回 ok=False + 结构化错误（不抛裸异常）。
    """
    cfg = load_config().b2b_push
    webhook = (inp.webhook_url or "").strip() or get_api_key(cfg.feishu_webhook_url_env) or ""
    webhook_source = "input" if (inp.webhook_url or "").strip() else ("env" if webhook else "missing")
    if not webhook:
        chain = build_chain(
            conclusion="飞书推送跳过：无 webhook",
            causal_analysis=f"入参未传 webhook_url 且环境变量 {cfg.feishu_webhook_url_env} 未设置",
            risk_note="请在「设置 → B 端运营」配置飞书机器人 webhook 后重试。",
        )
        return SkillOutput(
            data=FeishuPushPlan(ok=False, latency_ms=0.0, webhook_source="missing",
                                error=f"未设置 {cfg.feishu_webhook_url_env}，请在「设置 → B 端运营」配置飞书机器人 webhook"),
            reasoning=[chain], confidence=0.0, sample_size=0,
            degraded=True, degradation_reason="environment",
        )

    payload = {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {"title": {"tag": "plain_text", "content": inp.title}, "template": "blue"},
            "elements": [{"tag": "markdown", "content": inp.markdown}],
        },
    }

    start = perf_counter()
    try:
        body = post_json(webhook, payload, timeout_s=cfg.webhook_timeout_s)
    except PushError as exc:
        latency = (perf_counter() - start) * 1000.0
        chain = build_chain(
            conclusion="飞书推送失败",
            causal_analysis=f"POST 飞书 webhook → {type(exc).__name__}（{exc.category}）",
            risk_note="推送失败已结构化返回，请检查 webhook 地址与网络后重试。",
        )
        return SkillOutput(
            data=FeishuPushPlan(ok=False, latency_ms=latency, webhook_source=webhook_source,
                                error=str(exc), retriable=exc.retriable),
            reasoning=[chain], confidence=0.0, sample_size=0,
            degraded=True, degradation_reason=exc.category,
        )
    latency = (perf_counter() - start) * 1000.0

    # 飞书业务错误：HTTP 200 但 body.code != 0（如签名错误 / 频控）
    code = body.get("code", body.get("StatusCode", 0))
    if code not in (0, "0"):
        msg = body.get("msg") or body.get("StatusMessage") or "未知业务错误"
        chain = build_chain(
            conclusion="飞书推送被拒",
            causal_analysis=f"webhook 返回业务 code={code}：{msg}",
            risk_note="常见原因：机器人被移除、关键词安全设置不匹配、频控限流。",
        )
        return SkillOutput(
            data=FeishuPushPlan(ok=False, latency_ms=latency, webhook_source=webhook_source,
                                error=f"飞书业务错误 code={code}：{msg}"),
            reasoning=[chain], confidence=0.0, sample_size=0,
            degraded=True, degradation_reason="video",
        )

    chain = build_chain(
        conclusion=f"飞书卡片推送成功（{latency:.0f}ms）",
        causal_analysis=f"POST 飞书 webhook（{webhook_source}）→ code=0",
        risk_note="卡片内容为当日趋势摘要，仅推送到群，不落库。",
    )
    return SkillOutput(
        data=FeishuPushPlan(ok=True, latency_ms=latency, webhook_source=webhook_source),
        reasoning=[chain], confidence=0.95, sample_size=1,
    )
