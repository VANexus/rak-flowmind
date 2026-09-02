"""b2b_push_feishu / b2b_push_wecom / b2b_daily_digest 测试：云优先，无 mock 兜底。

去 mock 原则：
- 无 webhook → ok=False + 配置指引（绝不静默成功）；
- webhook POST 打桩在模块级 post_json / invoke（不 mock 业务逻辑）；
- 业务 code/errcode != 0 → 结构化失败；
- digest：榜单 degraded 照常进摘要并标注，绝不返回假数据。
"""
from __future__ import annotations

from types import SimpleNamespace

from flowmind.contracts import (
    ReliabilityMetrics,
    SkillError,
    SkillResult,
    new_trace,
)
from flowmind.skill import invoke
from flowmind.skills._push_common import PushError
import flowmind.skills.b2b_daily_digest as digest_mod
import flowmind.skills.b2b_push_feishu as feishu_mod
import flowmind.skills.b2b_push_wecom as wecom_mod


def _res(data) -> SkillResult:
    return SkillResult(
        ok=True, skill="x", version="0", trace=new_trace(),
        data=data, metrics=ReliabilityMetrics(latency_ms=1.0, confidence=0.9, sample_size=1),
    )


def _err_res(code: str, message: str) -> SkillResult:
    return SkillResult(
        ok=False, skill="x", version="0", trace=new_trace(),
        metrics=ReliabilityMetrics(latency_ms=0.0, confidence=0.0, sample_size=0),
        error=SkillError(code=code, message=message),
    )


# =====================================================================
# 1. b2b_push_feishu
# =====================================================================

def test_feishu_missing_webhook_degrades(monkeypatch):
    monkeypatch.setattr(feishu_mod, "get_api_key", lambda env: None)
    r = invoke("b2b_push_feishu", {"title": "T", "markdown": "M"})
    assert r.ok is True
    assert r.data.ok is False
    assert r.data.webhook_source == "missing"
    assert "FEISHU_WEBHOOK_URL" in r.data.error
    assert "设置 → B 端运营" in r.data.error


def test_feishu_success_via_env(monkeypatch):
    captured: dict = {}

    def _fake_post(url, payload, *, timeout_s):
        captured["url"] = url
        captured["payload"] = payload
        return {"code": 0, "msg": "success"}

    monkeypatch.setattr(feishu_mod, "get_api_key", lambda env: "https://hooks.feishu.example/xxx")
    monkeypatch.setattr(feishu_mod, "post_json", _fake_post)
    r = invoke("b2b_push_feishu", {"title": "日报", "markdown": "**TikTok**\n1. skincare"})
    assert r.ok is True
    assert r.data.ok is True
    assert r.data.webhook_source == "env"
    assert r.data.error is None
    # 卡片结构：interactive + header 标题 + markdown element
    assert captured["payload"]["msg_type"] == "interactive"
    assert captured["payload"]["card"]["header"]["title"]["content"] == "日报"
    assert "skincare" in captured["payload"]["card"]["elements"][0]["content"]


def test_feishu_input_webhook_overrides_env(monkeypatch):
    captured: dict = {}

    def _fake_post(url, payload, *, timeout_s):
        captured["url"] = url
        return {"code": 0}

    monkeypatch.setattr(feishu_mod, "get_api_key", lambda env: "https://env.example")
    monkeypatch.setattr(feishu_mod, "post_json", _fake_post)
    r = invoke("b2b_push_feishu", {
        "title": "T", "markdown": "M", "webhook_url": "https://input.example",
    })
    assert r.data.ok is True
    assert r.data.webhook_source == "input"
    assert captured["url"] == "https://input.example"


def test_feishu_business_code_error(monkeypatch):
    monkeypatch.setattr(feishu_mod, "get_api_key", lambda env: "https://hooks.feishu.example/xxx")
    monkeypatch.setattr(feishu_mod, "post_json", lambda url, payload, *, timeout_s: {"code": 19021, "msg": "签名错误"})
    r = invoke("b2b_push_feishu", {"title": "T", "markdown": "M"})
    assert r.data.ok is False
    assert "19021" in r.data.error and "签名错误" in r.data.error


def test_feishu_http_5xx_retriable(monkeypatch):
    monkeypatch.setattr(feishu_mod, "get_api_key", lambda env: "https://hooks.feishu.example/xxx")

    def _raise(url, payload, *, timeout_s):
        raise PushError("推送 HTTP 502", category="transient", retriable=True)

    monkeypatch.setattr(feishu_mod, "post_json", _raise)
    r = invoke("b2b_push_feishu", {"title": "T", "markdown": "M"})
    assert r.data.ok is False
    assert r.data.retriable is True


# =====================================================================
# 2. b2b_push_wecom
# =====================================================================

def test_wecom_missing_webhook_degrades(monkeypatch):
    monkeypatch.setattr(wecom_mod, "get_api_key", lambda env: None)
    r = invoke("b2b_push_wecom", {"title": "T", "markdown": "M"})
    assert r.data.ok is False
    assert "WECOM_WEBHOOK_URL" in r.data.error


def test_wecom_success_payload(monkeypatch):
    captured: dict = {}

    def _fake_post(url, payload, *, timeout_s):
        captured["payload"] = payload
        return {"errcode": 0, "errmsg": "ok"}

    monkeypatch.setattr(wecom_mod, "get_api_key", lambda env: "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=k")
    monkeypatch.setattr(wecom_mod, "post_json", _fake_post)
    r = invoke("b2b_push_wecom", {"title": "日报", "markdown": "1. water bottle"})
    assert r.data.ok is True
    assert r.data.webhook_source == "env"
    assert captured["payload"]["msgtype"] == "markdown"
    assert "日报" in captured["payload"]["markdown"]["content"]
    assert "water bottle" in captured["payload"]["markdown"]["content"]


def test_wecom_errcode_error(monkeypatch):
    monkeypatch.setattr(wecom_mod, "get_api_key", lambda env: "https://qyapi.example")
    monkeypatch.setattr(
        wecom_mod, "post_json",
        lambda url, payload, *, timeout_s: {"errcode": 93000, "errmsg": "webhook 无效"},
    )
    r = invoke("b2b_push_wecom", {"title": "T", "markdown": "M"})
    assert r.data.ok is False
    assert "93000" in r.data.error


# =====================================================================
# 3. b2b_daily_digest（编排：打桩 invoke，不 mock 数据内容）
# =====================================================================

def _trend_result(platform, kws, degraded=False, source="tikhub", warning=None):
    """构造与 b2b_keyword_trends 兼容的最小 data（字段鸭子类型访问）。"""
    return _res(SimpleNamespace(
        platform=platform, source=source, degraded=degraded,
        failure_category="environment" if degraded else None,
        keywords=[SimpleNamespace(word=w, heat=h, rank=i + 1) for i, (w, h) in enumerate(kws)],
        warning=warning,
    ))


def test_digest_assembles_markdown_and_pushes(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def _fake_invoke(skill_id, raw_args, trace=None):
        calls.append((skill_id, raw_args))
        if skill_id == "b2b_keyword_trends":
            p = raw_args["platform"]
            if p == "tiktok":
                return _trend_result(p, [("skincare", 100), ("bottle", 80)])
            if p == "instagram":
                return _trend_result(p, [], degraded=True, source="degraded(unresolved)")
            return _trend_result(p, [("water bottle", 5)], source="alibaba_hot_sell")
        if skill_id == "b2b_longtail_keywords":
            return _res(SimpleNamespace(
                keywords=[SimpleNamespace(word="bulk skincare sets", category="场景", search_intent="commercial")],
            ))
        if skill_id == "b2b_push_feishu":
            return _res(SimpleNamespace(ok=True, latency_ms=120.0, error=None))
        if skill_id == "b2b_push_wecom":
            return _res(SimpleNamespace(ok=False, latency_ms=0.0, error="webhook 失效"))
        raise AssertionError(f"unexpected skill {skill_id}")

    monkeypatch.setattr(digest_mod, "invoke", _fake_invoke)
    r = invoke("b2b_daily_digest", {"push_feishu": True, "push_wecom": True})
    assert r.ok is True
    plan = r.data

    # 三平台 section：2 真 1 降级
    assert [s.platform for s in plan.sections] == ["tiktok", "instagram", "alibaba"]
    assert plan.sections[0].keywords[0].word == "skincare"
    assert plan.sections[1].degraded is True
    assert plan.sections[1].keywords == []

    # markdown：真实词进榜、降级标注、长尾词进摘要；绝无假数据
    assert "TikTok" in plan.markdown and "skincare" in plan.markdown
    assert "Instagram**：数据源不可达" in plan.markdown
    assert "bulk skincare sets" in plan.markdown

    # 推送结果逐渠道结构化
    assert [(p.channel, p.ok) for p in plan.pushes] == [("feishu", True), ("wecom", False)]
    assert plan.pushes[1].error == "webhook 失效"

    # 编排顺序：趋势 ×3 → 长尾 → 推送 ×2
    ids = [c[0] for c in calls]
    assert ids == [
        "b2b_keyword_trends", "b2b_keyword_trends", "b2b_keyword_trends",
        "b2b_longtail_keywords", "b2b_push_feishu", "b2b_push_wecom",
    ]


def test_digest_push_disabled_skips_invoke(monkeypatch):
    def _fake_invoke(skill_id, raw_args, trace=None):
        if skill_id == "b2b_keyword_trends":
            return _trend_result(raw_args["platform"], [("skincare", 100)])
        if skill_id == "b2b_longtail_keywords":
            return _res(SimpleNamespace(keywords=[]))
        raise AssertionError(f"push skill should not be invoked: {skill_id}")

    monkeypatch.setattr(digest_mod, "invoke", _fake_invoke)
    r = invoke("b2b_daily_digest", {"push_feishu": False, "push_wecom": False})
    assert r.ok is True
    assert r.data.pushes == []


def test_digest_longtail_failure_does_not_block(monkeypatch):
    def _fake_invoke(skill_id, raw_args, trace=None):
        if skill_id == "b2b_keyword_trends":
            return _trend_result(raw_args["platform"], [("skincare", 100)])
        if skill_id == "b2b_longtail_keywords":
            return _err_res("INTERNAL", "未设置 AI_LLM_API_KEY")
        if skill_id == "b2b_push_feishu":
            return _res(SimpleNamespace(ok=True, latency_ms=100.0, error=None))
        raise AssertionError(f"unexpected skill {skill_id}")

    monkeypatch.setattr(digest_mod, "invoke", _fake_invoke)
    r = invoke("b2b_daily_digest", {"push_feishu": True})
    assert r.ok is True
    assert r.data.longtail_words == []
    assert r.data.longtail_error and "AI_LLM_API_KEY" in r.data.longtail_error
    assert r.data.pushes[0].ok is True  # 长尾失败仍推送趋势摘要


def test_digest_all_platforms_degraded_still_pushes(monkeypatch):
    """全平台降级 → 摘要仍编排成功（markdown 标注不可达），绝不造假数据。"""
    def _fake_invoke(skill_id, raw_args, trace=None):
        if skill_id == "b2b_keyword_trends":
            return _trend_result(raw_args["platform"], [], degraded=True)
        raise AssertionError(f"unexpected skill {skill_id}")

    monkeypatch.setattr(digest_mod, "invoke", _fake_invoke)
    r = invoke("b2b_daily_digest", {"push_feishu": False, "push_wecom": False})
    assert r.ok is True
    assert all(s.degraded for s in r.data.sections)
    assert r.data.longtail_words == []  # 无真实热词 → 不调长尾
    assert "数据源不可达" in r.data.markdown
