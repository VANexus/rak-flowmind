"""b2b_keyword_trends 技能测试：真实数据源 + degraded 契约。

去 mock 原则：
- adapter 正常 → keywords=真实数据，degraded=False；
- adapter 抛 TrendError environment → degraded=True，keywords=[], failure_category=environment, retriable=False；
- adapter 抛 TrendError transient → degraded=True，retriable=True；
- 非法 platform → VALIDATION 错误；
- 绝无 seed 演示数据兜底。
"""
from __future__ import annotations

import flowmind.skills.b2b_keyword_trends as mod
from flowmind.skill import invoke
from flowmind.skills._trend_adapters import TrendError


class _FakeAdapter:
    def __init__(self, items=None, exc=None, name="fake", session_cookie=""):
        self.items = items or []
        self.exc = exc
        self.name = name
        self.session_cookie = session_cookie

    def fetch(self, platform, *, industry_id=None, limit=20, keyword=None):
        if self.exc:
            raise self.exc
        return self.items


# =====================================================================
# 1. 正常数据路径
# =====================================================================

def test_trends_real_happy(monkeypatch):
    seen: dict = {}

    def _res(platform, cfg, **kwargs):
        seen.update(kwargs)
        return _FakeAdapter(
            items=[{"word": "dollyparton", "heat": 100, "delta": 5, "rank": 1, "industry": "通用", "source": "tikhub"}],
            name="tikhub",
            session_cookie=kwargs.get("session_cookie", ""),
        )

    monkeypatch.setattr(mod, "resolve_adapter", _res)
    r = invoke("b2b_keyword_trends", {"platform": "tiktok", "session_cookie": "sessionid=abc"})
    assert r.ok is True
    assert r.metrics.degraded is False
    assert len(r.data.keywords) == 1
    assert r.data.keywords[0].word == "dollyparton"
    assert r.data.keywords[0].source == "tikhub"
    assert r.data.warning is None
    assert seen["session_cookie"] == "sessionid=abc"


def test_trends_empty_real_but_ok(monkeypatch):
    """adapter 正常但空 → degraded=False，keywords=[]。"""
    monkeypatch.setattr(mod, "resolve_adapter", lambda p, cfg, **kwargs: _FakeAdapter(items=[], name="tikhub"))
    r = invoke("b2b_keyword_trends", {"platform": "tiktok"})
    assert r.ok is True
    assert r.metrics.degraded is False
    assert r.data.keywords == []


# =====================================================================
# 2. 错误 → degraded=True，keywords=[]，绝无 seed 兜底
# =====================================================================

def test_trends_degraded_environment_returns_empty_keywords(monkeypatch):
    exc = TrendError("未配置 AI_TRENDS_API_KEY", category="environment", retriable=False)
    monkeypatch.setattr(mod, "resolve_adapter", lambda p, cfg, **kwargs: _FakeAdapter(exc=exc))
    r = invoke("b2b_keyword_trends", {"platform": "tiktok"})
    assert r.ok is True
    assert r.metrics.degraded is True
    assert r.data.degraded is True
    assert r.data.failure_category == "environment"
    assert r.data.retriable is False
    assert r.data.keywords == []  # 云优先：空数组，绝不 fallback seed
    # TikTok 主路径走 TikHub：引导修复配置/网络，而非渠道登录
    assert r.data.warning and "AI_TRENDS_API_KEY" in r.data.warning
    assert "设置 → B 端运营" not in r.data.warning


def test_trends_degraded_instagram_guides_config_not_channel_login(monkeypatch):
    """IG 主路径走 TikHub：失败时引导修复配置，而非渠道登录。"""
    exc = TrendError("未配置 AI_TRENDS_API_KEY", category="environment", retriable=False)
    monkeypatch.setattr(mod, "resolve_adapter", lambda p, cfg, **kwargs: _FakeAdapter(exc=exc))
    r = invoke("b2b_keyword_trends", {"platform": "instagram", "keyword": "x"})
    assert r.data.degraded is True
    assert r.data.warning and "AI_TRENDS_API_KEY" in r.data.warning
    assert "渠道授权" not in r.data.warning


def test_trends_degraded_transient_retriable(monkeypatch):
    exc = TrendError("500", category="transient", retriable=True)
    monkeypatch.setattr(mod, "resolve_adapter", lambda p, cfg, **kwargs: _FakeAdapter(exc=exc, name="tikhub"))
    r = invoke("b2b_keyword_trends", {"platform": "tiktok"})
    assert r.ok is True
    assert r.data.degraded is True
    assert r.data.failure_category == "transient"
    assert r.data.retriable is True
    assert r.data.keywords == []


def test_trends_instagram_without_key_degrades(monkeypatch):
    """走真实 resolve_adapter：instagram 未配 AI_TRENDS_API_KEY → degraded=[]（无种子，无网络）。"""
    monkeypatch.setattr("flowmind.skills._secrets.get_api_key", lambda env: "")
    r = invoke("b2b_keyword_trends", {"platform": "instagram", "keyword": "skincare"})
    assert r.ok is True
    assert r.data.degraded is True
    assert r.data.failure_category == "environment"
    assert "AI_TRENDS_API_KEY" in r.data.warning
    assert r.data.keywords == []


def test_trends_instagram_without_keyword_degrades():
    """走真实 resolve_adapter：IG 缺关键词 → degraded=[]（TikHub 话题搜索必填关键词）。"""
    r = invoke(
        "b2b_keyword_trends",
        {"platform": "instagram", "session_cookie": "sessionid=x; csrftoken=y"},
    )
    assert r.ok is True
    assert r.data.degraded is True
    assert r.data.failure_category == "unknown"
    assert r.data.keywords == []


def test_trends_alibaba_unauthorized_degrades(monkeypatch):
    """走真实 resolve_adapter：alibaba 未授权 → degraded=[]（云优先，无种子）。"""
    monkeypatch.setattr("flowmind.skills._secrets.get_api_key", lambda env: "")
    r = invoke("b2b_keyword_trends", {"platform": "alibaba"})
    assert r.ok is True
    assert r.data.degraded is True
    assert r.data.failure_category == "environment"
    assert r.data.keywords == []


# =====================================================================
# 3. 入参校验
# =====================================================================

def test_trends_invalid_platform_validation():
    r = invoke("b2b_keyword_trends", {"platform": "weibo"})
    assert r.ok is False
    assert r.error.code == "VALIDATION"


def test_trends_limit_bound_validation():
    r = invoke("b2b_keyword_trends", {"platform": "tiktok", "limit": 0})
    assert r.ok is False
    assert r.error.code == "VALIDATION"
    r2 = invoke("b2b_keyword_trends", {"platform": "tiktok", "limit": 999})
    assert r2.ok is False
    assert r2.error.code == "VALIDATION"
