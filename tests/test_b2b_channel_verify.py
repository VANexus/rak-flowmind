"""b2b_channel_verify 技能测试：走 invoke() 信封层，mock httpx 响应。"""
from __future__ import annotations

import flowmind.skills.b2b_channel_verify as mod
from flowmind.skill import invoke


class _FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def _probe(monkeypatch, resp):
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None, follow_redirects=False):
        captured["url"] = url
        captured["headers"] = headers or {}
        return resp

    monkeypatch.setattr(mod.httpx, "get", fake_get)
    return captured


def test_verify_missing_sessionid_is_expired():
    r = invoke("b2b_channel_verify", {"platform": "tiktok", "cookie": "tt-target=ct; other=x"})
    assert r.ok is True
    assert r.data.status == "expired"
    assert "sessionid" in r.data.message


def test_verify_tiktok_active(monkeypatch):
    captured = _probe(
        monkeypatch,
        _FakeResp(200, {"data": {"user": {"nickname": "MyShop", "unique_id": "myshop"}}}),
    )
    r = invoke("b2b_channel_verify", {"platform": "tiktok", "cookie": "sessionid=abc; tt-target=ct"})
    assert r.ok is True
    assert r.data.status == "active"
    assert r.data.account == "MyShop"
    assert "sessionid=abc" in captured["headers"]["Cookie"]


def test_verify_instagram_expired_401(monkeypatch):
    _probe(monkeypatch, _FakeResp(401))
    r = invoke("b2b_channel_verify", {"platform": "instagram", "cookie": "sessionid=dead; csrftoken=t"})
    assert r.ok is True
    assert r.data.status == "expired"


def test_verify_redirect_means_expired(monkeypatch):
    _probe(monkeypatch, _FakeResp(302))
    r = invoke("b2b_channel_verify", {"platform": "instagram", "cookie": "sessionid=dead"})
    assert r.data.status == "expired"


def test_verify_rate_limit_is_risk_control(monkeypatch):
    _probe(monkeypatch, _FakeResp(429))
    r = invoke("b2b_channel_verify", {"platform": "tiktok", "cookie": "sessionid=x"})
    assert r.data.status == "risk_control"
