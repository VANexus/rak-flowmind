"""alibaba_product_list 技能测试：走 invoke() 信封层，mock client。"""
from __future__ import annotations

import flowmind.skills.alibaba_product_list as mod
from flowmind.skill import invoke


class _FakeClient:
    def __init__(self, app_key="", app_secret="", session=None, resp=None):
        self.app_key = app_key
        self.app_secret = app_secret
        self.session = session
        self.resp = resp or {}

    def call(self, method, biz_params):
        return self.resp


def test_product_list_unauthorized_degraded(monkeypatch):
    monkeypatch.setattr(mod, "new_client_from_config", lambda cfg: _FakeClient())
    r = invoke("alibaba_product_list", {})
    assert r.ok is True
    assert r.metrics.degraded is True
    assert r.data.authorized is False
    assert r.data.products == []
    assert "授权" in (r.data.warning or "")


def test_product_list_normalizes(monkeypatch):
    fake = _FakeClient(
        app_key="k", app_secret="s", session="sess",
        resp={"products": [
            {"productId": 1, "subject": "skincare set", "keywords": ["skincare", "cream"],
             "image": "http://img/1.jpg", "price": "5-8"},
        ]},
    )
    monkeypatch.setattr(mod, "new_client_from_config", lambda cfg: fake)
    r = invoke("alibaba_product_list", {})
    assert r.ok is True
    assert r.metrics.degraded is False
    assert r.data.authorized is True
    assert r.data.total == 1
    assert r.data.products[0].subject == "skincare set"
    assert r.data.products[0].keywords == ["skincare", "cream"]


def test_product_list_api_failure_degraded(monkeypatch):
    from flowmind.skills._alibaba_client import AlibabaAPIError

    class _Boom(_FakeClient):
        def call(self, method, biz_params):
            raise AlibabaAPIError("接口错误", category="video", retriable=False)

    monkeypatch.setattr(mod, "new_client_from_config", lambda cfg: _Boom(app_key="k", app_secret="s", session="s"))
    r = invoke("alibaba_product_list", {})
    assert r.ok is True
    assert r.metrics.degraded is True
    assert r.data.products == []