"""alibaba_product_post 技能测试：走 invoke() 信封层，mock client。"""
from __future__ import annotations

import flowmind.skills.alibaba_product_post as mod
from flowmind.skill import invoke


class _FakeClient:
    def __init__(self, app_key="", app_secret="", session=None, resp=None):
        self.app_key = app_key
        self.app_secret = app_secret
        self.session = session
        self.resp = resp or {}
        self.uploaded = None

    def upload_image(self, url):
        self.uploaded = url
        return url

    def call(self, method, biz_params):
        self.last_method = method
        self.last_biz = biz_params
        return self.resp


def test_post_unauthorized_raises(monkeypatch):
    monkeypatch.setattr(mod, "new_client_from_config", lambda cfg: _FakeClient())
    r = invoke("alibaba_product_post", {"subject": "serum", "image_url": "http://img/1.jpg"})
    assert r.ok is False
    assert "授权" in r.error.message


def test_post_success(monkeypatch):
    fake = _FakeClient(app_key="k", app_secret="s", session="s", resp={"product_id": 123, "str_product_id": "abc"})
    monkeypatch.setattr(mod, "new_client_from_config", lambda cfg: fake)
    r = invoke("alibaba_product_post", {
        "subject": "serum", "keywords": ["skincare"], "description": "d",
        "category_id": 999, "image_url": "http://img/1.jpg",
    })
    assert r.ok is True
    assert r.data.posted is True
    assert r.data.product_id == "123"
    assert fake.uploaded == "http://img/1.jpg"
    assert fake.last_method == "alibaba.icbu.open.product.post"
    # 主图以 image_file_url 直传
    assert fake.last_biz["param_product_post"]["product_image"]["image_file_list"][0]["image_file_url"] == "http://img/1.jpg"


def test_post_missing_subject_validation(monkeypatch):
    monkeypatch.setattr(mod, "new_client_from_config", lambda cfg: _FakeClient(app_key="k", app_secret="s", session="s"))
    r = invoke("alibaba_product_post", {})
    assert r.ok is False
    assert r.error.code == "VALIDATION"