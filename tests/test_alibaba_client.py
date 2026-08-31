"""_alibaba_client 单元测试：TOP 签名 + 客户端行为。"""
from __future__ import annotations

from flowmind.skills._alibaba_client import (
    AlibabaAPIError,
    AlibabaClient,
    new_client_from_config,
    sign_top_params,
)


class _Resp:
    def __init__(self, status: int, data):
        self.status_code = status
        self._data = data

    def json(self):
        return self._data


class _FakeHttp:
    def __init__(self, status: int, data):
        self._resp = _Resp(status, data)

    def post(self, url, *, data):
        self.last_data = data
        return self._resp


def test_sign_hmac_deterministic_and_uppercase():
    params = {"method": "x", "app_key": "k", "v": "2.0"}
    s1 = sign_top_params(params, "secret", "hmac")
    s2 = sign_top_params(params, "secret", "hmac")
    assert s1 == s2
    assert len(s1) == 32
    assert s1 == s1.upper()


def test_sign_md5_differs_from_hmac():
    params = {"a": "1", "b": "2"}
    assert sign_top_params(params, "secret", "md5") != sign_top_params(params, "secret", "hmac")


def test_call_requires_credentials():
    client = AlibabaClient(api_base="https://x", app_key="", app_secret="")
    try:
        client.call("alibaba.product.list", {})
    except AlibabaAPIError as e:
        assert e.category == "environment"


def test_call_success_and_sign_present():
    http = _FakeHttp(200, {"product_id": 1})
    client = AlibabaClient(
        api_base="https://x", app_key="k", app_secret="s", session="sess",
        client=http,
    )
    out = client.call("alibaba.product.list", {"pageNo": 1})
    assert out == {"product_id": 1}
    assert http.last_data["method"] == "alibaba.product.list"
    assert http.last_data["sign"]
    assert http.last_data["session"] == "sess"


def test_call_raises_on_error_response():
    http = _FakeHttp(200, {"error_response": {"code": 15, "msg": "签名错误"}})
    client = AlibabaClient(
        api_base="https://x", app_key="k", app_secret="s", session="sess",
        client=http,
    )
    try:
        client.call("alibaba.product.list", {})
    except AlibabaAPIError as e:
        assert "签名错误" in str(e)


def test_new_client_from_config_reads_env(monkeypatch):
    from flowmind.skills import _secrets

    monkeypatch.setattr(_secrets, "get_api_key", lambda env: {"ALIBABA_APP_KEY": "k", "ALIBABA_APP_SECRET": "s", "ALIBABA_SESSION": "sess"}.get(env))
    from flowmind.config import AlibabaConfig

    client = new_client_from_config(AlibabaConfig())
    assert client.app_key == "k"
    assert client.session == "sess"