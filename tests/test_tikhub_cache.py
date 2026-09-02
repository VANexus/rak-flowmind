"""TikHub 响应缓存 + 投机学习单测（打桩 http，不打真实网络）。

覆盖：
- soft_ttl 内两次同参调用只外呼 1 次（local 命中）
- soft_ttl 过期后转 live 真实外呼（无学习证据时永不投机）
- x-cache-ttl / x-cache-expiration / x-cache: hit 三种头学习 → speculative
- 外呼失败回落本地缓存（local_fallback，宁旧勿空）
- 不同参数不同缓存键
"""
from __future__ import annotations

import time

import pytest

from flowmind.skills._tikhub_cache import TikHubCache, get_last_cache_meta
from flowmind.skills._tikhub_client import TikHubClient, TikHubError


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = ""
        self.headers = headers or {}

    def json(self):
        return self._payload


class _FakeHttp:
    """固定响应的打桩 http.Client；记录每次调用以便断言外呼次数。"""

    def __init__(self, response: _FakeResponse | Exception):
        self.response = response
        self.calls: list[dict] = []

    def request(self, method, url, *, json=None, params=None, headers=None):
        self.calls.append({"method": method, "url": url, "json": json, "params": params})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _envelope(data) -> dict:
    return {"code": 200, "message": "ok", "data": data}


@pytest.fixture()
def cache(tmp_path):
    return TikHubCache(
        db_path=str(tmp_path / "tikhub_cache.db"),
        default_soft_ttl_s=60.0,
        max_free_window_s=3600.0,
    )


def _client(http, cache) -> TikHubClient:
    return TikHubClient(
        api_base="https://api.tikhub.test", api_key="k-test", timeout_s=5.0,
        client=http, cache=cache,
    )


def _age_entries(cache: TikHubCache, seconds: float) -> None:
    """把已有缓存条目人为变旧（绕开真实等待）。"""
    cache._conn.execute("UPDATE entries SET fetched_at = fetched_at - ?", (seconds,))
    cache._conn.commit()


# ── soft_ttl：窗口内直接回本地，绝不外呼 ────────────────────────────


def test_soft_ttl_serves_local_within_window(cache) -> None:
    http = _FakeHttp(_FakeResponse(_envelope({"items": [1]})))
    client = _client(http, cache)

    first = client.get("/api/v1/x", params={"a": 1})
    second = client.get("/api/v1/x", params={"a": 1})

    assert first == second == _envelope({"items": [1]})
    assert len(http.calls) == 1  # 第二次零外呼
    assert get_last_cache_meta() == {"mode": "local", "hit": True, "age_s": 0.0}


def test_different_params_different_key(cache) -> None:
    http = _FakeHttp(_FakeResponse(_envelope({"items": [1]})))
    client = _client(http, cache)

    client.get("/api/v1/x", params={"a": 1})
    client.get("/api/v1/x", params={"a": 2})

    assert len(http.calls) == 2


# ── 过期后 live：无学习证据时永不投机 ──────────────────────────────


def test_expired_goes_live_without_learning(cache) -> None:
    http = _FakeHttp(_FakeResponse(_envelope({"items": [1]})))
    client = _client(http, cache)

    client.get("/api/v1/x", params={"a": 1})
    _age_entries(cache, 120.0)  # 超过 soft_ttl=60
    client.get("/api/v1/x", params={"a": 1})

    assert len(http.calls) == 2
    assert get_last_cache_meta()["mode"] == "live"
    assert cache.learned_window("/api/v1/x") is None  # 无头证据 → 永远保守


# ── 投机学习：三种头证据 ───────────────────────────────────────────


def test_learn_ttl_header_enables_speculative(cache) -> None:
    http = _FakeHttp(_FakeResponse(_envelope({"items": [1]}), headers={"X-Cache-TTL": "3600"}))
    client = _client(http, cache)

    client.get("/api/v1/x", params={"a": 1})
    assert cache.learned_window("/api/v1/x") == 3600.0

    _age_entries(cache, 120.0)  # 过了 soft_ttl 但在免费窗内 → 投机外呼
    client.get("/api/v1/x", params={"a": 1})
    assert len(http.calls) == 2
    assert get_last_cache_meta()["mode"] == "speculative"


def test_learn_expire_header_window_clamped(cache) -> None:
    headers = {"X-Cache-Expiration": str(time.time() + 1200)}
    http = _FakeHttp(_FakeResponse(_envelope({"items": [1]}), headers=headers))
    client = _client(http, cache)

    client.get("/api/v1/x")
    window = cache.learned_window("/api/v1/x")
    assert window is not None and 1100 <= window <= 1200


def test_learn_hit_header_uses_max_window(cache) -> None:
    http = _FakeHttp(_FakeResponse(_envelope({"items": [1]}), headers={"X-Cache": "Hit"}))
    client = _client(http, cache)

    client.get("/api/v1/x")
    assert cache.learned_window("/api/v1/x") == 3600.0  # max_free_window_s


def test_no_headers_never_learns(cache) -> None:
    http = _FakeHttp(_FakeResponse(_envelope({"items": [1]})))
    client = _client(http, cache)

    client.get("/api/v1/x")
    assert cache.learned_window("/api/v1/x") is None


# ── 失败回落：宁给旧数据不给空态 ───────────────────────────────────


def test_failure_falls_back_to_stale_cache(cache) -> None:
    ok = _FakeResponse(_envelope({"items": [1]}))
    http = _FakeHttp(ok)
    client = _client(http, cache)

    client.get("/api/v1/x", params={"a": 1})
    _age_entries(cache, 120.0)

    http.response = httpx_exc = TikHubError("TikHub 余额不足 HTTP 402", category="environment")
    payload = client.get("/api/v1/x", params={"a": 1})

    assert payload == _envelope({"items": [1]})  # 旧缓存兜底
    meta = get_last_cache_meta()
    assert meta["mode"] == "local_fallback" and meta["hit"] is True and meta["age_s"] >= 120


def test_failure_without_cache_raises(cache) -> None:
    http = _FakeHttp(TikHubError("boom", category="environment"))
    client = _client(http, cache)

    with pytest.raises(TikHubError):
        client.get("/api/v1/x")


# ── 缓存键：参数顺序无关 ───────────────────────────────────────────


def test_make_key_order_insensitive() -> None:
    a = TikHubCache.make_key("GET", "/x", None, {"b": 2, "a": 1})
    b = TikHubCache.make_key("GET", "/x", None, {"a": 1, "b": 2})
    assert a == b
