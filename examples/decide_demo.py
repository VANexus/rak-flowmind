"""_decide 决策模型客户端演示 —— System One（Jev 云 / Laya 本地 sidecar）。

运行：PYTHONPATH=<repo>/src conda run -n flowmind python examples/decide_demo.py

展示：
1. 后端解析（auto/local/cloud 三态 + 两端皆无的显式失败）
2. happy path：本地 Laya sidecar 一次请求求值 choice/score/noul 三原语
3. 云回落：本地未配置但 TypeSafe key 已配置 → auto 选 cloud
4. 入参违约：choice 的 criteria 为空 / score 级别数越界 → DecideError（本地拦掉）
5. error：HTTP 500 → transient 可重试；HTTP 401 → environment 不可重试
6. error：两端均未配置 → environment 显式失败（不静默降级）

mock 方式：用 shim 替换 _decide.requests（不污染全局 requests），
不依赖真实 Laya sidecar / TypeSafe 网络。
"""

from __future__ import annotations

import os

import requests as _real_requests

import flowmind.skills._decide as d
from flowmind.config import ensure_env_loaded


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ── env 与传输替身 ─────────────────────────────────────────────────

_ENV_KEYS = ("FLOWMIND_DECIDE_BACKEND", "FLOWMIND_DECIDE_BASE_URL",
             "TYPESAFE_API_KEY", "TYPESAFE_BASE_URL")


def _set_env(**kw) -> dict:
    """设置决策相关 env，返回可恢复的旧值快照。"""
    old = {k: os.environ.get(k) for k in _ENV_KEYS}
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    for k, v in kw.items():
        if v is not None:
            os.environ[k] = v
    return old


def _restore_env(old: dict) -> None:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class _Resp:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = str(self._payload)

    def json(self):
        return self._payload


class _Shim:
    """requests 替身：只替换 post，exceptions 透传真实模块。"""

    exceptions = _real_requests.exceptions

    def __init__(self, handler):
        self._handler = handler
        self.calls: list[dict] = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json,
                           "headers": headers or {}, "timeout": timeout})
        return self._handler(url, json)


def _install(handler) -> _Shim:
    shim = _Shim(handler)
    d.requests = shim
    return shim


_ANSWERS = {
    "segment_is_speech": {"type": "noul", "noul": 0.94},
    # Laya 真实形状：confidence 未校准（温度表越界被钳到 0.5，见 _decide.answer_confidence），
    # answer_confidence 才是可靠的最大类概率。
    "region_kind": {
        "type": "choice", "choice": "subtitle",
        "probabilities": {"subtitle": 0.88, "logo": 0.07, "watermark": 0.05},
        "confidence": 0.0364, "answer_confidence": 0.88,
    },
    # TypeSafe 云形状：无 answer_confidence，只有校准过的 confidence → 回落取它。
    "translation_quality": {
        "type": "score", "score": 2.0,
        "probabilities": [0.05, 0.15, 0.80], "confidence": 0.80,
    },
}


def _ok_handler(url, payload):
    return _Resp(200, {"model": "laya-multilingual", "answers": _ANSWERS,
                       "usage": {"routing": {"model": "multilingual"}}})


_QUESTIONS = {
    "segment_is_speech": {
        "type": "noul",
        "instructions": "该分段是否为有效语音（而非纯音乐/噪声）？",
    },
    "region_kind": {
        "type": "choice",
        "instructions": "该检测区域属于哪一类？",
        "criteria": {"subtitle": "底部居中硬字幕", "logo": "角标台标",
                     "watermark": "半透明水印"},
    },
    "translation_quality": {
        "type": "score",
        "instructions": "译文口语化与长度控制是否达标？",
        "criteria": ["明显不合格", "勉强可用", "自然且长度合适"],
    },
}


def main() -> None:
    # 先触发 .env 加载：之后 pop 环境变量才会真正生效（load_dotenv 不覆盖已加载值，
    # 但首次 get_api_key 会从 .env 注入 TYPESAFE_API_KEY，必须先加载再 pop）
    ensure_env_loaded()

    old = _set_env()
    try:
        section("1) 后端解析：两端皆未配置 → 显式失败（不静默降级）")
        try:
            d.resolve_backend()
            raise AssertionError("未配置时应抛 DecideError")
        except d.DecideError as exc:
            print(f"  category : {exc.category}  retriable={exc.retriable}")
            print(f"  message  : {exc}")
            check(exc.category == "environment", "未配置应归 environment")
            check(d.is_configured() is False, "is_configured 应为 False")

        section("2) 后端解析：auto 三态（本地优先 → 云回落 → 显式报错）")
        _set_env(FLOWMIND_DECIDE_BASE_URL="http://127.0.0.1:31998",
                 TYPESAFE_API_KEY="k-123")
        check(d.resolve_backend() == "local", "auto + 本地已配置 → local（本地优先）")
        _set_env(TYPESAFE_API_KEY="k-123")
        check(d.resolve_backend() == "cloud", "auto + 仅云 key → cloud")
        _set_env(FLOWMIND_DECIDE_BASE_URL="http://127.0.0.1:31998")
        check(d.resolve_backend() == "local", "auto + 仅本地 → local")
        _set_env(FLOWMIND_DECIDE_BACKEND="local")
        try:
            d.resolve_backend()
            raise AssertionError("强制 local 但未配置地址应抛错")
        except d.DecideError as exc:
            check("local" in str(exc), "报错应指明 local 缺地址")
        _set_env(FLOWMIND_DECIDE_BACKEND="bogus")
        try:
            d.resolve_backend()
            raise AssertionError("非法开关值应抛错")
        except d.DecideError as exc:
            check(exc.category == "environment", "非法开关归 environment")
        print("  auto/local/cloud 三态与非法值均按预期")

        section("3) Happy path：本地 Laya sidecar 一次请求求值三原语")
        _set_env(FLOWMIND_DECIDE_BASE_URL="http://127.0.0.1:31998")
        shim = _install(_ok_handler)
        res = d.decide(
            {"segment": {"text": "本产品支持一键跨语言配音", "start": 42.5}},
            _QUESTIONS)
        print(f"  backend  : {res['backend']}")
        print(f"  model    : {res['model']}")
        for qid, ans in res["answers"].items():
            print(f"    • {qid}: {ans}")
        check(res["backend"] == "local", "应走本地后端")
        check(len(shim.calls) == 1, "三个问题应合并在一次请求内")
        call = shim.calls[0]
        check(call["url"] == "http://127.0.0.1:31998/decide", "应打到 sidecar /decide")
        check(call["headers"] == {}, "本地后端不带鉴权头")
        check(call["json"]["questions"] == _QUESTIONS, "questions 应原样透传")
        check(d.answer_confidence(res["answers"]["region_kind"]) == 0.88,
              "choice 应优先取 answer_confidence（Laya 的 confidence 未校准）")
        check(d.answer_confidence(res["answers"]["translation_quality"]) == 0.80,
              "无 answer_confidence 时回落 confidence（TypeSafe 云形状）")
        check(d.answer_confidence(res["answers"]["segment_is_speech"]) == 1.0,
              "noul 无独立置信度字段 → 1.0")

        section("4) 云回落：本地未配置但 TypeSafe key 已配置 → cloud")
        _set_env(TYPESAFE_API_KEY="ts-key-abc")
        shim = _install(lambda url, payload: _Resp(
            200, {"model": "jev-1.13.0", "answers": _ANSWERS,
                  "usage": {"input_tokens": 296, "output_tokens": 20}}))
        res = d.decide("Help! My payouts have been failing for 3 days.", _QUESTIONS)
        print(f"  backend  : {res['backend']}   model: {res['model']}")
        print(f"  usage    : {res['usage']}")
        check(res["backend"] == "cloud", "auto 应回落 cloud")
        check(shim.calls[0]["url"] == d.TYPESAFE_URL, "应打到 TypeSafe /v1/systemone")
        check(shim.calls[0]["headers"]["Authorization"] == "Bearer ts-key-abc",
              "云后端应带 Bearer 鉴权")

        section("4b) 云端点覆盖：TYPESAFE_BASE_URL 指向中转网关")
        _set_env(TYPESAFE_API_KEY="relay-key",
                 TYPESAFE_BASE_URL="https://relay.example.com")
        check(d._cloud_url() == "https://relay.example.com/v1/systemone",
              "填基址应自动补 /v1/systemone")
        _set_env(TYPESAFE_API_KEY="relay-key",
                 TYPESAFE_BASE_URL="https://relay.example.com/v1/systemone")
        check(d._cloud_url() == "https://relay.example.com/v1/systemone",
              "填完整端点应原样使用")
        _set_env(TYPESAFE_API_KEY="relay-key")
        check(d._cloud_url() == d.TYPESAFE_URL, "未覆盖时回落 TypeSafe 官方端点")
        print(f"  未覆盖 → {d.TYPESAFE_URL}")

        section("5) 入参违约：本地拦掉契约违规（不白跑一次远程调用）")
        shim = _install(_ok_handler)
        cases = [
            ({}, "空 questions"),
            ({"q": {"type": "noul"}}, "缺 instructions"),
            ({"q": {"type": "bogus", "instructions": "x"}}, "非法 type"),
            ({"q": {"type": "choice", "instructions": "x", "criteria": {}}},
             "choice criteria 为空"),
            ({"q": {"type": "score", "instructions": "x", "criteria": ["仅一级"]}},
             "score 级别数不足"),
        ]
        for qs, label in cases:
            try:
                d.decide("state", qs)
                raise AssertionError(f"{label} 应抛 DecideError")
            except d.DecideError as exc:
                print(f"  • {label}: {exc}")
        check(shim.calls == [], "入参违约应在本地拦掉，零远程调用")

        section("6) error：HTTP 500 → transient 可重试；401 → environment 不可重试")
        _install(lambda url, payload: _Resp(500, {"error": "boom"}))
        try:
            d.decide("state", _QUESTIONS)
            raise AssertionError("500 应抛 DecideError")
        except d.DecideError as exc:
            print(f"  500 → category={exc.category} retriable={exc.retriable}")
            check(exc.category == "transient" and exc.retriable is True,
                  "5xx 应 transient 且可重试")
        _set_env(TYPESAFE_API_KEY="bad-key")
        _install(lambda url, payload: _Resp(401, {"error": "unauthorized"}))
        try:
            d.decide("state", _QUESTIONS)
            raise AssertionError("401 应抛 DecideError")
        except d.DecideError as exc:
            print(f"  401 → category={exc.category} retriable={exc.retriable}")
            check(exc.category == "environment" and exc.retriable is False,
                  "401 应 environment 且不可重试")

        section("7) error：响应形状不符（缺 answers）→ unknown")
        _install(lambda url, payload: _Resp(200, {"model": "x"}))
        try:
            d.decide("state", _QUESTIONS)
            raise AssertionError("缺 answers 应抛 DecideError")
        except d.DecideError as exc:
            print(f"  category : {exc.category}   message: {exc}")
            check(exc.category == "unknown", "形状不符归 unknown")

        print("\n✅ decide_demo 全部通过")
    finally:
        _restore_env(old)
        d.requests = _real_requests


if __name__ == "__main__":
    main()
