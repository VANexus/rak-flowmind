"""b2b_push_* 共享：群机器人 webhook POST 与错误分类。

云优先原则：推送必须走真实 webhook（飞书自定义机器人 / 企微群机器人），
失败结构化抛出 PushError，绝不静默成功、绝无 mock。
"""
from __future__ import annotations

import httpx


class PushError(Exception):
    """webhook 推送失败。category/retriable 语义与 errors.py 一致。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def post_json(url: str, payload: dict, *, timeout_s: float) -> dict:
    """POST JSON 到 webhook，返回解析后的响应体。

    HTTP 5xx → transient（可重试）；4xx → video（凭证/参数类，不可重试）；
    非 JSON → unknown。供测试打桩（monkeypatch 本函数）。
    """
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(url, json=payload)
    except httpx.TimeoutException as exc:
        raise PushError("推送超时", category="environment", retriable=False) from exc
    except httpx.HTTPError as exc:
        raise PushError(f"推送连接失败：{type(exc).__name__}", category="environment", retriable=False) from exc

    if resp.status_code >= 500:
        raise PushError(f"推送 HTTP {resp.status_code}", category="transient", retriable=True)
    if resp.status_code >= 400:
        raise PushError(f"推送 HTTP {resp.status_code}", category="video", retriable=False)
    try:
        return resp.json()
    except ValueError as exc:
        raise PushError("webhook 响应不是合法 JSON", category="unknown", retriable=False) from exc
