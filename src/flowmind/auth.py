"""调用方凭证校验与租户上下文（ECO-ADR-0011 A 阶段）。

契约真源 = 生态 ADR `docs/adr/adr-0011-mcp-federation-auth.md`；本模块是
其 §4b token 格式的 **Python 侧实现**（Go: go-kernel/internal/httpserver，
TS: cross-dashboard 服务端），三侧必须逐字节一致——防漂移靠 ADR 里的
固定测试向量（`examples/auth_token_demo.py` 逐字断言同一组字面量）。

token 形状：``<base64url(JSON)> "." <hex HMAC-SHA256>``，payload 键序固定
``{"sub":..,"tid":..,"exp":<unix 秒>}``、无空白分隔符、base64url **不带填充**。

两个平面不要混：
  * ``secret`` 为空 → **鉴权整体关闭**（dev / 独立部署语义：现有 demo 与
    直连用法不受影响，任务不记租户）。
  * ``secret`` 已配 → 每个请求都必须带合法凭证；技能/仓储层拿不到凭证时
    **fail-closed**（拒绝访问，而不是退化成"看全部"）。安全上只有这两种
    取值，不存在"有鉴权但匿名可见全部"的中间态。
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

__all__ = [
    "Caller",
    "TokenError",
    "auth_enforced",
    "bind_caller",
    "current_caller",
    "tenant_scope",
    "use_caller",
    "verify_token",
]


class TokenError(Exception):
    """凭证非法。``code`` 决定对外语义（全部映射 401，仅 detail 不同）。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Caller:
    """一次调用的调用方身份（由已校验凭证派生，绝不可由请求头自填）。"""

    tenant_id: str
    principal: str


_caller: ContextVar[Caller | None] = ContextVar("flowmind_caller", default=None)


def auth_secret() -> str:
    """生效的校验密钥（env ``FLOWMIND_AUTH_SECRET`` → config ``auth.secret``）。"""
    from flowmind.config import get_config

    return (get_config().auth.secret or "").strip()


def auth_enforced() -> bool:
    """是否启用凭证校验（密钥非空即启用）。"""
    return bool(auth_secret())


def current_caller() -> Caller | None:
    """当前请求上下文中的调用方；未在上下文绑定（后台线程/启动恢复）为 None。"""
    return _caller.get()


def bind_caller(caller: Caller | None):
    """把调用方绑进当前上下文，返回还原用的 token（配合 ``use_caller``）。"""
    return _caller.set(caller)


@contextmanager
def use_caller(caller: Caller | None) -> Iterator[Caller | None]:
    """上下文绑定作用域（异常路径也保证 reset，避免租户串到下个请求）。"""
    token = _caller.set(caller)
    try:
        yield caller
    finally:
        _caller.reset(token)


def tenant_scope() -> tuple[bool, str | None]:
    """任务级访问的租户范围判定（三态，fail-closed）。

    返回 ``(是否强制隔离, 允许访问的租户 id)``：

    * ``(False, None)`` —— 未启用鉴权：不做租户过滤（dev / 独立部署）。
    * ``(True, <tid>)`` —— 已鉴权：仅可见该租户的任务。
    * ``(True, None)``  —— 启用鉴权但当前上下文无凭证：**拒绝一切任务访问**。
      这条分支就是防泄漏的关键——后台/脚本路径若忘了绑上下文，结果是
      "查不到"而不是"全能看到"。
    """
    if not auth_enforced():
        return False, None
    caller = current_caller()
    if caller is None:
        return True, None
    return True, caller.tenant_id


def verify_token(token: str, *, now: float | None = None) -> Caller:
    """校验 Bearer token 并还原调用方身份。

    顺序铁律：**先验签再解 JSON**——签名不匹配的载荷绝不进反序列化，
    避免把攻击者可控字节喂给 JSON 解析器。
    """
    secret = auth_secret()
    if not secret:
        raise TokenError("auth_disabled", "未配置 FLOWMIND_AUTH_SECRET，无法校验凭证")
    raw = (token or "").strip()
    b64, sep, sig = raw.partition(".")
    if not sep or not b64 or not sig:
        raise TokenError("malformed", "token 形状非法（期望 <b64>.<hex-hmac>）")

    expected = hmac.new(secret.encode("utf-8"), b64.encode("ascii"),
                        hashlib.sha256).hexdigest()
    # 恒时比较；非 hex 签名自然不相等（不会泄露“形状对但密钥错”的区别）
    if not hmac.compare_digest(expected, sig.strip().lower()):
        raise TokenError("signature", "token 签名不匹配")

    try:
        body = base64.urlsafe_b64decode(_pad(b64))
        payload = json.loads(body.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
        raise TokenError("malformed", "token 载荷无法解析") from exc
    if not isinstance(payload, dict):
        raise TokenError("malformed", "token 载荷不是 JSON object")

    exp = payload.get("exp")
    if not isinstance(exp, (int, float)) or isinstance(exp, bool):
        raise TokenError("malformed", "token 缺少合法的 exp")
    if (now if now is not None else time.time()) >= float(exp):
        raise TokenError("expired", "token 已过期")

    tid = payload.get("tid")
    if tid is not None and not isinstance(tid, str):
        raise TokenError("malformed", "token 的 tid 必须是字符串")
    sub = payload.get("sub")
    if sub is not None and not isinstance(sub, str):
        raise TokenError("malformed", "token 的 sub 必须是字符串")
    return Caller(tenant_id=tid or "", principal=sub or "")


def _pad(b64: str) -> bytes:
    """补齐 base64url 填充（签发侧不写填充，解码侧容忍两者）。"""
    return (b64 + "=" * (-len(b64) % 4)).encode("ascii")
