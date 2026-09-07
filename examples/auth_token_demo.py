"""auth_token_demo：调用方凭证校验（ECO-ADR-0011 A 阶段）自包含冒烟 demo。

运行：PYTHONPATH=src conda run -n flowmind python examples/auth_token_demo.py
前置：无（Starlette TestClient + 一个 echo 路由，不依赖 PG / MQTT / Milvus /
GPU / 任何 API key）

覆盖三段式：
1. happy —— 带合法 Bearer 的请求经 RakAuthMiddleware 放行，下游能读到
   已校验的租户；并逐字断言 ADR §4b 固定向量（与 go-kernel / TS 侧同源，
   格式一改三处必挂）。
2. 默认 —— 密钥未配（dev / 独立部署）时匿名放行，现有 demo 与直连用法
   不受影响；豁免前缀（health / manifest）任何情况下匿名可达。
3. 错误 —— 缺 token 401 / 过期 401（code=expired，区别于错签名）/ 篡改
   401（signature）/ 自填 X-Rak-Tenant 无效（归属只认 token）/ 启用鉴权
   而无凭证时任务读写 fail-closed（拒绝，而不是看全部）。

token 形状契约的真源 = docs/adr/adr-0011-mcp-federation-auth.md §4b。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sys

# 本 demo 自带密钥，绝不读用户 .env 的真实凭证
os.environ["FLOWMIND_AUTH_SECRET"] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

from starlette.applications import Starlette  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from flowmind import auth  # noqa: E402
from flowmind.config import reload_config  # noqa: E402
from flowmind.server_http import RakAuthMiddleware  # noqa: E402

SECRET_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
SECRET_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
TENANT_A = "11111111-1111-4111-8111-111111111111"
TENANT_B = "22222222-2222-4222-8222-222222222222"

# ADR §4b 固定向量（Go: internal/httpserver/rakauth_test.go 断言同一份字面量）
# 注意：单行字面量，禁止手工断行（跨行拼接曾漏字 —— 向量必须逐字节精确）
V1 = "eyJzdWIiOiJjcm9zcy1kYXNoYm9hcmQiLCJ0aWQiOiIxMTExMTExMS0xMTExLTQxMTEtODExMS0xMTExMTExMTExMTEiLCJleHAiOjQxMDI0NDQ4MDB9.39c59d213a85a11c19643a131d49e702a7e7eda3a58a0f4d5002ff90233e540d"
V2 = "eyJzdWIiOiIiLCJ0aWQiOiIyMjIyMjIyMi0yMjIyLTQyMjItODIyMi0yMjIyMjIyMjIyMjIiLCJleHAiOjQxMDI0NDQ4MDB9.0f3b44331fd1b4deef67891e018c48552cf42553a7f0e0b898da0a7bf99c2167"
V3 = "eyJzdWIiOiJwcm9iZS91LTkiLCJ0aWQiOiIzMzMzMzMzMy0zMzMzLTQzMzMtODMzMy0zMzMzMzMzMzMzMzMiLCJleHAiOjE3MDAwMDAwMDB9.47611ae3ce82b7cb62441e1f97c4d9af97457cbb0a07c83daf0090ef8bb742c2"


def _echo(request):
    """回显中间件派生的身份 —— 证明归属来自 token 而非自填头。"""
    return JSONResponse({
        "tenant_id": getattr(request.state, "tenant_id", None),
        "principal": getattr(request.state, "principal", None),
        "scope": list(auth.tenant_scope()),
    })


def _anonymous(request):
    return JSONResponse({"ok": True})


def _build_client(secret: str | None) -> TestClient:
    """按给定密钥构造挂了中间件的最小应用（None = 模拟未配置密钥）。"""
    os.environ["FLOWMIND_AUTH_SECRET"] = secret or "placeholder"
    cfg = reload_config()
    cfg.auth.secret = secret or ""      # 直接改生效对象，避免 .env 干扰
    app = Starlette(routes=[
        Route("/mcp", _echo, methods=["GET", "POST"]),
        Route("/api/v1/tasks", _echo, methods=["GET", "POST"]),
        Route("/api/v1/health", _anonymous, methods=["GET"]),
    ])
    app.add_middleware(RakAuthMiddleware)
    return TestClient(app)


def _sign(secret: str, tenant: str, sub: str, exp: int) -> str:
    """按 §4b 形状签发（与 Go / TS 侧同算法）。"""
    b64 = base64.urlsafe_b64encode(json.dumps(
        {"sub": sub, "tid": tenant, "exp": exp},
        separators=(",", ":"), ensure_ascii=False).encode()).rstrip(b"=").decode()
    sig = hmac.new(secret.encode(), b64.encode(), hashlib.sha256).hexdigest()
    return f"{b64}.{sig}"


def _future() -> int:
    import time
    return int(time.time()) + 3600


def part_happy() -> None:
    print("── 段 1：happy ──")
    client = _build_client(SECRET_A)

    # 向量逐字断言（格式对齐即三侧互验成立）
    for tok, tid, sub in ((V1, TENANT_A, "cross-dashboard"), (V2, TENANT_B, "")):
        caller = auth.verify_token(tok)
        assert (caller.tenant_id, caller.principal) == (tid, sub), caller
    assert _sign(SECRET_A, TENANT_A, "cross-dashboard", 4102444800) == V1
    print("  PASS ADR §4b 向量一致（V1/V2 校验 + Python 签名侧逐字节相同）")

    tok = _sign(SECRET_A, TENANT_A, "cross-dashboard", _future())
    for path in ("/mcp", "/api/v1/tasks"):
        resp = client.get(path, headers={"Authorization": f"Bearer {tok}"})
        assert resp.status_code == 200, (path, resp.status_code, resp.text)
        body = resp.json()
        assert body["tenant_id"] == TENANT_A, body
        assert body["scope"] == [True, TENANT_A], body
    print("  PASS MCP 与 REST 两条通道同受中间件保护，租户派生正确")

    # 自填 X-Rak-Tenant 不得改变归属（网关侧已剥离；这里防直连旁路）
    resp = client.get("/api/v1/tasks", headers={
        "Authorization": f"Bearer {tok}", "X-Rak-Tenant": "t-evil"})
    assert resp.json()["tenant_id"] == TENANT_A, resp.json()
    print("  PASS 自填 X-Rak-Tenant 无效（归属只认已校验 token）")


def part_default() -> None:
    print("── 段 2：默认（dev / 独立部署） ──")
    client = _build_client("")
    resp = client.get("/api/v1/tasks")
    assert resp.status_code == 200, resp.text
    assert resp.json()["scope"] == [False, None], resp.json()
    print("  PASS 密钥未配 → 匿名放行且不做租户过滤（现有用法零破坏）")

    for secret in (SECRET_A, ""):
        c = _build_client(secret)
        assert c.get("/api/v1/health").status_code == 200
    print("  PASS 豁免前缀（health/manifest）任何配置下匿名可达（集群探针不死）")


def part_error() -> None:
    print("── 段 3：错误 ──")
    client = _build_client(SECRET_A)

    resp = client.get("/api/v1/tasks")
    assert resp.status_code == 401 and resp.json()["error"] == "missing_token", resp.text
    resp = client.get("/api/v1/tasks", headers={"Authorization": "Basic abc"})
    assert resp.status_code == 401, resp.text
    print("  PASS 缺 / 非 Bearer 凭证 → 401")

    resp = client.get("/api/v1/tasks", headers={"Authorization": f"Bearer {V3}"})
    assert resp.status_code == 401 and resp.json()["error"] == "signature", resp.json()
    resp = client.get("/api/v1/tasks", headers={"Authorization": "Bearer " + _sign(
        SECRET_A, TENANT_A, "u", int(__import__("time").time()) - 60)})
    assert resp.status_code == 401 and resp.json()["error"] == "expired", resp.json()
    resp = client.get("/api/v1/tasks", headers={"Authorization": "Bearer " + (
        V1.split(".")[0] + "." + "0" * 64)})
    assert resp.status_code == 401 and resp.json()["error"] == "signature", resp.json()
    print("  PASS 错密钥 / 过期 / 篡改分别拒绝，错误类别可区分（401 不泄露凭证）")

    # 无 token 调 /mcp 被挡：证明 MCP 通道不能匿名绕过
    resp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert resp.status_code == 401, resp.text
    print("  PASS /mcp 无法匿名绕过")

    # 启用鉴权但上下文无凭证 → 任务读写 fail-closed（拒绝而非看全部）
    assert auth.tenant_scope() == (True, None)
    print("  PASS 无凭证上下文 fail-closed（tenant_scope=(True, None)）")


def main() -> int:
    reload_config()
    try:
        part_happy()
        part_default()
        part_error()
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("\nauth_token_demo: 全部 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
