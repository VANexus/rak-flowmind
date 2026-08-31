"""REST 发现 API 测试：验证 /api/v1/* 端点可用且 /mcp 未被破坏。

使用 httpx.AsyncClient + ASGITransport 直接对 FastMCP 的 Starlette 应用
（mcp.streamable_http_app()）发请求，无需起真实服务。
"""
from __future__ import annotations

import httpx
import pytest

from flowmind.server import mcp  # noqa: E402
from flowmind.server_rest import register_rest_routes  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """构造已挂载 REST 路由的 ASGI 客户端（共享 mcp 实例）。"""
    register_rest_routes(mcp)
    app = mcp.streamable_http_app()
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


@pytest.mark.asyncio
async def test_manifest_returns_skills(client):
    """GET /api/v1/manifest 返回 200，技能列表非空。"""
    resp = await client.get("/api/v1/manifest")
    assert resp.status_code == 200
    body = resp.json()
    assert "skills" in body
    assert len(body["skills"]) > 0


@pytest.mark.asyncio
async def test_manifest_detail(client):
    """GET /api/v1/manifest/inventory_risk 返回 200，id 匹配。"""
    resp = await client.get("/api/v1/manifest/inventory_risk")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "inventory_risk"


@pytest.mark.asyncio
async def test_manifest_unknown_404(client):
    """GET /api/v1/manifest/no_such_skill 返回 404，并附带可用列表。"""
    resp = await client.get("/api/v1/manifest/no_such_skill")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"] == "unknown_skill"
    assert "available" in body
    assert "inventory_risk" in body["available"]


@pytest.mark.asyncio
async def test_health(client):
    """GET /api/v1/health 返回 200，status=ok 且 skill_count>0。"""
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["skill_count"] > 0
    assert body["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_mcp_still_works(client):
    """确认挂 REST 路由后 /mcp 仍然存在（未被覆盖）。"""
    app = mcp.streamable_http_app()
    paths = [getattr(r, "path", None) for r in app.routes]
    assert "/mcp" in paths
