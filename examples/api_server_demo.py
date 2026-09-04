"""api_server_demo：独立 REST 后端（flowmind-api）自包含冒烟 demo。

运行：conda run -n flowmind python examples/api_server_demo.py
前置：无（子线程起 uvicorn 随机端口，不依赖外部服务与任何 API key）

覆盖：
1. 发现端点 —— health / manifest / 单技能 / 未知 id 404
2. 同步 invoke —— happy（trace 贯穿）+ NOT_FOUND + VALIDATION + invalid_json
3. 异步 job —— 提交/轮询/终态/列表 + 未知技能 404 + invalid_request
   + 「job succeeded 但 result.ok=False」语义

并发契约（与 server_api docstring 一致）：
  sync /invoke 不进 job lane —— 长 GPU 技能（localize_video 等）一律走 /jobs。
Job 为内存态，服务重启即丢——轮询方对 unknown_job 与被 TTL 回收做同类处理。
"""
from __future__ import annotations

import socket
import threading
import time

import httpx
import uvicorn

import flowmind.skills  # noqa: F401  触发 @skill 注册
from flowmind.server_api import create_app

_INVOKE_ARGS = {"items": [{"sku": "A", "on_hand": 10, "unit_cost": 1, "sales_30d": 1}]}


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _poll(url: str, timeout_s: float = 30.0) -> dict:
    """轮询 job 至终态。"""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = httpx.get(url, timeout=10)
        check(r.status_code == 200, f"轮询状态码 {r.status_code}")
        body = r.json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.2)
    raise AssertionError(f"job {timeout_s}s 内未到终态")


def main() -> None:
    port = _free_port()
    base = f"http://127.0.0.1:{port}/api/v1"
    server = uvicorn.Server(
        uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # 段0：就绪等待（server.started 标志 + health 探活，10s 超时）
    section("0. 启动服务（子线程 uvicorn，随机端口）")
    deadline = time.time() + 10
    ready = False
    while time.time() < deadline and not ready:
        if server.started:
            try:
                ready = httpx.get(f"{base}/health", timeout=2).status_code == 200
            except httpx.HTTPError:
                pass
        if not ready:
            time.sleep(0.1)
    check(ready, "服务 10s 内未就绪")
    print(f"   已就绪：{base}")

    # 段1：发现端点
    section("1. 发现端点")
    r = httpx.get(f"{base}/health", timeout=10)
    check(r.status_code == 200, f"health 状态码 {r.status_code} != 200")
    health = r.json()
    check(health["status"] == "ok" and health["skill_count"] > 0, f"health 内容异常: {health}")
    check("workers" in health.get("jobs", {}), "health 缺 jobs.workers 字段")
    print(f"   skill_count={health['skill_count']}  jobs={health['jobs']}")

    r = httpx.get(f"{base}/manifest", timeout=10)
    check(r.status_code == 200, f"manifest 状态码 {r.status_code}")
    manifest = r.json()
    check(len(manifest["skills"]) == health["skill_count"], "manifest 技能数与 health 不一致")
    check(all("input_schema" in s for s in manifest["skills"]), "manifest 技能缺 input_schema")
    print(f"   manifest 技能数: {len(manifest['skills'])}（示例: {manifest['skills'][0]['id']}）")

    r = httpx.get(f"{base}/manifest/inventory_risk", timeout=10)
    check(r.status_code == 200 and r.json()["id"] == "inventory_risk", "单技能发现失败")
    r = httpx.get(f"{base}/manifest/nope", timeout=10)
    body = r.json()
    check(r.status_code == 404 and body["error"] == "unknown_skill", "未知技能应 404 unknown_skill")
    check("inventory_risk" in body["available"], "available 列表应含 inventory_risk")
    print("   单查 OK；未知 id → 404 + available")

    # 段2：同步 invoke（happy）
    section("2. 同步 invoke（happy）")
    r = httpx.post(
        f"{base}/skills/inventory_risk/invoke",
        json=_INVOKE_ARGS,
        headers={"X-FlowMind-Trace-Id": "demo-trace-001"},
        timeout=30,
    )
    check(r.status_code == 200, f"invoke 状态码 {r.status_code}")
    body = r.json()
    check(body["ok"] is True, f"invoke ok != True: {body.get('error')}")
    check(body["trace"]["trace_id"] == "demo-trace-001", "X-FlowMind-Trace-Id 未贯穿")
    check(isinstance(body["metrics"]["latency_ms"], (int, float)), "latency_ms 缺失")
    check("summary" in (body.get("data") or {}), "data.summary 缺失")
    print(f"   ok=True  trace_id={body['trace']['trace_id']}"
          f"  latency={body['metrics']['latency_ms']:.1f}ms")

    # 段3：同步 invoke（错误段）
    section("3. 同步 invoke（错误段）")
    r = httpx.post(f"{base}/skills/nope/invoke", json={}, timeout=10)
    body = r.json()
    check(r.status_code == 404, f"未知技能状态码 {r.status_code} != 404")
    check(body["ok"] is False and body["error"]["code"] == "NOT_FOUND", "未知技能应 NOT_FOUND 信封")
    print("   未知技能 → 404 + NOT_FOUND 信封")

    r = httpx.post(f"{base}/skills/inventory_risk/invoke", json={}, timeout=10)
    body = r.json()
    check(r.status_code == 422, f"缺字段状态码 {r.status_code} != 422")
    check(body["ok"] is False and body["error"]["code"] == "VALIDATION", "缺字段应 VALIDATION 信封")
    check(body["error"]["details"]["errors"], "VALIDATION details.errors 应非空")
    print("   缺字段 → 422 + VALIDATION 信封（含 details.errors）")

    r = httpx.post(
        f"{base}/skills/inventory_risk/invoke",
        content="broken",
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    check(r.status_code == 400 and r.json()["error"] == "invalid_json", "非 JSON 体应 400 invalid_json")
    print("   非 JSON → 400 invalid_json（唯一无信封场景）")

    # 段4：异步 job（happy）
    section("4. 异步 job（happy）")
    r = httpx.post(f"{base}/jobs", json={"skill_id": "inventory_risk", "args": _INVOKE_ARGS}, timeout=10)
    check(r.status_code == 202, f"提交状态码 {r.status_code} != 202")
    submitted = r.json()
    job_id = submitted["job_id"]
    check(submitted["status"] == "queued", "新 job 应为 queued")
    check(submitted["created_at"] is not None, "提交响应缺 created_at")
    print(f"   job_id={job_id}")

    r = httpx.get(f"{base}/jobs/{job_id}", timeout=10)
    check(
        r.status_code == 200 and r.json()["status"] in ("queued", "running", "succeeded"),
        "job 首查失败",
    )

    final = _poll(f"{base}/jobs/{job_id}")
    check(final["status"] == "succeeded", f"job 终态 {final['status']} != succeeded")
    check(final["started_at"] and final["finished_at"], "终态 job 缺时间戳")
    check(final["created_at"] <= final["started_at"] <= final["finished_at"], "时间戳无序")
    check(final["result"]["ok"] is True and final["result"]["skill"] == "inventory_risk", "job result 异常")
    print("   终态 succeeded  result.ok=True  时间戳有序")

    r = httpx.get(f"{base}/jobs", timeout=10)
    listing = r.json()
    check(r.status_code == 200 and listing["count"] >= 1, "job 列表异常")
    check(any(j["job_id"] == job_id for j in listing["jobs"]), "job 列表缺刚提交的 job")
    print(f"   列表 count={listing['count']}")

    # 段5：异步 job（错误段）
    section("5. 异步 job（错误段）")
    r = httpx.post(f"{base}/jobs", json={"skill_id": "nope", "args": {}}, timeout=10)
    check(r.status_code == 404 and r.json()["error"] == "unknown_skill", "job 未知技能应 404")
    print("   未知技能 → 404 unknown_skill（不留孤儿 queued）")

    r = httpx.post(f"{base}/jobs", json={"args": {}}, timeout=10)
    check(r.status_code == 400 and r.json()["error"] == "invalid_request", "缺 skill_id 应 400 invalid_request")
    print("   缺 skill_id → 400 invalid_request")

    r = httpx.post(f"{base}/jobs", json={"skill_id": "inventory_risk", "args": {}}, timeout=10)
    check(r.status_code == 202, "已知技能但 args 非法应正常入队（202）")
    final = _poll(f"{base}/jobs/{r.json()['job_id']}")
    check(final["status"] == "succeeded", f"runner 未崩，job 应 succeeded，实际 {final['status']}")
    check(
        final["result"]["ok"] is False and final["result"]["error"]["code"] == "VALIDATION",
        "技能级失败应落在 result.error",
    )
    print("   args 非法 → job succeeded 但 result.ok=False + VALIDATION（关键语义）")

    # 段6：收尾
    section("6. 收尾")
    server.should_exit = True
    thread.join(timeout=5)
    check(not thread.is_alive(), "uvicorn 线程未在 5s 内退出")
    print("   服务已优雅关闭，demo 全部通过")


if __name__ == "__main__":
    main()
