"""api_server_demo：单端口 mcp-base-gpu 任务 REST 通道自包含冒烟 demo。

运行：PYTHONPATH=src conda run -n flowmind python examples/api_server_demo.py
前置：无（子线程起 uvicorn 随机端口，模块级 patch 内存 fake manager，
不依赖 PG / MQTT / Milvus / GPU / 任何 API key）

覆盖（server_http 单入口的 REST 半边；MCP 半边见探针脚本 / README）：
1. 发现端点 —— health（版本 + 组件状态）/ manifest / 未知 id 404
2. 提交 —— POST /api/v1/tasks happy（URL + 本地路径）→ 202 task_ids
3. 轮询 —— GET /api/v1/tasks/{id}：running → succeeded 终态（含 output_paths）
4. 下载 —— GET .../download?file=：200 流式内容比对 / 错名 404 /
   路径穿越 404 / 未知任务 404 / 缺参 400
5. 错误段 —— 非 JSON 400 / 入参校验 422 / 全部扩展名被拒 422
6. 背压 —— TaskQueueFull：一个未受理 → 429；中途满 → 202 + warning 部分受理

fake manager 模拟 TaskManager 生命周期（submit → running → succeeded 落
output_paths），与真实 TaskStore 行结构对齐；patch 点为
flowmind.server_tasks.get_task_manager（REST 端点的唯一取用入口）。
"""
from __future__ import annotations

import os
import socket
import tempfile
import threading
import time
from datetime import datetime, timezone

import httpx
import uvicorn

from flowmind.server import mcp
from flowmind.server_rest import register_rest_routes
from flowmind.server_tasks import register_task_routes
from flowmind.tasks import TaskQueueFull

# demo 用 /tmp 假路径（不落在 data_dir/uploads 沙箱内）：仅本 demo 进程
# 放行本地路径沙箱（生产禁设，见 localize_submit 模块 docstring）
# 沙箱在提交通道运行时读 env，import 后设置即可生效
os.environ.setdefault("FLOWMIND_ALLOW_ANY_PATH", "1")


class _FakeComponents:
    """health 探针的最小组件替身。"""

    def health_status(self) -> str:
        return "ok"

    def status(self) -> str:
        return "disabled"


class _FakeManager:
    """TaskManager 语义替身：submit → running →（0.2s 后）succeeded。

    fail_after 非 None 时：第 fail_after+1 次 submit 抛 TaskQueueFull
    （模拟 429 / 部分受理两种背压路径）。
    """

    def __init__(self, output_dir: str, fail_after: int | None = None):
        self.output_dir = output_dir
        self.fail_after = fail_after
        self._lock = threading.Lock()
        self._submitted = 0
        self.tasks: dict[str, dict] = {}
        self.store = _FakeComponents()
        self.events = _FakeComponents()

    def submit(self, skill_id: str, args: dict) -> str:
        with self._lock:
            if self.fail_after is not None and self._submitted >= self.fail_after:
                raise TaskQueueFull(
                    f"待处理任务已达上限 {self.fail_after}（demo 模拟），稍后重试")
            self._submitted += 1
            n = self._submitted
        task_id = f"demo-task-{n:03d}"
        out = f"{self.output_dir}/demo_{n}_sub.mp4"
        with open(out, "wb") as f:
            f.write(f"demo-output-{task_id}".encode())
        now = datetime.now(timezone.utc).isoformat()
        self.tasks[task_id] = {
            "task_id": task_id, "skill_id": skill_id, "args": args,
            "status": "running", "stage": "asr", "progress": 10.0,
            "error": None, "created_at": now, "started_at": now,
            "finished_at": None, "tenant_id": None, "output_paths": None,
        }
        timer = threading.Timer(
            0.2, self._finish, args=(task_id, out))
        timer.daemon = True
        timer.start()
        return task_id

    def _finish(self, task_id: str, out: str) -> None:
        rec = self.tasks[task_id]
        rec.update(
            status="succeeded", stage=None, progress=100.0,
            finished_at=datetime.now(timezone.utc).isoformat(),
            output_paths=[out],
        )

    def get_task(self, task_id: str) -> dict | None:
        return self.tasks.get(task_id)


def _install_fake_task_manager(output_dir: str, fail_after: int | None = None) -> _FakeManager:
    """patch server_tasks 模块级 get_task_manager（REST 端点唯一取用入口）。"""
    import flowmind.server_tasks as _st

    manager = _FakeManager(output_dir, fail_after)
    _st.get_task_manager = lambda: manager
    return manager


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _poll(client: httpx.Client, url: str, timeout_s: float = 10.0) -> dict:
    """轮询任务至 succeeded（fake 0.2s 后落终态）。"""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = client.get(url, timeout=10)
        check(r.status_code == 200, f"轮询状态码 {r.status_code}")
        body = r.json()
        if body["status"] == "succeeded":
            return body
        check(body["status"] in ("queued", "running"), f"异常状态 {body['status']}")
        time.sleep(0.05)
    raise AssertionError(f"任务 {timeout_s}s 内未到 succeeded")


def _start_server(port: int):
    """注册全部 REST 路由后以 uvicorn 起 streamable_http_app（自带 lifespan）。"""
    register_rest_routes(mcp)
    register_task_routes(mcp)
    server = uvicorn.Server(
        uvicorn.Config(mcp.streamable_http_app(), host="127.0.0.1",
                       port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


def main() -> None:
    port = _free_port()
    base = f"http://127.0.0.1:{port}/api/v1"
    server, thread = _start_server(port)

    # 段0：就绪等待
    section("0. 启动服务（子线程 uvicorn，随机端口）")
    with httpx.Client() as probe:
        deadline = time.time() + 10
        ready = False
        while time.time() < deadline and not ready:
            if server.started:
                try:
                    ready = probe.get(f"{base}/health", timeout=2).status_code == 200
                except httpx.HTTPError:
                    pass
            if not ready:
                time.sleep(0.1)
        check(ready, "服务 10s 内未就绪")
    print(f"   已就绪：{base}")

    with tempfile.TemporaryDirectory() as tmp, httpx.Client() as c:
        manager = _install_fake_task_manager(tmp)

        # 段1：发现端点
        section("1. 发现端点（health / manifest）")
        r = c.get(f"{base}/health", timeout=10)
        check(r.status_code == 200, f"health 状态码 {r.status_code}")
        health = r.json()
        check(health["status"] == "ok", f"health.status 异常: {health}")
        check(health["components"]["pg"] == "ok", "health 缺 components.pg=ok")
        check(health["components"]["mqtt"] == "disabled", "未配置 MQTT 应 disabled")
        check(health["components"]["milvus"] == "unverified", "未用过的 Milvus 应 unverified")
        check(health["skill_count"] == 7 and health["version"], f"health 元数据异常: {health}")
        print(f"   status=ok  version={health['version']}  components={health['components']}")

        r = c.get(f"{base}/manifest", timeout=10)
        manifest = r.json()
        check(r.status_code == 200 and len(manifest["skills"]) == 7, "manifest 应为 7 技能")
        check(all("input_schema" in s for s in manifest["skills"]), "manifest 缺 input_schema")
        print(f"   manifest 技能数: {len(manifest['skills'])}")

        r = c.get(f"{base}/manifest/nope", timeout=10)
        check(r.status_code == 404 and r.json()["error"] == "unknown_skill",
              "未知技能应 404 unknown_skill")
        print("   未知 id → 404 + available")

        # 段2：提交 happy（URL + 本地路径各一条）
        section("2. 提交 POST /api/v1/tasks（happy）")
        r = c.post(f"{base}/tasks", json={
            "videos": ["/tmp/demo_local.mp4", "https://example.com/v.mp4"],
            "target_lang": "en",
        }, timeout=10)
        check(r.status_code == 202, f"提交状态码 {r.status_code} != 202")
        submitted = r.json()
        check(len(submitted["task_ids"]) == 2 and submitted["accepted"] == 2,
              f"提交响应异常: {submitted}")
        check(submitted["rejected_count"] == 0 and submitted["skill_id"] == "localize_video",
              "提交元数据异常")
        print(f"   task_ids={submitted['task_ids']}")

        # 段3：轮询至终态
        section("3. 轮询 GET /api/v1/tasks/{id}")
        tid = submitted["task_ids"][0]
        final = _poll(c, f"{base}/tasks/{tid}")
        check(isinstance(final["output_paths"], list) and final["output_paths"],
              "终态任务缺 output_paths")
        check(final["progress"] == 100.0, "终态 progress 应 100")
        out_name = final["output_paths"][0].rsplit("/", 1)[-1]
        print(f"   {tid}: running → succeeded  output={out_name}")

        # 段4：下载（含穿越防护）
        section("4. 下载 GET /api/v1/tasks/{id}/download")
        r = c.get(f"{base}/tasks/{tid}/download", params={"file": out_name}, timeout=10)
        check(r.status_code == 200 and r.content == f"demo-output-{tid}".encode(),
              "产物下载内容异常")
        r = c.get(f"{base}/tasks/{tid}/download", params={"file": "wrong.mp4"}, timeout=10)
        check(r.status_code == 404 and r.json()["error"] == "file_not_found",
              "错误文件名应 404")
        r = c.get(f"{base}/tasks/{tid}/download",
                  params={"file": "../../etc/passwd"}, timeout=10)
        check(r.status_code == 404, "路径穿越序列应 404（basename 白名单外）")
        r = c.get(f"{base}/tasks/nope/download", params={"file": out_name}, timeout=10)
        check(r.status_code == 404 and r.json()["error"] == "unknown_task",
              "未知任务应 404 unknown_task")
        r = c.get(f"{base}/tasks/{tid}/download", timeout=10)
        check(r.status_code == 400, "缺 file 参数应 400")
        r = c.get(f"{base}/tasks/nope", timeout=10)
        check(r.status_code == 404 and r.json()["error"] == "unknown_task",
              "查询未知任务应 404")
        print("   200 内容比对 / 错名 404 / 穿越 404 / 未知任务 404 / 缺参 400 全通过")

        # 段5：错误段
        section("5. 提交错误段（400 / 422）")
        r = c.post(f"{base}/tasks", content="broken",
                   headers={"Content-Type": "application/json"}, timeout=10)
        check(r.status_code == 400 and r.json()["error"] == "invalid_json",
              "非 JSON 体应 400 invalid_json")
        r = c.post(f"{base}/tasks", json={}, timeout=10)
        check(r.status_code == 422 and r.json()["error"] == "validation",
              "缺 videos 应 422 validation")
        r = c.post(f"{base}/tasks", json={"videos": ["/tmp/a.avi", "/tmp/b.mov"]}, timeout=10)
        check(r.status_code == 422, "全部扩展名被拒应 422")
        print("   非 JSON 400 / 缺字段 422 / 全部扩展名被拒 422 全通过")

        # 段6：背压（429 / 部分受理 202）
        section("6. 背压（TaskQueueFull → 429 / 部分受理）")
        manager._submitted = 0  # 重置计数（前段已提交 2 个）
        manager.fail_after = 0  # 第 1 个 submit 即满：一个都没受理
        r = c.post(f"{base}/tasks", json={"videos": ["/tmp/a.mp4"]}, timeout=10)
        check(r.status_code == 429 and r.json()["error"] == "queue_full",
              f"队列满应 429 queue_full: {r.status_code} {r.text}")
        manager.fail_after = 1  # 第 2 个 submit 满：部分受理
        r = c.post(f"{base}/tasks", json={"videos": ["/tmp/a.mp4", "/tmp/b.mp4"]}, timeout=10)
        check(r.status_code == 202, f"部分受理应 202: {r.status_code}")
        partial = r.json()
        check(len(partial["task_ids"]) == 1 and "warning" in partial,
              f"部分受理响应异常: {partial}")
        print("   全满 → 429 queue_full；中途满 → 202 + warning（transient 可重提）")

    # 段7：收尾
    section("7. 收尾")
    server.should_exit = True
    thread.join(timeout=5)
    check(not thread.is_alive(), "uvicorn 线程未在 5s 内退出")
    print("   服务已优雅关闭，demo 全部通过")


if __name__ == "__main__":
    main()
