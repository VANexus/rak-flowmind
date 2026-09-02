"""
localize_batch 技能演示 —— 批量视频本地化编排。

运行：uv run python examples/localize_batch_demo.py

展示：
1. happy path：3 个视频 URL → batch_id + job_ids
2. 字幕策略 + 语言偏好的实际生效（来自 config 通用默认）
3. 错误分类（environment / video / transient）—— 让 Agent 知道下一步动作
4. **discover() 自动输出字段名** —— 避免猜错 `data.batch_id` / `data.job_ids` / `data.failure_category`

mock 后端写在文件内，不需要真起 video-localizer。
"""

from __future__ import annotations

import time
from typing import Any

import flowmind.skills  # noqa: F401  触发 @skill 注册
import flowmind.skills.localize_batch as lb
from flowmind.discover import field_names
from flowmind.skill import invoke


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


# ── mock 后端（用真实 HTTP 请求结构） ──────────────────────────────

def install_healthy_backend() -> list[dict]:
    """健康检查 OK + 批量提交返 200。返回 calls 列表供断言。"""
    posts: list[dict] = []
    def fake_get_health(url, timeout=None, **_kw):
        class _R:
            status_code = 200
            _json = {"status": "ok"}
            def raise_for_status(self): pass
            def json(self): return self._json
        return _R()
    def fake_post(url, json=None, timeout=None, **_kw):
        posts.append({"url": url, "payload": json})
        n_videos = len(json["video_paths"])  # 类体内取不到闭包参数，先在函数作用域算好
        class _R:
            status_code = 200
            _json = {"batch_id": f"batch-{int(time.time()*1000)}",
                     "job_ids": [f"job-{i}" for i in range(n_videos)],
                     "total": n_videos, "message": "submitted"}
            def raise_for_status(self): pass
            def json(self): return self._json
        return _R()
    lb.requests.get = fake_get_health
    lb.requests.post = fake_post
    return posts


def install_failing_backend(status_code: int) -> None:
    """健康 OK，但 POST 返错误状态码（模拟 VL 后端故障）。"""
    lb.requests.get = lambda url, timeout=None, **_kw: type("_R",(),{
        "status_code": 200, "raise_for_status": lambda self: None,
        "json": lambda self: {"status": "ok"}})()
    def fake_post(url, json=None, timeout=None, **_kw):
        code = status_code  # 类体内取不到闭包参数，先在函数作用域绑定
        class _R:
            status_code = code
            def raise_for_status(self): pass
            def json(self): return {}
        return _R()
    lb.requests.post = fake_post


def main() -> None:
    section("0) discover('localize_batch') —— Agent 自查字段")
    paths = field_names("localize_batch")
    for p, names in paths.items():
        print(f"  {p}:")
        for n in names:
            print(f"    • {n}")

    section("1) Happy path：3 个视频（自动批量化）")
    posts = install_healthy_backend()
    r = invoke("localize_batch", {
        "video_paths": [
            "https://cdn.example.com/promo-v1.mp4",
            "https://cdn.example.com/promo-v2.mp4",
            "https://cdn.example.com/promo-v3.mp4",
        ],
    })
    print(f"  ok         : {r.ok}")
    print(f"  trace_id   : {r.trace.trace_id[:8]}...")
    print(f"  latency_ms : {r.metrics.latency_ms:.2f}")
    print(f"  batch_id   : {r.data.batch_id}")           # ← discover() 告诉你的字段名
    print(f"  job_ids    : {r.data.job_ids}")
    print(f"  POST 次数  : {len(posts)}（1 批未超 max_videos_per_batch=100）")
    print(f"  字幕策略   : {r.data.remove_subtitles_strategy}（v0.3 唯一支持）")
    print(f"  cost_band  : {r.data.cost_band}")
    print(f"  reasoning  : {r.reasoning[0].conclusion}")

    section("2) 错误路径：VL 返 503 → transient（可重试）")
    install_failing_backend(503)
    r = invoke("localize_batch", {"video_paths": ["/data/v.mp4"]})
    print(f"  ok              : {r.ok}")
    print(f"  metrics.degraded: {r.metrics.degraded}")
    print(f"  failure_category: {r.data.failure_category}")  # 'transient'
    print(f"  retriable       : {r.data.retriable}（True → Agent 重试）")
    print(f"  warning         : {r.data.warning}")

    section("3) 错误路径：VL 返 404 → video（视频不存在，不重试）")
    install_failing_backend(404)
    r = invoke("localize_batch", {"video_paths": ["/data/missing.mp4"]})
    print(f"  failure_category: {r.data.failure_category}")  # 'video'
    print(f"  retriable       : {r.data.retriable}（False → Agent 检查视频路径）")

    section("4) 错误路径：ConnectionError → environment（先查网络）")
    install_healthy_backend()  # 先恢复健康检查 OK
    lb.requests.post = lambda url, json=None, timeout=None, **_kw: (_ for _ in ()).throw(
        __import__("requests").exceptions.ConnectionError("refused")
    )
    r = invoke("localize_batch", {"video_paths": ["/data/v.mp4"]})
    print(f"  failure_category: {r.data.failure_category}")  # 'environment'
    print(f"  retriable       : {r.data.retriable}（False → Agent 查 VL 服务）")


if __name__ == "__main__":
    main()