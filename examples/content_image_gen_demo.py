"""
content_image_gen 技能演示 —— 平台比例 AI 配图。

运行：uv run python examples/content_image_gen_demo.py

展示：
1. discover() 字段发现
2. mock 后端：按平台比例生成确定性占位图（离线可用）
3. auto 后端：有 CIYUANSKY_API_KEY 时走云 API，无 key 显式报错
"""

from __future__ import annotations

import flowmind.skills  # noqa: F401
from flowmind.discover import field_names
from flowmind.skill import invoke


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def main() -> None:
    section("0) discover('content_image_gen')")
    for p, names in field_names("content_image_gen").items():
        print(f"  {p}: {names}")

    for platform in ("xhs", "wechat", "douyin"):
        section(f"1) {platform} mock 后端配图")
        r = invoke("content_image_gen", {
            "platform": platform, "prompt": "通勤场景保温杯，暖色调", "count": 1, "backend": "mock",
        })
        if r.ok:
            print(f"  {r.data.width}x{r.data.height} backend={r.data.backend_used}")
            for img in r.data.images:
                print(f"  · {img.url}")
        else:
            print(f"  ok={r.ok} error={r.error}")

    section("2) auto 后端（云 API）")
    r = invoke("content_image_gen", {"platform": "xhs", "prompt": "夏日保冷，清透质感", "count": 1})
    if r.ok:
        for img in r.data.images:
            print(f"  · {img.url}")
    else:
        print(f"  （云 API 未配置/不可达）ok={r.ok} code={r.error.code} msg={r.error.message}")


if __name__ == "__main__":
    main()
