"""
content_hot_topics 技能演示 —— 平台热点雷达。

运行：uv run python examples/content_hot_topics_demo.py

展示：
1. discover() 字段发现
2. happy path：抓取真实热榜（需 HOT_TOPIC_API_BASE 可达）
3. degraded：聚合 API 不可达 → 种子兜底 + degraded=True（绝不静默）
"""

from __future__ import annotations

import flowmind.skills  # noqa: F401
from flowmind.discover import field_names
from flowmind.skill import invoke


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def main() -> None:
    section("0) discover('content_hot_topics')")
    for p, names in field_names("content_hot_topics").items():
        print(f"  {p}: {names}")

    for platform in ("douyin", "xhs", "wechat"):
        section(f"1) {platform} 热点雷达")
        r = invoke("content_hot_topics", {"platform": platform, "limit": 5})
        print(f"  ok={r.ok} degraded={r.metrics.degraded} source={r.data.source}")
        if r.data.warning:
            print(f"  ⚠ {r.data.warning}")
        for t in r.data.topics[:5]:
            print(f"  · {t.word}  heat={t.heat}  ({t.source})")


if __name__ == "__main__":
    main()
