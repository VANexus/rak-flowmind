"""
content_copywrite 技能演示 —— 平台化文案生成。

运行：uv run python examples/content_copywrite_demo.py

展示：
1. discover() 字段发现
2. happy path：小红书种草文案（需云 LLM）
3. 三平台风格差异调用
"""

from __future__ import annotations

import flowmind.skills  # noqa: F401
from flowmind.discover import field_names
from flowmind.skill import invoke


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def main() -> None:
    section("0) discover('content_copywrite')")
    for p, names in field_names("content_copywrite").items():
        print(f"  {p}: {names}")

    for platform, subject in (("xhs", "车载保温杯"), ("wechat", "保温杯品牌成长记"), ("douyin", "一键开盖保温杯")):
        section(f"1) {platform} 平台文案生成")
        r = invoke("content_copywrite", {"platform": platform, "subject": subject})
        if r.ok and r.data.title:
            print(f"  标题：{r.data.title}")
            print(f"  正文：{r.data.body[:80]}{'…' if len(r.data.body) > 80 else ''}")
            print(f"  标签：{r.data.tags}")
        else:
            print(f"  （云 LLM 不可达）ok={r.ok} error={r.error}")


if __name__ == "__main__":
    main()
