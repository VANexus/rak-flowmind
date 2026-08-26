"""
content_idea_design 技能演示 —— AI 选题思路。

运行：uv run python examples/content_idea_design_demo.py

展示：
1. discover() 自动字段发现
2. happy path：为「保温杯」生成小红书选题（需云 LLM 可达，即 LONGCAT_API_KEY 有效）
3. 无 key / LLM 不可达 → 结构化错误信封（错误永不静默）
"""

from __future__ import annotations

import flowmind.skills  # noqa: F401
from flowmind.discover import field_names
from flowmind.skill import invoke


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def main() -> None:
    section("0) discover('content_idea_design') —— Agent 自查字段")
    for p, names in field_names("content_idea_design").items():
        print(f"  {p}: {names}")

    section("1) Happy path：小红书选题 3 条")
    r = invoke("content_idea_design", {"platform": "xhs", "subject": "车载保温杯", "count": 3})
    if r.ok and r.data.ideas:
        for idea in r.data.ideas:
            print(f"  [{idea.angle}] {idea.title}\n       ↳ {idea.reason}")
        print(f"  推理链：{r.reasoning[0].conclusion}")
    else:
        print(f"  （云 LLM 不可达）ok={r.ok} error={r.error}")

    section("2) 边界：wechat 长文选题 2 条")
    r = invoke("content_idea_design", {"platform": "wechat", "subject": "品牌内容方法论", "count": 2})
    if r.ok and r.data.ideas:
        for idea in r.data.ideas:
            print(f"  [{idea.angle}] {idea.title}")
    else:
        print(f"  （云 LLM 不可达）ok={r.ok} error={r.error}")


if __name__ == "__main__":
    main()
