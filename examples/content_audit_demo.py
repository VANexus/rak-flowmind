"""
content_audit 技能演示 —— 平台规则审计（规则扫描 + LLM 复核）。

运行：uv run python examples/content_audit_demo.py

展示：
1. discover() 字段发现
2. 违规文案审计（绝对化用语 → error）——规则扫描离线可用
3. LLM 复核（需云 LLM；失败不影响规则结果）
"""

from __future__ import annotations

import flowmind.skills  # noqa: F401
from flowmind.discover import field_names
from flowmind.skill import invoke


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def main() -> None:
    section("0) discover('content_audit')")
    for p, names in field_names("content_audit").items():
        print(f"  {p}: {names}")

    cases = [
        ("xhs", "全网最低价的保温杯", "加微信领优惠，能降血压。", []),
        ("wechat", "一个品牌的成长记", "转发到朋友圈即可领取专属优惠。", ["#转发抽奖"]),
        ("douyin", "一键开盖保温杯", "稳赚不赔的投资好物。", []),
        ("xhs", "316 不锈钢内胆", "夏天保冷、冬天保暖，450ml 容量。", ["通勤好物"]),
    ]
    for i, (platform, title, body, tags) in enumerate(cases, 1):
        section(f"{i}) {platform} 审计")
        r = invoke("content_audit", {"platform": platform, "title": title, "body": body, "tags": tags})
        d = r.data
        print(f"  passed={d.passed} llm_reviewed={d.llm_reviewed} findings={len(d.findings)}")
        for f in d.findings[:6]:
            print(f"  [{f.severity.upper()}][{f.category}] {f.message}（命中「{f.matched_text}」）")
            print(f"       ↳ {f.suggestion}")


if __name__ == "__main__":
    main()
