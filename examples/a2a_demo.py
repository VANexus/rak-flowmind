"""FlowMind A2A 集成 demo：展示 Agent Card 发现 + 任务委托全流程。

运行：uv run python examples/a2a_demo.py
前置：需设置 LONGCAT_API_KEY 环境变量（编排器 LLM）。

展示：
1. Agent Card 发现 —— A2A 客户端自描述入口
2. 任务委托（tasks/send）—— JSON-RPC 编排请求
3. 结果 + 推理链（include_reasoning=true）—— CoT 按需暴露
4. **discover() 自动输出字段名** —— 避免猜错编排器输出字段
"""
from __future__ import annotations

import os

import httpx

import flowmind.skills  # noqa: F401  触发 @skill 注册


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def main() -> None:
    base_url = os.environ.get("FLOWMIND_BASE_URL", "http://localhost:8001")

    print("=== FlowMind A2A Demo ===")

    # 1. 发现 Agent Card
    section("1. 发现 Agent Card")
    resp = httpx.get(f"{base_url}/.well-known/agent.json", timeout=10)
    resp.raise_for_status()
    card = resp.json()
    print(f"   名称: {card['name']}")
    print(f"   描述: {card['description']}")
    print(f"   版本: {card.get('version', 'n/a')}")
    skills = card.get("skills", [])
    print(f"   技能数: {len(skills)}")
    if skills:
        print(f"   技能分组示例: {skills[0]['id']} — {skills[0].get('description', '')[:40]}")

    # 2. 提交任务
    goal = os.environ.get("A2A_DEMO_GOAL", "帮我分析库存风险，SKU 为 A001")
    section(f"2. 提交任务: {goal}")
    resp = httpx.post(
        f"{base_url}/a2a",
        json={
            "jsonrpc": "2.0",
            "id": "demo-req",
            "method": "tasks/send",
            "params": {
                "id": "demo-task",
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": goal}],
                },
                "metadata": {"include_reasoning": True},
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    result = resp.json()["result"]
    print(f"   任务 ID: {result['id']}")
    print(f"   状态: {result['status']['state']}")
    if result["status"].get("degraded"):
        print("   ⚠️ 部分完成（degraded）")

    if result.get("artifacts"):
        text = result["artifacts"][0]["parts"][0]["text"]
        print(f"   输出: {text[:200]}{'...' if len(text) > 200 else ''}")

    if result.get("history"):
        print(f"   推理链 ({len(result['history'])} 步):")
        for i, r in enumerate(result["history"], 1):
            print(f"     [{i}] {r[:80]}{'...' if len(str(r)) > 80 else ''}")

    # 3. 完成
    section("3. Demo 完成")
    print("   A2A 流：Agent Card → tasks/send → 编排 → 结果 ✓")


if __name__ == "__main__":
    main()
