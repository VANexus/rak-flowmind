"""Agent Card 生成：从注册表动态构建分组能力描述。"""
from __future__ import annotations

from flowmind.skill import registry

# 技能分组映射：分组 id → 匹配前缀/技能 id 列表
_SKILL_GROUPS: dict[str, dict] = {
    "content": {
        "description": "内容创作：选题、文案、审核、生图、发布、爬取",
        "prefixes": ("content_", "crawler_"),
        "extra": ("marketing_image_gen",),
    },
    "video": {
        "description": "视频处理：本地化、字幕、配音、状态查询",
        "prefixes": ("localize_",),
        "extra": (),
    },
    "data": {
        "description": "数据分析：库存风险、飞书知识库检索",
        "prefixes": (),
        "extra": ("inventory_risk", "feishu_kb_search"),
    },
}


def _group_skills() -> list[dict]:
    """将注册表技能分组，返回 Agent Card skills 列表。"""
    reg = registry()
    skills = []
    for group_id, meta in _SKILL_GROUPS.items():
        matched = [
            sid for sid in reg
            if any(sid.startswith(p) for p in meta["prefixes"]) or sid in meta["extra"]
        ]
        if matched:
            skills.append({
                "id": group_id,
                "description": meta["description"],
                "skills": matched,
            })
    return skills


def build_agent_card(base_url: str = "http://localhost:8001") -> dict:
    """构建 FlowMind Agent Card。

    Args:
        base_url: 外部可访问的基地址（用于生成端点 URL）。

    Returns:
        Agent Card JSON 兼容 dict。
    """
    return {
        "name": "FlowMind",
        "description": "统一工具总线：内容创作、视频处理、数据分析",
        "url": f"{base_url}/a2a",
        "version": "1.0.0",
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "skills": [
            {"id": s["id"], "description": s["description"]}
            for s in _group_skills()
        ],
    }
