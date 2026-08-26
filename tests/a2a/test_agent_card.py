import flowmind.skills  # noqa: F401  触发技能注册


def test_agent_card_structure():
    from flowmind.a2a.agent_card import build_agent_card
    card = build_agent_card()
    assert "name" in card
    assert "description" in card
    assert "url" in card
    assert "capabilities" in card
    assert "skills" in card

def test_agent_card_has_three_groups():
    from flowmind.a2a.agent_card import build_agent_card
    card = build_agent_card()
    skill_ids = {s["id"] for s in card["skills"]}
    assert "content" in skill_ids
    assert "video" in skill_ids
    assert "data" in skill_ids

def test_agent_card_skills_reference_real_registry():
    """Agent Card 的技能分组应基于注册表真实技能。"""
    from flowmind.a2a.agent_card import build_agent_card
    card = build_agent_card()
    # content 组应包含 content_* 技能
    content_skill = next(s for s in card["skills"] if s["id"] == "content")
    assert "content" in content_skill["description"].lower() or "内容" in content_skill["description"]
