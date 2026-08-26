def test_planner_prompt_includes_skills():
    from flowmind.orchestrator.prompts import build_planner_prompt
    prompt = build_planner_prompt(goal="写篇小红书", skill_group="content", max_steps=5)
    assert "content_idea_design" in prompt or "content" in prompt
    assert "写篇小红书" in prompt

def test_planner_prompt_respects_max_steps():
    from flowmind.orchestrator.prompts import build_planner_prompt
    prompt = build_planner_prompt(goal="test", skill_group=None, max_steps=3)
    assert "3" in prompt

def test_summarizer_prompt_includes_results():
    from flowmind.orchestrator.prompts import build_summarizer_prompt
    prompt = build_summarizer_prompt(step_results=[{"skill": "x", "ok": True}])
    assert "x" in prompt
