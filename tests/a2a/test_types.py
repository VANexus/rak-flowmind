def test_a2a_task_to_request_extracts_goal():
    from flowmind.a2a.types import a2a_task_to_request
    task_msg = {
        "role": "user",
        "parts": [{"type": "text", "text": "帮我写篇小红书文案"}],
    }
    req = a2a_task_to_request(task_msg, metadata={})
    assert req["goal"] == "帮我写篇小红书文案"
    assert req["skill_group"] is None

def test_a2a_task_to_request_extracts_skill_group():
    from flowmind.a2a.types import a2a_task_to_request
    task_msg = {
        "role": "user",
        "parts": [{"type": "text", "text": "处理这个视频"}],
    }
    req = a2a_task_to_request(task_msg, metadata={"skill_group": "video"})
    assert req["skill_group"] == "video"

def test_result_to_a2a_task_completed():
    from flowmind.a2a.types import result_to_a2a_task
    result = {
        "output": {"text": "完成了"},
        "reasoning": [],
        "degraded": False,
        "error": None,
    }
    task = result_to_a2a_task(task_id="abc", result=result, include_reasoning=False)
    assert task["id"] == "abc"
    assert task["status"]["state"] == "completed"

def test_result_to_a2a_task_failed():
    from flowmind.a2a.types import result_to_a2a_task
    result = {
        "output": None,
        "reasoning": [],
        "degraded": False,
        "error": "无法处理",
    }
    task = result_to_a2a_task(task_id="abc", result=result, include_reasoning=False)
    assert task["status"]["state"] == "failed"

def test_result_to_a2a_task_degraded():
    from flowmind.a2a.types import result_to_a2a_task
    result = {
        "output": {"text": "部分完成"},
        "reasoning": [],
        "degraded": True,
        "error": None,
    }
    task = result_to_a2a_task(task_id="abc", result=result, include_reasoning=False)
    assert task["status"]["state"] == "completed"
    assert task["status"]["degraded"] is True
