"""_llm_client 测试：通用 Anthropic 兼容 LLM 客户端（httpx MockTransport 注入）。

覆盖：
- 请求格式（endpoint / Bearer key / model / system 顶层字段 / user 消息）
- 成功：解析 content[type=text] 的 JSON 对象
- 推理模型 thinking 块在前 → 取第一个 type=text
- 无 key 显式报错（云优先：不静默降级）
- 非法 JSON / 非对象回复 → 结构化报错
- 网络错误分类：超时=environment / 5xx=transient(可重试) / 4xx=video
"""
from __future__ import annotations

import json

import httpx
import pytest

from flowmind.skills._llm_client import LLMClientError, llm_json


def _client(captured: dict, content: str, status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(status, json={
            "content": [{"type": "text", "text": content}],
        })

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_llm_json_request_format():
    captured: dict = {}
    reply = json.dumps({"title": "测试", "body": "内容"}, ensure_ascii=False)
    client = _client(captured, reply)

    out = llm_json(prompt='{"subject":"保温杯"}', system="你是文案助手", api_key="k-test", client=client)

    assert captured["url"].endswith("/v1/messages")
    assert captured["auth"] == "Bearer k-test"
    body = captured["body"]
    assert body["model"] == "LongCat-2.0"
    assert body["system"] == "你是文案助手"  # Anthropic 兼容：system 是顶层字段
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == '{"subject":"保温杯"}'
    assert out == {"title": "测试", "body": "内容"}


def test_llm_json_skips_thinking_blocks():
    """推理模型 thinking 块在前，取第一个 type=text。"""
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["seen"] = True
        return httpx.Response(200, json={
            "content": [
                {"type": "thinking", "thinking": "内部推理..."},
                {"type": "text", "text": '{"ok": true}'},
            ],
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = llm_json(prompt="p", system="s", api_key="k", client=client)
    assert out == {"ok": True}


def test_llm_json_raises_on_empty_key():
    with pytest.raises(ValueError, match="AI_LLM_API_KEY"):
        llm_json(prompt="p", system="s", api_key="", client=_client({}, "{}"))


def test_llm_json_rejects_invalid_json():
    client = _client({}, "抱歉，我无法生成")
    with pytest.raises(LLMClientError) as ei:
        llm_json(prompt="p", system="s", api_key="k", client=client)
    assert ei.value.category == "unknown"
    assert ei.value.retriable is False


def test_llm_json_rejects_non_object():
    client = _client({}, json.dumps(["a", "b"]))
    with pytest.raises(LLMClientError, match="不是 JSON 对象"):
        llm_json(prompt="p", system="s", api_key="k", client=client)


def test_llm_json_http_5xx_transient():
    client = _client({}, "", status=503)
    with pytest.raises(LLMClientError) as ei:
        llm_json(prompt="p", system="s", api_key="k", client=client)
    assert ei.value.category == "transient"
    assert ei.value.retriable is True


def test_llm_json_http_4xx_video():
    client = _client({}, "", status=422)
    with pytest.raises(LLMClientError) as ei:
        llm_json(prompt="p", system="s", api_key="k", client=client)
    assert ei.value.category == "video"
    assert ei.value.retriable is False


def test_llm_json_accepts_fenced_json():
    """LLM 用 ```json 围栏包 JSON 也能解析。"""
    client = _client({}, '```json\n{"ok": 1}\n```')
    out = llm_json(prompt="p", system="s", api_key="k", client=client)
    assert out == {"ok": 1}
