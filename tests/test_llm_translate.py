"""_llm_translate 测试：LongCat Anthropic 兼容客户端（httpx MockTransport 注入）。

覆盖：
- 请求格式（endpoint/模型/messages 结构/Bearer key）
- 分批翻译：多句按批切分、批间上下文传递
- 响应解析：JSON 数组译文提取
- 无 key 显式报错（云优先：不静默降级）
- 注入防护：LLM 返回非法 JSON → 报错不吞
- 网络错误分类透传
"""
from __future__ import annotations

import json

import httpx
import pytest

from flowmind.skills._llm_translate import LLMTtranslateError, translate_segments


def _client(captured: dict, content: str) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": content}],
        })

    return httpx.Client(transport=httpx.MockTransport(handler))


SEGMENTS = [
    {"index": 0, "begin": 0.0, "end": 2.0, "text": "大家好"},
    {"index": 1, "begin": 2.0, "end": 4.5, "text": "今天讲发动机保养"},
]


def test_translate_request_format():
    captured: dict = {}
    reply = json.dumps([
        {"index": 0, "text": "Hello everyone"},
        {"index": 1, "text": "Today we cover engine maintenance"},
    ], ensure_ascii=False)
    client = _client(captured, reply)

    out = translate_segments(
        SEGMENTS, target_lang="en", source_lang="zh",
        api_key="k-test", client=client,
    )
    assert captured["url"].endswith("/v1/messages")
    assert captured["auth"] == "Bearer k-test"
    body = captured["body"]
    assert body["model"] == "LongCat-2.0"
    # Anthropic 兼容协议：system 是顶层字段
    assert "字幕" in body["system"]
    roles = [m["role"] for m in body["messages"]]
    assert set(roles) == {"user"}
    # 用户消息里带原文与语向
    user_text = body["messages"][0]["content"]
    assert "大家好" in user_text and "en" in user_text
    # 译文按 index 对齐返回
    assert out[0]["text"] == "Hello everyone"
    assert out[1]["begin"] == SEGMENTS[1]["begin"], "时间轴字段原样保留"


def test_translate_raises_on_empty_key():
    with pytest.raises(ValueError, match="LONGCAT_API_KEY"):
        translate_segments(SEGMENTS, target_lang="en",
                           api_key="", client=_client({}, ""))


def test_translate_rejects_invalid_json_reply():
    """LLM 返回非 JSON → 结构化报错，绝不静默吞掉。"""
    client = _client({}, "抱歉，我无法翻译")
    with pytest.raises(LLMTtranslateError):
        translate_segments(SEGMENTS, target_lang="th",
                           api_key="k", client=client)


def test_translate_http_error_propagates_status():
    """5xx → transient 可重试；4xx → video。分类供技能层转 degraded。"""

    def handler_500(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(LLMTtranslateError) as ei:
        translate_segments(
            SEGMENTS, target_lang="en", api_key="k",
            client=httpx.Client(transport=httpx.MockTransport(handler_500)),
        )
    assert ei.value.category == "transient"
    assert ei.value.retriable is True


def test_translate_batches_long_input():
    """超过 batch_size 的句子列表分多次请求，全部译文合并。"""
    many = [{"index": i, "begin": float(i), "end": i + 1.0, "text": f"句{i}"}
            for i in range(7)]
    seen_batch_sizes: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        n = body["messages"][-1]["content"].count("index")
        seen_batch_sizes.append(n)
        reply = json.dumps(
            [{"index": i, "text": f"t{i}"} for i in range(n)], ensure_ascii=False)
        return httpx.Response(200, json={"content": [{"type": "text", "text": reply}]})

    out = translate_segments(
        many, target_lang="en", api_key="k", batch_size=3,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert len(out) == 7
    assert seen_batch_sizes == [3, 3, 1]
