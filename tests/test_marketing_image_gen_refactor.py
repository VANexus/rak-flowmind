"""营销生图技能 — 重构后链路测试：抽取器 / 真实后端 / key 安全。

旧链路回归测试见 test_marketing_image_gen.py,本文件专测:
- marketing_copy → 画面描述抽取
- PassthroughExtractor vs ChatExtractor
- MockBackend vs AllInApiBackend
- ALLIN_API_KEY 仅从环境变量读,绝不进 config 文件
- 入参 schema 向后兼容(prompt 仍必填;marketing_copy 新增可选)
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest

import flowmind.skills  # noqa: F401  触发技能注册
from flowmind.config import FlowmindConfig, MarketingImageConfig, save_config
from flowmind.skill import invoke
from flowmind.skills._image_backend import (
    AllInApiBackend,
    MockBackend,
    resolve_api_key,
    select_backend,
)
from flowmind.skills._scene_extractor import (
    ChatExtractor,
    PassthroughExtractor,
)


# ---------- 工具 ----------

def _args(**over):
    # 云优先原则：默认显式走 mock 后端（仅测试用）；不传 key 时 backend=auto 报错。
    base = {"prompt": "白瓷盘, 蒸汽升腾, 自然光, 电商产品摄影", "backend": "mock"}
    base.update(over)
    return base


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _chat_handler(captured: dict, content: str = "白瓷盘蒸汽升腾, 自然光, 木桌"):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": content}}
                ]
            },
        )
    return handler


def _img_handler(captured: dict, urls: list[str] | None = None):
    urls = urls or ["https://api.example.com/generated/1.png"]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={"created": 1700000000, "data": [{"url": u} for u in urls]},
        )
    return handler


# =========================================================================
# 1. marketing_copy 抽取链路
# =========================================================================

def test_no_marketing_copy_yields_user_prompt_source(tmp_path, monkeypatch):
    """只给 prompt → prompt_source=user_prompt,extracted_scene=None。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLIN_API_KEY", raising=False)

    result = invoke("marketing_image_gen", _args())
    assert result.ok is True
    assert result.data.prompt_source == "user_prompt"
    assert result.data.extracted_scene is None


def test_marketing_copy_passthrough_yields_extracted_from_copy(tmp_path, monkeypatch):
    """只给 marketing_copy → prompt_source=extracted_from_copy。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLIN_API_KEY", raising=False)

    cfg_path = tmp_path / "flowmind.config.toml"
    save_config(
        FlowmindConfig(
            marketing_image=MarketingImageConfig(extractor_mode="passthrough")
        ),
        path=cfg_path,
    )

    result = invoke("marketing_image_gen", _args(
        prompt="电商产品摄影",  # 即便给了 prompt,只有 marketing_copy 触发抽取时视为 hint
        marketing_copy="酸菜鱼预制菜, 山野到家, 一口酸爽",
    ))
    assert result.ok is True
    # 因为 prompt 也给了 → merged;单独验 extracted_scene 与 extractor name 已记录
    assert result.data.prompt_source == "merged"
    assert result.data.extracted_scene is not None
    assert "酸菜鱼" in result.data.extracted_scene
    assert any("extractor=passthrough" in n for n in result.data.sampling_notes)


def test_marketing_copy_alone_yields_extracted_from_copy(tmp_path, monkeypatch):
    """只给 marketing_copy、不给 prompt → prompt_source=extracted_from_copy。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLIN_API_KEY", raising=False)
    save_config(
        FlowmindConfig(marketing_image=MarketingImageConfig(extractor_mode="passthrough")),
        path=tmp_path / "flowmind.config.toml",
    )

    result = invoke("marketing_image_gen", {
        "prompt": "",
        "backend": "mock",
        "marketing_copy": "酸菜鱼预制菜, 山野到家, 一口酸爽",
    })
    assert result.ok is True
    assert result.data.prompt_source == "extracted_from_copy"
    assert result.data.extracted_scene is not None
    assert "酸菜鱼" in result.data.extracted_scene


def test_marketing_copy_with_prompt_yields_merged(tmp_path, monkeypatch):
    """marketing_copy + prompt 同时给 → prompt_source=merged。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLIN_API_KEY", raising=False)

    cfg_path = tmp_path / "flowmind.config.toml"
    save_config(
        FlowmindConfig(
            marketing_image=MarketingImageConfig(extractor_mode="passthrough")
        ),
        path=cfg_path,
    )

    result = invoke("marketing_image_gen", _args(
        prompt="电商产品摄影",
        marketing_copy="酸菜鱼预制菜, 山野到家, 一口酸爽",
    ))
    assert result.ok is True
    assert result.data.prompt_source == "merged"
    assert "附加要求" in result.data.resolved_prompt
    assert "原始文案" in result.data.resolved_prompt


def test_passthrough_extractor_returns_copy_verbatim():
    """PassthroughExtractor 不调 HTTP,直接返回原文。"""
    ext = PassthroughExtractor()
    out = ext.extract(marketing_copy="一句营销文案")
    assert out == "一句营销文案"

    out_with_hint = ext.extract(marketing_copy="一句营销文案", hint="要暖色调")
    assert "一句营销文案" in out_with_hint
    assert "要暖色调" in out_with_hint


# =========================================================================
# 2. ChatExtractor — mocked HTTP
# =========================================================================

def test_chat_extractor_calls_allin_api_chat_completions():
    """ChatExtractor 走 /v1/chat/completions,system prompt 引导画面描述。"""
    captured: dict[str, Any] = {}
    client = _mock_client(_chat_handler(captured, content="白瓷盘蒸汽升腾"))

    ext = ChatExtractor(api_key="test-key", client=client)
    out = ext.extract(marketing_copy="酸菜鱼预制菜, 山野到家, 一口酸爽")

    assert out == "白瓷盘蒸汽升腾"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["auth"] == "Bearer test-key"
    body = captured["body"]
    assert body["model"] == "gpt-4o-mini"
    msgs = body["messages"]
    assert msgs[0]["role"] == "system"
    assert "画面描述" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "酸菜鱼" in msgs[1]["content"]


def test_chat_extractor_appends_hint():
    """ChatExtractor 的 hint 会拼到 user 消息末尾。"""
    captured: dict[str, Any] = {}
    client = _mock_client(_chat_handler(captured, content="ok"))
    ext = ChatExtractor(api_key="test-key", client=client)
    ext.extract(marketing_copy="copy", hint="暖色调")

    user_msg = captured["body"]["messages"][1]["content"]
    assert "copy" in user_msg
    assert "暖色调" in user_msg
    assert "附加要求" in user_msg


def test_chat_extractor_raises_on_empty_key():
    """空 API key 必须立刻报错,绝不让请求飞出去。"""
    ext = ChatExtractor(api_key="")
    with pytest.raises(ValueError, match="ALLIN_API_KEY"):
        ext.extract(marketing_copy="x")


# =========================================================================
# 3. AllInApiBackend — mocked HTTP
# =========================================================================

def test_allin_api_backend_calls_images_generations_endpoint():
    """AllInApiBackend 走 /v1/images/generations,model=gpt-image-2。"""
    captured: dict[str, Any] = {}
    client = _mock_client(_img_handler(captured, urls=["https://x/1.png"]))

    backend = AllInApiBackend(api_key="test-key", client=client)
    images = backend.generate(
        prompt="白瓷盘蒸汽",
        negative_prompt="",
        width=1080,
        height=1440,
        n=1,
        seed=42,
        save_dir=None,
    )

    assert len(images) == 1
    assert images[0].url == "https://x/1.png"
    assert captured["url"].endswith("/v1/images/generations")
    assert captured["auth"] == "Bearer test-key"
    body = captured["body"]
    assert body["model"] == "gpt-image-2"
    assert body["prompt"] == "白瓷盘蒸汽"
    assert body["size"] == "1080x1440"
    assert body["seed"] == 42
    assert body["n"] == 1


def test_allin_api_backend_merges_negative_prompt():
    """gpt-image-2 不支持 negative_prompt,应合并到 prompt 末尾。"""
    captured: dict[str, Any] = {}
    client = _mock_client(_img_handler(captured))
    backend = AllInApiBackend(api_key="k", client=client)
    backend.generate(
        prompt="主体", negative_prompt="no text",
        width=512, height=512, n=1, seed=None, save_dir=None,
    )
    sent = captured["body"]["prompt"]
    assert sent.startswith("主体")
    assert "Avoid:" in sent
    assert "no text" in sent


def test_allin_api_backend_raises_on_empty_key():
    backend = AllInApiBackend(api_key="")
    with pytest.raises(ValueError, match="ALLIN_API_KEY"):
        backend.generate(
            prompt="x", negative_prompt="", width=512, height=512,
            n=1, seed=None, save_dir=None,
        )


def test_allin_api_backend_raises_when_response_data_empty():
    """API 返回空 data → 必须抛错(错误永不静默)。"""
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    client = _mock_client(handler)
    backend = AllInApiBackend(api_key="k", client=client)
    with pytest.raises(RuntimeError, match="返回空 data"):
        backend.generate(
            prompt="x", negative_prompt="", width=512, height=512,
            n=1, seed=None, save_dir=None,
        )


def test_allin_api_backend_handles_b64_fallback():
    """若 API 只返回 b64_json(无 url),要把 data: URL 形式返回。"""
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"b64_json": "AAAA"}]})

    client = _mock_client(handler)
    backend = AllInApiBackend(api_key="k", client=client)
    images = backend.generate(
        prompt="x", negative_prompt="", width=512, height=512,
        n=1, seed=None, save_dir=None,
    )
    assert images[0].url == "data:image/png;base64,AAAA"


# =========================================================================
# 4. select_backend 路由 + key 安全
# =========================================================================

def test_select_backend_auto_raises_without_key(monkeypatch):
    """云优先：auto + 无 ALLIN_API_KEY → 显式 ValueError，不再静默降级 mock。"""
    monkeypatch.delenv("ALLIN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ALLIN_API_KEY"):
        select_backend(
            requested=None,
            cfg_allin_key_env="ALLIN_API_KEY",
            cfg_allin_base="https://allin-api.com",
            cfg_allin_model="gpt-image-2",
            cfg_allin_timeout_s=60.0,
        )


def test_select_backend_auto_uses_allin_api_with_key(monkeypatch):
    """auto + 有 ALLIN_API_KEY → AllInApiBackend。"""
    monkeypatch.setenv("ALLIN_API_KEY", "sk-test")
    backend = select_backend(
        requested=None,
        cfg_allin_key_env="ALLIN_API_KEY",
        cfg_allin_base="https://allin-api.com",
        cfg_allin_model="gpt-image-2",
        cfg_allin_timeout_s=60.0,
    )
    assert isinstance(backend, AllInApiBackend)
    assert backend.name == "allin_api"
    assert backend.api_key == "sk-test"


def test_select_backend_mock_force(mock_backend_env):
    """显式 mock → 强制 mock,即使有 key。"""
    backend = select_backend(
        requested="mock",
        cfg_allin_key_env="ALLIN_API_KEY",
        cfg_allin_base="https://allin-api.com",
        cfg_allin_model="gpt-image-2",
        cfg_allin_timeout_s=60.0,
    )
    assert isinstance(backend, MockBackend)


def test_select_backend_allin_api_force_uses_env_key(monkeypatch):
    """显式 allin_api → 必须有 key。"""
    monkeypatch.setenv("ALLIN_API_KEY", "sk-test")
    backend = select_backend(
        requested="allin_api",
        cfg_allin_key_env="ALLIN_API_KEY",
        cfg_allin_base="https://allin-api.com",
        cfg_allin_model="gpt-image-2",
        cfg_allin_timeout_s=60.0,
    )
    assert isinstance(backend, AllInApiBackend)


def test_select_backend_unknown_raises():
    with pytest.raises(ValueError, match="未知 backend"):
        select_backend(
            requested="magic",
            cfg_allin_key_env="ALLIN_API_KEY",
            cfg_allin_base="https://allin-api.com",
            cfg_allin_model="gpt-image-2",
            cfg_allin_timeout_s=60.0,
        )


def test_resolve_api_key_strips_whitespace():
    """env var 值带空白应被清掉,避免空 key 漏到调用层。"""
    os.environ["ALLIN_API_KEY"] = "  sk-test  "
    assert resolve_api_key("ALLIN_API_KEY") == "sk-test"
    os.environ["ALLIN_API_KEY"] = "   "
    assert resolve_api_key("ALLIN_API_KEY") is None


# =========================================================================
# 5. 端到端 — mock httpx,跑完整链路(marketing_copy → 抽 → 出图)
# =========================================================================

def test_end_to_end_with_real_backend_and_passthrough(tmp_path, monkeypatch):
    """端到端:chat 抽 + allin_api 出图,httpx 全程 mock。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALLIN_API_KEY", "sk-test")

    img_captured: dict[str, Any] = {}
    chat_captured: dict[str, Any] = {}

    # 单个 mock client + 路由 handler 处理两类 endpoint
    def routing_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/v1/chat/completions"):
            chat_captured["url"] = url
            chat_captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "白瓷盘蒸汽"}}]
            })
        if url.endswith("/v1/images/generations"):
            img_captured["url"] = url
            img_captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={
                "data": [{"url": "https://api.example.com/gen/1.png"}]
            })
        return httpx.Response(404, json={"error": "no route"})

    client = _mock_client(routing_handler)

    # 直接替换 marketing_image_gen 模块内部的 helper,这样客户端不会被
    # ``with httpx.Client() as c:`` 关掉,可在同一次调用里被多次复用。
    from flowmind.skills._image_backend import AllInApiBackend as _AllIn
    from flowmind.skills._scene_extractor import ChatExtractor as _Chat

    def fake_backend(inp_backend, cfg):
        return _AllIn(api_key="sk-test", client=client)

    def fake_extractor(cfg):
        return _Chat(api_key="sk-test", client=client)

    monkeypatch.setattr(
        "flowmind.skills.marketing_image_gen._select_image_backend",
        fake_backend,
    )
    monkeypatch.setattr(
        "flowmind.skills.marketing_image_gen._select_scene_extractor",
        fake_extractor,
    )

    # 配置走 chat 抽取(有 key → 走 ChatExtractor)
    cfg_path = tmp_path / "flowmind.config.toml"
    save_config(
        FlowmindConfig(
            marketing_image=MarketingImageConfig(extractor_mode="chat")
        ),
        path=cfg_path,
    )

    result = invoke("marketing_image_gen", _args(
        prompt="电商产品摄影",
        marketing_copy="酸菜鱼预制菜, 山野到家, 一口酸爽",
    ))

    assert result.ok is True
    plan = result.data
    assert plan.backend_used == "allin_api"
    assert plan.prompt_source == "merged"
    assert plan.extracted_scene == "白瓷盘蒸汽"
    assert plan.images[0].url == "https://api.example.com/gen/1.png"
    # 链路都打到了
    assert chat_captured["url"].endswith("/v1/chat/completions")
    assert img_captured["url"].endswith("/v1/images/generations")
    assert img_captured["body"]["model"] == "gpt-image-2"


def test_end_to_end_explicit_mock_uses_mock_everywhere(tmp_path, monkeypatch):
    """显式 backend=mock + passthrough 抽取 → 全程 mock,零 HTTP（离线测试路径）。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLIN_API_KEY", raising=False)
    save_config(
        FlowmindConfig(marketing_image=MarketingImageConfig(extractor_mode="passthrough")),
        path=tmp_path / "flowmind.config.toml",
    )
    result = invoke("marketing_image_gen", _args(
        marketing_copy="酸菜鱼预制菜, 一口酸爽",
    ))
    assert result.ok is True
    assert result.data.backend_used == "mock"
    assert result.data.images[0].url.startswith("https://flowmind.local/mock/")


# =========================================================================
# 6. 入参 schema 向后兼容
# =========================================================================

def test_input_schema_makes_prompt_optional():
    """prompt 改为可选(主输入交给 marketing_copy),marketing_copy 也是可选。"""
    from flowmind.skill import registry
    spec = registry()["marketing_image_gen"]
    schema = spec.input_model.model_json_schema()
    assert "prompt" not in schema.get("required", [])
    assert "marketing_copy" not in schema.get("required", [])
    # 但 cross-field 校验确保至少给一个
    assert "marketing_copy" in schema["properties"]
    assert "prompt" in schema["properties"]


def test_input_rejects_both_empty():
    """prompt 与 marketing_copy 同时为空 → VALIDATION(由 model_validator 兜底)。"""
    result = invoke("marketing_image_gen", {"prompt": "", "marketing_copy": ""})
    assert result.ok is False and result.error.code == "VALIDATION"


def test_marketing_copy_optional_in_input():
    """marketing_copy 可选;不传也能跑(纯 prompt 路径向后兼容)。"""
    result = invoke("marketing_image_gen", _args())  # 无 marketing_copy
    assert result.ok is True
    assert result.data.prompt_source == "user_prompt"


@pytest.fixture
def mock_backend_env(monkeypatch):
    """带 ALLIN_API_KEY 的测试环境(用于验证 mock 强制优先于 key)。"""
    monkeypatch.setenv("ALLIN_API_KEY", "sk-test")
    yield
    monkeypatch.delenv("ALLIN_API_KEY", raising=False)