"""营销生图技能 — 重构后链路测试：云优先，彻底禁用 backend=mock。

去 mock 原则：
- backend="mock" 与 auto 无 key 都抛 ValueError（显式 mock 已禁用）
- 有 ALLIN_API_KEY 时 auto/allin_api → AllInApiBackend
- httpx 全程打桩，测抽取 + 出图完整链路
- 离线测试必须显式 monkeypatch 掉 select_backend，不能走 MockBackend
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest

import flowmind.skills  # noqa: F401
from flowmind.config import FlowmindConfig, MarketingImageConfig, save_config
from flowmind.skill import invoke
from flowmind.skills._image_backend import (
    AllInApiBackend,
    resolve_api_key,
    select_backend,
)
from flowmind.skills._scene_extractor import (
    ChatExtractor,
    PassthroughExtractor,
)


# ---------- 工具 ----------

def _real_args(**over):
    """云优先：默认走 auto 后端；无 key 必然 ValueError（单测自行打桩）。"""
    base = {"prompt": "白瓷盘, 蒸汽升腾, 自然光, 电商产品摄影"}
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
# 1. marketing_copy 抽取链路（所有 backend="mock" 改为打桩 _select_image_backend → AllInApiBackend mock-httpx）
# =========================================================================

def _force_allin_stub(client: httpx.Client, api_key: str = "sk-test"):
    """离线测试 helper：把 select_backend 替换成固定 AllInApiBackend(mock httpx)。

    注：_select_scene_extractor 的 cfg 实参是 FlowmindConfig → cfg.marketing_image.extractor_mode；
    而 _select_image_backend 里 cfg 就是 marketing_image 子配置本身（传的是 cfg.marketing_image）。
    为安全统一：先 try cfg.marketing_image.extractor_mode，except AttributeError 再 try cfg.extractor_mode。
    """
    from flowmind.skills._image_backend import AllInApiBackend as _AllIn
    from flowmind.skills._scene_extractor import (
        ChatExtractor as _Chat,
        PassthroughExtractor as _Pass,
    )

    def _fake_backend(inp_backend, cfg):
        return _AllIn(api_key=api_key, client=client)

    def _fake_extractor(cfg):
        mode = "passthrough"
        try:
            mode = cfg.marketing_image.extractor_mode
        except AttributeError:
            try:
                mode = cfg.extractor_mode
            except AttributeError:
                mode = "passthrough"
        if mode == "chat":
            return _Chat(api_key=api_key, client=client)
        return _Pass()

    return _fake_backend, _fake_extractor


def test_no_marketing_copy_yields_user_prompt_source(tmp_path, monkeypatch):
    """只给 prompt → prompt_source=user_prompt；后端强制 AllIn mock httpx 出图。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLIN_API_KEY", raising=False)

    captured: dict[str, Any] = {}
    client = _mock_client(_img_handler(captured))
    fb, fe = _force_allin_stub(client)
    monkeypatch.setattr("flowmind.skills.marketing_image_gen._select_image_backend", fb)
    monkeypatch.setattr("flowmind.skills.marketing_image_gen._select_scene_extractor", fe)

    result = invoke("marketing_image_gen", _real_args())
    assert result.ok is True
    assert result.data.prompt_source == "user_prompt"
    assert result.data.extracted_scene is None
    assert result.data.backend_used == "allin_api"
    assert captured["url"].endswith("/v1/images/generations")


def test_marketing_copy_passthrough_yields_merged(tmp_path, monkeypatch):
    """marketing_copy + prompt + passthrough 抽取 → merged。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLIN_API_KEY", raising=False)

    cfg_path = tmp_path / "flowmind.config.toml"
    save_config(
        FlowmindConfig(
            marketing_image=MarketingImageConfig(extractor_mode="passthrough")
        ),
        path=cfg_path,
    )

    captured: dict[str, Any] = {}
    client = _mock_client(_img_handler(captured))
    fb, fe = _force_allin_stub(client)
    monkeypatch.setattr("flowmind.skills.marketing_image_gen._select_image_backend", fb)
    monkeypatch.setattr("flowmind.skills.marketing_image_gen._select_scene_extractor", fe)

    result = invoke("marketing_image_gen", _real_args(
        prompt="电商产品摄影",
        marketing_copy="酸菜鱼预制菜, 山野到家, 一口酸爽",
    ))
    assert result.ok is True
    assert result.data.prompt_source == "merged"
    assert result.data.extracted_scene is not None
    assert "酸菜鱼" in result.data.extracted_scene
    assert any("extractor=passthrough" in n for n in result.data.sampling_notes)


def test_marketing_copy_alone_yields_extracted_from_copy(tmp_path, monkeypatch):
    """只给 marketing_copy → extracted_from_copy。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLIN_API_KEY", raising=False)
    save_config(
        FlowmindConfig(marketing_image=MarketingImageConfig(extractor_mode="passthrough")),
        path=tmp_path / "flowmind.config.toml",
    )

    captured: dict[str, Any] = {}
    client = _mock_client(_img_handler(captured))
    fb, fe = _force_allin_stub(client)
    monkeypatch.setattr("flowmind.skills.marketing_image_gen._select_image_backend", fb)
    monkeypatch.setattr("flowmind.skills.marketing_image_gen._select_scene_extractor", fe)

    result = invoke("marketing_image_gen", {
        "prompt": "",
        "marketing_copy": "酸菜鱼预制菜, 山野到家, 一口酸爽",
    })
    assert result.ok is True
    assert result.data.prompt_source == "extracted_from_copy"
    assert result.data.extracted_scene is not None
    assert "酸菜鱼" in result.data.extracted_scene


def test_marketing_copy_with_prompt_yields_merged(tmp_path, monkeypatch):
    """marketing_copy + prompt → merged；resolved_prompt 含「附加要求/原始文案」。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLIN_API_KEY", raising=False)

    cfg_path = tmp_path / "flowmind.config.toml"
    save_config(
        FlowmindConfig(
            marketing_image=MarketingImageConfig(extractor_mode="passthrough")
        ),
        path=cfg_path,
    )

    captured: dict[str, Any] = {}
    client = _mock_client(_img_handler(captured))
    fb, fe = _force_allin_stub(client)
    monkeypatch.setattr("flowmind.skills.marketing_image_gen._select_image_backend", fb)
    monkeypatch.setattr("flowmind.skills.marketing_image_gen._select_scene_extractor", fe)

    result = invoke("marketing_image_gen", _real_args(
        prompt="电商产品摄影",
        marketing_copy="酸菜鱼预制菜, 山野到家, 一口酸爽",
    ))
    assert result.ok is True
    assert result.data.prompt_source == "merged"
    assert "附加要求" in result.data.resolved_prompt
    assert "原始文案" in result.data.resolved_prompt


def test_passthrough_extractor_returns_copy_verbatim():
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
    captured: dict[str, Any] = {}
    client = _mock_client(_chat_handler(captured, content="ok"))
    ext = ChatExtractor(api_key="test-key", client=client)
    ext.extract(marketing_copy="copy", hint="暖色调")
    user_msg = captured["body"]["messages"][1]["content"]
    assert "copy" in user_msg
    assert "暖色调" in user_msg
    assert "附加要求" in user_msg


def test_chat_extractor_raises_on_empty_key():
    ext = ChatExtractor(api_key="")
    with pytest.raises(ValueError, match="ALLIN_API_KEY"):
        ext.extract(marketing_copy="x")


# =========================================================================
# 3. AllInApiBackend — mocked HTTP
# =========================================================================

def test_allin_api_backend_calls_images_generations_endpoint():
    captured: dict[str, Any] = {}
    client = _mock_client(_img_handler(captured, urls=["https://x/1.png"]))
    backend = AllInApiBackend(api_key="test-key", client=client)
    images = backend.generate(
        prompt="白瓷盘蒸汽", negative_prompt="",
        width=1080, height=1440, n=1, seed=42, save_dir=None,
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
# 4. select_backend 路由 + key 安全（云优先：彻底禁 backend=mock）
# =========================================================================

def test_select_backend_mock_explicitly_disabled(monkeypatch):
    """显式 backend="mock" → 必须抛 ValueError（云优先：一切生图走云 API）。"""
    monkeypatch.setenv("ALLIN_API_KEY", "sk-test")
    with pytest.raises(ValueError, match="显式 mock 出图后端已禁用"):
        select_backend(
            requested="mock",
            cfg_allin_key_env="ALLIN_API_KEY",
            cfg_allin_base="https://allin-api.com",
            cfg_allin_model="gpt-image-2",
            cfg_allin_timeout_s=60.0,
        )


def test_select_backend_auto_raises_without_key(monkeypatch):
    """auto + 无 key → ValueError，不再静默降级 mock。"""
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


def test_select_backend_allin_api_force_uses_env_key(monkeypatch):
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
    os.environ["ALLIN_API_KEY"] = "  sk-test  "
    assert resolve_api_key("ALLIN_API_KEY") == "sk-test"
    os.environ["ALLIN_API_KEY"] = "   "
    assert resolve_api_key("ALLIN_API_KEY") is None


# =========================================================================
# 5. 端到端 — mock httpx，跑完整链路（marketing_copy → 抽 → 出图）
# =========================================================================

def test_end_to_end_with_real_backend_and_passthrough(tmp_path, monkeypatch):
    """端到端：chat 抽 + allin_api 出图，httpx 全程 mock。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALLIN_API_KEY", "sk-test")

    img_captured: dict[str, Any] = {}
    chat_captured: dict[str, Any] = {}

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

    # 直接替换模块内部 helper，防止 ``with httpx.Client() as c:`` 重新建 client
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

    cfg_path = tmp_path / "flowmind.config.toml"
    save_config(
        FlowmindConfig(
            marketing_image=MarketingImageConfig(extractor_mode="chat")
        ),
        path=cfg_path,
    )

    result = invoke("marketing_image_gen", _real_args(
        prompt="电商产品摄影",
        marketing_copy="酸菜鱼预制菜, 山野到家, 一口酸爽",
    ))

    assert result.ok is True
    plan = result.data
    assert plan.backend_used == "allin_api"
    assert plan.prompt_source == "merged"
    assert plan.extracted_scene == "白瓷盘蒸汽"
    assert plan.images[0].url == "https://api.example.com/gen/1.png"
    assert chat_captured["url"].endswith("/v1/chat/completions")
    assert img_captured["url"].endswith("/v1/images/generations")
    assert img_captured["body"]["model"] == "gpt-image-2"


def test_end_to_end_explicit_mock_disabled(tmp_path, monkeypatch):
    """显式 backend="mock" 必须 ValueError（云优先：绝无离线占位图）。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLIN_API_KEY", raising=False)
    save_config(
        FlowmindConfig(marketing_image=MarketingImageConfig(extractor_mode="passthrough")),
        path=tmp_path / "flowmind.config.toml",
    )
    # 直接调 invoke，backend=mock，不打桩任何 helper → 走真实 select_backend
    result = invoke("marketing_image_gen", {
        "prompt": "",
        "backend": "mock",
        "marketing_copy": "酸菜鱼预制菜, 一口酸爽",
    })
    assert result.ok is False
    # ValueError 被技能运行时包成 INTERNAL 错误，error.message 含「显式 mock」
    assert result.error.code == "INTERNAL"
    assert "显式 mock 出图后端已禁用" in result.error.message


# =========================================================================
# 6. 入参 schema 向后兼容
# =========================================================================

def test_input_schema_makes_prompt_optional():
    from flowmind.skill import registry
    spec = registry()["marketing_image_gen"]
    schema = spec.input_model.model_json_schema()
    assert "prompt" not in schema.get("required", [])
    assert "marketing_copy" not in schema.get("required", [])
    assert "marketing_copy" in schema["properties"]
    assert "prompt" in schema["properties"]


def test_input_rejects_both_empty():
    result = invoke("marketing_image_gen", {"prompt": "", "marketing_copy": ""})
    assert result.ok is False and result.error.code == "VALIDATION"


def test_marketing_copy_optional_in_input(tmp_path, monkeypatch):
    """无 marketing_copy，纯 prompt 路径 → user_prompt；后端强制打桩 AllIn mock httpx。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLIN_API_KEY", raising=False)

    captured: dict[str, Any] = {}
    client = _mock_client(_img_handler(captured))
    fb, fe = _force_allin_stub(client)
    monkeypatch.setattr("flowmind.skills.marketing_image_gen._select_image_backend", fb)
    monkeypatch.setattr("flowmind.skills.marketing_image_gen._select_scene_extractor", fe)

    result = invoke("marketing_image_gen", _real_args())
    assert result.ok is True
    assert result.data.prompt_source == "user_prompt"
