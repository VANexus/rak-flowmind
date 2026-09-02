"""营销生图技能测试：参数推断、四段式链、错误路径、确定性测试后端。

去 mock 原则：显式 mock 出图后端已在 select_backend 禁用（云优先）。
本文件在技能层打桩 _select_image_backend 注入确定性测试后端（仅测试用，
不覆盖业务逻辑），参数推断 / 规则 / 尺寸等路径照常走真实代码。
标记 @pytest.mark.real_backend 的用例不打桩（测后端选择与错误路径）。
"""
from __future__ import annotations

import hashlib

import pytest

import flowmind.skills  # noqa: F401  触发技能注册
import flowmind.skills.marketing_image_gen as mig_mod
from flowmind.config import (
    FlowmindConfig,
    MarketingImageConfig,
    save_config,
)
from flowmind.skill import invoke, registry
from flowmind.skills._image_backend import GeneratedImage, _sanitize_save_dir


# ---------- 工具 ----------

def _args(prompt: str = "酸菜鱼预制菜, 白瓷盘, 自然光, 电商产品摄影", **over):
    base = {"prompt": prompt}
    base.update(over)
    return base


class _FakeTestBackend:
    """确定性测试后端：URL 由 sha256 派生，纯本地可复现（仅测试用）。"""

    name = "fake_test"

    def generate(self, *, prompt, negative_prompt, width, height, n, seed, save_dir):
        if seed is None:
            seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF
        safe = _sanitize_save_dir(save_dir) if save_dir else None
        images: list[GeneratedImage] = []
        for i in range(n):
            img_seed = seed + i
            h = hashlib.sha256()
            for p in (prompt, negative_prompt, width, height, img_seed):
                h.update(str(p).encode("utf-8"))
                h.update(b"|")
            img_id = h.hexdigest()[:12]
            images.append(GeneratedImage(
                index=i + 1,
                url=f"https://test.local/img/{img_id}.png?w={width}&h={height}",
                local_path=f"{safe}/{img_id}.png" if safe else None,
                width=width,
                height=height,
                seed=img_seed,
            ))
        return images


@pytest.fixture(autouse=True)
def _fake_image_backend(request, monkeypatch):
    """默认在技能层打桩出图后端；@pytest.mark.real_backend 用例不打桩。"""
    if request.node.get_closest_marker("real_backend"):
        yield None
        return
    monkeypatch.setattr(mig_mod, "_select_image_backend", lambda backend, cfg: _FakeTestBackend())
    yield


# ---------- 基础 ----------

def test_registers_in_registry():
    """技能必须在注册表里,且 id / name / version 正确。"""
    spec = registry().get("marketing_image_gen")
    assert spec is not None
    assert spec.name == "营销生图（多平台多风格）"
    assert spec.version == "0.1.0"


def test_basic_call_returns_one_image():
    """最小调用:prompt 必填,无 platform/style 推断 → 1 张图。"""
    result = invoke("marketing_image_gen", _args())
    assert result.ok is True
    plan = result.data
    assert plan.num_variants == 1
    assert len(plan.images) == 1
    assert plan.images[0].url.startswith("https://test.local/img/")
    assert plan.images[0].width > 0 and plan.images[0].height > 0
    # 信封四要素齐全
    assert result.trace.trace_id
    assert result.metrics.sample_size == 1
    assert result.metrics.latency_ms >= 0
    assert result.metrics.confidence == 1.0
    # 四段式推理链四要素齐全
    chain = result.reasoning[0]
    assert chain.conclusion and chain.causal_analysis and chain.risk_note


# ---------- 参数推断 ----------

def test_platform_default_resolves_from_config(tmp_path, monkeypatch):
    """未指定 platform 时回退到 config.default_platform。"""
    cfg = FlowmindConfig(marketing_image=MarketingImageConfig(default_platform="douyin"))
    path = tmp_path / "flowmind.config.toml"
    save_config(cfg, path=path)
    monkeypatch.chdir(tmp_path)

    result = invoke("marketing_image_gen", _args())
    assert result.ok is True
    assert result.data.platform == "douyin"
    # douyin 默认 9:16
    assert result.data.aspect_ratio == "9:16"


def test_style_default_resolves_from_config(tmp_path, monkeypatch):
    """未指定 style 时回退到 config.default_style。"""
    cfg = FlowmindConfig(marketing_image=MarketingImageConfig(default_style="promo"))
    path = tmp_path / "flowmind.config.toml"
    save_config(cfg, path=path)
    monkeypatch.chdir(tmp_path)

    result = invoke("marketing_image_gen", _args())
    assert result.ok is True
    assert result.data.style == "promo"


def test_aspect_ratio_defaulted_from_platform():
    """未指定 aspect_ratio 时按 platform 取 config 默认。"""
    result = invoke("marketing_image_gen", _args(platform="douyin"))
    assert result.data.aspect_ratio == "9:16"
    result2 = invoke("marketing_image_gen", _args(platform="banner"))
    assert result2.data.aspect_ratio == "16:9"


def test_aspect_ratio_explicit_wins_over_platform_default():
    """显式 aspect_ratio 应压过平台默认。"""
    result = invoke("marketing_image_gen", _args(platform="douyin", aspect_ratio="1:1"))
    assert result.data.aspect_ratio == "1:1"


def test_aspect_ratio_auto_falls_back_to_platform_default():
    """aspect_ratio='auto' 应按平台默认推断。"""
    result = invoke("marketing_image_gen", _args(platform="xiaohongshu", aspect_ratio="auto"))
    assert result.data.aspect_ratio == "3:4"


def test_negative_prompt_user_wins_over_default():
    """用户传入 negative_prompt 必须采用用户值。"""
    user_neg = "no people, no letters, no logo"
    result = invoke("marketing_image_gen", _args(negative_prompt=user_neg))
    assert result.data.negative_prompt == user_neg


def test_negative_prompt_default_added_when_missing(tmp_path, monkeypatch):
    """未传 negative_prompt 时使用配置默认,且 sampling_notes 有提示。"""
    cfg = FlowmindConfig(
        marketing_image=MarketingImageConfig(default_negative_prompt="no text, no watermark, no blur")
    )
    path = tmp_path / "flowmind.config.toml"
    save_config(cfg, path=path)
    monkeypatch.chdir(tmp_path)

    result = invoke("marketing_image_gen", _args())
    assert result.ok is True
    assert "no watermark" in result.data.negative_prompt
    assert any("negative_prompt" in n for n in result.data.sampling_notes)


def test_brand_override_applied():
    """brand 入参非空 → 数据透传到 plan,并写入 sampling_notes。"""
    brand = {"name": "万味山", "primary_color": "#C8102E", "tagline": "把厨房搬上山"}
    result = invoke("marketing_image_gen", _args(brand=brand))
    assert result.ok is True
    assert result.data.brand_name == "万味山"  # type: ignore[attr-defined]
    assert any("brand" in n for n in result.data.sampling_notes)


@pytest.mark.real_backend
def test_explicit_mock_backend_rejected():
    """去 mock 契约：backend='mock' 显式被拒（云优先，绝不静默出假图）。"""
    result = invoke("marketing_image_gen", _args(backend="mock"))
    assert result.ok is False
    assert "mock" in result.error.message


@pytest.mark.real_backend
def test_backend_auto_requires_key():
    """云优先：backend 缺省（auto）且无 AI_IMAGE_API_KEY → 显式报错，不静默降级 mock。"""
    import os

    saved = os.environ.pop("AI_IMAGE_API_KEY", None)
    try:
        result = invoke("marketing_image_gen", {"prompt": "测试"})
        assert result.ok is False
        assert result.error.code == "INTERNAL"
        assert "AI_IMAGE_API_KEY" in result.error.message
    finally:
        if saved is not None:
            os.environ["AI_IMAGE_API_KEY"] = saved


# ---------- 多版本 / 尺寸 / 成本 ----------

def test_num_variants_three_returns_three():
    """num_variants=3 → 3 张图,3 个不同 seed。"""
    result = invoke("marketing_image_gen", _args(num_variants=3))
    assert len(result.data.images) == 3
    seeds = {img.seed for img in result.data.images}
    assert len(seeds) == 3  # 互不相同
    # 3 张 URL 互不相同
    urls = [img.url for img in result.data.images]
    assert len(set(urls)) == 3


def test_num_variants_cap_four_via_validation():
    """num_variants=5 → VALIDATION。Pydantic 约束 le=4。"""
    result = invoke("marketing_image_gen", _args(num_variants=5))
    assert result.ok is False and result.error.code == "VALIDATION"


def test_num_variants_zero_via_validation():
    result = invoke("marketing_image_gen", _args(num_variants=0))
    assert result.ok is False and result.error.code == "VALIDATION"


def test_dimensions_match_aspect_ratio():
    """image.width/height 必须与 aspect_ratio 严格成比例。"""
    result = invoke("marketing_image_gen", _args(platform="douyin"))  # 9:16
    img = result.data.images[0]
    # 1080x1920 (douyin) ⇒ 9:16
    ratio = img.width / img.height
    assert abs(ratio - 9 / 16) < 0.01


def test_estimated_cost_credit_matches_num_variants(tmp_path, monkeypatch):
    """estimated_cost_credit = num_variants * cfg.credit_per_image。"""
    cfg = FlowmindConfig(marketing_image=MarketingImageConfig(credit_per_image=2))
    path = tmp_path / "flowmind.config.toml"
    save_config(cfg, path=path)
    monkeypatch.chdir(tmp_path)

    result = invoke("marketing_image_gen", _args(num_variants=3))
    assert result.ok is True
    assert result.data.estimated_cost_credit == 3 * 2


def test_save_dir_resolution(tmp_path):
    """save_dir 入参 → images[].local_path 应拼出本地路径。"""
    result = invoke("marketing_image_gen", _args(save_dir=str(tmp_path)))
    img = result.data.images[0]
    assert img.local_path is not None
    assert img.local_path.startswith(str(tmp_path))


# ---------- 确定性 / 种子 ----------

def test_same_seed_same_url():
    """相同 prompt+seed+platform → URL 完全一致。"""
    r1 = invoke("marketing_image_gen", _args(seed=42))
    r2 = invoke("marketing_image_gen", _args(seed=42))
    assert r1.data.images[0].url == r2.data.images[0].url


def test_different_seed_different_url():
    """seed 不同 → URL 不同。"""
    r1 = invoke("marketing_image_gen", _args(seed=42))
    r2 = invoke("marketing_image_gen", _args(seed=43))
    assert r1.data.images[0].url != r2.data.images[0].url


def test_default_seed_deterministic():
    """不传 seed → 用 prompt 的 sha256 派生,固定不变。"""
    r1 = invoke("marketing_image_gen", _args(prompt="固定 prompt"))
    r2 = invoke("marketing_image_gen", _args(prompt="固定 prompt"))
    assert r1.data.images[0].url == r2.data.images[0].url


# ---------- 推理链 ----------

def test_reasoning_chain_has_triggered_rules_and_evidence():
    """四段式链必须含 triggered_rules（来自 evaluate_rules）+ evidence。"""
    result = invoke("marketing_image_gen", _args(platform="douyin"))
    chain = result.reasoning[0]
    # 至少一条规则命中（platform 用 douyin 默认与 xiaohongshu 不同⇒ 命中 MIG-PLATFORM-01）
    rule_ids = [r.rule_id for r in chain.triggered_rules]
    assert rule_ids, "应至少有一条触发规则"
    # 至少一条证据
    assert chain.evidence, "应至少有一条证据"
    # 因子链上 conclusion/因果/风险 都齐
    assert "营销图" in chain.conclusion
    assert "douyin" in chain.conclusion
    assert chain.risk_note


def test_explicit_inputs_skip_default_rules():
    """全部字段显式提供时,MIG-PLATFORM/STYLE/RATIO-01 不应命中。"""
    result = invoke(
        "marketing_image_gen",
        _args(
            platform="douyin",
            style="tech",
            aspect_ratio="9:16",
            negative_prompt="no people",
        ),
    )
    chain = result.reasoning[0]
    rule_ids = [r.rule_id for r in chain.triggered_rules]
    assert "MIG-PLATFORM-01" not in rule_ids
    assert "MIG-STYLE-01" not in rule_ids
    assert "MIG-RATIO-01" not in rule_ids
    assert "MIG-NEGATIVE-01" not in rule_ids


# ---------- 错误路径 ----------

def test_empty_prompt_is_validation_error():
    result = invoke("marketing_image_gen", _args(prompt=""))
    assert result.ok is False and result.error.code == "VALIDATION"


def test_unknown_platform_is_validation_error():
    result = invoke("marketing_image_gen", _args(platform="unknown_platform"))
    assert result.ok is False and result.error.code == "VALIDATION"


def test_unknown_style_is_validation_error():
    result = invoke("marketing_image_gen", _args(style="gibberish_style"))
    assert result.ok is False and result.error.code == "VALIDATION"


def test_unknown_backend_is_validation_error():
    result = invoke("marketing_image_gen", _args(backend="magic_ai"))
    assert result.ok is False and result.error.code == "VALIDATION"


def test_unknown_skill_is_not_found_error():
    result = invoke("marketing_image_gen_doesnt_exist", _args())
    assert result.ok is False and result.error.code == "NOT_FOUND"


# ---------- Manifest 发现 ----------

def test_manifest_lists_marketing_image_gen():
    """manifest.build_manifest 必须把新技能也输出,且 input_schema 是合法 JSON Schema。"""
    from flowmind.manifest import build_manifest

    manifest = build_manifest()
    ids = [s["id"] for s in manifest["skills"]]
    assert "marketing_image_gen" in ids
    entry = next(s for s in manifest["skills"] if s["id"] == "marketing_image_gen")
    assert entry["name"] == "营销生图（多平台多风格）"
    assert entry["version"] == "0.1.0"
    assert entry["input_schema"]["type"] == "object"
    assert "prompt" in entry["input_schema"]["properties"]
    assert entry["reliability_profile"]["deterministic"] is True
