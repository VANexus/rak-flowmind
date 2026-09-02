"""content_image_gen 技能：平台比例 AI 配图。

复用 _image_backend.AllInApiBackend（OpenAI 兼容 /v1/images/generations），
把 api_base 指向 ciyuansky（config.image_api_base）。平台 → 像素尺寸：
xhs 1080x1440（3:4 图文）、wechat 1920x1080（16:9 头图）、douyin 1080x1920（9:16 短视频封面）。

云优先：无 key 显式报错（raise），不静默降级；显式 backend="mock" 仅用于测试。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import PLATFORMS, ContentPlatform, build_chain
from flowmind.skills._image_backend import AllInApiBackend, MockBackend
from flowmind.skills._secrets import get_api_key

_VERSION = "0.1.0"

Backend = str  # "mock" | "auto"（auto=有 key 走真实，无 key 报错）


class ContentImageInput(BaseModel):
    """AI 配图入参。"""
    platform: ContentPlatform = Field(description="目标平台：xhs / wechat / douyin")
    prompt: str = Field(min_length=1, max_length=2000, description="画面描述")
    count: int = Field(default=1, ge=1, le=4, description="生成张数（1-4）")
    negative_prompt: str | None = Field(default=None, max_length=500)
    backend: Backend | None = Field(default="auto", description="mock 仅测试；auto=云 API")


class ContentImageResult(BaseModel):
    """单张生成结果。"""
    index: int
    url: str


class ContentImagePlan(BaseModel):
    """AI 配图业务载荷。"""
    platform: str
    width: int
    height: int
    backend_used: str
    images: list[ContentImageResult]


@skill(id="content_image_gen", name="平台比例 AI 配图", version=_VERSION)
def content_image_gen(inp: ContentImageInput) -> SkillOutput[ContentImagePlan]:
    """按平台比例（xhs 3:4 / wechat 16:9 / douyin 9:16）生成配图。

    数据流：平台→尺寸 → 选后端（auto=有 key 走云 API，无 key 显式报错）→ 生成 → 计划 + 推理链。
    """
    cfg = load_config().content
    width, height = PLATFORMS[inp.platform]["pixels"]

    chosen = (inp.backend or "auto").lower()
    if chosen == "mock":
        backend = MockBackend()
    elif chosen in ("auto", "allin_api"):
        api_key = get_api_key(cfg.image_api_key_env)
        if not api_key:
            raise ValueError(
                f"未设置环境变量 {cfg.image_api_key_env}。云优先原则：真实配图必须走云 API；"
                "如需离线测试请显式传 backend='mock'。"
            )
        backend = AllInApiBackend(
            api_base=cfg.image_api_base,
            api_key=api_key,
            model=cfg.image_model,
            timeout_s=cfg.image_timeout_s,
        )
    else:
        raise ValueError(f"未知 backend：{chosen}")

    negative = inp.negative_prompt or "no text, no watermark, no blurry, no distorted, no extra fingers"

    raw = backend.generate(
        prompt=inp.prompt,
        negative_prompt=negative,
        width=width,
        height=height,
        n=inp.count,
        seed=None,
        save_dir=None,
    )
    images = [ContentImageResult(index=g.index, url=g.url) for g in raw]

    chain = build_chain(
        conclusion=f"为 {inp.platform} 生成 {len(images)} 张 {width}x{height} 配图（backend={backend.name}）",
        causal_analysis=f"平台 {inp.platform} → 像素 {width}x{height}；提示词 {len(inp.prompt)} 字",
        risk_note="AI 生图非确定性，仅作创意草稿；正式发布前请人工挑选并确认版权。",
    )
    return SkillOutput(
        data=ContentImagePlan(
            platform=inp.platform, width=width, height=height,
            backend_used=backend.name, images=images,
        ),
        reasoning=[chain],
        confidence=0.9,
        sample_size=len(images),
    )
