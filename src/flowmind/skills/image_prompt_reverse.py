"""image_prompt_reverse 技能：上传风格封面 → 视觉 LLM 反推提示词。

把用户过往效果好的封面图反推成生成式提示词 + 风格标签 + 默认负面词，
用于固化为「生图 skill」模板。错误契约：普通 raise（invoke() 套信封为 INTERNAL）；无 key 显式报错。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import build_chain
from flowmind.skills._secrets import get_api_key
from flowmind.skills._vision_prompt import reverse_prompt

_VERSION = "0.1.0"


class ReversePromptInput(BaseModel):
    """提示词反推入参。"""
    image_url: str = Field(min_length=1, description="参考图 URL")
    hint: str | None = Field(default=None, max_length=500, description="补充说明/目标品类")


class ReversePromptPlan(BaseModel):
    """反推业务载荷。"""
    prompt: str
    style_tags: list[str]
    negative_prompt: str


@skill(id="image_prompt_reverse", name="生图提示词反推", version=_VERSION)
def image_prompt_reverse(inp: ReversePromptInput) -> SkillOutput[ReversePromptPlan]:
    """反推参考图的生成式提示词（视觉 LLM）。

    数据流：入参校验 → 视觉云 LLM 反推 → 字段兜底清洗 → ReversePromptPlan + 推理链。
    """
    cfg = load_config().image_skill
    api_key = get_api_key(cfg.reverse_prompt_key_env)
    if not api_key:
        raise ValueError(
            f"未设置环境变量 {cfg.reverse_prompt_key_env}。云优先原则：提示词反推必须走视觉云 LLM。"
        )

    obj = reverse_prompt(
        image_url=inp.image_url,
        hint=inp.hint,
        api_key=api_key,
        api_base=cfg.reverse_prompt_api_base,
        model=cfg.reverse_prompt_model,
        timeout_s=cfg.reverse_prompt_timeout_s,
    )

    prompt = str(obj.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("视觉 LLM 未返回有效的 prompt")

    style_tags: list[str] = []
    raw_tags = obj.get("style_tags")
    if isinstance(raw_tags, list):
        style_tags = [str(t).strip() for t in raw_tags[:5] if str(t).strip()]

    negative = str(obj.get("negative_prompt") or "").strip() or (
        "no text, no watermark, no blurry, no distorted faces"
    )

    chain = build_chain(
        conclusion=f"反推提示词成功（风格标签 {len(style_tags)} 个）",
        causal_analysis="视觉 LLM 观察参考图 → 生成式提示词 + 风格标签 + 负面词",
        risk_note="反推提示词非确定，建议生成后人工挑选并与原图对比校准风格。",
    )
    return SkillOutput(
        data=ReversePromptPlan(prompt=prompt, style_tags=style_tags, negative_prompt=negative),
        reasoning=[chain], confidence=0.85, sample_size=1,
    )