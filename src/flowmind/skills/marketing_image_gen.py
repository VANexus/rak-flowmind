"""营销生图技能:多平台多风格的营销视觉生成。

对外契约:
- 入参(MarketingImageInput):prompt 必填;marketing_copy 可选(用户原始文案,
  走 SceneExtractor 抽取画面描述后并入 prompt);其它字段 Agent 从上下文推断。
- 出参(MarketingImagePlan):生成计划 + 候选图(四段式因果推理链同时返回)。

底层可插拔:
- 画面描述提取:PassthroughExtractor(默认/无网络)/ ChatExtractor(走 allin-api)。
- 图像生成后端:MockBackend(确定性占位)/ AllInApiBackend(模型 gpt-image-2)。

安全:AllInApiBackend / ChatExtractor 的 API key 只从环境变量读取,
不进 config 文件、不进 commit。生产部署由运维导出 ALLIN_API_KEY。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from flowmind.config import MarketingImageConfig, load_config
from flowmind.contracts import Evidence, ReasoningChain, SkillOutput
from flowmind.rules import Rule, evaluate_rules
from flowmind.skill import skill
from flowmind.skills._image_backend import (
    GeneratedImage as _GeneratedImage,
    select_backend,
)
from flowmind.skills._scene_extractor import select_extractor

_VERSION = "0.1.0"

Platform = Literal[
    "wechat_moment",
    "xiaohongshu",
    "douyin",
    "taobao_main",
    "taobao_detail",
    "banner",
    "weibo",
    "video_cover",
    "generic",
]
Style = Literal[
    "minimal",
    "literary",
    "promo",
    "festival",
    "tech",
    "editorial",
    "warm",
    "auto",
]
Backend = Literal["mock", "allin_api", "auto"]
AspectRatio = Literal["1:1", "3:4", "9:16", "4:3", "16:9", "21:9", "2:3", "3:2", "auto"]

_VALID_ASPECT_RATIOS = {"1:1", "3:4", "9:16", "4:3", "16:9", "21:9", "2:3", "3:2"}


class BrandInfo(BaseModel):
    """覆盖默认品牌配置(请求内透传,不必让用户重复说品牌)。"""
    name: str | None = None
    primary_color: str | None = None
    secondary_color: str | None = None
    logo_url: str | None = None
    font_pref: str | None = None
    tagline: str | None = None


class MarketingImageInput(BaseModel):
    """营销生图入参。prompt 与 marketing_copy 至少给一个。

    - ``marketing_copy``:用户原始营销文案,走 SceneExtractor 抽画面描述。
    - ``prompt``:显式画面描述;若同时给 marketing_copy,会作为附加要求并入。
    - 至少给一个(都为空时 Pydantic 校验失败 → 框架返回 VALIDATION)。
    """
    prompt: str = Field(default="", max_length=4096)
    marketing_copy: str | None = Field(default=None, max_length=4096)
    platform: Platform | None = None
    style: Style | None = None
    aspect_ratio: AspectRatio | None = None
    negative_prompt: str | None = None
    brand: BrandInfo | None = None
    reference_image_url: str | None = None
    num_variants: int = Field(default=1, ge=1, le=4)
    seed: int | None = None
    save_dir: str | None = None
    backend: Backend | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> "MarketingImageInput":
        if not (self.prompt and self.prompt.strip()) and not (self.marketing_copy and self.marketing_copy.strip()):
            raise ValueError("prompt 与 marketing_copy 至少给一个")
        return self


class GeneratedImage(BaseModel):
    """单张候选图。"""
    index: int
    url: str
    local_path: str | None = None
    width: int
    height: int
    seed: int | None = None


class MarketingImagePlan(BaseModel):
    """营销生图技能的业务载荷。"""
    prompt: str
    resolved_prompt: str
    prompt_source: str  # "user_prompt" | "extracted_from_copy" | "merged"
    extracted_scene: str | None = None
    platform: str
    style: str
    aspect_ratio: str
    width: int
    height: int
    backend: str  # 用户请求/默认;auto/mock/allin_api/...
    backend_used: str  # 实际调用的后端名;mock | allin_api
    num_variants: int
    images: list[GeneratedImage]
    negative_prompt: str
    estimated_cost_credit: int
    brand_name: str | None = None
    sampling_notes: list[str] = Field(default_factory=list)


# --- helpers ---------------------------------------------------------------

def _platform_label(p: str) -> str:
    table = {
        "wechat_moment": "朋友圈",
        "xiaohongshu": "小红书",
        "douyin": "抖音",
        "taobao_main": "淘宝主图",
        "taobao_detail": "淘宝详情",
        "banner": "Banner",
        "weibo": "微博",
        "video_cover": "视频封面",
        "generic": "通用",
    }
    return table.get(p, p)


def _sanitize_final_prompt(text: str, *, max_len: int = 2000) -> str:
    """对最终 image prompt 做最后一道脱敏，防 extractor 之外的注入。

    - 截断（防超长）
    - 去除代码围栏/反引号（防扮演指令的标记污染）
    - 拒绝已知注入 pattern（命中返回空触发上游回退）
    """
    if not text:
        return ""
    s = text.strip()
    if len(s) > max_len:
        s = s[:max_len].rsplit(" ", 1)[0] or s[:max_len]
    s = s.replace("```", "").replace("`", "")
    lower = s.lower()
    for pat in (
        "ignore previous", "ignore above", "disregard prior",
        "<|im_start|>", "<|im_end|>",
        # 不拦截 "system:" "user:" "assistant:" —— 它们在合法正文里常见
    ):
        if pat in lower:
            return ""
    return s.strip()


def _dimensions(aspect: str, hint: tuple[int, int]) -> tuple[int, int]:
    w_hint, h_hint = hint
    try:
        a, b = (int(x) for x in aspect.split(":", 1))
    except (ValueError, AttributeError):
        return w_hint, h_hint
    if a <= 0 or b <= 0:
        return w_hint, h_hint
    base = max(w_hint, h_hint)
    if b >= a:
        height = base
        width = max(1, round(base * a / b))
    else:
        width = base
        height = max(1, round(base * b / a))
    return int(width), int(height)


def _build_rules(cfg: MarketingImageConfig) -> list[Rule]:
    """推理链的第二、三段由 evaluate_rules 自动产出。"""
    return [
        Rule(
            id="MIG-PLATFORM-01",
            name="平台未指定回退默认",
            expression=f"platform 输入为空 → 默认 {cfg.default_platform}",
            predicate=lambda m: m["platform_input"] is None,
            evidence=lambda m: [
                Evidence(
                    metric="platform 输入", value="未提供",
                    threshold=cfg.default_platform, comparison="==",
                    window="本次请求",
                ),
            ],
        ),
        Rule(
            id="MIG-STYLE-01",
            name="风格未指定回退默认",
            expression=f"style 输入为空 → 默认 {cfg.default_style}",
            predicate=lambda m: m["style_input"] is None,
            evidence=lambda m: [
                Evidence(
                    metric="style 输入", value="未提供",
                    threshold=cfg.default_style, comparison="==",
                    window="本次请求",
                ),
            ],
        ),
        Rule(
            id="MIG-RATIO-01",
            name="比例由平台默认",
            expression="aspect_ratio 未指定或为 auto → 按平台默认",
            predicate=lambda m: m["aspect_ratio_input"] in (None, "auto"),
            evidence=lambda m: [
                Evidence(
                    metric="aspect_ratio 输入",
                    value=m["aspect_ratio_input"] or "auto",
                    threshold=m["aspect_ratio_resolved"],
                    comparison="==", window="本次请求",
                ),
            ],
        ),
        Rule(
            id="MIG-NEGATIVE-01",
            name="默认 negative_prompt 已追加",
            expression="negative_prompt 输入为空 → 用 config 默认",
            predicate=lambda m: m["negative_prompt_added"],
            evidence=lambda m: [
                Evidence(
                    metric="default_negative_prompt 长度",
                    value=m["negative_prompt_length"],
                    threshold=len(cfg.default_negative_prompt),
                    comparison="==",
                ),
            ],
        ),
        Rule(
            id="MIG-BRAND-01",
            name="品牌覆盖",
            expression="brand 输入非空 → 命中=应用请求 brand 覆盖",
            predicate=lambda m: m["brand_overridden"],
            evidence=lambda m: [
                Evidence(
                    metric="brand.name 覆盖值",
                    value=m["brand_name_override"] or "未提供",
                    threshold="未提供", comparison="覆盖",
                    window="本次请求",
                ),
            ],
        ),
        Rule(
            id="MIG-VARIANTS-01",
            name="多版本测款",
            expression=f"num_variants > 1 → 按 {cfg.max_variants} 张上限生成候选",
            predicate=lambda m: m["num_variants"] > 1,
            evidence=lambda m: [
                Evidence(
                    metric="num_variants",
                    value=m["num_variants"],
                    threshold=cfg.max_variants,
                    comparison="<=", window="本次请求",
                ),
            ],
        ),
        Rule(
            id="MIG-DEGRADE-01",
            name="backend=auto 路由",
            expression="backend=auto → 由配置 + 环境变量自动挑选后端",
            predicate=lambda m: m["backend"] == "auto",
            evidence=lambda m: [
                Evidence(
                    metric="backend 输入", value=m["backend"],
                    threshold="auto", comparison="==",
                    window="本次请求",
                ),
            ],
        ),
        Rule(
            id="MIG-EXTRACT-01",
            name="营销文案已抽取画面描述",
            expression="marketing_copy 提供 → 走 SceneExtractor",
            predicate=lambda m: m["marketing_copy_provided"],
            evidence=lambda m: [
                Evidence(
                    metric="extracted_scene 长度",
                    value=m["extracted_scene_length"],
                    threshold=10,
                    comparison=">=", window="本次请求",
                ),
            ],
        ),
    ]


def _select_image_backend(inp_backend: str | None, cfg: MarketingImageConfig):
    """根据入参 + cfg 选后端。auto 在无 ALLIN_API_KEY 时回落 mock。"""
    return select_backend(
        requested=inp_backend,
        cfg_allin_key_env=cfg.allin_api_key_env,
        cfg_allin_base=cfg.allin_api_base,
        cfg_allin_model=cfg.allin_api_image_model,
        cfg_allin_timeout_s=cfg.allin_api_timeout_s,
    )


def _select_scene_extractor(cfg: MarketingImageConfig):
    return select_extractor(
        mode=cfg.extractor_mode,
        cfg_api_base=cfg.allin_api_base,
        cfg_api_key_env=cfg.allin_api_key_env,
        cfg_extractor_model=cfg.extractor_model,
        cfg_extractor_timeout_s=cfg.extractor_timeout_s,
    )


@skill(id="marketing_image_gen", name="营销生图（多平台多风格）", version=_VERSION)
def marketing_image_gen(inp: MarketingImageInput) -> SkillOutput[MarketingImagePlan]:
    """解析并规划营销视觉生成任务,产出生成计划与候选图。

    流程:
    1. 推断 platform / style / aspect_ratio(回退 config 默认值)
    2. 若提供 marketing_copy,先走 SceneExtractor 抽取画面描述
    3. 拼装 final_prompt
    4. 选 backend(mock / allin_api / auto),生成 num_variants 张图
    5. 组装四段式因果推理链返回
    """
    cfg = load_config().marketing_image
    notes: list[str] = []

    platform = inp.platform or cfg.default_platform
    if platform not in cfg.platform_aspect_ratio:
        raise ValueError(f"未知 platform：{platform}")
    style = inp.style or cfg.default_style

    if inp.aspect_ratio and inp.aspect_ratio != "auto":
        aspect_ratio = inp.aspect_ratio
    else:
        aspect_ratio = cfg.platform_aspect_ratio[platform]

    if aspect_ratio not in _VALID_ASPECT_RATIOS:
        raise ValueError(f"非法 aspect_ratio：{aspect_ratio}")

    if inp.negative_prompt:
        negative_prompt = inp.negative_prompt
        neg_added = False
    else:
        negative_prompt = cfg.default_negative_prompt
        neg_added = True
        notes.append("未提供 negative_prompt,已自动追加默认排斥词")

    brand_overridden = inp.brand is not None
    brand_name = inp.brand.name if inp.brand else None
    backend = inp.backend or cfg.default_backend

    if brand_overridden:
        notes.append(f"已使用请求内 brand 覆盖;name={brand_name!r}")
    else:
        notes.append("使用 config 默认 brand profile(请求内未提供)")

    # --- 1) 画面描述抽取 ---
    extracted_scene: str | None = None
    if inp.marketing_copy:
        extractor = _select_scene_extractor(cfg)
        extracted_scene = extractor.extract(
            marketing_copy=inp.marketing_copy,
            hint=inp.prompt,
        ).strip()
        notes.append(f"已抽取画面描述(extractor={extractor.name});长度={len(extracted_scene)}")

    # --- 2) 拼 final_prompt ---
    # 安全:把 extracted_scene 与 marketing_copy 视为不透明内容（可能来自不可信
    # 的 LLM 或用户），用明确分隔符包起来防 image model 误把内容当指令。
    if inp.marketing_copy and extracted_scene:
        if inp.prompt:
            final_prompt = (
                f"{extracted_scene}\n\n"
                f"附加要求：{inp.prompt}\n\n"
                f"原始文案：{inp.marketing_copy}"
            )
            prompt_source = "merged"
        else:
            final_prompt = (
                f"{extracted_scene}\n\n原始文案：{inp.marketing_copy}"
            )
            prompt_source = "extracted_from_copy"
    else:
        final_prompt = inp.prompt
        prompt_source = "user_prompt"

    # 防御深度:对最终 prompt 也脱一次（防注入 pattern 跨过 extractor 边界）
    final_prompt = _sanitize_final_prompt(final_prompt)

    # --- 3) 选后端 + 生成 ---
    backend_obj = _select_image_backend(inp.backend, cfg)
    backend_used = backend_obj.name
    notes.append(f"实际调用后端：{backend_used}")

    hint = cfg.platform_pixel_hint.get(platform, (1024, 1024))
    width, height = _dimensions(aspect_ratio, hint)

    raw_images = backend_obj.generate(
        prompt=final_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        n=inp.num_variants,
        seed=inp.seed,
        save_dir=inp.save_dir,
    )
    images = [_to_plan_image(g) for g in raw_images]

    # --- 4) 推理链 ---
    metrics = {
        "platform_input": inp.platform,
        "platform": platform,
        "style_input": inp.style,
        "style": style,
        "aspect_ratio_input": inp.aspect_ratio,
        "aspect_ratio_resolved": aspect_ratio,
        "negative_prompt_added": neg_added,
        "negative_prompt_length": len(negative_prompt),
        "brand_overridden": brand_overridden,
        "brand_name_override": brand_name,
        "num_variants": inp.num_variants,
        "backend": backend,
        "marketing_copy_provided": inp.marketing_copy is not None,
        "extracted_scene_length": len(extracted_scene) if extracted_scene else 0,
    }
    hits, evidence = evaluate_rules(_build_rules(cfg), metrics)

    chain = ReasoningChain(
        conclusion=(
            f"为本次请求产出 {len(images)} 张 {aspect_ratio} {style} 营销图;"
            f"目标平台：{platform}({_platform_label(platform)});"
            f"backend：{backend_used}。"
        ),
        triggered_rules=hits,
        evidence=evidence,
        causal_analysis=(
            f"prompt 来源={prompt_source},提取画面长度="
            f"{len(extracted_scene) if extracted_scene else 0};"
            f"平台={platform},风格={style},比例={aspect_ratio};"
            f"命中 {len(hits)} 条规则,按既定优先级推断参数。"
        ),
        risk_note=(
            "本技能默认后端优先 allin_api(若环境变量 ALLIN_API_KEY 已设置),"
            "否则回落 mock。真实出图非确定性,仅作为创意草稿;"
            "正式投放前请人工挑选或调整 prompt 重试。"
        ),
    )

    return SkillOutput(
        data=MarketingImagePlan(
            prompt=inp.prompt,
            resolved_prompt=final_prompt,
            prompt_source=prompt_source,
            extracted_scene=extracted_scene,
            platform=platform,
            style=style,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            backend=backend,
            backend_used=backend_used,
            num_variants=inp.num_variants,
            images=images,
            negative_prompt=negative_prompt,
            estimated_cost_credit=len(images) * cfg.credit_per_image,
            brand_name=brand_name,
            sampling_notes=notes,
        ),
        reasoning=[chain],
        confidence=1.0,
        sample_size=len(images),
    )


def _to_plan_image(g: _GeneratedImage) -> GeneratedImage:
    return GeneratedImage(
        index=g.index,
        url=g.url,
        local_path=g.local_path,
        width=g.width,
        height=g.height,
        seed=g.seed,
    )