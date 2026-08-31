"""alibaba_product_post 技能：一键上传商品到阿里国际站。

走官方开放 API `alibaba.icbu.open.product.post`（国际站商品开放接口）。
未授权/接口失败 → 结构化错误（ok=False，invoke() 套信封供前端提示），绝不静默。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._alibaba_client import AlibabaAPIError, new_client_from_config
from flowmind.skills._content_common import build_chain

_VERSION = "0.1.0"


class ProductTrade(BaseModel):
    """交易信息（FOB 价格区间等）。"""
    price_range_min: str | None = None
    price_range_max: str | None = None
    min_order_quantity: str | None = None


class ProductPostInput(BaseModel):
    """发布入参（对齐 alibaba.icbu.open.product.post）。"""
    subject: str = Field(min_length=1, max_length=256, description="产品主题")
    keywords: list[str] = Field(default_factory=list, max_length=3, description="关键字")
    description: str = Field(default="", max_length=10000, description="产品描述")
    category_id: int | None = Field(default=None, description="类目 ID")
    image_url: str = Field(default="", description="主图 URL")
    group_id: int | None = Field(default=None, description="产品组 ID")
    trade: ProductTrade | None = None


class ProductPostPlan(BaseModel):
    """发布结果载荷。"""
    product_id: str = ""
    str_product_id: str = ""
    posted: bool
    warnings: list[str] = Field(default_factory=list)


@skill(id="alibaba_product_post", name="国际站一键上传", version=_VERSION)
def alibaba_product_post(inp: ProductPostInput) -> SkillOutput[ProductPostPlan]:
    """把生成的 Listing 发布到阿里国际站。

    数据流：组装 client → 校验授权 → 主图 URL 直传 → call(alibaba.icbu.open.product.post)
    → ProductPostPlan；未授权/失败抛结构化错误（ok=False）。
    """
    cfg = load_config().alibaba
    client = new_client_from_config(cfg)

    if not client.app_key or not client.app_secret or not client.session:
        raise ValueError(
            "尚未在阿里国际站开放平台授权（缺少 ALIBABA_APP_KEY/APP_SECRET/SESSION）。"
            "请运营完成授权后重试，当前 Listing 已可作为草稿保存。"
        )

    warnings: list[str] = []
    if len(inp.keywords) < 1:
        warnings.append("关键词为空：建议至少 1 个核心关键词再发布")
    if len(inp.keywords) > 3:
        warnings.append(f"关键词超过 3 个（{len(inp.keywords)}）：国际站只会取前 3 个")
    if len(inp.subject) < 10:
        warnings.append(f"标题过短（{len(inp.subject)} 字）：建议 ≥ 30 字、≤ 128 字以提升曝光")
    if len(inp.description) < 100:
        warnings.append(f"详情过短（{len(inp.description)} 字）：建议 ≥ 500 字、突出卖点与采购场景")
    if not inp.image_url:
        warnings.append("主图 URL 为空：将只上传文字信息，强烈建议补主图")
    elif not (inp.image_url.startswith("http://") or inp.image_url.startswith("https://")):
        warnings.append(f"主图 URL 不是绝对地址：{inp.image_url}")
    if inp.category_id is None:
        warnings.append("未提供类目 ID（category_id）：后台会分配默认类目，请运营确认后修改")

    image_url = client.upload_image(inp.image_url)
    if not image_url and inp.image_url:
        warnings.append(f"主图上传失败（原始 URL: {inp.image_url[:100]}）")

    product_post: dict = {
        "subject": inp.subject,
        "keywords": inp.keywords,
        "description": inp.description,
        "product_image": {"image_file_list": [{"image_file_url": image_url, "image_watermark": False}]},
    }
    if inp.category_id:
        product_post["category_id"] = inp.category_id
    if inp.group_id:
        product_post["group_id"] = inp.group_id
    if inp.trade:
        trade: dict = {}
        for k in ("price_range_min", "price_range_max", "min_order_quantity"):
            v = getattr(inp.trade, k, None)
            if v is not None:
                trade[k] = v
        if trade:
            product_post["product_trade"] = trade

    try:
        resp = client.call("alibaba.icbu.open.product.post", {"param_product_post": product_post})
    except AlibabaAPIError:
        raise

    pid = str(resp.get("product_id") or "")
    spid = str(resp.get("str_product_id") or "")
    posted = bool(pid or spid)
    if not posted:
        raise ValueError("国际站接口未返回 product_id，发布可能失败，请核查后台草稿箱")

    chain = build_chain(
        conclusion=f"商品已发布到阿里国际站（product_id={pid or spid}）",
        causal_analysis="alibaba.icbu.open.product.post 调用成功，主图经 image_file_url 直传",
        risk_note="发布成功请到国际站后台确认类目/属性完整；如需编辑请走编辑接口。",
    )
    return SkillOutput(
        data=ProductPostPlan(product_id=pid, str_product_id=spid, posted=True, warnings=warnings),
        reasoning=[chain], confidence=1.0, sample_size=1,
    )