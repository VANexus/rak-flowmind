"""tiktok_shop_intel 技能：TikTok Shop 选品情报（真实电商数据，绝不 mock）。

数据全部来自 TikHub Shop Web 系列：
- search：关键词搜商品（价格/评分/评论数/销量/店铺/标签/链接）；
- suggest：搜索词联想（扩词）；
- categories：Shop 商品类目树；
- detail：商品深度详情（图集/规格/SKU/店铺评分，匿名详情无价格，价格以 search 为准）；
- reviews：商品评论 V2（评分分布/带图/已验证购买）；
- seller：指定商家在售商品列表。

源不可用时统一 degraded 空态，绝不返回假数据。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._content_common import build_chain
from flowmind.skills._intel_common import fail_fields, intel_client
from flowmind.skills._tikhub_intel_parse import (
    parse_product_detail,
    parse_product_reviews,
    parse_shop_categories,
    parse_shop_page,
    parse_shop_products,
)

_VERSION = "0.1.0"
ShopIntelAction = Literal["search", "suggest", "categories", "detail", "reviews", "seller"]


class ShopProduct(BaseModel):
    product_id: str
    title: str = ""
    image_url: str = ""
    price: str = ""
    original_price: str = ""
    discount: str = ""
    currency: str = ""
    rating: float | None = None
    review_count: int | None = None
    sold_count: int | None = None
    seller_id: str = ""
    seller_name: str = ""
    brand: str = ""
    url: str = ""
    labels: list[str] = []


class ShopReview(BaseModel):
    review_id: str
    rating: int | None = None
    time: str = ""
    verified: bool = False
    incentivized: bool = False
    reviewer: str = ""
    text: str = ""
    images: list[str] = []
    sku_spec: str = ""
    country: str = ""


class ShopIntelInput(BaseModel):
    action: ShopIntelAction = Field(description="search/suggest/categories/detail/reviews/seller")
    keyword: str | None = Field(default=None, description="search/suggest 必填")
    product_id: str | None = Field(default=None, description="detail/reviews 必填")
    seller_id: str | None = Field(default=None, description="seller 必填")
    region: str = Field(default="US")
    limit: int = Field(default=20, ge=1, le=50)
    offset: int = Field(default=0, ge=0)
    page: int = Field(default=1, ge=1, description="reviews 页码")


class ShopIntelPlan(BaseModel):
    action: str
    source: str = "tikhub"
    degraded: bool = False
    products: list[ShopProduct] = []
    page: dict = Field(default_factory=dict)
    suggestions: list[str] = []
    categories: list[dict] = []
    detail: dict = Field(default_factory=dict)
    reviews: list[ShopReview] = []
    review_summary: dict = Field(default_factory=dict)
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


@skill(id="tiktok_shop_intel", name="TikTokShop选品情报", version=_VERSION)
def tiktok_shop_intel(inp: ShopIntelInput) -> SkillOutput[ShopIntelPlan]:
    """TikTok Shop 选品：搜商品/扩词/类目/详情/评论/商家商品；失败 degraded 空态。"""
    degraded = False
    warning = failure_category = None
    retriable = False
    products: list[dict] = []
    page: dict = {}
    suggestions: list[str] = []
    categories: list[dict] = []
    detail: dict = {}
    reviews: list[dict] = []
    review_summary: dict = {}
    result_n = 0

    try:
        client = intel_client()
        if inp.action == "search":
            kw = (inp.keyword or "").strip()
            if not kw:
                raise ValueError("search 需要关键词 keyword")
            raw = client.shop_search_products(keyword=kw, region=inp.region, offset=inp.offset)
            products = parse_shop_products(raw)[: inp.limit]
            page = parse_shop_page(raw)
            page["size"] = len(products)
            result_n = len(products)
        elif inp.action == "suggest":
            if not (inp.keyword or "").strip():
                raise ValueError("suggest 需要关键词 keyword")
            suggestions = client.shop_search_suggest(keyword=inp.keyword, region=inp.region)
            result_n = len(suggestions)
        elif inp.action == "categories":
            categories = parse_shop_categories(client.shop_categories(region=inp.region))
            result_n = len(categories)
        elif inp.action == "detail":
            if not (inp.product_id or "").strip():
                raise ValueError("detail 需要 product_id")
            detail = parse_product_detail(client.shop_product_detail(
                product_id=inp.product_id.strip(), region=inp.region))
            result_n = len(detail.get("images", []))
        elif inp.action == "reviews":
            if not (inp.product_id or "").strip():
                raise ValueError("reviews 需要 product_id")
            parsed = parse_product_reviews(client.shop_product_reviews(
                product_id=inp.product_id.strip(), region=inp.region, page_start=inp.page))
            reviews = parsed["reviews"][: inp.limit]
            review_summary = parsed["summary"]
            result_n = len(reviews)
        elif inp.action == "seller":
            if not (inp.seller_id or "").strip():
                raise ValueError("seller 需要 seller_id")
            raw = client.shop_seller_products(seller_id=inp.seller_id.strip(), region=inp.region)
            products = parse_shop_products(raw)[: inp.limit]
            result_n = len(products)
    except Exception as exc:
        degraded = True
        fb = fail_fields(exc)
        failure_category, retriable, warning = fb["failure_category"], fb["retriable"], fb["warning"]

    chain = build_chain(
        conclusion=f"选品情报 {inp.action} {'降级' if degraded else '成功'}：{result_n} 项（源 tikhub）",
        causal_analysis=f"TikHub Shop Web action={inp.action} → 解析 {result_n} 项",
        risk_note="Shop 数据随上架/销量实时变化；匿名详情不含价格，价格以搜索列表为准。",
    )
    return SkillOutput(
        data=ShopIntelPlan(
            action=inp.action, degraded=degraded,
            products=[ShopProduct(**p) for p in products], page=page,
            suggestions=suggestions, categories=categories, detail=detail,
            reviews=[ShopReview(**r) for r in reviews], review_summary=review_summary,
            failure_category=failure_category, retriable=retriable, warning=warning,
        ),
        reasoning=[chain], confidence=0.0 if degraded else 0.9,
        sample_size=result_n, degraded=degraded, degradation_reason=failure_category,
    )
