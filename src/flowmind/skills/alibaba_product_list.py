"""alibaba_product_list 技能：拉取阿里国际站在线商品（商品池）。

走阿里国际站开放 API（alibaba.product.list）；未授权（无 AppKey/Secret/Session）时
返回 degraded SkillOutput（空商品列表 + 明确 warning，不抛裸异常），保证推荐链路
在未授权阶段也能走通（商品池可为空，前端降级提示）。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills._alibaba_client import AlibabaAPIError, new_client_from_config
from flowmind.skills._content_common import build_chain

_VERSION = "0.1.0"


class ProductListInput(BaseModel):
    """商品拉取入参。"""
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=50, ge=1, le=100, description="每页条数")
    status: Literal["all", "onSelling"] = "onSelling"


class AlibabaProduct(BaseModel):
    """单条在线商品。"""
    product_id: str
    subject: str
    keywords: list[str] = Field(default_factory=list)
    image_url: str = ""
    price: str = ""
    status: str = "onSelling"


class ProductListPlan(BaseModel):
    """商品列表业务载荷。"""
    total: int
    products: list[AlibabaProduct]
    authorized: bool
    warning: str | None = None


@skill(id="alibaba_product_list", name="国际站在线商品拉取", version=_VERSION)
def alibaba_product_list(inp: ProductListInput) -> SkillOutput[ProductListPlan]:
    """拉取阿里国际站在线商品作为「货品一键上架」的商品池。

    数据流：组装 client → 校验授权 → call(alibaba.product.list) → 归一化 → 计划 + 推理链；
    未授权 → degraded（空列表 + 授权提示）。抓取失败 → degraded（不抛，供前端降级）。
    """
    cfg = load_config().alibaba
    client = new_client_from_config(cfg)

    if not client.app_key or not client.app_secret or not client.session:
        chain = build_chain(
            conclusion="未授权，跳过商品拉取",
            causal_analysis="缺少 ALIBABA_APP_KEY/APP_SECRET/SESSION（需运营在开放平台授权）",
            risk_note="未授权阶段商品池为空；推荐/生成仍可基于手动导入运行。",
        )
        return SkillOutput(
            data=ProductListPlan(total=0, products=[], authorized=False,
                                 warning="尚未在阿里国际站开放平台授权，无法拉取在线商品"),
            reasoning=[chain], confidence=0.0, sample_size=0,
            degraded=True, degradation_reason="unauthorized",
        )

    try:
        resp = client.call("alibaba.product.list", {
            "pageNo": inp.page,
            "pageSize": inp.page_size,
        })
    except AlibabaAPIError as exc:
        chain = build_chain(
            conclusion="商品拉取失败",
            causal_analysis=f"alibaba.product.list → {type(exc).__name__}（{exc.category}）",
            risk_note="接口失败已结构化返回，未授权或网络问题请联系运营核查凭证。",
        )
        return SkillOutput(
            data=ProductListPlan(total=0, products=[], authorized=True, warning=exc.args[0]),
            reasoning=[chain], confidence=0.0, sample_size=0,
            degraded=True, degradation_reason=exc.category,
        )

    products = _normalize(resp)
    chain = build_chain(
        conclusion=f"拉取到 {len(products)} 个在线商品",
        causal_analysis="alibaba.product.list 返回归一化为 AlibabaProduct 列表",
        risk_note="商品为店铺在线态快照，推荐结果仅作参考。",
    )
    return SkillOutput(
        data=ProductListPlan(total=len(products), products=products, authorized=True),
        reasoning=[chain], confidence=0.9, sample_size=len(products),
    )


def _normalize(resp: dict) -> list[AlibabaProduct]:
    """宽容解析不同接口返回结构，归一化为 AlibabaProduct 列表。"""
    for key in ("products", "product_list", "result", "data"):
        v = resp.get(key)
        if isinstance(v, list):
            items = v
            break
        if isinstance(v, dict):
            for inner in ("products", "product_list", "list", "records"):
                if isinstance(v.get(inner), list):
                    items = v[inner]
                    break
            else:
                continue
            break
    else:
        items = []

    out: list[AlibabaProduct] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        pid = it.get("productId") or it.get("product_id") or it.get("id")
        subject = it.get("subject") or it.get("name") or ""
        if not pid or not subject:
            continue
        kw = it.get("keywords") or it.get("keyword") or []
        if isinstance(kw, str):
            kw = [x.strip() for x in kw.split(",") if x.strip()]
        img = it.get("image") or it.get("imageUrl") or it.get("mainImage") or ""
        if isinstance(img, dict):
            img = img.get("url") or img.get("fullPath") or ""
        out.append(AlibabaProduct(
            product_id=str(pid),
            subject=str(subject),
            keywords=[str(k) for k in (kw if isinstance(kw, list) else [])],
            image_url=str(img),
            price=str(it.get("price") or ""),
            status=str(it.get("status") or "onSelling"),
        ))
    return out