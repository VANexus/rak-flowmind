"""参考技能：库销比/库存风险分析。

纯确定性计算，无外部依赖。阈值来自 config（用户配置 > 通用默认）。
输出每 SKU 分析 + 汇总，并对被标记 SKU 生成四段式因果推理链。
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from flowmind.config import InventoryConfig, load_config
from flowmind.contracts import Evidence, ReasoningChain, SkillOutput
from flowmind.rules import Rule, evaluate_rules
from flowmind.skill import skill

_VERSION = "0.1.0"
_LEVEL_ORDER = {"危险": 3, "预警": 2, "关注": 1, "健康": 0}


class InventoryItem(BaseModel):
    """单个 SKU 的库存与销量记录。"""
    sku: str
    on_hand: int = Field(ge=0)          # 在库数量
    unit_cost: float = Field(ge=0)      # 单位成本
    sales_30d: int = Field(ge=0)        # 近30天销量
    price: float | None = None
    category: str | None = None


class InventoryInput(BaseModel):
    """库销比技能入参：至少一条记录。"""
    items: list[InventoryItem]

    @field_validator("items")
    @classmethod
    def _non_empty(cls, v: list[InventoryItem]) -> list[InventoryItem]:
        if not v:
            raise ValueError("items 不能为空")
        return v


class SkuAnalysis(BaseModel):
    """单 SKU 分析结果。"""
    sku: str
    on_hand: int
    sales_30d: int
    dsi: float | None                    # 周转天数；无动销为 None
    inventory_sales_ratio: float | None  # 库销比；无动销为 None
    capital_occupied: float
    risk_level: str
    hit_rules: list[str] = Field(default_factory=list)


class InventorySummary(BaseModel):
    """全局汇总。"""
    total_capital_occupied: float
    dead_stock_capital: float
    level_counts: dict[str, int]
    top_risks: list[str]


class InventoryReport(BaseModel):
    """库销比技能业务载荷。"""
    items: list[SkuAnalysis]
    summary: InventorySummary
    currency: str


def _metrics(item: InventoryItem) -> dict:
    """计算单 SKU 指标；无动销时 DSI/库销比为 None。"""
    dsi = None if item.sales_30d == 0 else item.on_hand / (item.sales_30d / 30.0)
    ratio = None if item.sales_30d == 0 else item.on_hand / item.sales_30d
    return {
        "sku": item.sku,
        "on_hand": item.on_hand,
        "sales_30d": item.sales_30d,
        "unit_cost": item.unit_cost,
        "dsi": dsi,
        "ratio": ratio,
        "capital": item.on_hand * item.unit_cost,
    }


def _rules(cfg: InventoryConfig) -> list[Rule]:
    """基于配置阈值构造规则集。"""
    return [
        Rule(
            id="INV-P01", name="滞销积压", expression="sales_30d==0 且 on_hand>0",
            predicate=lambda m: m["sales_30d"] == 0 and m["on_hand"] > 0,
            evidence=lambda m: [
                Evidence(metric="近30天销量", value=m["sales_30d"], threshold=0, comparison="==", window="近30天"),
                Evidence(metric="在库数量", value=m["on_hand"], threshold=0, comparison=">"),
            ],
        ),
        Rule(
            id="INV-P02", name="周转过慢", expression=f"DSI > {cfg.dsi_warn_max:g}",
            predicate=lambda m: m["dsi"] is not None and m["dsi"] > cfg.dsi_warn_max,
            evidence=lambda m: [
                Evidence(metric="周转天数(DSI)", value=round(m["dsi"], 1), threshold=cfg.dsi_warn_max, comparison=">", window="近30天"),
            ],
        ),
        Rule(
            id="INV-P03", name="慢周转+资金占用高",
            expression=f"DSI > {cfg.dsi_watch_max:g} 且 资金占用 > {cfg.capital_high:g}",
            predicate=lambda m: m["dsi"] is not None and m["dsi"] > cfg.dsi_watch_max and m["capital"] > cfg.capital_high,
            evidence=lambda m: [
                Evidence(metric="周转天数(DSI)", value=round(m["dsi"], 1), threshold=cfg.dsi_watch_max, comparison=">"),
                Evidence(metric="资金占用", value=round(m["capital"], 2), threshold=cfg.capital_high, comparison=">"),
            ],
        ),
        Rule(
            id="INV-P04", name="断货风险", expression=f"DSI < {cfg.dsi_low:g}",
            predicate=lambda m: m["dsi"] is not None and m["dsi"] < cfg.dsi_low,
            evidence=lambda m: [
                Evidence(metric="周转天数(DSI)", value=round(m["dsi"], 1), threshold=cfg.dsi_low, comparison="<", window="近30天"),
            ],
        ),
    ]


def _level(m: dict, cfg: InventoryConfig) -> str:
    """按指标与阈值判定风险等级。"""
    if m["sales_30d"] == 0 and m["on_hand"] > 0:
        return "危险"
    dsi = m["dsi"]
    if dsi is None:  # 零库存零动销
        return "健康"
    if dsi < cfg.dsi_low:
        return "预警"  # 断货风险
    if dsi <= cfg.dsi_healthy_max:
        base = "健康"
    elif dsi <= cfg.dsi_watch_max:
        base = "关注"
    elif dsi <= cfg.dsi_warn_max:
        base = "预警"
    else:
        base = "危险"
    if base == "关注" and m["capital"] > cfg.capital_high:
        base = "预警"  # 资金占用高则升级
    return base


def _advice(m: dict, level: str, cfg: InventoryConfig) -> str:
    """按情形给出处置建议。断货阈值取自 cfg.dsi_low。"""
    if m["sales_30d"] == 0 and m["on_hand"] > 0:
        return "立即清仓去化，停止补货"
    if m["dsi"] is not None and m["dsi"] < cfg.dsi_low:
        return "加快补货，防止断货"
    if level in ("预警", "危险"):
        return "促销/调价加速去化"
    if level == "关注":
        return "关注动销，控制补货节奏"
    return "保持现状"


def _chain(m: dict, level: str, cfg: InventoryConfig, hits, evidence) -> ReasoningChain:
    """组装四段式因果推理链。"""
    dsi_txt = "无动销" if m["dsi"] is None else f"{m['dsi']:.1f} 天"
    return ReasoningChain(
        conclusion=f"SKU {m['sku']} 风险等级：{level}；建议：{_advice(m, level, cfg)}",
        triggered_rules=hits,
        evidence=evidence,
        causal_analysis=(
            f"周转天数={dsi_txt}、资金占用={m['capital']:.2f}{cfg.currency}，"
            f"命中 {len(hits)} 条规则，综合判定为「{level}」。"
        ),
        risk_note=_advice(m, level, cfg),
    )


@skill(id="inventory_risk", name="库销比/库存风险分析", version=_VERSION)
def inventory_risk(inp: InventoryInput) -> SkillOutput[InventoryReport]:
    """分析每个 SKU 的周转与资金占用，输出风险分级、汇总与推理链。"""
    cfg = load_config().inventory
    rules = _rules(cfg)

    analyses: list[SkuAnalysis] = []
    chains: list[ReasoningChain] = []
    level_counts: dict[str, int] = {"健康": 0, "关注": 0, "预警": 0, "危险": 0}
    total_capital = 0.0
    dead_capital = 0.0

    for item in inp.items:
        m = _metrics(item)
        level = _level(m, cfg)
        hits, evidence = evaluate_rules(rules, m)
        level_counts[level] += 1
        total_capital += m["capital"]
        if m["sales_30d"] == 0 and m["on_hand"] > 0:
            dead_capital += m["capital"]

        analyses.append(SkuAnalysis(
            sku=item.sku,
            on_hand=item.on_hand,
            sales_30d=item.sales_30d,
            dsi=None if m["dsi"] is None else round(m["dsi"], 1),
            inventory_sales_ratio=None if m["ratio"] is None else round(m["ratio"], 2),
            capital_occupied=round(m["capital"], 2),
            risk_level=level,
            hit_rules=[h.rule_id for h in hits],
        ))
        if level != "健康":
            chains.append(_chain(m, level, cfg, hits, evidence))

    # Top 风险：按等级严重度、其次资金占用排序，取前 5 个 SKU
    ranked = sorted(
        [a for a in analyses if a.risk_level != "健康"],
        key=lambda a: (_LEVEL_ORDER[a.risk_level], a.capital_occupied),
        reverse=True,
    )
    top_risks = [a.sku for a in ranked[:5]]

    report = InventoryReport(
        items=analyses,
        summary=InventorySummary(
            total_capital_occupied=round(total_capital, 2),
            dead_stock_capital=round(dead_capital, 2),
            level_counts=level_counts,
            top_risks=top_risks,
        ),
        currency=cfg.currency,
    )
    return SkillOutput(data=report, reasoning=chains, confidence=1.0, sample_size=len(inp.items))
