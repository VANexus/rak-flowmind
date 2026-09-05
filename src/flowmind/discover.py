"""运行时发现工具：让 Agent / 调试器拿到技能的完整契约。

背景：Agent 调 `invoke("skill_id", args)` 拿到 SkillResult 后，要想知道 `data`
下面有哪些字段（是 `data.top_k` 还是 `data.hits`？是 `summary.level_counts` 还是
`summary.band`？），以前只能读源码。`discover()` 把 input + output JSON Schema
一次性暴露。

用法：
    from flowmind.discover import discover
    info = discover()                      # 全部技能
    info = discover("localize_status")     # 单个技能
"""
from __future__ import annotations

from flowmind.skill import SkillSpec, registry


def skill_entry(spec: SkillSpec) -> dict:
    """把一个 SkillSpec 组装成对外发现 dict（discover 与 manifest 共用，防漂移）。"""
    return {
        "id": spec.id,
        "name": spec.name,
        "version": spec.version,
        "description": spec.description,
        "category": spec.category,
        "tags": spec.tags,
        "input_schema": spec.input_model.model_json_schema() if spec.input_model else None,
        "output_schema": spec.output_model.model_json_schema() if spec.output_model else None,
        "reliability_profile": {
            "deterministic": True,
            "emits_reasoning_chain": True,
            "typical_latency_ms": "<50",
            "confidence": 1.0,
        },
    }


def discover(skill_id: str | None = None) -> dict | list[dict]:
    """返回技能发现信息。

    - 不传 skill_id → 返回所有技能的列表（dict 数组）
    - 传 skill_id → 返回单个技能的完整 dict；未知 id 抛 KeyError

    每个 skill dict 包含：id / name / version / description /
    input_schema / output_schema / reliability_profile。
    """
    if skill_id is None:
        return [skill_entry(spec) for spec in registry().values()]

    if skill_id not in registry():
        available = sorted(registry().keys())
        raise KeyError(
            f"未知技能：{skill_id!r}。可用：{available}。\n"
            f"提示：用 discover() 不带参数可看全部。"
        )

    return skill_entry(registry()[skill_id])


def field_names(skill_id: str) -> dict[str, list[str]]:
    """返回某个技能返回数据（`r.data`）下的字段路径列表（含嵌套）。

    Agent 想知道 `data.summary.level_counts` 还是 `data.summary.band`？调这个。

    返回 dict 形如：
        {
          "data": ["items", "summary", "currency"],
          "data.items[]": ["sku", "on_hand", "sales_30d", "dsi", ...],
          "data.summary": ["total_capital_occupied", "dead_stock_capital", ...],
        }
    """
    info = discover(skill_id)
    schema = info.get("output_schema") or {}
    return _flatten_schema(schema, prefix="data")


def _flatten_schema(schema: dict, prefix: str = "") -> dict[str, list[str]]:
    """递归把 JSON Schema 摊成路径 → 字段名列表。"""
    result: dict[str, list[str]] = {}
    props = schema.get("properties", {})
    if props:
        direct_names = []
        for name in props:
            direct_names.append(name)
            sub_schema = props[name]
            sub_prefix = f"{prefix}.{name}" if prefix else name
            if sub_schema.get("type") == "array":
                item_schema = sub_schema.get("items", {})
                # 处理 $ref 到 $defs
                if "$ref" in item_schema:
                    defs_name = item_schema["$ref"].split("/")[-1]
                    item_schema = schema.get("$defs", {}).get(defs_name, {})
                sub_result = _flatten_schema(item_schema, prefix=f"{sub_prefix}[]")
                result.update(sub_result)
            elif sub_schema.get("type") == "object" or "$ref" in sub_schema:
                # 嵌套对象
                resolved = sub_schema
                if "$ref" in sub_schema:
                    defs_name = sub_schema["$ref"].split("/")[-1]
                    resolved = schema.get("$defs", {}).get(defs_name, {})
                sub_result = _flatten_schema(resolved, prefix=sub_prefix)
                result.update(sub_result)
        if direct_names:
            result[prefix] = direct_names
    return result