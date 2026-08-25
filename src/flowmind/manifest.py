"""能力清单：由注册表生成机器可读描述，供 Agent 发现与挂载。

设计原则（v0.3.2 起）：**完整 schema 暴露**——同时返回 input_schema 与 output_schema，
外加从函数 docstring 提取的 description。Agent 拿到 manifest 后无需读源码即可：
- 知道传什么字段
- 知道返回什么字段
- 知道每个字段的含义（中文 docstring）
"""
from __future__ import annotations

from flowmind.skill import registry


def build_manifest() -> dict:
    """生成能力清单。每个技能附：
    - id / name / version
    - description（从函数 docstring 第一行提取）
    - input_schema（完整 JSON Schema）
    - output_schema（完整 JSON Schema，让 Agent 知道返回什么字段）
    - reliability_profile（确定性 / 推理链 / 典型延迟 / 置信度）
    """
    skills = []
    for spec in registry().values():
        entry: dict = {
            "id": spec.id,
            "name": spec.name,
            "version": spec.version,
            "description": spec.description,
            "input_schema": spec.input_model.model_json_schema() if spec.input_model else None,
            "output_schema": spec.output_model.model_json_schema() if spec.output_model else None,
            "reliability_profile": {
                "deterministic": True,     # 纯确定性计算
                "emits_reasoning_chain": True,
                "typical_latency_ms": "<50",
                "confidence": 1.0,
            },
        }
        skills.append(entry)
    return {"skills": skills}