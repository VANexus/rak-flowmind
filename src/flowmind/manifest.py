"""能力清单：由注册表生成机器可读描述，供 Agent 发现与挂载。

与 discover() 共用同一个 entry 构造器（skill_entry），保证两份视图不漂移。
"""
from __future__ import annotations

from flowmind.discover import skill_entry
from flowmind.skill import registry


def build_manifest() -> dict:
    """生成能力清单。每个技能附：id / name / version / description /
    input_schema / output_schema / reliability_profile。
    """
    return {"skills": [skill_entry(spec) for spec in registry().values()]}
