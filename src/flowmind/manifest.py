"""能力清单：由注册表生成机器可读描述，供 Agent 发现与挂载。

与 discover() 共用同一个 entry 构造器（skill_entry），保证两份视图不漂移。
"""
from __future__ import annotations

from flowmind.discover import skill_entry
from flowmind.pipeline_steps import stage_schema, steps_for
from flowmind.skill import registry


def build_manifest() -> dict:
    """生成能力清单。每个技能附：id / name / version / description /
    input_schema / output_schema / reliability_profile。

    有流水线步骤的技能额外附 ``stage_schema``（步骤 id/中文名/依赖/可跳过性/
    进度区间），前端据此渲染步骤选择器与分步进度，**无需硬编码步骤名**。
    """
    skills = []
    for spec in registry().values():
        entry = skill_entry(spec)
        if steps_for(spec.id):
            entry["stage_schema"] = stage_schema(spec.id)
        skills.append(entry)
    return {"skills": skills}
