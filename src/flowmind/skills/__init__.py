"""技能包：导入各技能以触发 @skill 注册。

**族（family）筛选**：env ``FLOWMIND_SKILL_FAMILIES``（逗号分隔）显式选择要
注册的技能族；**缺省（未设置）= 注册全部族**（行为与历史完全一致）。

族是「一个功能包一个 prefix」在网关侧的落地手段：网关把每个 backend 的工具
聚合为 ``{prefix}__{remote_tool}``，而 prefix 每 backend 只有一个。因此 dsp 族
以**独立进程/部署**注册为 ``prefix=dsp`` 的 backend，其注册表里只应有 dsp 的
技能（契约 §6「``dsp__`` 前缀下无多余工具」）：

    FLOWMIND_SKILL_FAMILIES=dsp \\
    FLOWMIND_FEDERATION_BACKEND_ID=dsp \\
    FLOWMIND_FEDERATION_PREFIX=dsp \\
    FLOWMIND_MCP_PORT=8003 FLOWMIND_FEDERATION_REGISTER=1 ... server_http

历史（保留）：``224a194 合并修复（2026-09-07）``——合并曾把 35 个不存在或已按
``f72fb2d`` 裁剪决策移除的技能模块写入 import 列表，导致 ``import flowmind.skills``
直接 ImportError、服务无法启动。故本表**显式列出实际存在的模块**（族表即清单），
其余 ``content_*`` / ``b2b_*`` / ``alibaba_*`` / ``crawler_*`` / ``tiktok_*`` 等
待源码分支合入后再逐个恢复，**禁止幽灵 import 入库**——模块不存在时在本文件当场
报错（不静默跳过）。
"""
from __future__ import annotations

import importlib
import os

# 族表（唯一真源）：族名 → 该族技能模块名（按字母序）。显式列名 = 延续
# 「禁止幽灵 import」纪律：任一模块缺失，import 阶段即抛 ModuleNotFoundError。
FAMILIES: dict[str, tuple[str, ...]] = {
    "localize": (
        "localize_cancel",
        "localize_download",
        "localize_retry",
        "localize_search",
        "localize_status",
        "localize_submit",
        "localize_video",
    ),
    "dsp": (
        "dsp_bgm_match",
        "dsp_cart_sticker",
        "dsp_cover_pick",
        "dsp_image_to_video",
        "dsp_render_compose",
        "dsp_subtitle_build",
        "dsp_timeline_build",
        "dsp_tts_generate",
    ),
}


def selected_families() -> tuple[str, ...]:
    """按 env ``FLOWMIND_SKILL_FAMILIES`` 选出要注册的族；缺省 = 全部。

    未知族名**当场报错**（拼错族名静默注册成空集，会让整条链路无声失灵）。
    """
    raw = os.environ.get("FLOWMIND_SKILL_FAMILIES", "").strip()
    if not raw:
        return tuple(FAMILIES)
    wanted = [f.strip() for f in raw.split(",") if f.strip()]
    unknown = [f for f in wanted if f not in FAMILIES]
    if unknown:
        raise ValueError(
            f"FLOWMIND_SKILL_FAMILIES 含未知族：{unknown}；可用族：{sorted(FAMILIES)}")
    return tuple(wanted)


def _register_families() -> None:
    """导入被选中的族模块（导入即经 @skill 装饰器登记进注册表）。"""
    for family in selected_families():
        for module in FAMILIES[family]:
            importlib.import_module(f"flowmind.skills.{module}")


_register_families()
