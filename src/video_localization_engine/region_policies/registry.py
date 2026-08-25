"""L2 RegionPolicy Registry — orchestrator 通过 Registry 自动选择适用 policy。"""
from __future__ import annotations

from typing import Iterable, List

from video_localization_engine.region_policies.base import SubtitleRegionPolicy
from video_localization_engine.types.video import VideoMeta
from video_localization_engine.utils.registry import RegistryBase


class RegionPolicyRegistry(RegistryBase[type[SubtitleRegionPolicy]]):
    """注册 SubtitleRegionPolicy 类 (factory)。"""
    pass


def applicable_policies(meta: VideoMeta) -> List[SubtitleRegionPolicy]:
    """根据 VideoMeta 自动选所有适用 policy 实例。"""
    out: List[SubtitleRegionPolicy] = []
    for name in RegionPolicyRegistry.available():
        cls = RegionPolicyRegistry.get(name)
        inst = cls()
        if inst.is_applicable(meta):
            out.append(inst)
    return out


# 默认注册
from video_localization_engine.region_policies.policies import (
    BottomHorizontalPolicy, BottomPortraitPolicy,
    TopNewsPolicy, CustomPolicy,
)
RegionPolicyRegistry.register("bottom_horizontal", BottomHorizontalPolicy)
RegionPolicyRegistry.register("bottom_portrait", BottomPortraitPolicy)
RegionPolicyRegistry.register("top_news", TopNewsPolicy)
# CustomPolicy 不注册 (需要参数, 不是单例)