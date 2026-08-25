"""L2: Region policies — where might subtitles be?"""
from video_localization_engine.region_policies.base import (
    SubtitleRegionPolicy, CompositeRegionPolicy,
)
from video_localization_engine.region_policies.policies import (
    BottomHorizontalPolicy, BottomPortraitPolicy,
    TopNewsPolicy, CustomPolicy,
)
from video_localization_engine.region_policies.registry import (
    RegionPolicyRegistry, applicable_policies,
)

__all__ = [
    "SubtitleRegionPolicy",
    "CompositeRegionPolicy",
    "BottomHorizontalPolicy",
    "BottomPortraitPolicy",
    "TopNewsPolicy",
    "CustomPolicy",
    "RegionPolicyRegistry",
    "applicable_policies",
]
