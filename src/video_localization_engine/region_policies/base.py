"""L2 SubtitleRegionPolicy 协议定义 + 抽象基类。

设计原则:
- policy 只输出 RegionProposal (polygon + weight + source)
- policy 不知道"是不是字幕", 只表达"这里可能是字幕"
- policy 自带 is_applicable(), 由 orchestrator 根据 VideoMeta 自动选择
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from video_localization_engine.types.detection import RegionProposal
from video_localization_engine.types.video import FramePacket, VideoMeta


class SubtitleRegionPolicy(ABC):
    """字幕区域策略。"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_applicable(self, meta: VideoMeta) -> bool:
        """该 policy 是否适用于这个视频。
        注意: 这里也只是判定"是否可能", 不做字幕判断。
        """
        ...

    @abstractmethod
    def propose(self, packet: FramePacket) -> List[RegionProposal]:
        """返回这一帧的字幕候选 RegionProposal。

        可以返回多个 proposal, 每个有独立 weight。
        weight 越大表示该 policy 越"主张"这是字幕区。
        """
        ...


class CompositeRegionPolicy(SubtitleRegionPolicy):
    """组合多个 policy, 取并集 + max weight。

    orchestrator 用来合并多个 region proposal 来源。
    """

    def __init__(self, policies: List[SubtitleRegionPolicy], name: str = "composite"):
        self._policies = policies
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def is_applicable(self, meta: VideoMeta) -> bool:
        return any(p.is_applicable(meta) for p in self._policies)

    def propose(self, packet: FramePacket) -> List[RegionProposal]:
        out: List[RegionProposal] = []
        for p in self._policies:
            if p.is_applicable(packet.meta):
                out.extend(p.propose(packet))
        return out