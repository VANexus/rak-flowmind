"""4 个内置 RegionPolicy 实现。

不假设位置是字幕, 只表达"这里可能是字幕区"。
所有 weight 默认基于: 出现在底部 + 横向中段 + 屏幕较宽, 但仍允许调用方
根据实际视频调权重 (weight 是相对值, 0-1)。
"""
from __future__ import annotations

from typing import List

from video_localization_engine.region_policies.base import SubtitleRegionPolicy
from video_localization_engine.types.detection import RegionProposal
from video_localization_engine.types.video import FramePacket, VideoMeta, Orientation


def _make_full_width_band(meta: VideoMeta, y_top: float, y_bot: float,
                          weight: float, source: str, desc: str) -> RegionProposal:
    """构造一条横跨全宽的 region proposal。"""
    return RegionProposal(
        polygon=[
            (0, int(y_top * meta.height)),
            (meta.width, int(y_top * meta.height)),
            (meta.width, int(y_bot * meta.height)),
            (0, int(y_bot * meta.height)),
        ],
        weight=weight,
        source=source,
        description=desc,
    )


class BottomHorizontalPolicy(SubtitleRegionPolicy):
    """横屏底部居中字幕带。

    适用: LANDSCAPE 视频。底部 ~65-95% 高度, 横向 10-90% 宽。
    weight = 0.95 (底部 + 横屏 + 中段, 高置信度"可能字幕")。
    """

    @property
    def name(self) -> str:
        return "bottom_horizontal"

    def is_applicable(self, meta: VideoMeta) -> bool:
        return meta.orientation == Orientation.LANDSCAPE

    def propose(self, packet: FramePacket) -> List[RegionProposal]:
        meta = packet.meta
        return [_make_full_width_band(
            meta,
            y_top=0.65, y_bot=0.95,
            weight=0.95,
            source=f"policy:{self.name}",
            desc="横屏底部居中字幕带",
        )]


class BottomPortraitPolicy(SubtitleRegionPolicy):
    """竖屏短视频底部字幕带。

    适用: PORTRAIT 视频。底部 ~75-98% 高度, 横向 5-95% 宽。
    weight = 0.92。
    """

    @property
    def name(self) -> str:
        return "bottom_portrait"

    def is_applicable(self, meta: VideoMeta) -> bool:
        return meta.orientation == Orientation.PORTRAIT

    def propose(self, packet: FramePacket) -> List[RegionProposal]:
        meta = packet.meta
        return [_make_full_width_band(
            meta,
            y_top=0.75, y_bot=0.98,
            weight=0.92,
            source=f"policy:{self.name}",
            desc="竖屏短视频底部字幕带",
        )]


class TopNewsPolicy(SubtitleRegionPolicy):
    """横屏顶部字幕带 (新闻 / 标题字幕)。

    适用: LANDSCAPE。顶部 ~5-25% 高度。
    weight = 0.70 (顶部字幕相对少见, 给一个中等 weight 让 classifier 决定)。
    """

    @property
    def name(self) -> str:
        return "top_news"

    def is_applicable(self, meta: VideoMeta) -> bool:
        return meta.orientation == Orientation.LANDSCAPE

    def propose(self, packet: FramePacket) -> List[RegionProposal]:
        meta = packet.meta
        return [_make_full_width_band(
            meta,
            y_top=0.05, y_bot=0.25,
            weight=0.70,
            source=f"policy:{self.name}",
            desc="新闻 / 标题字幕带",
        )]


class CustomPolicy(SubtitleRegionPolicy):
    """用户自定义 region。用户传入 (y_top_ratio, y_bot_ratio, weight) 即可。

    适用: 任何方向。
    """

    def __init__(self, name: str, y_top_ratio: float, y_bot_ratio: float,
                 weight: float = 0.5,
                 orientation_filter: Orientation | None = None,
                 description: str = "用户自定义"):
        self._name = name
        self._y_top = y_top_ratio
        self._y_bot = y_bot_ratio
        self._weight = weight
        self._orientation_filter = orientation_filter
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    def is_applicable(self, meta: VideoMeta) -> bool:
        if self._orientation_filter is None:
            return True
        return meta.orientation == self._orientation_filter

    def propose(self, packet: FramePacket) -> List[RegionProposal]:
        meta = packet.meta
        return [_make_full_width_band(
            meta,
            y_top=self._y_top, y_bot=self._y_bot,
            weight=self._weight,
            source=f"policy:{self._name}",
            desc=self._description,
        )]