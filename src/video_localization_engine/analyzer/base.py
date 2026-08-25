"""L1 VideoAnalyzer 协议定义 + 抽象基类。

不假设字幕位置, 不假设分辨率, 不假设 codec。
只负责: 读视频 + 推 VideoMeta + 推 FramePacket 流。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional

import numpy as np

from video_localization_engine.types.video import FramePacket, Orientation, VideoMeta


class VideoAnalyzer(ABC):
    """视频分析器抽象基类。

    使用模式:
        analyzer = OpenCVVideoAnalyzer(path)
        meta = analyzer.meta
        for pkt in analyzer:
            ...
        analyzer.close()
    """

    @property
    @abstractmethod
    def meta(self) -> VideoMeta: ...

    @abstractmethod
    def __iter__(self) -> Iterator[FramePacket]: ...

    @abstractmethod
    def seek(self, frame_id: int) -> FramePacket: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> "VideoAnalyzer":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def derive_orientation(width: int, height: int) -> Orientation:
    """纯函数: 像素宽高 → Orientation。"""
    if width > height:
        return Orientation.LANDSCAPE
    if height > width:
        return Orientation.PORTRAIT
    return Orientation.SQUARE