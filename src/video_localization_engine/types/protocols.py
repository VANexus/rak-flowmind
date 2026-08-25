"""VLE backend 协议 (interfaces only — implementations live in each layer's package).

这些 Protocol 是 backend 必须满足的最小契约。任何用户写的 backend 只要满足
Protocol 就能注册进 Registry, 被 orchestrator 调用。

不允许在协议里写:
  - 任何阈值常量 (如 y > 0.8)
  - 任何特定算法名称 (如 PaddleOCR / ProPainter)
  - 任何特定视频类型的 hard code
"""
from __future__ import annotations

from typing import Iterator, List, Protocol, runtime_checkable

import numpy as np

from video_localization_engine.types.detection import (
    FrameTextCandidates,
    RegionProposal,
    TextCandidate,
)
from video_localization_engine.types.instance import SubtitleInstance
from video_localization_engine.types.video import FramePacket, VideoMeta


# ============================================================
# L1 VideoAnalyzer
# ============================================================
@runtime_checkable
class VideoAnalyzerProtocol(Protocol):
    """视频输入分析器。提供 VideoMeta + 帧迭代器。"""

    @property
    def meta(self) -> VideoMeta: ...

    def __iter__(self) -> Iterator[FramePacket]: ...

    def seek(self, frame_id: int) -> FramePacket: ...

    def close(self) -> None: ...


# ============================================================
# L2 RegionPolicy
# ============================================================
@runtime_checkable
class RegionPolicyProtocol(Protocol):
    """字幕区域策略: 给定一个 frame packet, 返回候选 RegionProposal。"""

    @property
    def name(self) -> str: ...

    def is_applicable(self, meta: VideoMeta) -> bool:
        """此 policy 是否适用于该视频 (基于 orientation/content_type_hint)"""
        ...

    def propose(self, packet: FramePacket) -> List[RegionProposal]:
        """返回此 frame 的候选区域 proposals (空列表 = 不发表意见)"""
        ...


# ============================================================
# L3 TextDetector
# ============================================================
@runtime_checkable
class TextDetectorProtocol(Protocol):
    """文字检测器: 输入一帧, 输出该帧所有 TextCandidate。"""

    @property
    def name(self) -> str: ...

    @property
    def supported_languages(self) -> List[str]: ...

    def detect(self, packet: FramePacket) -> List[TextCandidate]: ...

    def warmup(self) -> None:
        """可选: 预热模型。"""
        ...


# ============================================================
# L4 SubtitleManager 子协议
# ============================================================
@runtime_checkable
class InstanceTrackerProtocol(Protocol):
    """跨帧 instance tracker: 输入新一帧的 candidates, 输出 instance 状态更新。"""

    def update(
        self, candidates: List[TextCandidate], packet: FramePacket
    ) -> List["InstanceUpdate"]:
        ...

    def tick(self, packet: FramePacket) -> List[SubtitleInstance]:
        """推进一帧时间, 返回刚刚 finished 的 instances"""
        ...


@runtime_checkable
class SubtitleClassifierProtocol(Protocol):
    """instance 分类器: 给出 instance 是不是字幕的评分。"""

    def score(
        self, instance: SubtitleInstance, proposals: List[RegionProposal]
    ) -> "InstanceScore":
        ...

    def decide(self, instance: SubtitleInstance, score: "InstanceScore") -> str:
        """返回 'subtitle' / 'logo' / 'ui' / 'watermark' / 'ambiguous'"""
        ...


@runtime_checkable
class SubtitleManagerProtocol(Protocol):
    """L4 顶层入口。"""

    def feed(
        self,
        candidates: List[TextCandidate],
        proposals: List[RegionProposal],
        packet: FramePacket,
    ) -> None: ...

    def finish(self) -> List[SubtitleInstance]:
        """所有帧处理完, 返回所有 finalized instance"""
        ...


# ============================================================
# L5 MaskGenerator
# ============================================================
@runtime_checkable
class MaskBackendProtocol(Protocol):
    """Mask backend: 输入 frame + instances, 输出 HxW uint8 binary mask。"""

    @property
    def name(self) -> str: ...

    def build(
        self, instances: List[SubtitleInstance], packet: FramePacket
    ) -> np.ndarray:
        """返回 binary mask, 0/255 or 0/1"""
        ...


# ============================================================
# L6 InpaintingEngine
# ============================================================
@runtime_checkable
class InpaintingBackendProtocol(Protocol):
    """Inpainting backend: 输入 frame + mask, 输出修复后 frame。"""

    @property
    def name(self) -> str: ...

    def is_available(self) -> bool: ...

    def inpaint(self, packet: FramePacket, mask: np.ndarray) -> np.ndarray: ...


# ============================================================
# data classes used by protocols
# ============================================================
from dataclasses import dataclass
from video_localization_engine.types.detection import BBox, Polygon
from video_localization_engine.types.instance import InstanceScore


@dataclass
class InstanceUpdate:
    """Tracker 单次 update 输出: 给 manager 一个 instance 的最新状态。"""
    instance_id: str
    bbox: BBox
    polygon: Polygon
    text: str
    confidence: float
    # 与上一帧同一个 instance 的关联度 (用于 buffer 决定 keep/miss)
    match_score: float = 1.0
    detector_id: str = "unknown"