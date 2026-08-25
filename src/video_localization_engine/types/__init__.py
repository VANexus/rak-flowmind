"""VLE core data types — the contract between all layers.

三个核心对象:
  - VideoMeta / FramePacket        : L1 输入
  - TextCandidate / RegionProposal : L2/L3 中间产物
  - SubtitleInstance               : L4 跨帧稳定身份 (L1-L6 与 L7 的数据协议)

Phase C 新增:
  - TextLineCandidate              : T1 单帧单行候选
  - SubtitleCandidate              : T2 跨帧聚合候选 (NEW/ACTIVE/ENDING/CLOSED)
"""
from video_localization_engine.types.video import (
    Orientation,
    VideoMeta,
    FramePacket,
)
from video_localization_engine.types.detection import (
    TextCandidate,
    FrameTextCandidates,
    RegionProposal,
)
from video_localization_engine.types.instance import (
    InstanceStatus,
    SubtitleInstance,
    InstanceScore,
    SubtitleTrack,
)
from video_localization_engine.types.candidates import (
    SubtitleCandidateState,
    TextLineCandidate,
    SubtitleCandidate,
)

__all__ = [
    "Orientation",
    "VideoMeta",
    "FramePacket",
    "TextCandidate",
    "FrameTextCandidates",
    "RegionProposal",
    "InstanceStatus",
    "SubtitleInstance",
    "InstanceScore",
    "SubtitleTrack",
    "SubtitleCandidateState",
    "TextLineCandidate",
    "SubtitleCandidate",
]
