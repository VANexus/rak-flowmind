"""L7 数据结构: AudioSegment / LocalizedSubtitle / LocalizedTrack。

L7 不直接 import L5/L6 内部类型 (除了 VideoMeta/SubtitleInstance/BBox/Polygon),
通过 SubtitleInstance 拿数据, 通过 MaskArtifact 不接触。

AudioSegment 持有 numpy 音频数组;不直接暴露给业务层以外的 code path。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from video_localization_engine.types.detection import BBox, Polygon
from video_localization_engine.types.video import VideoMeta


@dataclass
class AudioSegment:
    """单 instance 的合成音频段 (mono float32)。"""
    instance_id: str
    start_ms: int
    end_ms: int
    sample_rate: int
    samples: np.ndarray
    translated_text: str
    confidence: float = 1.0

    @property
    def duration_ms(self) -> int:
        return int(len(self.samples) / max(self.sample_rate, 1) * 1000)


@dataclass
class LocalizedSubtitle:
    """单 instance 的本地化结果。"""
    instance_id: str
    source_text: str
    target_text: str
    source_locale: str
    target_locale: str
    start_ms: int
    end_ms: int
    bbox: Optional[BBox] = None
    polygon: Optional[Polygon] = None
    audio: Optional[AudioSegment] = None
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class LocalizedTrack:
    """L7 完整输出。"""
    video_meta: VideoMeta
    target_locale: str
    subtitles: List[LocalizedSubtitle] = field(default_factory=list)
    rendered_video_path: Optional[str] = None
    audio_path: Optional[str] = None
    pipeline_version: str = "0.1.0"
