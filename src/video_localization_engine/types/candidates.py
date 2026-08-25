"""Phase C 新增类型: TextLineCandidate + SubtitleCandidate + 状态枚举。

T1 输出: TextLineCandidate — 单帧单行候选
T2 输出: SubtitleCandidate — 跨帧聚合, 4 状态生命周期
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from video_localization_engine.types.detection import (
    BBox, FrameTextCandidates, Polygon, TextCandidate,
)


class SubtitleCandidateState(str, Enum):
    """SubtitleCandidate 4 状态生命周期。"""
    NEW = "new"             # 首次观测
    ACTIVE = "active"       # 跨帧匹配中, 持续观测
    ENDING = "ending"       # 上一帧未匹配, 处于 grace period
    CLOSED = "closed"       # grace period 过期, 提交给 T3


@dataclass
class TextLineCandidate:
    """T1 输出: 单帧单行候选。

    Phase C 默认 1 TextCandidate = 1 行 (PaddleOCR 输出粒度)。
    未来扩展可加 multi-char 合并。
    """
    polygon: Polygon
    bbox: BBox
    text: str
    confidence: float
    language: Optional[str]
    detector_id: str
    frame_id: int
    timestamp_ms: int
    char_count: int
    # 来源
    source_candidate: Optional[TextCandidate] = None

    @classmethod
    def from_text_candidate(
        cls, c: TextCandidate, frame_id: int, timestamp_ms: int
    ) -> "TextLineCandidate":
        return cls(
            polygon=c.polygon,
            bbox=c.bbox,
            text=c.text,
            confidence=c.confidence,
            language=c.language,
            detector_id=c.detector_id,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            char_count=c.char_count or len(c.text.strip()),
            source_candidate=c,
        )


@dataclass
class SubtitleCandidate:
    """T2 输出: 跨帧聚合的候选。

    不判定"是不是字幕", 只描述"这个文本序列在多帧连续出现"。
    状态由 SubtitleCandidateBuffer 管理。
    """
    candidate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: SubtitleCandidateState = SubtitleCandidateState.NEW

    # 时间窗口
    first_frame: int = 0
    last_frame: int = 0
    last_matched_frame: int = 0     # 上一次成功匹配的帧
    frame_count: int = 0             # 成功匹配的帧数
    duration_ms: int = 0
    grace_frames_left: int = 0       # ENDING 状态剩余 grace 帧数

    # 内容历史
    text_history: List[str] = field(default_factory=list)
    # 几何历史
    polygon_history: List[Polygon] = field(default_factory=list)
    bbox_history: List[BBox] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    timestamp_history: List[int] = field(default_factory=list)

    # 上一次匹配到的 TextLineCandidate, 用于下一帧 match 参考
    last_polygon: Optional[Polygon] = None
    last_text: Optional[str] = None

    # raw features (Phase C 仅保存, 不评分)
    features: Dict[str, float] = field(default_factory=dict)

    def add_match(self, line: TextLineCandidate) -> None:
        """记录一次成功匹配。"""
        if self.state == SubtitleCandidateState.ENDING:
            self.state = SubtitleCandidateState.ACTIVE
        elif self.state == SubtitleCandidateState.NEW:
            self.state = SubtitleCandidateState.ACTIVE
        self.last_frame = line.frame_id
        self.last_matched_frame = line.frame_id
        self.last_polygon = line.polygon
        self.last_text = line.text
        self.frame_count += 1
        self.duration_ms = max(self.duration_ms, line.timestamp_ms)
        if not self.text_history or self.text_history[-1] != line.text:
            self.text_history.append(line.text)
        self.polygon_history.append(line.polygon)
        self.bbox_history.append(line.bbox)
        self.confidence_history.append(line.confidence)
        self.timestamp_history.append(line.timestamp_ms)
        if self.first_frame == 0:
            self.first_frame = line.frame_id

    def mark_missing(self, grace_frames: int = 2) -> None:
        """本帧未匹配到 (从 ACTIVE → ENDING, 或 ENDING grace--).

        grace_frames: 第一次进入 ENDING 时的 grace 帧数 (来自 SubtitleCandidateBuffer.config).
        """
        if self.state == SubtitleCandidateState.ACTIVE:
            self.state = SubtitleCandidateState.ENDING
            self.grace_frames_left = grace_frames
        elif self.state == SubtitleCandidateState.ENDING:
            self.grace_frames_left -= 1
            if self.grace_frames_left <= 0:
                self.state = SubtitleCandidateState.CLOSED

    @property
    def representative_text(self) -> str:
        """跨帧出现次数最多的文本。"""
        if not self.text_history:
            return ""
        from collections import Counter
        return Counter(self.text_history).most_common(1)[0][0]

    def compute_raw_features(self) -> None:
        """计算 raw features, 不评分, 仅描述。

        Phase D 的 classifier 拿这些 features 计算 score。
        """
        if not self.bbox_history:
            return
        # 平均 confidence
        self.features["avg_confidence"] = (
            sum(self.confidence_history) / len(self.confidence_history)
            if self.confidence_history else 0.0
        )
        # text stability: 1 - (unique_texts / total_observations)
        if self.text_history:
            self.features["text_stability"] = (
                1.0 - len(set(self.text_history)) / len(self.text_history)
            )
        # bbox stability: 1 - (mean bbox variance)
        if len(self.bbox_history) >= 2:
            xs1 = [b[0] for b in self.bbox_history]
            ys1 = [b[1] for b in self.bbox_history]
            xs2 = [b[2] for b in self.bbox_history]
            ys2 = [b[3] for b in self.bbox_history]
            # normalize by frame size — 这里仅给绝对值方差, classifier 自适配
            vx = sum((x - sum(xs1) / len(xs1)) ** 2 for x in xs1) / len(xs1)
            vy = sum((y - sum(ys1) / len(ys1)) ** 2 for y in ys1) / len(ys1)
            # stability: 1 - normalized variance (高 = 稳定)
            avg = (sum(xs1) + sum(ys1) + sum(xs2) + sum(ys2)) / (
                4 * len(self.bbox_history)
            )
            if avg > 0:
                var_total = (vx + vy) ** 0.5 / avg
                self.features["centroid_stability"] = max(0.0, 1.0 - var_total)
        # char count
        self.features["char_count"] = float(
            self.text_history[0] and len(self.text_history[0]) or 0
        )
        # duration
        self.features["duration_ms"] = float(self.duration_ms)