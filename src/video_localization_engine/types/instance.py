"""L4 核心: SubtitleInstance — 跨帧稳定身份。

SubtitleInstance 是 VLE 最重要的输出, 也是 L1-L6 与未来 L7 的数据协议。
它不绑定任何视频, 不绑定分辨率, 不绑定算法。

属性:
  - 跨帧稳定身份 (instance_id)
  - 时间窗口 (first_frame, last_frame, frame_count)
  - 文字内容 (稳定化)
  - 几何 (代表 bbox / 代表 polygon, 多帧投票或均值)
  - 分类结果 (is_subtitle + score 维度)
  - 来源 (哪些帧、哪个 detector、哪些 region_policy 投了票)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from video_localization_engine.types.detection import BBox, Polygon, TextCandidate


class InstanceStatus(str, Enum):
    """Instance 生命周期。"""
    CANDIDATE = "candidate"      # 跨帧匹配中, 还不确定是不是字幕
    ACTIVE = "active"            # 确认是字幕 (classifier score 达标)
    REJECTED = "rejected"        # classifier 判定非字幕 (logo/ui/watermark)
    FINISHED = "finished"        # 不再出现, 已 finalize


@dataclass
class InstanceScore:
    """分类器评分, 多维度。"""
    region_score: float = 0.0        # 落在 region_policy 提案里的程度
    persistence_score: float = 0.0   # 持续时长信号
    arrangement_score: float = 0.0   # 排列/字符数
    font_score: float = 0.0          # 字体一致性
    ui_exclusion_score: float = 0.0  # UI/logo 排除信号 (高 = 像 UI)

    @property
    def total(self) -> float:
        # 默认加权, 实际由 classifier 决定; 此处给一个稳健的 fallback
        return max(
            0.0,
            self.region_score
            + self.persistence_score
            + self.arrangement_score
            + self.font_score
            - self.ui_exclusion_score,
        )


@dataclass
class SubtitleInstance:
    """跨帧稳定的字幕身份。

    Phase C 设计:
      - 不在 L4 写硬评分, 仅持有 raw features
      - 评分由 Phase D 的 classifier 完成
      - text_observations / polygon_history / features 都是 raw 观测
    """
    instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: InstanceStatus = InstanceStatus.CANDIDATE

    # 时间窗口
    first_frame: int = 0
    last_frame: int = 0
    frame_count: int = 0                     # 出现过的帧数 (去重)
    duration_ms: int = 0                     # last_frame.timestamp - first_frame.timestamp

    # 文字
    representative_text: str = ""            # 跨帧最稳定的代表文本
    text_observations: List[str] = field(default_factory=list)

    # 几何 — 代表值 (多帧均值或投票)
    representative_bbox: Optional[BBox] = None
    representative_polygon: Optional[Polygon] = None
    # 跨帧所有观测的 bbox, 供 mask builder 取用
    observed_bboxes: List[BBox] = field(default_factory=list)
    observed_frame_ids: List[int] = field(default_factory=list)
    observed_polygons: List[Polygon] = field(default_factory=list)  # Phase C 新增

    # 多行信息 (同一 instance 由多 polygon 拼成时填)
    is_multiline: bool = False
    multiline_count: int = 1

    # raw features (Phase C 仅保存, 不评分)
    features: Dict[str, float] = field(default_factory=dict)

    # 评分 (Phase D classifier 填)
    score: InstanceScore = field(default_factory=InstanceScore)
    classification_reason: str = ""          # 人类可读的判定原因

    # 来源追溯
    detector_id: str = "unknown"
    locale: Optional[str] = None             # 推测语种
    source_candidates: List[TextCandidate] = field(default_factory=list)

    def update_with(self, candidate: TextCandidate, frame_id: int, timestamp_ms: int):
        """新一帧观测并入。"""
        self.last_frame = frame_id
        self.frame_count += 1
        self.duration_ms = max(self.duration_ms, timestamp_ms)
        if self.first_frame == 0:
            self.first_frame = frame_id
        # bbox 累积
        if candidate.bbox:
            self.observed_bboxes.append(candidate.bbox)
            self.observed_frame_ids.append(frame_id)
            self.observed_polygons.append(candidate.polygon)
        # text 累积 (用于后面做代表文本投票)
        if candidate.text and candidate.text not in self.text_observations:
            self.text_observations.append(candidate.text)
        # 来源
        self.source_candidates.append(candidate)
        # 更新 detector_id (后写覆盖前写, 通常不会变)
        self.detector_id = candidate.detector_id
        if not self.locale and candidate.language:
            self.locale = candidate.language

    def finalize(self):
        """不再观测后 finalize: 计算代表 bbox / text / features 综合。"""
        # 代表文本: 选出现次数最多
        from collections import Counter
        if self.text_observations:
            counter = Counter(self.text_observations)
            self.representative_text = counter.most_common(1)[0][0]
        # 代表 bbox: 跨帧均值
        if self.observed_bboxes:
            xs1 = [b[0] for b in self.observed_bboxes]
            ys1 = [b[1] for b in self.observed_bboxes]
            xs2 = [b[2] for b in self.observed_bboxes]
            ys2 = [b[3] for b in self.observed_bboxes]
            self.representative_bbox = (
                sum(xs1) / len(xs1),
                sum(ys1) / len(ys1),
                sum(xs2) / len(xs2),
                sum(ys2) / len(ys2),
            )
        # raw features: Phase C 不强制, Phase D 的 classifier 决定需要哪些
        if self.observed_bboxes:
            self.features["avg_confidence"] = (
                sum(c.confidence for c in self.source_candidates)
                / max(len(self.source_candidates), 1)
            )
            self.features["char_count"] = float(
                len(self.representative_text) if self.representative_text else 0
            )
            self.features["duration_ms"] = float(self.duration_ms)
            self.features["frame_count"] = float(self.frame_count)


@dataclass
class SubtitleTrack:
    """一个完整视频跑完 VLE L1-L6 的结果。

    这是要写进 .vle.json 的对象。
    """
    video_meta_path: str                  # 引用 VideoMeta 的序列化
    instances: List[SubtitleInstance] = field(default_factory=list)
    # 原始 detector 输出 (per frame), 用于 L7 重生成或多语言重跑
    frame_candidates: List[FrameTextCandidates] = field(default_factory=list)
    # 区域政策被采纳的版本 (用于 reproducibility)
    region_policies_used: List[str] = field(default_factory=list)
    detector_id: str = "unknown"
    pipeline_version: str = "0.1.0"