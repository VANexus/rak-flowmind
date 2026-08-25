"""T2: TextLineCandidate → SubtitleCandidate (跨帧聚合 + 4 状态生命周期)。

不做:
  - region 评分
  - "是不是字幕" 判断
  - 最终 status 决定

只做:
  - 跨帧 IoU / centroid / text similarity 匹配
  - 累积 duration, frame_count, text_history, polygon_history
  - 状态转换: NEW → ACTIVE → ENDING → CLOSED
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from video_localization_engine.types.candidates import (
    SubtitleCandidate,
    SubtitleCandidateState,
    TextLineCandidate,
)
from video_localization_engine.utils.geometry import bbox_iou, polygon_centroid


@dataclass
class SubtitleCandidateConfig:
    """Phase C 配置 — 不含硬字幕阈值, 仅匹配参数。"""
    iou_match_threshold: float = 0.30          # IoU >= 此值视为同一 instance
    centroid_distance_max: float = 100.0      # 重心距离最大像素 (IoU 不足时 fallback)
    text_similarity_min: float = 0.5          # 文本相似度最低 (Jaccard)
    grace_frames: int = 2                     # ENDING 状态保持多少帧


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _poly_to_bbox(poly) -> tuple:
    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def _match_score(candidate: SubtitleCandidate, line: TextLineCandidate,
                 cfg: SubtitleCandidateConfig) -> float:
    """计算候选与新一帧 line 的匹配分数 (0-1)。

    评分维度 (无硬字幕判断, 仅"是不是同一个 line"):
      - bbox IoU (主要)
      - centroid distance (IoU 不足时 fallback)
      - text similarity (辅助)
    """
    if candidate.last_polygon is None:
        return 0.0
    iou = bbox_iou(line.bbox, _poly_to_bbox(candidate.last_polygon))
    if iou >= cfg.iou_match_threshold:
        score = iou
    else:
        last_cx, last_cy = polygon_centroid(candidate.last_polygon)
        cur_cx = (line.bbox[0] + line.bbox[2]) / 2
        cur_cy = (line.bbox[1] + line.bbox[3]) / 2
        dist = ((cur_cx - last_cx) ** 2 + (cur_cy - last_cy) ** 2) ** 0.5
        if dist > cfg.centroid_distance_max:
            return 0.0
        score = max(0.0, 1.0 - dist / cfg.centroid_distance_max) * 0.5
    if candidate.last_text:
        sim = _text_similarity(line.text, candidate.last_text)
        if sim < cfg.text_similarity_min:
            score *= 0.3
        elif sim >= 0.8:
            score = min(1.0, score * 1.1)
    return score


class SubtitleCandidateBuffer:
    """T2 buffer: 维护活跃候选列表, 每帧 feed 一组 TextLine。"""

    def __init__(self, config: Optional[SubtitleCandidateConfig] = None):
        self.config = config or SubtitleCandidateConfig()
        self._active: List[SubtitleCandidate] = []
        self._closed: List[SubtitleCandidate] = []

    @property
    def active(self) -> List[SubtitleCandidate]:
        return [c for c in self._active if c.state != SubtitleCandidateState.CLOSED]

    @property
    def closed(self) -> List[SubtitleCandidate]:
        return list(self._closed)

    def feed(self, lines: List[TextLineCandidate]) -> List[SubtitleCandidate]:
        """输入当前帧所有 line, 更新 buffer。

        返回本帧新 closed 的 candidates (T3 用)。
        """
        # 1. 每条 line 找最佳匹配 active candidate
        matched_cand_ids = set()
        for line in lines:
            best_cand: Optional[SubtitleCandidate] = None
            best_score = 0.0
            for cand in self._active:
                if cand.state == SubtitleCandidateState.CLOSED:
                    continue
                if cand.candidate_id in matched_cand_ids:
                    continue
                s = _match_score(cand, line, self.config)
                if s > best_score:
                    best_score = s
                    best_cand = cand
            if best_cand is not None and best_score > 0.0:
                best_cand.add_match(line)
                matched_cand_ids.add(best_cand.candidate_id)
            else:
                # 2. 没匹配到 → 新建 candidate
                nc = SubtitleCandidate()
                nc.add_match(line)
                self._active.append(nc)
                matched_cand_ids.add(nc.candidate_id)

        # 3. active 中本帧未匹配的 → mark_missing
        newly_closed: List[SubtitleCandidate] = []
        for cand in self._active:
            if cand.state == SubtitleCandidateState.CLOSED:
                continue
            if cand.candidate_id not in matched_cand_ids:
                cand.mark_missing(grace_frames=self.config.grace_frames)
                if cand.state == SubtitleCandidateState.CLOSED:
                    newly_closed.append(cand)

        # 4. 把已 closed 的从 active 移到 closed
        for c in newly_closed:
            self._closed.append(c)
            self._active.remove(c)
        return newly_closed

    def finalize(self) -> List[SubtitleCandidate]:
        """所有帧结束: 强制关闭所有活跃候选。"""
        for cand in self._active:
            if cand.state != SubtitleCandidateState.CLOSED:
                cand.state = SubtitleCandidateState.CLOSED
                self._closed.append(cand)
        self._active = []
        return list(self._closed)

    def compute_all_features(self) -> None:
        """对所有 closed 候选计算 raw features。"""
        for c in self._closed:
            c.compute_raw_features()