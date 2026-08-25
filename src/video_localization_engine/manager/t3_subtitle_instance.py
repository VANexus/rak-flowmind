"""T3: SubtitleCandidate → SubtitleInstance。

把 closed 候选"打包"成 SubtitleInstance。
- 不写 region score / ui_exclusion score (留给 Phase D classifier)
- 不决定 status 最终值 (CANDIDATE / ACTIVE / REJECTED)
- 仅做结构化: 收 features, 写 representative bbox/text
"""
from __future__ import annotations

from typing import List

from video_localization_engine.types.candidates import SubtitleCandidate
from video_localization_engine.types.instance import (
    InstanceStatus,
    SubtitleInstance,
)


class SubtitleInstanceExtractor:
    """T3: SubtitleCandidate → SubtitleInstance.

    设计:
      - Phase C 不判定 subtitle / logo / ui
      - 所有 instance 默认 status = CANDIDATE (待 Phase D 决定)
      - features 透传
    """

    def extract_one(self, cand: SubtitleCandidate) -> SubtitleInstance:
        if not cand.features:
            cand.compute_raw_features()
        # 构造一个 TextCandidate (用于 fill SubtitleInstance 字段)
        from video_localization_engine.types.detection import TextCandidate
        rep_text = cand.representative_text
        rep_polygon = cand.polygon_history[0] if cand.polygon_history else []
        rep_bbox = cand.bbox_history[0] if cand.bbox_history else (0, 0, 0, 0)
        avg_conf = cand.features.get("avg_confidence", 0.0)
        # 包装成 TextCandidate 让 SubtitleInstance.update_with 接管
        tc = TextCandidate(
            polygon=rep_polygon,
            text=rep_text,
            confidence=avg_conf,
            detector_id="manager_phase_c",
            bbox=rep_bbox,
        )
        inst = SubtitleInstance(status=InstanceStatus.CANDIDATE)
        inst.update_with(tc, cand.first_frame, cand.timestamp_history[0] if cand.timestamp_history else 0)
        # 透传所有后续观测 (用 timestamp_history 推 frame_id)
        n = len(cand.bbox_history)
        for i in range(1, n):
            ts = cand.timestamp_history[i] if i < len(cand.timestamp_history) else 0
            ttc = TextCandidate(
                polygon=cand.polygon_history[i],
                text=cand.text_history[i] if i < len(cand.text_history) else rep_text,
                confidence=cand.confidence_history[i] if i < len(cand.confidence_history) else avg_conf,
                detector_id="manager_phase_c",
                bbox=cand.bbox_history[i],
            )
            frame_id = cand.first_frame + i if n > 0 else 0
            inst.update_with(ttc, frame_id, ts)
        # finalize (写 representative_bbox/text/features)
        inst.finalize()
        # 透传 buffer 的 features (Phase D 用)
        inst.features.update(cand.features)
        # 状态历史
        inst.classification_reason = (
            f"phase_c_extracted: text_count={len(cand.text_history)}, "
            f"frame_count={cand.frame_count}, duration={cand.duration_ms}ms"
        )
        return inst

    def extract_all(self, candidates: List[SubtitleCandidate]) -> List[SubtitleInstance]:
        return [self.extract_one(c) for c in candidates]