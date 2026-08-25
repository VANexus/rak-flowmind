"""把 .vle.json 反序列化为 SubtitleTrack + VideoMeta + SubtitleInstance 列表。

Phase E `run_localize_from_vle` 用: 跳过 L1-L4, 复用 Phase D 的产物。
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from video_localization_engine.types.detection import (
    BBox,
    FrameTextCandidates,
    Polygon,
)
from video_localization_engine.types.instance import (
    InstanceScore,
    InstanceStatus,
    SubtitleInstance,
    SubtitleTrack,
)
from video_localization_engine.types.video import Orientation, VideoMeta


def deserialize_video_meta(raw: Dict[str, Any], fallback_path: str = "") -> VideoMeta:
    """从 .vle.json 的 video 段构造 VideoMeta。"""
    return VideoMeta(
        source_path=raw.get("source_path") or fallback_path,
        width=int(raw["width"]),
        height=int(raw["height"]),
        fps=float(raw["fps"]),
        frame_count=int(raw["frame_count"]),
        duration_ms=int(raw["duration_ms"]),
        orientation=Orientation(raw.get("orientation", "landscape")),
        has_audio=bool(raw.get("has_audio", False)),
        content_type_hint=raw.get("content_type_hint"),
        source_locale=raw.get("source_locale"),
    )


def deserialize_instance(raw: Dict[str, Any]) -> SubtitleInstance:
    """从 .vle.json 的 instances[i] 段构造 SubtitleInstance。"""
    inst = SubtitleInstance(instance_id=raw["instance_id"])
    inst.status = InstanceStatus(raw.get("status", "candidate"))
    inst.first_frame = int(raw.get("first_frame", 0))
    inst.last_frame = int(raw.get("last_frame", 0))
    inst.frame_count = int(raw.get("frame_count", 0))
    inst.duration_ms = int(raw.get("duration_ms", 0))
    inst.representative_text = raw.get("representative_text", "")
    inst.text_observations = list(raw.get("text_observations", []))
    rb = raw.get("representative_bbox")
    inst.representative_bbox = tuple(rb) if rb else None  # type: ignore[assignment]
    rp = raw.get("representative_polygon")
    inst.representative_polygon = [tuple(p) for p in rp] if rp else None
    inst.observed_bboxes = [tuple(b) for b in raw.get("observed_bboxes", [])]
    inst.observed_frame_ids = list(raw.get("observed_frame_ids", []))
    inst.observed_polygons = [
        [tuple(p) for p in poly] for poly in raw.get("observed_polygons", [])
    ]
    inst.is_multiline = bool(raw.get("is_multiline", False))
    inst.multiline_count = int(raw.get("multiline_count", 1))
    inst.features = dict(raw.get("features", {}))
    score_raw = raw.get("score", {}) or {}
    inst.score = InstanceScore(
        region_score=float(score_raw.get("region_score", 0.0)),
        persistence_score=float(score_raw.get("persistence_score", 0.0)),
        arrangement_score=float(score_raw.get("arrangement_score", 0.0)),
        font_score=float(score_raw.get("font_score", 0.0)),
        ui_exclusion_score=float(score_raw.get("ui_exclusion_score", 0.0)),
    )
    inst.classification_reason = raw.get("classification_reason", "")
    inst.detector_id = raw.get("detector_id", "unknown")
    inst.locale = raw.get("locale")
    return inst


def deserialize_track(raw: Dict[str, Any]) -> Tuple[VideoMeta, List[SubtitleInstance], SubtitleTrack]:
    """返回 (VideoMeta, instances, SubtitleTrack)。"""
    meta = deserialize_video_meta(raw["video"], fallback_path=raw.get("video", {}).get("source_path", ""))
    instances = [deserialize_instance(i) for i in raw.get("instances", [])]
    frame_candidates = [
        FrameTextCandidates(
            frame_id=fc.get("frame_id", 0),
            timestamp_ms=fc.get("timestamp_ms", 0),
            width=fc.get("width", 0),
            height=fc.get("height", 0),
            candidates=[],
        )
        for fc in raw.get("frame_candidates", [])
    ]
    track = SubtitleTrack(
        video_meta_path=raw["video"].get("source_path", ""),
        instances=list(instances),
        frame_candidates=frame_candidates,
        region_policies_used=list(raw.get("region_policies_used", [])),
        detector_id=raw.get("detector_id", "unknown"),
    )
    return meta, instances, track
