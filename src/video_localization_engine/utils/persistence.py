""".vle.json 序列化 — SubtitleTrack <-> JSON。

设计:
- JSON 是 orchestrator 的 checkpoint 格式
- 可以是 L1-L6 跑完的中间产物 (供 L7 重生成)
- 也可以是 L7 跑完的最终结果

所有 dataclass 用 asdict + 手工序列化 (numpy array 不直接 JSON)。
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from video_localization_engine.types.instance import (
    InstanceScore,
    InstanceStatus,
    SubtitleInstance,
    SubtitleTrack,
)
from video_localization_engine.types.video import Orientation, VideoMeta


_VLE_VERSION = "0.1.0"


def _video_meta_to_dict(meta: VideoMeta) -> Dict[str, Any]:
    return {
        "source_path": meta.source_path,
        "width": meta.width,
        "height": meta.height,
        "fps": meta.fps,
        "frame_count": meta.frame_count,
        "duration_ms": meta.duration_ms,
        "orientation": meta.orientation.value,
        "has_audio": meta.has_audio,
        "content_type_hint": meta.content_type_hint,
        "source_locale": meta.source_locale,
    }


def _instance_to_dict(inst: SubtitleInstance) -> Dict[str, Any]:
    return {
        "instance_id": inst.instance_id,
        "status": inst.status.value,
        "first_frame": inst.first_frame,
        "last_frame": inst.last_frame,
        "frame_count": inst.frame_count,
        "duration_ms": inst.duration_ms,
        "representative_text": inst.representative_text,
        "text_observations": list(inst.text_observations),
        "representative_bbox": list(inst.representative_bbox) if inst.representative_bbox else None,
        "representative_polygon": list(inst.representative_polygon) if inst.representative_polygon else None,
        "observed_bboxes": [list(b) for b in inst.observed_bboxes],
        "observed_frame_ids": list(inst.observed_frame_ids),
        "observed_polygons": [[list(p) for p in poly] for poly in inst.observed_polygons],
        "is_multiline": inst.is_multiline,
        "multiline_count": inst.multiline_count,
        "features": dict(inst.features),
        "score": asdict(inst.score),
        "classification_reason": inst.classification_reason,
        "detector_id": inst.detector_id,
        "locale": inst.locale,
    }


def _frame_candidates_to_dict(fc) -> Dict[str, Any]:
    return {
        "frame_id": fc.frame_id,
        "timestamp_ms": fc.timestamp_ms,
        "width": fc.width,
        "height": fc.height,
        "candidates": [
            {
                "polygon": [list(p) for p in c.polygon],
                "text": c.text,
                "confidence": c.confidence,
                "language": c.language,
                "detector_id": c.detector_id,
                "char_count": c.char_count,
                "bbox": list(c.bbox) if c.bbox else None,
            }
            for c in fc.candidates
        ],
    }


def save_track(track: SubtitleTrack, video_meta: VideoMeta, path: str) -> None:
    """写 .vle.json。"""
    payload = {
        "version": _VLE_VERSION,
        "video": _video_meta_to_dict(video_meta),
        "instances": [_instance_to_dict(i) for i in track.instances],
        "frame_candidates": [_frame_candidates_to_dict(fc) for fc in track.frame_candidates],
        "region_policies_used": list(track.region_policies_used),
        "detector_id": track.detector_id,
        "pipeline_version": track.pipeline_version,
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def load_track(path: str) -> Dict[str, Any]:
    """读 .vle.json, 返回原始 dict (供 L7 等后续层反序列化)。"""
    return json.loads(Path(path).read_text())