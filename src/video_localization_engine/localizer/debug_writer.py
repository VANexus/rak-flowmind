"""L7 debug artifact writer — translated_subtitles / final preview / localized_audio / full_pipeline.vle.json。"""
from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import List

import cv2
import numpy as np

from video_localization_engine.localizer.types import LocalizedTrack


def _pick_middle_frame_index(items: List) -> int:
    return len(items) // 2 if items else 0


def write_localize_artifacts(
    *,
    track: LocalizedTrack,
    results: list,
    rendered_frames: List[np.ndarray],
    full_audio: np.ndarray,
    sample_rate: int,
    out_dir: str,
) -> List[str]:
    """把 L7 产物落盘, 返回写入文件路径列表。"""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    # 1. translated_subtitles.png — 中间帧 + 翻译字幕叠加
    if rendered_frames:
        mid = _pick_middle_frame_index(rendered_frames)
        p = out_path / "translated_subtitles.png"
        cv2.imwrite(str(p), rendered_frames[mid])
        written.append(str(p))

        # 2. final_video_preview.png (= rendered_frames[mid])
        p = out_path / "final_video_preview.png"
        cv2.imwrite(str(p), rendered_frames[mid])
        written.append(str(p))

    # 3. localized_audio.wav
    if full_audio is not None and len(full_audio) > 0:
        wav_path = out_path / "localized_audio.wav"
        audio = np.clip(full_audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        written.append(str(wav_path))

    # 4. full_pipeline.vle.json — LocalizedTrack 序列化
    payload = {
        "version": "0.2.0",   # Phase E 引入, 与 Phase D 0.1.0 区分
        "video": {
            "source_path": track.video_meta.source_path,
            "width": track.video_meta.width,
            "height": track.video_meta.height,
            "fps": track.video_meta.fps,
            "frame_count": track.video_meta.frame_count,
            "duration_ms": track.video_meta.duration_ms,
        },
        "target_locale": track.target_locale,
        "rendered_video_path": track.rendered_video_path,
        "audio_path": track.audio_path,
        "subtitles": [
            {
                "instance_id": s.instance_id,
                "source_text": s.source_text,
                "target_text": s.target_text,
                "source_locale": s.source_locale,
                "target_locale": s.target_locale,
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "bbox": list(s.bbox) if s.bbox else None,
                "polygon": [list(p) for p in s.polygon] if s.polygon else None,
                "features": dict(s.features),
                "audio_ms": s.audio.duration_ms if s.audio else 0,
            }
            for s in track.subtitles
        ],
        "pipeline_version": track.pipeline_version,
    }
    p = out_path / "full_pipeline.vle.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    written.append(str(p))
    return written
