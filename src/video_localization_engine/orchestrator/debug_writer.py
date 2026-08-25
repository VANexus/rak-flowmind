"""Debug artifact writer — 把 orchestrator 结果落盘供人工与测试验证。

产出 (/tmp/vle_debug/<fixture>/):
  - track.vle.json            SubtitleTrack 序列化
  - subtitles_vis.png         中间帧 + representative bbox + text 标注
  - mask_overlay.png          中间帧 + mask 红色半透明 overlay
  - before.png                中间帧原始
  - after.png                 中间帧 inpaint 后 (仅 enable_inpaint)
  - side_by_side.png          before | after 并排 (仅 enable_inpaint)
  - per_frame/frame_*.png     按 sample_rate 抽帧, 含 mask overlay
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from video_localization_engine.orchestrator.pipeline import PipelineFrameResult
from video_localization_engine.types.instance import SubtitleTrack
from video_localization_engine.types.video import VideoMeta
from video_localization_engine.utils.persistence import save_track


def _pick_middle_frame(results: List[PipelineFrameResult]) -> Optional[PipelineFrameResult]:
    """选最具代表性的中间帧。

    优先级:
      1. 中位 frame_id
      2. 若空, 返回第一个非空结果
    """
    if not results:
        return None
    mid_idx = len(results) // 2
    return results[mid_idx]


def _draw_subtitles_vis(img: np.ndarray, frame_result: PipelineFrameResult) -> np.ndarray:
    """画每个 active instance 的 representative_bbox + text。"""
    out = img.copy()
    for inst in frame_result.instances_active:
        if not inst.representative_bbox:
            continue
        x1, y1, x2, y2 = [int(v) for v in inst.representative_bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{inst.instance_id[:8]} {inst.representative_text[:20]!r}"
        # text above bbox
        ty = max(y1 - 6, 12)
        cv2.putText(out, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)
    return out


def _draw_mask_overlay(img: np.ndarray, frame_result: PipelineFrameResult) -> np.ndarray:
    """画 union mask 红色半透明 overlay。"""
    if not frame_result.mask_artifacts:
        return img.copy()
    from video_localization_engine.mask.mask_generator import MaskGenerator
    union = MaskGenerator.union(frame_result.mask_artifacts)
    if union is None:
        return img.copy()
    mask = union.materialize()
    color_layer = np.zeros_like(img)
    color_layer[mask > 0] = (0, 0, 255)  # BGR red
    return cv2.addWeighted(img, 0.6, color_layer, 0.4, 0)


def _side_by_side(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """水平拼接 before | after, 并标注。"""
    if before.shape != after.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]))
    h, w = before.shape[:2]
    sep = np.full((h, 4, 3), 128, dtype=np.uint8)
    canvas = np.hstack([before, sep, after])
    cv2.putText(canvas, "before", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "after", (w + 14, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def write_debug_artifacts(
    *,
    track: SubtitleTrack,
    results: List[PipelineFrameResult],
    meta: VideoMeta,
    out_dir: str,
    enable_inpaint: bool = True,
    sample_rate: int = 1,
) -> List[str]:
    """把所有 debug artifact 写到 out_dir,返回写入的文件路径列表。"""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    # 1. track.vle.json
    vle_path = out_path / "track.vle.json"
    save_track(track, meta, str(vle_path))
    written.append(str(vle_path))

    middle = _pick_middle_frame(results)
    if middle is None:
        return written

    # 2. subtitles_vis.png
    p = out_path / "subtitles_vis.png"
    cv2.imwrite(str(p), _draw_subtitles_vis(middle.before_image, middle))
    written.append(str(p))

    # 3. mask_overlay.png
    p = out_path / "mask_overlay.png"
    cv2.imwrite(str(p), _draw_mask_overlay(middle.before_image, middle))
    written.append(str(p))

    # 4. before.png
    p = out_path / "before.png"
    cv2.imwrite(str(p), middle.before_image)
    written.append(str(p))

    # 5/6. after.png + side_by_side.png (仅 enable_inpaint)
    if enable_inpaint:
        p = out_path / "after.png"
        cv2.imwrite(str(p), middle.after_image)
        written.append(str(p))
        p = out_path / "side_by_side.png"
        cv2.imwrite(str(p), _side_by_side(middle.before_image, middle.after_image))
        written.append(str(p))

    # 7. per_frame/
    if sample_rate > 0 and results:
        per_dir = out_path / "per_frame"
        per_dir.mkdir(exist_ok=True)
        for i, fr in enumerate(results):
            if i % sample_rate != 0:
                continue
            overlay = _draw_mask_overlay(fr.before_image, fr)
            if enable_inpaint:
                canvas = _side_by_side(fr.before_image, fr.after_image)
                cv2.imwrite(str(per_dir / f"frame_{fr.frame_id:04d}_compare.png"), canvas)
            cv2.imwrite(str(per_dir / f"frame_{fr.frame_id:04d}_overlay.png"), overlay)
            written.append(str(per_dir / f"frame_{fr.frame_id:04d}_overlay.png"))

    return written
