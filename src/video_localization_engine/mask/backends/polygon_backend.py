"""PolygonMaskBackend — 第一个 mask backend。

只做 polygon rasterize (cv2.fillPoly + 可选 cv2.dilate)。
未来 SAM / Segmentation backend 同 build() 签名,产出不同 mask_type 的 artifact。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from video_localization_engine.mask.artifact import (
    MaskArtifact,
    MaskType,
    _normalize_dilation,
)
from video_localization_engine.mask.base import MaskBackend
from video_localization_engine.types.instance import SubtitleInstance
from video_localization_engine.types.video import FramePacket


def _pick_polygon_for_frame(instance: SubtitleInstance, frame_id: int) -> Optional[List]:
    """从 instance.observed_polygons 中挑当前 frame 的 polygon。

    优先级:
      1. exact: observed_frame_ids[i] == frame_id
      2. fallback: 取最近一次(≤ frame_id)的 polygon,仅当 frame_id 在 instance 寿命内
         (frame_id ∈ [first_frame, last_frame]) 才允许 fallback,避免"死而复生"
    """
    if not instance.observed_polygons or not instance.observed_frame_ids:
        return None
    ids = instance.observed_frame_ids
    polys = instance.observed_polygons
    # exact match
    if frame_id in ids:
        idx = ids.index(frame_id)
        return polys[idx]
    # fallback: 最近一次匹配 (≤ frame_id)
    cand_idx = None
    for i, fid in enumerate(ids):
        if fid <= frame_id:
            cand_idx = i
        else:
            break
    if cand_idx is None:
        return None
    # 仅在 instance 寿命窗口内允许 fallback
    if instance.first_frame <= frame_id <= instance.last_frame:
        return polys[cand_idx]
    return None


class PolygonMaskBackend(MaskBackend):
    """polygon rasterize backend — 默认 mask backend。"""

    @property
    def name(self) -> str:
        return "polygon"

    def build(self, instances: List[SubtitleInstance], packet: FramePacket,
              dilation_px: int = 0,
              dilation_xy: Optional[Tuple[int, int]] = None) -> List[MaskArtifact]:
        # 优先级: dilation_xy 显式 > dilation_px (向后兼容)
        if dilation_xy is not None:
            dilation = tuple(dilation_xy)
        else:
            dilation = (int(dilation_px), int(dilation_px))
        h, w = packet.height, packet.width
        artifacts: List[MaskArtifact] = []
        for inst in instances:
            poly = _pick_polygon_for_frame(inst, packet.frame_id)
            # polygons 字段语义: List[Polygon] — 每个 polygon 是一组点。
            # 一个 instance 当前帧对应一个 polygon, 包成单元素列表。
            poly_list = [poly] if poly else None
            artifacts.append(MaskArtifact(
                frame_id=packet.frame_id,
                timestamp_ms=packet.timestamp_ms,
                mask_type=MaskType.POLYGON,
                polygons=poly_list,
                source_instance_id=inst.instance_id,
                confidence=inst.features.get("avg_confidence", 1.0),
                width=w,
                height=h,
                dilation_xy=dilation,
            ))
        return artifacts
