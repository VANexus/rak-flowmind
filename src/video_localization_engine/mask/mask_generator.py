"""L5 门面: MaskGenerator。

不做 rasterize, 只调度 backend + 包装 MaskArtifact。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from video_localization_engine.mask.artifact import MaskArtifact, _normalize_dilation
from video_localization_engine.mask.base import MaskBackend
from video_localization_engine.mask.registry import MaskBackendRegistry
from video_localization_engine.types.instance import SubtitleInstance
from video_localization_engine.types.video import FramePacket


class MaskGenerator:
    """L5 入口: 给一帧 packet + 当前 active instances, 返回每个 instance 的 MaskArtifact。

    用法:
        mg = MaskGenerator(backend_name="polygon", dilation_px=2)
        artifacts = mg.generate_for_frame(active_instances, packet)
        # artifacts[i].materialize() → np.ndarray HxW uint8

    P10: 新增 x/y 独立 dilation.
        mg = MaskGenerator(dilation_px=12, dilation_y=20)  # y 方向更宽
    """

    def __init__(self, backend_name: str = "polygon",
                 dilation_px: int = 2,
                 dilation_y: Optional[int] = None):
        backend_cls = MaskBackendRegistry.get(backend_name)
        self.backend: MaskBackend = backend_cls()
        self.dilation_px = int(dilation_px)
        # dilation_y = None → 回退: x 方向用 dilation_px, y 方向也用 dilation_px
        # dilation_y = 0  → 显式 0, y 不扩张 (回到纯 x 扩张)
        self.dilation_y: int = (
            self.dilation_px if dilation_y is None else int(dilation_y)
        )
        self.dilation_xy: Tuple[int, int] = _normalize_dilation(
            (self.dilation_px, self.dilation_y)
        )
        self.backend_name = backend_name

    def generate_for_frame(self, instances: List[SubtitleInstance],
                           packet: FramePacket) -> List[MaskArtifact]:
        return self.backend.build(instances, packet, dilation_xy=self.dilation_xy)

    @staticmethod
    def union(artifacts: List[MaskArtifact]) -> "MaskArtifact | None":
        """把所有 artifact 的 polygons 合成一个 union artifact (供 inpaint 一步走)。

        返回 None 当 artifacts 为空。dilation_xy 沿用第一个 artifact 的。
        """
        if not artifacts:
            return None
        first = artifacts[0]
        all_polys = []
        max_conf = 0.0
        for a in artifacts:
            if a.polygons:
                all_polys.extend(a.polygons)
            max_conf = max(max_conf, a.confidence)
        return MaskArtifact(
            frame_id=first.frame_id,
            timestamp_ms=first.timestamp_ms,
            mask_type=first.mask_type,
            polygons=all_polys,
            source_instance_id="+".join(a.source_instance_id[:8] for a in artifacts),
            confidence=max_conf,
            width=first.width,
            height=first.height,
            dilation_xy=first.dilation_xy,
        )
