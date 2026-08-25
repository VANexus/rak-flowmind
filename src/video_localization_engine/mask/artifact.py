"""L5 核心数据结构: MaskArtifact。

不直接暴露 backend 内部实现 (np.ndarray / file path) 给业务层;
业务层只看到 MaskArtifact, 通过 .materialize() 在 backend 控制下物化。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

from video_localization_engine.types.detection import Polygon

if TYPE_CHECKING:
    from video_localization_engine.types.instance import SubtitleInstance


class MaskType(str, Enum):
    """Mask 来源类型 — 决定 materialize() 的实现策略。"""
    POLYGON = "polygon"             # polygon rasterize (cv2.fillPoly)
    SAM = "sam"                     # future: Segment Anything
    SEGMENTATION = "segmentation"   # future: 通用语义分割
    BOUNDING_BOX = "bounding_box"   # bbox 矩形 mask (兜底)


# 兼容类型: int (老 API), Tuple[int, int] (新 x/y 独立 dilation)
DilationSpec = Union[int, Tuple[int, int]]


def _normalize_dilation(d: DilationSpec) -> Tuple[int, int]:
    """把 int / (x, y) 统一成 (dx, dy). int n → (n, n). 负值截到 0."""
    if isinstance(d, int):
        return (max(0, d), max(0, d))
    dx, dy = d
    return (max(0, int(dx)), max(0, int(dy)))


@dataclass
class MaskArtifact:
    """单 instance × 单 frame 的 mask 产物。

    设计:
      - 业务层(L6 InpaintingEngine, orchestrator)只看到 MaskArtifact, 不直接接收 np.ndarray
      - 每个 instance 各自一份 artifact (orchestrator 决定是否 union)
      - materialize() 由 backend 提供;这里仅是 ABC 接口,子类化时覆盖
    """
    frame_id: int
    timestamp_ms: int
    mask_type: MaskType
    polygons: Optional[List[Polygon]] = None     # POLYGON backend 填
    mask_path: Optional[str] = None              # SAM/Segmentation backend 填(磁盘 lazy)
    source_instance_id: str = ""
    confidence: float = 1.0
    width: int = 0
    height: int = 0
    # backend 私有 — materialize 时使用的 dilation (x, y) 像素,backend 在 build 时填
    # 兼容旧字段名 dilation_px (int): 通过 @property 暴露为 (n, n)
    dilation_xy: Tuple[int, int] = (0, 0)

    @property
    def dilation_px(self) -> int:
        """向后兼容: 当 x == y 时返回 int, 否则报 NotImplementedError 提示调用方用 dilation_xy."""
        dx, dy = self.dilation_xy
        if dx == dy:
            return dx
        raise NotImplementedError(
            f"dilation_px requires x==y, got {(dx, dy)}; use .dilation_xy instead"
        )

    def materialize(self) -> np.ndarray:
        """物化为 HxW uint8 0/255 二值 mask。

        默认实现:polygon rasterize (cv2.fillPoly)。后端可继承覆盖,
        例如 SAMBackend 从 mask_path 读 .npy / .png。
        """
        dx, dy = self.dilation_xy
        if self.mask_type == MaskType.POLYGON:
            return _polygons_to_mask(self.polygons, self.width, self.height, (dx, dy))
        if self.mask_type == MaskType.BOUNDING_BOX:
            return _polygons_to_mask(self.polygons, self.width, self.height, (dx, dy))
        if self.mask_type in (MaskType.SAM, MaskType.SEGMENTATION):
            if not self.mask_path:
                raise ValueError(
                    f"MaskArtifact.mask_type={self.mask_type} requires mask_path"
                )
            import cv2
            m = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise IOError(f"failed to load mask from {self.mask_path}")
            if m.shape != (self.height, self.width):
                m = cv2.resize(m, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            return (m > 0).astype(np.uint8) * 255
        raise ValueError(f"unsupported mask_type: {self.mask_type}")


def _polygons_to_mask(polygons: Optional[List[Polygon]],
                      width: int, height: int,
                      dilation: DilationSpec = 0) -> np.ndarray:
    """把 1+ polygon rasterize 成 HxW uint8 0/255 二值 mask。

    流程:
      1. cv2.fillPoly 把每个 polygon 画成 255
      2. dilation (x, y) > 0 → 用非方形 kernel (2*dx+1, 2*dy+1) dilate
         (字幕下边沿阴影通常 > 横向间距, dy 应略大于 dx)
      3. morph close (dilation → erosion): 填中文字笔画的微小间隙,
         让 inpainting 后不会残留笔画碎片。kernel 大小用 max(dx, dy) 同 (2n+1, 2n+1)
    """
    import cv2
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygons:
        return mask
    for poly in polygons:
        if not poly or len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    dx, dy = _normalize_dilation(dilation)
    if dx > 0 or dy > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dx + 1, 2 * dy + 1))
        mask = cv2.dilate(mask, k, iterations=1)
        # morph close: dilate → erode, 填笔画间隙;仅改 bitmap, polygon 坐标不动
        close_k = max(dx, dy)
        if close_k > 0:
            kc = cv2.getStructuringElement(
                cv2.MORPH_RECT, (2 * close_k + 1, 2 * close_k + 1)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
    return mask
