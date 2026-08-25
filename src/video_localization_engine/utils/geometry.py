"""几何工具: polygon IoU, bbox 距离, 点包含等。"""
from __future__ import annotations

from typing import List, Tuple

Point = Tuple[float, float]
Polygon = List[Point]
BBox = Tuple[float, float, float, float]


def polygon_iou(p1: Polygon, p2: Polygon) -> float:
    """两 polygon 像素级 IoU (用 cv2 contour)。"""
    import cv2
    import numpy as np
    a = np.array(p1, dtype=np.int32)
    b = np.array(p2, dtype=np.int32)
    # 计算 union bbox
    xs = [min(min(pt[0] for pt in p1), min(pt[0] for pt in p2)),
          max(max(pt[0] for pt in p1), max(pt[0] for pt in p2))]
    ys = [min(min(pt[1] for pt in p1), min(pt[1] for pt in p2)),
          max(max(pt[1] for pt in p1), max(pt[1] for pt in p2))]
    w = int(xs[1] - xs[0]) + 2
    h = int(ys[1] - ys[0]) + 2
    if w <= 0 or h <= 0:
        return 0.0
    ca = np.zeros((h, w), dtype=np.uint8)
    cb = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(ca, [a - np.array([xs[0] - 1, ys[0] - 1])], 1)
    cv2.fillPoly(cb, [b - np.array([xs[0] - 1, ys[0] - 1])], 1)
    inter = int((ca & cb).sum())
    union = int((ca | cb).sum())
    return inter / union if union else 0.0


def bbox_iou(b1: BBox, b2: BBox) -> float:
    """两轴对齐 bbox 的 IoU。"""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union else 0.0


def polygon_centroid(p: Polygon) -> Point:
    xs = [pt[0] for pt in p]
    ys = [pt[1] for pt in p]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def bbox_from_polygon(p: Polygon) -> BBox:
    xs = [pt[0] for pt in p]
    ys = [pt[1] for pt in p]
    return (min(xs), min(ys), max(xs), max(ys))


def point_in_bbox(pt: Point, bbox: BBox, margin: float = 0.0) -> bool:
    return (
        bbox[0] - margin <= pt[0] <= bbox[2] + margin
        and bbox[1] - margin <= pt[1] <= bbox[3] + margin
    )


def polygon_in_bbox(poly: Polygon, bbox: BBox, iou_thresh: float = 0.5) -> bool:
    """polygon 重心落在 bbox 内 (简化判断, 不做 IoU)。"""
    cx, cy = polygon_centroid(poly)
    return point_in_bbox((cx, cy), bbox)