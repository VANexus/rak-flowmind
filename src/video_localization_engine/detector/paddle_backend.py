"""PaddleOCR Backend — L3 TextDetector 实现。

不判断字幕, 不生成 instance, 只输出 TextCandidate。

设计:
- 支持 lang = "ch", "en", "chinese_cht", 多语言
- 预热避免首次 30s 延迟
- polygon 输出 4 点 ([TL, TR, BR, BL])
- 输出 OCR 置信度
- (P8) 调低 det_db_thresh / det_db_box_thresh 让候选 bbox 更敏感, 不漏小字
- (P8) merge 间距 < merge_max_gap_px 的相邻 bbox, 减少小字间隙空隙
"""
from __future__ import annotations

import logging
import os
from typing import List, Tuple

import cv2
import numpy as np

os.environ.setdefault("FLAGS_log_level", "3")
logging.getLogger("ppocr").setLevel(logging.ERROR)

from paddleocr import PaddleOCR

from video_localization_engine.detector.base import TextDetector
from video_localization_engine.types.detection import TextCandidate
from video_localization_engine.types.video import FramePacket


def _bbox_iou_close(a: Tuple[int, int, int, int],
                    b: Tuple[int, int, int, int],
                    max_gap: int) -> bool:
    """两个 bbox 若水平/垂直距离 ≤ max_gap 视为相邻, 应当 merge。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    # 任一方向上 (垂直/水平) 的最小间隔
    h_gap = max(0, max(ax1, bx1) - min(ax2, bx2))  # 水平方向不重叠的距离
    v_gap = max(0, max(ay1, by1) - min(ay2, by2))  # 垂直方向不重叠的距离
    return h_gap <= max_gap and v_gap <= max_gap


def _union_close_bboxes(bboxes: List[Tuple[int, int, int, int]],
                        max_gap: int) -> List[Tuple[int, int, int, int]]:
    """两两 union 间距 ≤ max_gap 的 bbox, 返回合并后的 bbox 列表。"""
    if not bboxes:
        return []
    merged = True
    boxes = list(bboxes)
    while merged:
        merged = False
        new_boxes: List[Tuple[int, int, int, int]] = []
        used = [False] * len(boxes)
        for i, a in enumerate(boxes):
            if used[i]:
                continue
            cur = a
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                b = boxes[j]
                if _bbox_iou_close(cur, b, max_gap):
                    cur = (min(cur[0], b[0]), min(cur[1], b[1]),
                           max(cur[2], b[2]), max(cur[3], b[3]))
                    used[j] = True
                    merged = True
            used[i] = True
            new_boxes.append(cur)
        boxes = new_boxes
    return boxes


class PaddleOCRDetector(TextDetector):
    """PaddleOCR 后端 — L3 TextDetector 默认实现。

    (P8) 新增参数:
      det_db_thresh: 文字概率阈值 (默认 0.2, 原 0.3, 越低越敏感)
      det_db_box_thresh: bbox 阈值 (默认 0.3, 原 0.5, 越低越能保留低置信 bbox)
      merge_close_bboxes: 是否 union 间距 ≤ merge_max_gap_px 的相邻 bbox
      merge_max_gap_px: 相邻 bbox 合并的最大间距 (默认 30px)
    """

    def __init__(
        self,
        lang: str = "ch",
        conf_thresh: float = 0.3,
        use_angle_cls: bool = False,
        det_db_thresh: float = 0.2,
        det_db_box_thresh: float = 0.3,
        merge_close_bboxes: bool = True,
        merge_max_gap_px: int = 30,
    ):
        self._lang = lang
        self._conf_thresh = conf_thresh
        self._use_angle_cls = use_angle_cls
        self._det_db_thresh = float(det_db_thresh)
        self._det_db_box_thresh = float(det_db_box_thresh)
        self._merge_close = bool(merge_close_bboxes)
        self._merge_gap = max(1, int(merge_max_gap_px))
        self._ocr = None

    @property
    def name(self) -> str:
        return f"paddleocr_{self._lang}"

    @property
    def supported_languages(self) -> List[str]:
        return ["ch", "en", "chinese_cht", "fr", "german", "korean", "japan"]

    def warmup(self) -> None:
        if self._ocr is None:
            self._ocr = PaddleOCR(
                use_angle_cls=self._use_angle_cls,
                lang=self._lang,
                show_log=False,
                det_db_thresh=self._det_db_thresh,
                det_db_box_thresh=self._det_db_box_thresh,
            )

    def detect(self, packet: FramePacket) -> List[TextCandidate]:
        if self._ocr is None:
            self.warmup()

        results = self._ocr.ocr(packet.image, cls=self._use_angle_cls)
        # results 格式: [[ [poly, (text, conf)], ... ], ...] 或 [] for empty
        if not results or not results[0]:
            return []

        # 第一遍: 收集候选 (polygon + text + conf)
        candidates: List[TextCandidate] = []
        bboxes_xyxy: List[Tuple[int, int, int, int]] = []
        for line in results[0]:
            if not line:
                continue
            poly, (text, conf) = line
            if conf < self._conf_thresh:
                continue
            if not text or not text.strip():
                continue

            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            xyxy = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            poly_tuples = [(float(p[0]), float(p[1])) for p in poly]
            candidates.append(TextCandidate(
                polygon=poly_tuples,
                text=text.strip(),
                confidence=float(conf),
                language=self._lang,
                detector_id=self.name,
            ))
            bboxes_xyxy.append(xyxy)

        # (P8) Merge 间距 ≤ merge_gap 的相邻 bbox, 把合并后 bbox 作为 mask 候选的扩张基线
        if self._merge_close and candidates:
            merged_boxes = _union_close_bboxes(bboxes_xyxy, self._merge_gap)
            # 把每个 candidate 的 polygon 替换成它所在 group 的 union bbox
            # 简化: 若某 candidate 的 bbox 被并入比它大的 box, 用 union 替换 polygon
            new_polys: List[List[Tuple[float, float]]] = []
            for cand, orig in zip(candidates, bboxes_xyxy):
                # 找覆盖 orig 的最大 merged box
                best = None
                best_area = -1
                for m in merged_boxes:
                    ox1, oy1, ox2, oy2 = orig
                    mx1, my1, mx2, my2 = m
                    # orig 是否在 m 内 (允许小幅外溢, 即 union)
                    contains = (mx1 <= ox1 and my1 <= oy1
                                and mx2 >= ox2 and my2 >= oy2)
                    if contains:
                        area = (mx2 - mx1) * (my2 - my1)
                        if area > best_area:
                            best_area = area
                            best = m
                if best is not None:
                    bx1, by1, bx2, by2 = best
                    new_polys.append([
                        (float(bx1), float(by1)),
                        (float(bx2), float(by1)),
                        (float(bx2), float(by2)),
                        (float(bx1), float(by2)),
                    ])
                else:
                    new_polys.append(cand.polygon)
            # 写回
            for cand, np_ in zip(candidates, new_polys):
                cand.polygon = np_

        return candidates