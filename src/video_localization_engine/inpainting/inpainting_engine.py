"""L6 门面: InpaintingEngine。

不直接调 cv2.inpaint, 只调度 backend。

可选字幕复检 (字幕擦不干净 → 扩展 mask → 重 inpaint, 最多 N 轮):
  - 启用: 构造时传 ocr_checker (任意 callable(image, mask) -> list[BBox])
  - 启用: 调用 inpaint_frame(..., retest_ocr=True)
  - 轮数 / 扩展 dilation 比例可调
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from video_localization_engine.inpainting.base import InpaintingBackend
from video_localization_engine.inpainting.registry import InpaintingBackendRegistry
from video_localization_engine.mask.artifact import MaskArtifact
from video_localization_engine.types.video import FramePacket


_log = logging.getLogger(__name__)


BBox = Tuple[int, int, int, int]   # (x, y, w, h)
OcrChecker = Callable[[np.ndarray, np.ndarray], Sequence[BBox]]


class InpaintingEngine:
    """L6 入口: 输入一帧 + mask, 返回 inpaint 后图像。

    用法:
        engine = InpaintingEngine(backend_name="opencv", algorithm="telea", radius=3)
        out = engine.inpaint_frame(packet, mask_artifact)
        # 字幕复检:
        engine = InpaintingEngine(..., ocr_checker=ocr_fn)
        out = engine.inpaint_frame(packet, mask_artifact, retest_ocr=True)
    """

    def __init__(
        self,
        backend_name: str = "opencv",
        ocr_checker: Optional[OcrChecker] = None,
        retest_max_rounds: int = 5,
        retest_extra_dilation_ratio: float = 1.5,
        **backend_kwargs,
    ):
        backend_cls = InpaintingBackendRegistry.get(backend_name)
        self.backend: InpaintingBackend = backend_cls(**backend_kwargs)
        self.backend_name = backend_name
        # 字幕复检 — 通过外部 callable 注入 OCR, 避免 inpainting ↔ detector 循环依赖
        self.ocr_checker = ocr_checker
        self.retest_max_rounds = max(1, retest_max_rounds)
        self.retest_extra_dilation_ratio = float(retest_extra_dilation_ratio)

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------
    def inpaint_frame(self, packet: FramePacket, mask_artifact: MaskArtifact,
                      *, retest_ocr: bool = False) -> np.ndarray:
        """inpaint 一帧. retest_ocr=True 时先一次过, 再用 OCR 复检残余字幕.

        复检机制:
          - 仅在 OCR 检测到非空文字 → 扩展 mask → 重 inpaint
          - 轮数 ≤ retest_max_rounds (防死循环)
          - 即使 3 轮后仍有残余, 也不抛 — 仅记 warning
        """
        mask = mask_artifact.materialize()
        if not retest_ocr or self.ocr_checker is None:
            return self.backend.inpaint(packet, mask)

        image = self.backend.inpaint(packet, mask)
        height, width = mask.shape[:2]
        # 复检 ROI = mask 覆盖矩形 + padding 5% — 避免漏掉残影刚溢出 mask 的像素
        pad_x = max(8, int(width * 0.05))
        pad_y = max(8, int(height * 0.05))
        ys, xs = np.where(mask > 0)
        if ys.size == 0 or xs.size == 0:
            return image
        x0 = max(0, int(xs.min()) - pad_x)
        y0 = max(0, int(ys.min()) - pad_y)
        x1 = min(width, int(xs.max()) + pad_x)
        y1 = min(height, int(ys.max()) + pad_y)

        current_mask = mask.copy()
        for round_idx in range(1, self.retest_max_rounds + 1):
            crop_img = image[y0:y1, x0:x1]
            crop_mask = current_mask[y0:y1, x0:x1]
            try:
                residual_bboxes = self.ocr_checker(crop_img, crop_mask)
            except Exception as e:  # OCR 失败不应让 L6 崩
                _log.warning("ocr residual check failed (round=%d): %s", round_idx, e)
                return image
            if not residual_bboxes:
                break
            # 把 ROI 坐标还原到全局, 并扩 50% dilation
            extra = self._expand_bboxes(residual_bboxes, x0, y0,
                                        extra_ratio=self.retest_extra_dilation_ratio)
            current_mask = self._union_mask(current_mask, extra, height, width)
            # 同步 ROI: 取所有残余 bbox 的并集
            ys, xs = np.where(current_mask > 0)
            x0 = max(0, int(xs.min()) - pad_x)
            y0 = max(0, int(ys.min()) - pad_y)
            x1 = min(width, int(xs.max()) + pad_x)
            y1 = min(height, int(ys.max()) + pad_y)
            image = self.backend.inpaint(packet, current_mask)
        else:
            # 走到 for-else 表示最后一轮仍残留 — 不崩
            _log.warning(
                "inpaint residual check: %d rounds exhausted, possible residual text",
                self.retest_max_rounds,
            )
        return image

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _expand_bboxes(bboxes: Sequence[BBox], offset_x: int, offset_y: int,
                       extra_ratio: float) -> List[BBox]:
        """bboxes 坐标从 ROI 偏移到全局, 每边再扩 extra_ratio * max(w,h)"""
        out: List[BBox] = []
        for (bx, by, bw, bh) in bboxes:
            bx += offset_x
            by += offset_y
            extra = int(max(bw, bh) * extra_ratio)
            out.append((bx - extra, by - extra, bw + 2 * extra, bh + 2 * extra))
        return out

    @staticmethod
    def _union_mask(base: np.ndarray, bboxes: Sequence[BBox],
                    height: int, width: int) -> np.ndarray:
        """把 bboxes 画进 base mask (uint8 0/255), 返回新 mask"""
        union = base.copy()
        if not bboxes:
            return union
        for (x, y, w, h) in bboxes:
            if w <= 0 or h <= 0:
                continue
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(width, x + w)
            y1 = min(height, y + h)
            if x1 > x0 and y1 > y0:
                union[y0:y1, x0:x1] = 255
        return union
