# -*- coding: utf-8 -*-
"""OCR-guided stroke-level masks with temporal triggering (fast).

- RapidOCR (~4-5s/frame on CPU) is only invoked every BASELINE frames AND
  whenever the per-frame cheap "band stroke map" changes drastically
  (XOR ratio vs previous frame) - i.e. at subtitle switch points.
- All other frames reuse the latest accepted boxes and just recompute the
  cheap pixel-level strokes inside them.
Outputs: masks_ocr/00000.png ... + QA overlays in mask_ocr_previews/.
"""
import os

import cv2
import numpy as np

VIDEO = r'D:\Desktop\obj114514\ad738c4bcfdb9f0d41eeb5ede9a06a2e_raw.mp4'
OUT_DIR = r'D:\Desktop\obj114514\work\masks_ocr'
PREV_DIR = r'D:\Desktop\obj114514\work\mask_ocr_previews'

BAND_Y0, BAND_Y1 = 930, 1150
WM = (32, 27, 228, 66)
MIN_BOX_H, MIN_BOX_W = 26, 40
DILATE_FINAL = 3
WHITE_V, WHITE_S, DARK_V = 165, 80, 85

BASELINE = 6            # OCR at least every N frames
XOR_TRIG = 0.30         # stroke-map change ratio that forces an OCR call
DET_SCALE = 0.75

NEAR_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
OUT_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
DIL_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE_FINAL + 1,) * 2)
OPEN3 = np.ones((3, 3), np.uint8)
CLOSE5 = np.ones((5, 5), np.uint8)


def band_stroke_map(roi):
    """Cheap full-band stroke map used only as a change trigger."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    white = ((gray >= WHITE_V) & (S <= WHITE_S)).astype(np.uint8)
    dark = (gray <= DARK_V).astype(np.uint8)
    core = (white * cv2.dilate(dark, NEAR_K)) > 0
    core = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_OPEN, OPEN3)
    outline = (dark * cv2.dilate(core, OUT_K)) > 0
    return (core | outline).astype(np.uint8)


def stroke_mask_in_box(roi):
    return cv2.morphologyEx(
        band_stroke_map(roi), cv2.MORPH_CLOSE, CLOSE5)


def xor_ratio(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return 1.0 - inter / union


def main():
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PREV_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('frames', total, flush=True)

    boxes = []              # latest accepted OCR boxes
    prev_strokes = None
    n_ocr = 0
    prev_idx = {0, 45, 300, 700, 1350, 1450, 2000, total - 1}
    areas = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        band = frame[BAND_Y0:BAND_Y1]
        cur_strokes = band_stroke_map(band)

        need_ocr = (prev_strokes is None or idx % BASELINE == 0
                    or xor_ratio(cur_strokes, prev_strokes) > XOR_TRIG)
        if need_ocr:
            small = cv2.resize(frame, None, fx=DET_SCALE, fy=DET_SCALE,
                               interpolation=cv2.INTER_AREA)
            out = engine(small)
            res = out[0]
            boxes = []
            for box in (res or []):
                pts = np.array(box[0], dtype=np.float32) / DET_SCALE
                x0, y0 = pts[:, 0].min(), pts[:, 1].min()
                x1, y1 = pts[:, 0].max(), pts[:, 1].max()
                in_band = (y0 >= BAND_Y0 - 10) and (y1 <= BAND_Y1 + 10)
                big = (y1 - y0) >= MIN_BOX_H and (x1 - x0) >= MIN_BOX_W
                if in_band and big:
                    boxes.append((x0, y0, x1, y1))
                elif (x1 > WM[0] and x0 < WM[0] + WM[2]
                      and y1 > WM[1] and y0 < WM[1] + WM[3]):
                    boxes.append((x0, y0, x1, y1))
            n_ocr += 1
        prev_strokes = cur_strokes

        mask = np.zeros((h, w), np.uint8)
        for (x0, y0, x1, y1) in boxes:
            px0, py0 = max(int(x0) - 5, 0), max(int(y0) - 5, 0)
            px1, py1 = min(int(x1) + 5, w), min(int(y1) + 5, h)
            strokes = stroke_mask_in_box(frame[py0:py1, px0:px1])
            if strokes.sum() < 0.01 * strokes.size:
                continue
            mask[py0:py1, px0:px1] |= strokes * 255

        wx, wy, ww, wh = WM
        mask[wy:wy + wh, wx:wx + ww] = 255
        mask = cv2.dilate(mask, DIL_K)

        cv2.imwrite(os.path.join(OUT_DIR, '%05d.png' % idx), mask)
        areas.append(float((mask > 0).sum()) / (h * w))
        if idx in prev_idx:
            prev = frame.copy()
            m = mask > 0
            prev[m] = (0.4 * np.array([0, 0, 255]) + 0.6 * frame[m]).astype(np.uint8)
            cv2.putText(prev, 'idx=%d boxes=%d ocr=%d area=%.2f%%'
                        % (idx, len(boxes), n_ocr, areas[-1] * 100),
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(PREV_DIR, 'prev_%05d.png' % idx), prev)
        idx += 1
        if idx % 250 == 0:
            print('%d/%d ocr_calls=%d mean_area=%.2f%%'
                  % (idx, total, n_ocr, 100 * np.mean(areas)), flush=True)
    cap.release()
    print('done %d masks, ocr_calls=%d, mean=%.2f%% max=%.2f%%'
          % (idx, n_ocr, 100 * np.mean(areas), 100 * np.max(areas)))


if __name__ == '__main__':
    main()
