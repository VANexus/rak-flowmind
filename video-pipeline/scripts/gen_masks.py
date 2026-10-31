# -*- coding: utf-8 -*-
"""Generate per-frame masks for burned-in subtitle band + fixed top-left watermark.

Output: <out_dir>/00000.png ... (grayscale, 255 = inpaint)
Also writes a few overlay previews for visual QA.
"""
import os
import sys
import cv2
import numpy as np

VIDEO = sys.argv[1] if len(sys.argv) > 1 else r'D:\Desktop\obj114514\ad738c4bcfdb9f0d41eeb5ede9a06a2e_raw.mp4'
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else r'D:\Desktop\obj114514\work\masks'
PREV_DIR = os.path.join(OUT_DIR, '_previews')

# subtitle band (720x1280 frame): 1-2 lines observed at y ~960..1110
BAND_Y0, BAND_Y1 = 930, 1150
# fixed watermark bbox (x, y, w, h) measured from frames, with margin
WM = (32, 27, 228, 66)

WHITE_V = 165      # min value for subtitle fill
WHITE_S = 100      # max saturation for subtitle fill (exclude colored brights)
DARK_V = 85        # max value for subtitle outline
NEAR_K = 11        # dilation kernel for "white pixel has dark outline nearby"
OUTLINE_K = 13     # dilation kernel to pick up the black outline
DILATE_K = 7       # final dilation before saving (ProPainter dilates more on its own)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PREV_DIR, exist_ok=True)
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit('cannot open video')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'frames={total} fps={fps}')

    near_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (NEAR_K, NEAR_K))
    out_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OUTLINE_K, OUTLINE_K))
    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_K, DILATE_K))
    open_k = np.ones((3, 3), np.uint8)
    close_k = np.ones((5, 5), np.uint8)

    prev_idx = {0, 1, 220, 450, 700, 1000, 1400, 1800, 2200, total - 1}
    idx = 0
    areas = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), np.uint8)

        roi = frame[BAND_Y0:BAND_Y1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        S, V = hsv[:, :, 1], hsv[:, :, 2]

        white = ((gray >= WHITE_V) & (S <= WHITE_S)).astype(np.uint8)
        dark = (gray <= DARK_V).astype(np.uint8)

        # white fill that has a dark outline within ~5px (kills big flat white areas)
        core = (white * cv2.dilate(dark, near_k)) > 0
        core = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_OPEN, open_k)
        # black outline hugging that fill
        outline = (dark * cv2.dilate(core, out_k)) > 0
        text = (core | outline).astype(np.uint8)
        text = cv2.morphologyEx(text, cv2.MORPH_CLOSE, close_k)
        text = cv2.dilate(text, dil_k)

        mask[BAND_Y0:BAND_Y1] = text * 255
        wx, wy, ww, wh = WM
        mask[wy:wy + wh, wx:wx + ww] = 255

        cv2.imwrite(os.path.join(OUT_DIR, f'{idx:05d}.png'), mask)
        area = float((mask > 0).sum()) / (h * w)
        areas.append(area)

        if idx in prev_idx:
            prev = frame.copy()
            m = mask > 0
            overlay = prev.copy()
            overlay[m] = (0.4 * np.array([0, 0, 255]) + 0.6 * prev[m]).astype(np.uint8)
            cv2.putText(overlay, f'idx={idx} area={area*100:.2f}%', (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(PREV_DIR, f'prev_{idx:05d}.png'), overlay)
        idx += 1
        if idx % 500 == 0:
            print(f'{idx}/{total}')

    cap.release()
    print(f'done: {idx} masks, mean area {100*np.mean(areas):.2f}% '
          f'min {100*np.min(areas):.2f}% max {100*np.max(areas):.2f}%')


if __name__ == '__main__':
    main()
