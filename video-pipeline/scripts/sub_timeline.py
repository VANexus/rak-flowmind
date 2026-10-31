# -*- coding: utf-8 -*-
"""Detect actual burned-subtitle display intervals by measuring subtitle-like
text coverage per frame inside the known band (same heuristic as gen_masks)."""
import cv2
import numpy as np
import json

VIDEO = r'D:\Desktop\obj114514\ad738c4bcfdb9f0d41eeb5ede9a06a2e_raw.mp4'
OUT = r'D:\Desktop\obj114514\work\asr\sub_intervals.json'
BAND_Y0, BAND_Y1 = 930, 1150

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
near_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
areas = []
idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    roi = frame[BAND_Y0:BAND_Y1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    white = ((gray >= 165) & (S <= 100)).astype(np.uint8)
    dark = (gray <= 85).astype(np.uint8)
    core = (white * cv2.dilate(dark, near_k)) > 0
    core = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # count text-ish pixels (core only, no watermark rect, no dilation)
    areas.append(float(core.sum()))
    idx += 1
cap.release()

areas = np.array(areas)
thr = max(120.0, 0.15 * np.percentile(areas, 95))
on = areas > thr

intervals = []
start = None
for i, v in enumerate(on):
    if v and start is None:
        start = i
    elif not v and start is not None:
        if i - start >= 8:  # ignore <0.27s flicker
            intervals.append((round(start / fps, 3), round(i / fps, 3)))
        start = None
if start is not None:
    intervals.append((round(start / fps, 3), round(len(on) / fps, 3)))

print(f'frames={idx} fps={fps} thr={thr:.0f} -> {len(intervals)} display intervals')
for a, b in intervals:
    print(f'  {a:7.2f} -> {b:7.2f}  ({b-a:.2f}s)')

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump({'fps': fps, 'threshold': float(thr),
               'intervals': [[a, b] for a, b in intervals]}, f, indent=2)
print('saved', OUT)
