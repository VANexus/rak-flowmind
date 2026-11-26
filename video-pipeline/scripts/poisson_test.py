# -*- coding: utf-8 -*-
"""Poisson-blend experiment (HANDOFF V2 §5 step 3).

For sample frames of a finished ProPainter chunk, compare:
  left  = ProPainter output as-is
  right = cv2.seamlessClone of the inpainted content back into the ORIGINAL
          frame under the OCR mask (gradient-domain blending, seam removal)

Usage: python poisson_test.py [chunk_tag ...]     (default: ch000 ch005 ch010)
"""
import os
import sys

import cv2
import numpy as np

WORK = r'D:\Desktop\obj114514\work'
SRC_VIDEO = r'D:\Desktop\obj114514\ad738c4bcfdb9f0d41eeb5ede9a06a2e_raw.mp4'
MASKS = os.path.join(WORK, 'masks_ocr')
OUT = os.path.join(WORK, 'poisson_test')
CHUNK, OVERLAP = 140, 20


def plan_starts(total):
    step = CHUNK - OVERLAP
    starts = list(range(0, max(total - CHUNK, 0) + 1, step))
    last = total - CHUNK
    if last > 0 and (not starts or last > starts[-1]):
        starts.append(last)
    return starts


def main():
    tags = sys.argv[1:] or ['ch000', 'ch005', 'ch010', 'ch015']
    os.makedirs(OUT, exist_ok=True)
    cap = cv2.VideoCapture(SRC_VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    starts = plan_starts(total)

    for tag in tags:
        k = int(tag[2:])
        s = starts[k]
        fdir = os.path.join(WORK, 'results_chunks', tag, 'frames')
        if not os.path.isdir(fdir):
            print(tag, 'not finished yet, skip')
            continue
        names = sorted(os.listdir(fdir))
        for name in (names[0], names[len(names) // 2], names[-1]):
            i = int(name.split('.')[0])
            g = s + i
            pp = cv2.imread(os.path.join(fdir, name))
            cap.set(cv2.CAP_PROP_POS_FRAMES, g)
            ok, orig = cap.read()
            m = cv2.imread(os.path.join(MASKS, '%05d.png' % g), 0)
            if not (ok and pp is not None and m is not None):
                continue
            mask = (m > 0).astype(np.uint8) * 255
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            # seamlessClone needs the mask strictly inside the image with
            # margin; give it a bbox crop around the mask
            pad = 12
            x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1] - 1)
            y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0] - 1)
            src_crop = pp[y0:y1, x0:x1]
            dst_crop = orig[y0:y1, x0:x1]
            mask_crop = mask[y0:y1, x0:x1]
            cx, cy = (x0 + x1) // 2 - x0, (y0 + y1) // 2 - y0
            try:
                blended = cv2.seamlessClone(
                    src_crop, dst_crop, mask_crop, (cx, cy), cv2.NORMAL_CLONE)
                out = orig.copy()
                out[y0:y1, x0:x1] = blended
            except cv2.error as e:
                print('clone failed', tag, name, e)
                out = pp
            canvas = np.concatenate([pp, out], axis=1)
            cv2.putText(canvas, '%s f%d (L=PP R=poisson)' % (tag, g),
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(OUT, '%s_%05d.png' % (tag, g)), canvas)
            print('saved', tag, g)
    cap.release()


if __name__ == '__main__':
    main()
