# -*- coding: utf-8 -*-
"""Chunked ProPainter run for long videos (fits 12GB VRAM / 32GB RAM).

- splits the video into overlapping chunks (CHUNK frames, OVERLAP overlap)
- runs inference_propainter.py per chunk in a fresh subprocess
  (--save_frames writes PNGs; --mask_dilation 0 because masks are pre-dilated)
- assembles the results (each chunk skips its overlap so stitched frames come
  from the chunk with the most future context), encodes
  output/clean/<base>_clean.mp4 with the original audio.

Usage: python run_pp_chunked.py <video>
"""
import os
import shutil
import subprocess
import sys
import time

import cv2

FFMPEG = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'
WORK = r'D:\Desktop\obj114514\work'
ROOT = r'D:\Desktop\obj114514'
PP_DIR = os.path.join(WORK, 'ProPainter')
PYTHON = r'D:\conda_envs\pytorch_d\python.exe'

CHUNK = 120          # frames per chunk (full resolution; host-RAM constrained)
OVERLAP = 20         # frames of shared context between neighboring chunks
STEP = CHUNK - OVERLAP
MASKS = os.path.join(WORK, 'masks_ocr')   # OCR stroke-level masks (HANDOFF V2 §5)
FPS = 30
RESIZE = 1.0         # FULL resolution (no upscale softness)


def plan_chunks(total):
    starts = list(range(0, max(total - CHUNK, 0) + 1, STEP))
    last = total - CHUNK
    if last > 0 and (not starts or last > starts[-1]):
        starts.append(last)
    return starts


def main():
    src = sys.argv[1]
    stem = os.path.splitext(os.path.basename(src))[0]
    base = stem[:-4] if stem.endswith('_raw') else stem
    clean_out = os.path.join(ROOT, 'output', 'clean', base + '_clean.mp4')
    os.makedirs(os.path.dirname(clean_out), exist_ok=True)

    cap = cv2.VideoCapture(src)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    starts = plan_chunks(total)
    print('total frames', total)
    print('chunks:', [(s, min(s + CHUNK, total)) for s in starts], flush=True)

    input_root = os.path.join(WORK, 'pp_input')
    mask_root = os.path.join(WORK, 'pp_masks')
    results_root = os.path.join(WORK, 'results_chunks')

    for k, s in enumerate(starts):
        e = min(s + CHUNK, total)
        tag = 'ch%03d' % k
        done_marker = os.path.join(results_root, tag, 'frames',
                                   '%04d.png' % (e - s - 1))
        if os.path.exists(done_marker):
            print(f'chunk {k}: cached, skip', flush=True)
            continue

        input_dir = os.path.join(input_root, tag)
        mask_dir = os.path.join(mask_root, tag)
        for d in (input_dir, mask_dir):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(input_dir)
        os.makedirs(mask_dir)

        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        idx = 0
        while idx < e - s:
            ok, img = cap.read()
            if not ok:
                break
            cv2.imwrite(os.path.join(input_dir, '%05d.png' % idx), img)
            shutil.copyfile(os.path.join(MASKS, '%05d.png' % (s + idx)),
                            os.path.join(mask_dir, '%05d.png' % idx))
            idx += 1
        if idx != e - s:
            raise SystemExit(f'only read {idx} frames for chunk {k}')

        t0 = time.time()
        env = dict(os.environ)
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        r = subprocess.run(
            [PYTHON, 'inference_propainter.py',
             '-i', input_dir, '-m', mask_dir, '-o', results_root,
             '--save_frames', '--mask_dilation', '4', '--save_fps', str(FPS),
             '--fp16', '--resize_ratio', str(RESIZE),
             '--neighbor_length', '8', '--subvideo_length', '40',
             '--raft_iter', '20'],
            cwd=PP_DIR, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            print(r.stderr[-6000:])
            raise SystemExit(f'chunk {k} FAILED')
        if not os.path.exists(done_marker):
            raise SystemExit(f'chunk {k}: output missing {done_marker}')
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(mask_dir, ignore_errors=True)
        print(f'chunk {k}: frames {s}-{e} done in {time.time()-t0:.0f}s', flush=True)

    cap.release()

    # ---- assemble, skipping overlaps ----
    asm_dir = os.path.join(WORK, 'assembled')
    shutil.rmtree(asm_dir, ignore_errors=True)
    os.makedirs(asm_dir)
    out_i = 0
    prev_end = 0
    for k, s in enumerate(starts):
        e = min(s + CHUNK, total)
        fdir = os.path.join(results_root, 'ch%03d' % k, 'frames')
        skip = max(0, prev_end - s)
        for i in range(skip, e - s):
            src_png = os.path.join(fdir, '%04d.png' % i)
            dst_png = os.path.join(asm_dir, '%05d.png' % out_i)
            img = cv2.imread(src_png)
            if img is None:
                raise SystemExit('missing frame ' + src_png)
            if img.shape[1] != ow or img.shape[0] != oh:
                img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(dst_png, img)
            out_i += 1
        prev_end = e
        print(f'assemble chunk {k}: skipped {skip} -> total {out_i}', flush=True)
    if out_i != total:
        raise SystemExit(f'assembled {out_i} != {total}')

    # ---- encode clean video with original audio ----
    tmp_mp4 = os.path.join(WORK, 'clean_video_noaudio.mp4')
    r = subprocess.run([FFMPEG, '-y', '-v', 'error',
                        '-framerate', str(FPS),
                        '-i', os.path.join(asm_dir, '%05d.png'),
                        '-c:v', 'libx264', '-crf', '17', '-preset', 'medium',
                        '-pix_fmt', 'yuv420p', tmp_mp4])
    if r.returncode != 0:
        raise SystemExit('encode failed')
    r = subprocess.run([FFMPEG, '-y', '-v', 'error',
                        '-i', tmp_mp4, '-i', src,
                        '-map', '0:v:0', '-map', '1:a:0',
                        '-c:v', 'copy', '-c:a', 'copy', '-shortest', clean_out])
    if r.returncode != 0:
        raise SystemExit('mux failed')
    print('CLEAN ->', clean_out, flush=True)


if __name__ == '__main__':
    main()
