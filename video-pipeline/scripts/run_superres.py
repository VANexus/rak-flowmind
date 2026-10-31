# -*- coding: utf-8 -*-
"""4K super-resolution for the clean video (HANDOFF V2 §5 step 4).

Pipeline: clean.mp4 -> frames(PNG) -> realesrgan-ncnn-vulkan x4 (Vulkan GPU)
        -> scale 2880x5120 down to 2160x3840 (standard 4K portrait, lanczos)
        -> h264 nvenc encode + original audio
Keeps the 720p master as <base>_clean720.mp4 and replaces
output/clean/<base>_clean.mp4 with the 4K version.

Usage: python run_superres.py <clean_mp4>
"""
import os
import shutil
import subprocess
import sys
import time

import cv2

FFMPEG = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'
WORK = r'D:\Desktop\obj114514\work'
RA_DIR = os.path.join(WORK, 'realesrgan')
RA_EXE = os.path.join(RA_DIR, 'realesrgan-ncnn-vulkan.exe')
TMP_FRAMES = os.path.join(WORK, 'sr_in')
SR_OUT = os.path.join(WORK, 'sr_out')
TARGET = (2160, 3840)


def run(cmd, **kw):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        raise SystemExit(f'command failed rc={r.returncode}')
    return r


def main():
    src = sys.argv[1]
    stem = os.path.splitext(os.path.basename(src))[0]
    master720 = os.path.join(os.path.dirname(src), stem + '720.mp4')

    if not os.path.exists(master720):
        shutil.copyfile(src, master720)
        print('kept 720p master ->', master720)

    # 1. extract frames
    shutil.rmtree(TMP_FRAMES, ignore_errors=True)
    os.makedirs(TMP_FRAMES)
    run([FFMPEG, '-y', '-v', 'error', '-i', master720,
         os.path.join(TMP_FRAMES, '%05d.png')])

    # 2. ncnn x4 upscaling on Vulkan GPU
    shutil.rmtree(SR_OUT, ignore_errors=True)
    os.makedirs(SR_OUT)
    t0 = time.time()
    run([RA_EXE, '-i', TMP_FRAMES, '-o', SR_OUT, '-n', 'realesrgan-x4plus',
         '-s', '4', '-g', '1', '-f', 'png'], cwd=RA_DIR)
    print('upscale done in %.0fs' % (time.time() - t0), flush=True)

    # 3. scale to standard 4K portrait + encode + mux audio
    tmp4k = os.path.join(WORK, 'clean_4k_tmp.mp4')
    vf = f"scale={TARGET[0]}:{TARGET[1]}:flags=lanczos"
    nvenc_ok = subprocess.run(
        [FFMPEG, '-y', '-v', 'error', '-f', 'lavfi',
         '-i', 'nullsrc=s=256x256:d=0.1', '-c:v', 'hevc_nvenc', '-f', 'null', '-'],
        capture_output=True).returncode == 0
    enc = (['-c:v', 'hevc_nvenc', '-preset', 'p5', '-cq', '21', '-rc', 'vbr']
           if nvenc_ok else ['-c:v', 'libx264', '-crf', '18', '-preset', 'medium'])
    print('encoder:', 'nvenc' if nvenc_ok else 'libx264')
    run([FFMPEG, '-y', '-v', 'error', '-framerate', '30',
         '-i', os.path.join(SR_OUT, '%05d.png'),
         '-vf', vf, '-pix_fmt', 'yuv420p', *enc, tmp4k])
    run([FFMPEG, '-y', '-v', 'error', '-i', tmp4k, '-i', master720,
         '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'copy',
         '-shortest', src])
    os.remove(tmp4k)
    shutil.rmtree(TMP_FRAMES, ignore_errors=True)
    shutil.rmtree(SR_OUT, ignore_errors=True)
    print('4K CLEAN ->', src, flush=True)


if __name__ == '__main__':
    main()
