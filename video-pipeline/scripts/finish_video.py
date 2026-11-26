# -*- coding: utf-8 -*-
"""Assemble final deliverables for one video:
  A) output/clean/<base>_clean.mp4   (ProPainter result + original audio)
  B) output/en/<base>_en.mp4 + <base>.en.srt  (burned English subs + EN VO mix)

Usage: python finish_video.py <original_video>
Requires: work/results/<stem>/inpaint_out.mp4 and work/en/mix.wav + en.srt
"""
import os
import shutil
import subprocess
import sys

FFMPEG = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'
FFPROBE = r'C:\Program Files\ffmpeg\bin\ffprobe.exe'
WORK = r'D:\Desktop\obj114514\work'
ROOT = r'D:\Desktop\obj114514'

# ASS style tuned for 720x1280: white text, thin black outline, sits in the
# (now cleaned) subtitle band around y=960..1060.
STYLE = ("FontName=Arial,FontSize=12,PrimaryColour=&H00FFFFFF&,"
         "OutlineColour=&H00000000&,BorderStyle=1,Outline=1,Shadow=0,"
         "Alignment=2,MarginV=44,Spacing=0,Bold=0")


def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print('CMD FAIL:', ' '.join(cmd))
        print(r.stdout[-2000:])
        print(r.stderr[-4000:])
        raise SystemExit(1)
    return r


def ffprobe_json(path):
    out = subprocess.check_output(
        [FFPROBE, '-v', 'error', '-show_streams', '-show_format',
         '-of', 'json', path], text=True)
    import json
    return json.loads(out)


def main():
    src = sys.argv[1]
    stem = os.path.splitext(os.path.basename(src))[0]
    base = stem[:-4] if stem.endswith('_raw') else stem

    clean = os.path.join(ROOT, 'output', 'clean', base + '_clean.mp4')
    en_dir = os.path.join(ROOT, 'output', 'en')
    en_out = os.path.join(en_dir, base + '_en.mp4')
    srt_out = os.path.join(en_dir, base + '.en.srt')
    os.makedirs(os.path.dirname(clean), exist_ok=True)
    os.makedirs(en_dir, exist_ok=True)

    inpaint = os.path.join(WORK, 'results', stem, 'inpaint_out.mp4')
    if not os.path.exists(clean) and not os.path.exists(inpaint):
        raise SystemExit('missing ProPainter output: ' + inpaint)

    # ---------- A: clean video ----------
    if not os.path.exists(clean):
        run([FFMPEG, '-y', '-v', 'error',
             '-i', inpaint, '-i', src,
             '-map', '0:v:0', '-map', '1:a:0',
             '-c:v', 'libx264', '-crf', '17', '-preset', 'medium',
             '-pix_fmt', 'yuv420p',
             '-c:a', 'copy', '-shortest', clean])
    print('clean ->', clean)

    # ---------- B: English version ----------
    mix = os.path.join(WORK, 'en', 'mix.wav')
    srt = os.path.join(WORK, 'en', 'en.srt')
    if not os.path.exists(en_out):
        # hardware encode for >=1440px sources (4K libx264 would take forever)
        vinfo = next(s for s in ffprobe_json(clean)['streams']
                     if s['codec_type'] == 'video')
        if vinfo['width'] >= 1440:
            enc = ['-c:v', 'hevc_nvenc', '-preset', 'p5', '-cq', '21', '-rc', 'vbr']
        else:
            enc = ['-c:v', 'libx264', '-crf', '17', '-preset', 'medium']
        run([FFMPEG, '-y', '-v', 'error',
             '-i', clean, '-i', mix,
             '-vf', "subtitles=en/en.srt:force_style='" + STYLE + "'",
             '-map', '0:v:0', '-map', '1:a:0',
             *enc, '-pix_fmt', 'yuv420p',
             '-c:a', 'aac', '-b:a', '192k',
             '-shortest', en_out], cwd=WORK)
    shutil.copyfile(srt, srt_out)
    print('en    ->', en_out)

    for p in (clean, en_out):
        info = ffprobe_json(p)
        v = next(s for s in info['streams'] if s['codec_type'] == 'video')
        a = next((s for s in info['streams'] if s['codec_type'] == 'audio'), None)
        print(os.path.basename(p),
              'dur=%.3f' % float(info['format']['duration']),
              v['width'], 'x', v['height'], v['codec_name'],
              'audio=' + (a['codec_name'] if a else 'NONE'),
              'fps=' + v.get('avg_frame_rate', '?'))


if __name__ == '__main__':
    main()
