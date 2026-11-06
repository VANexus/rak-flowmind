# -*- coding: utf-8 -*-
"""Place English TTS clips on the original slot timeline (local atempo when a
clip overflows its slot), mix with the instrumental stem, write the English SRT.

Inputs : en.json, en/tts/seg_*.wav, separated/htdemucs/orig/no_vocals.wav
Outputs: en/vo.wav, en/mix.wav, en/en.srt, en/plan.json
"""
import json
import os
import subprocess
import sys

BASE = r'D:\Desktop\obj114514\work'
EN_JSON = os.path.join(BASE, 'en', 'en.json')
TTS_DIR = (sys.argv[1] if len(sys.argv) > 1
           else os.path.join(BASE, 'en', 'tts'))
BG = os.path.join(BASE, 'separated', 'htdemucs', 'orig', 'no_vocals.wav')
OUT_DIR = os.path.join(BASE, 'en')
FFMPEG = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'
FFPROBE = r'C:\Program Files\ffmpeg\bin\ffprobe.exe'

VIDEO_DUR = 82.245
GAP = 0.10
MAX_TEMPO = 1.2
BG_GAIN = 0.9
VO_GAIN = 1.85


def dur_of(path):
    out = subprocess.check_output(
        [FFPROBE, '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=nw=1:nk=1', path], text=True).strip()
    return float(out)


def fmt_srt(t):
    ms = int(round(max(t, 0) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def main():
    with open(EN_JSON, encoding='utf-8') as f:
        segs = json.load(f)['segments']
    n = len(segs)
    for i, s in enumerate(segs):
        s['wav'] = os.path.join(TTS_DIR, 'seg_%03d.wav' % i)
        s['tts_dur'] = dur_of(s['wav'])

    tempos = [1.0] * n
    t0s, ends = [], []
    for _ in range(5):
        t0s, ends, prev = [], [], None
        for i, s in enumerate(segs):
            d = s['tts_dur'] / tempos[i]
            st = s['start'] if prev is None else max(s['start'], prev + GAP)
            t0s.append(st)
            prev = st + d
            ends.append(prev)
        changed = False
        for i in range(n):
            limit = (segs[i + 1]['start'] - 0.08) if i + 1 < n else (VIDEO_DUR - 0.12)
            avail = limit - t0s[i]
            d = segs[i]['tts_dur'] / tempos[i]
            if avail > 0 and d > avail:
                new = min(MAX_TEMPO, d / max(avail, 0.30))
                if new > tempos[i] + 0.005:
                    tempos[i] = new
                    changed = True
        if not changed:
            break

    if ends[-1] > VIDEO_DUR - 0.10:
        print('WARNING: still overflowing: last_end=%.2f' % ends[-1])

    for i, s in enumerate(segs):
        s['tempo'] = round(tempos[i], 4)
        s['play_start'] = round(t0s[i], 3)
        s['play_end'] = round(ends[i], 3)
    stretched = [i for i in range(n) if tempos[i] > 1.01]
    print('tempo applied on segs:', {i: round(tempos[i], 2) for i in stretched})
    print('first=%.2f last_end=%.2f video=%.2f' % (t0s[0], ends[-1], VIDEO_DUR))

    # --- SRT -------------------------------------------------------------
    lines = []
    for i, s in enumerate(segs):
        nxt = segs[i + 1]['play_start'] if i + 1 < n else VIDEO_DUR
        end = min(nxt - 0.05, s['play_end'] + 0.55, VIDEO_DUR)
        end = max(end, s['play_start'] + 0.8)
        lines.append('%d\n%s --> %s\n%s\n' % (
            i + 1, fmt_srt(s['play_start']), fmt_srt(end), s['text_en']))
    with open(os.path.join(OUT_DIR, 'en.srt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    # --- VO track --------------------------------------------------------
    fc = []
    inputs = ['-i', BG]
    for i, s in enumerate(segs):
        inputs += ['-i', s['wav']]
        ms = int(round(s['play_start'] * 1000))
        chain = '[%d:a]' % (i + 1)
        if s['tempo'] > 1.001:
            chain += 'atempo=%.5f,' % s['tempo']
        chain += ('aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,'
                  'adelay=%d|%d[v%d]' % (ms, ms, i))
        fc.append(chain)
    fc.append(''.join('[v%d]' % i for i in range(n)) +
              'amix=inputs=%d:duration=longest:normalize=0[vo]' % n)

    vo = os.path.join(OUT_DIR, 'vo.wav')
    subprocess.run([FFMPEG, '-y', '-v', 'error'] + inputs + [
        '-filter_complex', ';'.join(fc),
        '-map', '[vo]', '-ar', '44100', '-ac', '2', vo], check=True)
    print('vo.wav dur %.2f' % dur_of(vo))

    # --- mix VO + instrumental ------------------------------------------
    mix = os.path.join(OUT_DIR, 'mix.wav')
    fc2 = ('[0:a]volume=%s[bg];' % BG_GAIN +
           '[1:a]volume=%s[v];' % VO_GAIN +
           '[bg][v]amix=inputs=2:duration=first:normalize=0,'
           'alimiter=limit=0.95:level=disabled[mx]')
    subprocess.run([FFMPEG, '-y', '-v', 'error', '-i', BG, '-i', vo,
                    '-filter_complex', fc2, '-map', '[mx]',
                    '-ar', '44100', '-ac', '2', mix], check=True)
    print('mix.wav dur %.2f' % dur_of(mix))

    with open(os.path.join(OUT_DIR, 'plan.json'), 'w', encoding='utf-8') as f:
        json.dump({'segments': segs}, f, ensure_ascii=False, indent=2)
    print('wrote en.srt, plan.json')


if __name__ == '__main__':
    main()
