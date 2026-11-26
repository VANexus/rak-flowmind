# -*- coding: utf-8 -*-
"""Trim leading/trailing silence from TTS wavs in place (keeps 25ms pad)."""
import glob
import os
import sys
import wave
import array

TTS_DIR = sys.argv[1] if len(sys.argv) > 1 else r'D:\Desktop\obj114514\work\en\tts'
PAD = int(0.025 * 44100)   # 25 ms kept on each side
MIN_SILENCE = int(0.05 * 44100)  # only trim if silence longer than this


def process(path):
    w = wave.open(path, 'rb')
    ch, sw, fr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
    raw = w.readframes(n)
    w.close()
    if sw != 2:
        print('skip non-16bit', path)
        return 0, 0
    a = array.array('h')
    a.frombytes(raw)
    frames = len(a) // ch
    thr = 500  # ~1.5% of full scale

    # per-frame max abs across channels
    def frame_ok(i):
        return max(abs(a[i * ch + c]) for c in range(ch)) > thr

    first = 0
    while first < frames and not frame_ok(first):
        first += 1
    last = frames - 1
    while last >= 0 and not frame_ok(last):
        last -= 1
    if first == 0 and last == frames - 1:
        return 0, 0
    if first >= last:
        return 0, 0
    head_trim = first - PAD if first > MIN_SILENCE else 0
    tail_trim = (frames - 1 - last) - PAD if (frames - 1 - last) > MIN_SILENCE else 0
    if head_trim <= 0 and tail_trim <= 0:
        return 0, 0
    s = max(0, head_trim)
    e = frames - max(0, tail_trim)
    seg = a[s * ch:e * ch]
    out = wave.open(path, 'wb')
    out.setnchannels(ch)
    out.setsampwidth(2)
    out.setframerate(fr)
    out.writeframes(seg.tobytes())
    out.close()
    return (e - s) - frames, frames - (e - s)


total_saved = 0
files = sorted(glob.glob(os.path.join(TTS_DIR, '*.wav')))
for p in files:
    _, saved = process(p)
    total_saved += saved
print(f'trimmed {len(files)} clips, saved {total_saved/44100:.2f}s total')
