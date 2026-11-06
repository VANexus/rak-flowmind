# -*- coding: utf-8 -*-
import sys
import json

video = sys.argv[1] if len(sys.argv) > 1 else r'D:\Desktop\obj114514\ad738c4bcfdb9f0d41eeb5ede9a06a2e_raw.mp4'
out_json = sys.argv[2] if len(sys.argv) > 2 else r'D:\Desktop\obj114514\work\asr\asr.json'
out_srt = out_json.replace('.json', '.zh.srt')

from faster_whisper import WhisperModel

model = WhisperModel('medium', device='cpu', compute_type='int8')
segments, info = model.transcribe(video, language='zh', vad_filter=True,
                                  beam_size=5, condition_on_previous_text=False)

segs = []
for s in segments:
    seg = {'start': round(s.start, 3), 'end': round(s.end, 3), 'text': s.text.strip()}
    segs.append(seg)
    print(f'[{s.start:7.2f} -> {s.end:7.2f}] {s.text}', flush=True)

import os
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump({'language': info.language, 'segments': segs}, f, ensure_ascii=False, indent=2)


def fmt(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


with open(out_srt, 'w', encoding='utf-8') as f:
    for i, seg in enumerate(segs, 1):
        f.write(f'{i}\n{fmt(seg["start"])} --> {fmt(seg["end"])}\n{seg["text"]}\n\n')

print(f'saved {len(segs)} segments -> {out_srt}')
