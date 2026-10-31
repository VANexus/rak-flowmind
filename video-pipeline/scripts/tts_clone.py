# -*- coding: utf-8 -*-
"""Generate English clips with GPT-SoVITS using the ORIGINAL speaker's voice.

Starts api_v2.py (if not already up), then POSTs /tts for every segment of
en.json using the demucs-isolated original vocals as the reference audio.
Output: en/tts_clone/seg_XXX.wav
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

BASE = r'D:\Desktop\obj114514\work'
GPT_DIR = os.path.join(BASE, 'GPT-SoVITS')
PY = r'D:\conda_envs\gptsovits\python.exe'
EN_JSON = os.path.join(BASE, 'en', 'en.json')
OUT_DIR = os.path.join(BASE, 'en', 'tts_clone')
PORT = 9880

REF1 = os.path.join(BASE, 'en', 'ref_speaker2.wav')   # 43.0-50.8s of the host (7.8s, primary ref)
REF2 = os.path.join(BASE, 'en', 'ref_speaker.wav')    # 14.4-23.6s of the host (9.2s, aux ref)
REF1_TEXT = '我们拥有15年化妆品资深专业的研发团队，支持自主研发打样、来图打样，满足客户各种定制需求。'
REF2_TEXT = '生产的产品包括口红、眼影、高光、指甲油、眼线笔、粉底、眉笔、睫毛膏。'


def server_up():
    try:
        urllib.request.urlopen(f'http://127.0.0.1:{PORT}/', timeout=2)
        return True
    except urllib.error.HTTPError:
        # server answered (no root route) => it is up
        return True
    except Exception:
        return False


def tts_one(text, out_path):
    payload = json.dumps({
        'text': text,
        'text_lang': 'en',
        'ref_audio_path': REF1,
        'aux_ref_audio_paths': [REF2],
        'prompt_text': REF1_TEXT,
        'prompt_lang': 'zh',
        'media_type': 'wav',
        'seed': 42,
        'text_split_method': 'cut5',
    }).encode('utf-8')
    req = urllib.request.Request(
        f'http://127.0.0.1:{PORT}/tts', data=payload,
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', 'replace')[:800]
        raise RuntimeError(f'HTTP {e.code}: {body}')
    if len(data) < 1000:
        raise RuntimeError('bad response: %r...' % data[:200])
    with open(out_path, 'wb') as f:
        f.write(data)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(EN_JSON, encoding='utf-8') as f:
        segs = json.load(f)['segments']

    if not server_up():
        print('starting api_v2 server ...', flush=True)
        env = dict(os.environ)
        # torchcodec (torchaudio.load backend) needs shared ffmpeg DLLs
        ff_bin = (r'D:\Desktop\obj114514\work\ffmpeg_shared'
                  r'\ffmpeg-master-latest-win64-gpl-shared\bin')
        env['PATH'] = ff_bin + ';' + env['PATH']
        subprocess.Popen(
            [PY, 'api_v2.py', '-a', '127.0.0.1', '-p', str(PORT)],
            cwd=GPT_DIR, env=env,
            stdout=open(os.path.join(BASE, 'gptsovits_server.log'), 'w'),
            stderr=subprocess.STDOUT)
        for i in range(120):
            time.sleep(3)
            if server_up():
                break
        else:
            raise SystemExit('server did not come up; see work/gptsovits_server.log')
    print('server up', flush=True)

    only = set(int(x) for x in sys.argv[1:]) if len(sys.argv) > 1 else None
    for i, s in enumerate(segs):
        if only is not None and i not in only:
            continue
        out = os.path.join(OUT_DIR, 'seg_%03d.wav' % i)
        if os.path.exists(out) and (only is None or i not in only):
            continue
        t0 = time.time()
        tts_one(s['text_en'], out)
        print('%3d  %.1fs  %s' % (i, time.time() - t0, s['text_en'][:60]), flush=True)
    print('ALL CLIPS DONE', flush=True)


if __name__ == '__main__':
    main()
