"""
localize_video 技能演示 —— 全云端视频本地化流水线（mock 全链路）。

运行：uv run python examples/localize_video_demo.py

流水线：ffmpeg 提音轨 → 百炼 Paraformer ASR → LongCat 翻译
→ 阿里云 OCR 定位原字幕区（文本以 ASR 为准）→ CosyVoice 克隆 TTS → 合成。

本 demo mock 所有云调用与 ffmpeg，展示编排与输出形状；
真打需 export DASHSCOPE_API_KEY / LONGCAT_API_KEY 并提供公网可访问视频 URL。
"""

from __future__ import annotations

from pathlib import Path

import flowmind.skills  # noqa: F401  触发 @skill 注册
import flowmind.skills.localize_video as lv
from flowmind.discover import field_names
from flowmind.skill import invoke


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def install_mock_pipeline() -> None:
    segs = [
        {"index": 0, "begin": 0.0, "end": 2.1, "text": "大家好，欢迎收看本期节目"},
        {"index": 1, "begin": 2.1, "end": 4.8, "text": "今天我们来聊发动机保养"},
    ]
    translated = [{**s, "text": t} for s, t in
                  zip(segs, ["Hi everyone, welcome", "Today: engine maintenance"])]
    from pathlib import Path

    lv.get_api_key = lambda env: "mock-key"
    lv._media.extract_audio = lambda v, o, **kw: o
    lv._media.probe_media = lambda p: (4.8, 640, 360)
    lv._media.extract_frame = lambda v, t, o: (Path(o).write_bytes(b"") or o)
    lv._media.burn_subtitles = lambda vp, op, ass, erase_regions=None: (
        Path(op).write_bytes(b"x") or op)
    lv._media.mix_audio = lambda vp, d, op, keep_background=False: op
    lv._cloud_asr.transcribe_local = lambda wav, api_key, **kw: segs
    lv._locate_region = lambda src, dur, wd, key, cfg, **kw: {"x": 120, "y": 900, "w": 640, "h": 70}
    lv._llm_translate.translate_segments = lambda s, **kw: translated
    lv._cloud_tts.synthesize_segments = lambda segments, **kw: [
        f"{kw['out_dir']}/seg_{x['index']:04d}.mp3" for x in segments]
    lv._concat_wavs = lambda wavs, out: out


def main() -> None:
    Path("/tmp/demo.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")  # 预检用假 mp4

    section("0) discover('localize_video') —— Agent 自查字段")
    for p, names in field_names("localize_video").items():
        print(f"  {p}: {names}")

    section("1) Happy path：全链路（含克隆配音）")
    install_mock_pipeline()
    r = invoke("localize_video", {
        "video_path": "/tmp/demo.mp4",
        "target_lang": "en",
        "voice_id": "cosyvoice-v3.5-plus-demo-abc",
        "output_path": "/tmp/demo_localized.mp4",
    })
    print(f"  ok              : {r.ok}")
    print(f"  output_path     : {r.data.output_path}")
    print(f"  时长            : {r.data.duration_seconds}s")
    print(f"  ASR 句数        : {r.data.asr_segment_count}")
    print(f"  原字幕区已擦除  : {r.data.subtitle_region_erased}")
    print(f"  使用音色        : {r.data.voice_used}")
    print(f"  trace_id        : {r.trace.trace_id[:8]}...")
    print(f"  推理结论        : {r.reasoning[0].conclusion}")

    section("2) 无 key：显式 degraded，不静默降级")
    lv.get_api_key = lambda env: None
    r = invoke("localize_video", {"video_path": "/tmp/demo.mp4"})
    print(f"  degraded        : {r.metrics.degraded}")
    print(f"  failure_category: {r.data.failure_category}（environment → 先配 key）")
    print(f"  warning         : {r.data.warning}")

    section("3) 文件不存在：video 类（修输入）")
    lv.get_api_key = lambda env: "k"
    r = invoke("localize_video", {"video_path": "/no/such/file.mp4"})
    print(f"  failure_category: {r.data.failure_category}（video）")
    print(f"  retriable       : {r.data.retriable}")


if __name__ == "__main__":
    main()
