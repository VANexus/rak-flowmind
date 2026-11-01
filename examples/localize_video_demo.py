"""
localize_video 技能演示 —— 全云端视频本地化流水线（mock 全链路）。

运行：uv run python examples/localize_video_demo.py

流水线（九阶段）：ffmpeg 提音轨 → 百炼 Paraformer ASR → 说话人分离（CAM++）
→ 阿里云 OCR 定位原字幕区（文本以 ASR 为准）→ LongCat 翻译 → 字幕擦除
→ CosyVoice 克隆 TTS → 混音合成 → 字幕向量化。

本 demo mock 所有云调用与 ffmpeg，展示编排与输出形状；
真打需 export DASHSCOPE_API_KEY / LONGCAT_API_KEY 并提供公网可访问视频 URL。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import flowmind.skills  # noqa: F401  触发 @skill 注册
import flowmind.skills.localize_video as lv
from flowmind.discover import field_names
from flowmind.skill import invoke

MOCK_CALLS = {}


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def install_mock_pipeline() -> None:
    """基础形态：跳过说话人分离 + 云 TTS（不加载 funasr / qwen-tts 真模型）。"""
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
    lv._media.mix_audio = lambda vp, d, op, keep_background=False, bg_path=None: op
    lv._inpaint.available = lambda: True      # mock LaMa 遮罩传递，不加载模型

    def mock_erase(src, regions, out_path, workdir, solid_regions=None):
        MOCK_CALLS["lama"] = {
            "subtitle_regions": regions,
            "solid_regions": solid_regions,
        }
        Path(out_path).write_bytes(b"x")
        return out_path

    lv._inpaint.erase_regions = mock_erase
    lv._local_tts.available = lambda: False   # demo 走云 TTS 分支（不加载 4GB 本地模型）
    lv._music_sep.available = lambda: False   # demo 不做真人声分离（无 demucs 依赖）
    lv._diarize.available = lambda: False     # demo 跳过说话人分离（不跑真 funasr 模型）
    lv._local_asr.available = lambda: False   # auto 后端回落到已 mock 的云 ASR
    lv._local_ocr.available = lambda: False   # auto 后端回落到已 mock 的云 OCR
    lv._cloud_asr.transcribe_local = lambda wav, api_key, **kw: segs
    lv._locate_region = lambda src, dur, wd, key, cfg, **kw: [
        {"x": 120, "y": 900, "w": 640, "h": 70}]
    lv._llm_translate.translate_segments = lambda s, **kw: translated
    lv._cloud_tts.synthesize_segments = lambda segments, **kw: [
        f"{kw['out_dir']}/seg_{x['index']:04d}.mp3" for x in segments]
    lv._mix_timed_audio = lambda segs, dubs, dur, out, **kw: (
        out, {"segments": len(segs), "compressed": 0, "overflow": 0,
              "max_overflow_ratio": 0.0, "worst_index": None})
    lv._replace_audio = lambda vp, audio, out: out


def install_mock_diarize_ok() -> None:
    """「分离成功」形态：在基础 mock 之上叠加两人对话 → 逐说话人音色克隆。

    先在基础形态上重置，再叠加 diarize-ok 与本地 TTS 的 mock：
    - ``_diarize.diarize`` 返回两个轮次，``assign_speakers``（真实纯函数）据此
      给 ASR 句段打 spk0/spk1 标签；
    - ``build_speaker_refs`` 返回含 ref_audio（真实落盘的占位文件）/ref_text/
      seconds/low_confidence 的完整结构；
    - 翻译 mock **透传入参**（``_split_long_segments`` 已把 ``speaker`` 键带到
      display 句段），逐说话人路由才能命中；
    - 本地 TTS 走通（``tts_backend=auto`` + 无 ``voice_id`` 才走本地克隆分支），
      故本形态下调用 invoke **不要**传 voice_id。
    """
    install_mock_pipeline()
    ref_dir = Path(tempfile.mkdtemp(prefix="demo_speaker_refs_"))

    def _ref(sid: str) -> dict:
        ref_path = ref_dir / f"ref_{sid}.wav"
        ref_path.write_bytes(b"")   # 占位：仅需真实存在的路径，流水线不读其内容
        return {"ref_audio": str(ref_path), "ref_text": f"{sid} 参考转写",
                "seconds": 6.0, "low_confidence": False}

    lv._diarize.available = lambda: True
    lv._diarize.diarize = lambda wav, **kw: [
        {"begin": 0.0, "end": 2.1, "speaker": "spk0"},
        {"begin": 2.1, "end": 4.8, "speaker": "spk1"},
    ]
    lv._diarize.build_speaker_refs = lambda segs, wav, out, **kw: {
        "spk0": _ref("spk0"), "spk1": _ref("spk1")}
    lv._local_tts.available = lambda: True
    lv._local_tts.synthesize_segments = lambda segs, **kw: [
        f"{kw['out_dir']}/seg_{s['index']:04d}.wav" for s in segs]
    lv._llm_translate.translate_segments = lambda segs, **kw: [
        {**s, "text": f"[en] {s.get('text', '')}"} for s in segs]


def main() -> None:
    Path("/tmp/demo.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")  # 预检用假 mp4

    section("0) discover('localize_video') —— Agent 自查字段")
    for p, names in field_names("localize_video").items():
        print(f"  {p}: {names}")

    section("1) Happy path：全链路（跳过分离 + 云克隆配音）")
    install_mock_pipeline()
    r = invoke("localize_video", {
        "video_path": "/tmp/demo.mp4",
        "target_lang": "en",
        "voice_id": "cosyvoice-v3.5-plus-demo-abc",
        "output_path": "/tmp/demo_localized.mp4",
        "watermark_bbox": (0.04, 0.02, 0.37, 0.08),
    })
    if not r.ok:
        raise AssertionError(f"mock localization failed: {r.error}")
    assert r.data.watermark_removed is True
    assert MOCK_CALLS["lama"]["solid_regions"] == [
        {"x": 26, "y": 7, "w": 211, "h": 22},
    ]
    print(f"  ok              : {r.ok}")
    print(f"  output_path     : {r.data.output_path}")
    print(f"  时长            : {r.data.duration_seconds}s")
    print(f"  ASR 句数        : {r.data.asr_segment_count}")
    print(f"  原字幕区已擦除  : {r.data.subtitle_region_erased}")
    print(f"  水印区已送修复  : {r.data.watermark_removed}")
    print(f"  使用音色        : {r.data.voice_used}")
    print(f"  分离后端        : {r.data.diarize_backend}")
    print(f"  说话人条数      : {len(r.data.quality['speakers'])}")
    print(f"  trace_id        : {r.trace.trace_id[:8]}...")
    print(f"  推理结论        : {r.reasoning[0].conclusion}")
    assert r.ok is True
    assert r.data.voice_used == "cosyvoice-v3.5-plus-demo-abc"
    assert r.data.diarize_backend == "none"
    assert len(r.data.quality["speakers"]) == 0
    assert "说话人 0 位" in r.reasoning[0].conclusion

    section("1b) 多人对话：逐说话人音色克隆 + quality 自检")
    install_mock_diarize_ok()
    r = invoke("localize_video", {
        "video_path": "/tmp/demo.mp4",
        "target_lang": "en",
        "output_path": "/tmp/demo_multi.mp4",
    })
    quality = r.data.quality
    print(f"  ok              : {r.ok}")
    print(f"  output_path     : {r.data.output_path}")
    print(f"  使用音色        : {r.data.voice_used}")
    print(f"  分离后端        : {r.data.diarize_backend}")
    print(f"  说话人条数      : {len(quality['speakers'])}")
    for spk in quality["speakers"]:
        print(f"    - {spk}")
    print(f"  配音统计(dub)   : {quality['dub']}")
    print(f"  译文超预算句段  : {quality['translate_over_budget']}")
    print(f"  擦除覆盖框数    : {quality['erase_coverage']}")
    print(f"  推理结论        : {r.reasoning[0].conclusion}")
    print(f"  因果分析        : {r.reasoning[0].causal_analysis}")
    assert r.ok is True
    assert r.data.voice_used == "clone:multi_speaker(n=2)"
    assert r.data.diarize_backend == "cam++"
    assert len(quality["speakers"]) == 2
    assert {spk["id"] for spk in quality["speakers"]} == {"spk0", "spk1"}
    assert all(set(spk) == {"id", "segments", "ref_seconds", "low_confidence"}
               for spk in quality["speakers"])
    assert set(quality["dub"]) == {
        "segments", "compressed", "overflow", "max_overflow_ratio", "worst_index"}
    assert quality["translate_over_budget"] == []
    # 擦除框数取自 OCR 候选区经分辨率过滤后的结果（demo 假分辨率下可为 0）
    assert isinstance(quality["erase_coverage"], int)
    assert "说话人 2 位" in r.reasoning[0].conclusion
    assert "说话人分离[cam++]" in r.reasoning[0].causal_analysis

    section("2) 无 key：显式 degraded，不静默降级")
    install_mock_pipeline()   # 恢复基础形态（跳过分离 + 云 TTS）
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
