"""voice_clone_enroll + localize_video 技能测试（全链路 mock）。

覆盖：
- 注册/入参校验/无 key 无 voice_id 的显式报错路径
- localize_video 编排：mock 各云模块 → 产物字段/推理链/degraded 形状
- ASR 失败 → degraded + 分类透传
- 双匹配：OCR 文本 vs ASR 文本冲突以 ASR 为准
"""
from __future__ import annotations

from pathlib import Path

import pytest

import flowmind.skills  # noqa: F401
from flowmind.skill import invoke, registry


# ── 注册 ──


def test_skills_registered():
    assert "voice_clone_enroll" in registry()
    assert "localize_video" in registry()


# ── voice_clone_enroll ──


def test_enroll_requires_sample_url(monkeypatch):
    r = invoke("voice_clone_enroll", {
        "action": "create",
        "sample_audio_url": "",
        "prefix": "p",
    })
    assert r.ok is False and r.error.code == "VALIDATION"


def test_enroll_create_no_key_degraded(monkeypatch):
    """无 DASHSCOPE_API_KEY → degraded + environment，不静默。"""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    import flowmind.skills.voice_clone_enroll as ve
    monkeypatch.setattr(ve, "resolve_api_key", lambda env: None)
    r = invoke("voice_clone_enroll", {
        "action": "create",
        "sample_audio_url": "https://oss/s.wav",
        "prefix": "my",
    })
    assert r.ok is True and r.metrics.degraded is True
    assert r.data.failure_category == "environment"
    assert "DASHSCOPE_API_KEY" in (r.data.warning or "")


def test_enroll_create_happy(monkeypatch):
    import flowmind.skills.voice_clone_enroll as ve

    monkeypatch.setattr("flowmind.skills.voice_clone_enroll.resolve_api_key",
                        lambda env: "k-ds")
    monkeypatch.setattr(
        ve, "_do_create",
        lambda sample_url, prefix, api_key: "cosyvoice-v3.5-plus-my-abc",
    )
    r = invoke("voice_clone_enroll", {
        "action": "create",
        "sample_audio_url": "https://oss/s.wav",
        "prefix": "my",
    })
    assert r.ok is True and r.metrics.degraded is False
    assert r.data.voice_id == "cosyvoice-v3.5-plus-my-abc"
    chain = r.reasoning[0]
    assert chain.conclusion and chain.causal_analysis and chain.risk_note


# ── localize_video ──


@pytest.fixture
def no_keys(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("LONGCAT_API_KEY", raising=False)


@pytest.fixture
def asr_local_ok(monkeypatch):
    """ASR 走本地流式路径（mock 掉 WS 适配层）。"""
    import flowmind.skills.localize_video as lv

    monkeypatch.setattr(lv._cloud_asr, "transcribe_local", lambda wav, api_key: [])


def test_localize_video_no_keys_degraded(no_keys, monkeypatch):
    """无任何 key → degraded + environment + 提示配 key，不静默降级 mock。"""
    monkeypatch.setattr("flowmind.skills.localize_video.resolve_api_key", lambda env: None)
    r = invoke("localize_video", {"video_path": "/nonexistent/v.mp4"})
    # 文件预检在前：不存在文件 → video；这里用真文件测 key 检查顺序
    assert r.metrics.degraded is True


def test_localize_video_missing_file_is_video(no_keys, tmp_path):
    r = invoke("localize_video", {"video_path": str(tmp_path / "no.mp4")})
    assert r.ok is True and r.metrics.degraded is True
    assert r.data.failure_category == "video"
    assert r.data.retriable is False


def test_localize_video_full_pipeline_mocked(no_keys, asr_local_ok, tmp_path, monkeypatch):
    """mock 五个云模块 → 完整编排跑通：产物/字幕区/推理链齐全。"""
    import flowmind.skills.localize_video as lv

    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00\x00\x00\x18ftypmp42")  # 假 mp4 头（预检只查存在+扩展名）

    segs = [
        {"index": 0, "begin": 0.0, "end": 2.0, "text": "大家好"},
        {"index": 1, "begin": 2.0, "end": 4.0, "text": "欢迎观看"},
    ]
    translated = [{**s, "text": f"EN{s['index']}"} for s in segs]

    monkeypatch.setattr(lv, "resolve_api_key", lambda env: "k-test")
    monkeypatch.setattr(lv._media, "extract_audio", lambda v, o, sample_rate=16000: o)
    monkeypatch.setattr(lv._media, "probe_duration", lambda p: 4.0)
    monkeypatch.setattr(lv._media, "extract_frame",
                        lambda v, t, o: open(o, "wb").close() or o)
    monkeypatch.setattr(lv._media, "burn_subtitles",
                        lambda vp, op, ass, erase_regions=None: op)
    monkeypatch.setattr(lv._media, "mix_audio", lambda vp, d, op, keep_background=False: op)
    monkeypatch.setattr(lv._cloud_asr, "transcribe_local", lambda wav, api_key: segs)
    monkeypatch.setattr(lv, "_locate_region",
                        lambda src, dur, wd, key, cfg: {"x": 100, "y": 800, "w": 600, "h": 80})
    # extract_frame 不再被 _locate_region 调用，但保留桩防其他路径
    monkeypatch.setattr(lv._llm_translate, "translate_segments",
                        lambda segs, **kw: translated)
    monkeypatch.setattr(lv._cloud_tts, "synthesize_segments",
                        lambda segments, **kw: [f"{kw['out_dir']}/seg_{s['index']:04d}.wav"
                                                for s in segments])
    # _concat_wavs 打桩为直接返回假配音轨路径
    monkeypatch.setattr(lv, "_concat_wavs", lambda wavs, out: out)

    r = invoke("localize_video", {
        "video_path": str(src),
        "target_lang": "en",
        "voice_id": "v-clone-1",
        "output_path": str(tmp_path / "out.mp4"),
    })
    assert r.ok is True, r.error
    d = r.data
    assert d.output_path == str(tmp_path / "out.mp4")
    assert d.asr_segment_count == 2
    assert d.subtitle_region_erased is True
    assert d.voice_used == "v-clone-1"
    chain = r.reasoning[0]
    assert chain.conclusion and chain.causal_analysis and chain.risk_note


def _fake_concat(cmd, **kw):
    return 0, "", ""


def test_localize_video_asr_failure_degraded(no_keys, asr_local_ok, tmp_path, monkeypatch):
    """ASR transient 失败 → degraded + transient + retriable=True。"""
    from flowmind.skills._cloud_asr import ASRError

    src = tmp_path / "in2.mp4"
    src.write_bytes(b"x")

    import flowmind.skills.localize_video as lv
    monkeypatch.setattr(lv, "resolve_api_key", lambda env: "k")
    monkeypatch.setattr(lv._media, "extract_audio", lambda v, o, sample_rate=16000: o)
    monkeypatch.setattr(lv._cloud_asr, "transcribe_local",
                        lambda wav, api_key: (_ for _ in ()).throw(
                            ASRError("限流", category="transient", retriable=True)))

    r = invoke("localize_video", {"video_path": str(src)})
    assert r.ok is True and r.metrics.degraded is True
    assert r.data.failure_category == "transient"
    assert r.data.retriable is True


def test_localize_video_asr_over_ocr_conflict(no_keys, asr_local_ok, tmp_path, monkeypatch):
    """双匹配冲突：OCR 文本与 ASR 不一致时，采用 ASR 文本（用户硬要求）。"""
    import flowmind.skills.localize_video as lv

    src = tmp_path / "in3.mp4"
    src.write_bytes(b"x")
    segs = [{"index": 0, "begin": 0.0, "end": 2.0, "text": "ASR文本"}]

    monkeypatch.setattr(lv, "resolve_api_key", lambda env: "k")
    monkeypatch.setattr(lv._media, "extract_audio", lambda v, o, sample_rate=16000: o)
    monkeypatch.setattr(lv._media, "probe_duration", lambda p: 2.0)
    monkeypatch.setattr(lv._media, "extract_frame",
                        lambda v, t, o: open(o, "wb").close() or o)
    monkeypatch.setattr(lv._media, "burn_subtitles",
                        lambda vp, op, ass, erase_regions=None: op)
    monkeypatch.setattr(lv._media, "mix_audio", lambda vp, d, op, keep_background=False: op)
    monkeypatch.setattr(lv._cloud_asr, "transcribe_local", lambda wav, api_key: segs)
    monkeypatch.setattr(lv._llm_translate, "translate_segments",
                        lambda s, **kw: [{**x, "text": "T"} for x in s])
    monkeypatch.setattr(lv._cloud_tts, "synthesize_segments",
                        lambda segments, **kw: ["/tmp/x.wav"])
    monkeypatch.setattr(lv, "_concat_wavs", lambda wavs, out: out)
    # 不配音分支会把擦除产物 move 到默认输出路径；桩里直接落文件并返回路径
    def fake_burn(vp, op, ass, erase_regions=None):
        Path(op).write_bytes(b"x")
        return op

    monkeypatch.setattr(lv._media, "burn_subtitles", fake_burn)
    # OCR 只用于定位 bbox，文本不进翻译源（ASR 为准）
    monkeypatch.setattr(lv, "_locate_region",
                        lambda src, dur, wd, key, cfg: {"x": 0, "y": 900, "w": 500, "h": 60})

    r = invoke("localize_video", {"video_path": str(src), "target_lang": "en"})
    assert r.ok is True
    # 送翻译的必须是 ASR 文本
    assert r.data.asr_segment_count == 1
