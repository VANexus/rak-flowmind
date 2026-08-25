"""_media 模块测试：ffmpeg 薄封装。

覆盖：
- 命令拼装正确（不真跑 ffmpeg，monkeypatch 注入 fake runner）
- extract_audio：输入输出路径、采样率参数
- probe_media：解析 ffprobe JSON（时长+分辨率）
- burn_subtitles / erase_region：滤镜串拼装
- 真 ffmpeg 冒烟（生成 2s 测试视频 → 提取音轨 → 探时长）
"""
from __future__ import annotations

import json

import pytest

from flowmind.skills import _media


# ── 命令拼装（monkeypatch 注入，避免模块属性泄漏） ──


def test_extract_audio_builds_expected_command(monkeypatch):
    calls = []

    def fake_run(cmd, **_kw):
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(_media, "run_ffmpeg", fake_run)
    out = _media.extract_audio("/in/v.mp4", "/out/v.wav", sample_rate=16000)
    assert out == "/out/v.wav"
    cmd = " ".join(calls[0])
    assert "-i" in cmd and "/in/v.mp4" in cmd
    assert "/out/v.wav" in cmd
    assert "16000" in cmd
    assert "-vn" in cmd, "提取音轨必须丢弃视频流"


def test_extract_audio_raises_on_failure(monkeypatch):
    def fake_run(cmd, **_kw):
        return 1, "", "boom"

    monkeypatch.setattr(_media, "run_ffmpeg", fake_run)
    with pytest.raises(_media.MediaError, match="boom"):
        _media.extract_audio("/in/v.mp4", "/out/v.wav")


def test_probe_media_parses_ffprobe_json(monkeypatch):
    payload = {"format": {"duration": "12.5"},
               "streams": [{"width": 1920, "height": 1080}]}

    def fake_run(cmd, **_kw):
        return 0, json.dumps(payload), ""

    monkeypatch.setattr(_media, "run_ffprobe", fake_run)
    duration, w, h = _media.probe_media("/v.mp4")
    assert duration == pytest.approx(12.5)
    assert (w, h) == (1920, 1080)


def test_burn_subtitles_includes_filter_and_output(monkeypatch):
    calls = []

    def fake_run(cmd, **_kw):
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(_media, "run_ffmpeg", fake_run)
    _media.burn_subtitles(
        "/in.mp4", "/out.mp4", "/subs.ass",
        erase_regions=[{"x": 100, "y": 800, "w": 600, "h": 80}],
    )
    cmd = " ".join(calls[0])
    assert "delogo=x=100:y=800:w=600:h=80" in cmd
    assert "ass=/subs.ass" in cmd
    assert "/out.mp4" in calls[0][-1]


def test_erase_only_without_subs(monkeypatch):
    calls = []

    def fake_run(cmd, **_kw):
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(_media, "run_ffmpeg", fake_run)
    _media.burn_subtitles("/in.mp4", "/out.mp4", None,
                          erase_regions=[{"x": 0, "y": 900, "w": 1280, "h": 60}])
    cmd = " ".join(calls[0])
    assert "delogo" in cmd
    assert "ass=" not in cmd


def test_mix_audio_replaces_original(monkeypatch):
    calls = []

    def fake_run(cmd, **_kw):
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(_media, "run_ffmpeg", fake_run)
    _media.mix_audio("/video.mp4", "/dub.wav", "/final.mp4", keep_background=False)
    cmd = " ".join(calls[0])
    assert "-map" in cmd
    assert "/dub.wav" in cmd


# ── 真 ffmpeg 冒烟 ──


def test_real_ffmpeg_smoke(tmp_path):
    """生成 2s 测试视频 → 探时长 → 提取音轨 → 音轨可探。"""
    src = tmp_path / "t.mp4"
    wav = tmp_path / "t.wav"
    rc, _, err = _media.run_ffmpeg([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-shortest", str(src),
    ])
    assert rc == 0, err[:200]
    assert _media.probe_media(str(src))[0] == pytest.approx(2.0, abs=0.3)
    out = _media.extract_audio(str(src), str(wav), sample_rate=16000)
    assert out == str(wav) and wav.exists()
