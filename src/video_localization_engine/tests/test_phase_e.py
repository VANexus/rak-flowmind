"""Phase E 测试 — L7 Localizer (Translator + TTS + Renderer + Compositor + 端到端)。

覆盖:
  - 4 个 backend Protocol conformance (单测)
  - MockTranslator / MockTtsBackend / OpenCVRenderer / FFmpegCompositor (单测)
  - TimingAligner (单测: 补静音/调速/截断)
  - 端到端: landscape / news / portrait (run_localize 产 mp4)
  - run_localize_from_vle 复用 Phase D 的 .vle.json (跳过 L1-L4)
  - debug artifacts 落地
  - 不变量: 业务层不直接 import 具体 API / 无 hard-code 字体参数
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from video_localization_engine.localizer import (
    AudioSegment,
    Compositor,
    CompositorRegistry,
    FFmpegCompositor,
    LocalizedSubtitle,
    LocalizedTrack,
    MockTranslator,
    MockTtsBackend,
    OpenCVRenderer,
    RendererRegistry,
    SubtitleRenderer,
    TimingAligner,
    TtsEngine,
    Translator,
    TranslatorRegistry,
    TtsRegistry,
    write_localize_artifacts,
)
from video_localization_engine.localizer.protocols import (
    CompositorBackend,
    RendererBackend,
    TranslatorBackend,
    TtsBackend,
)
from video_localization_engine.orchestrator import PipelineConfig, VideoLocalizationPipeline
from video_localization_engine.types.video import Orientation, VideoMeta


FIXTURES = Path("/tmp/vle_fixtures")
LANDSCAPE = FIXTURES / "landscape_person.mp4"
PORTRAIT = FIXTURES / "portrait_short.mp4"
SCREENCAST = FIXTURES / "screencast_ui.mp4"
NEWS = FIXTURES / "news_subtitle.mp4"


# =====================================================================
# L7.1 Translator
# =====================================================================
def test_TranslatorBackend_protocol_conformance():
    """MockTranslator 实现 TranslatorBackend Protocol。"""
    assert isinstance(MockTranslator(), TranslatorBackend)
    print("✓ TranslatorBackend protocol: MockTranslator OK")


def test_Translator_mock_prefix():
    t = Translator(backend_name="mock")
    out = t.translate("你好世界", source_locale="zh", target_locale="en")
    assert out == "[en] 你好世界", f"got {out!r}"
    print(f"✓ MockTranslator: {out!r}")


# =====================================================================
# L7.2 TTS
# =====================================================================
def test_TtsBackend_protocol_conformance():
    assert isinstance(MockTtsBackend(), TtsBackend)
    print("✓ TtsBackend protocol: MockTtsBackend OK")


def test_Tts_mock_silence():
    t = TtsEngine(backend_name="mock")
    s = t.synth("Hello world", target_locale="en", sample_rate=24000)
    assert isinstance(s, np.ndarray)
    assert s.dtype == np.float32
    assert s.ndim == 1
    assert s.shape[0] > 0
    # mock 应输出全零 (静音)
    assert float(np.abs(s).max()) == 0.0
    print(f"✓ MockTtsBackend silence: shape={s.shape}, dtype={s.dtype}")


# =====================================================================
# L7.3 TimingAligner
# =====================================================================
def test_TimingAligner_pad_silence():
    """短音频补静音到目标时长。"""
    a = TimingAligner(sample_rate=1000)
    audio = np.ones(500, dtype=np.float32)   # 0.5s @ 1000Hz
    out = a.align(audio, target_duration_ms=1000)   # 目标 1.0s
    assert out.shape[0] == 1000
    # 前 500 是 1, 后 500 是 0
    assert float(out[:500].mean()) == 1.0
    assert float(out[500:].mean()) == 0.0
    print(f"✓ TimingAligner pad: 500 → {out.shape[0]} (尾部静音)")


def test_TimingAligner_speed_up():
    """长音频 ≤ 20% 调速到目标时长。"""
    a = TimingAligner(sample_rate=1000, max_speed_change=0.20)
    audio = np.ones(1100, dtype=np.float32)   # 1.1s @ 1000Hz
    out = a.align(audio, target_duration_ms=1000)   # ratio=1.1 → 调速
    assert out.shape[0] == 1000
    print(f"✓ TimingAligner speed-up: 1100 → {out.shape[0]} (调速)")


def test_TimingAligner_truncate():
    """超过 20% 调速范围 → 截断。"""
    a = TimingAligner(sample_rate=1000, max_speed_change=0.20)
    audio = np.ones(2000, dtype=np.float32)   # 2.0s, 目标 1.0s, ratio=2.0
    out = a.align(audio, target_duration_ms=1000)
    assert out.shape[0] == 1000
    print(f"✓ TimingAligner truncate: 2000 → {out.shape[0]} (截断)")


# =====================================================================
# L7.4 Renderer
# =====================================================================
def test_RendererBackend_protocol_conformance():
    assert isinstance(OpenCVRenderer(), RendererBackend)
    print("✓ RendererBackend protocol: OpenCVRenderer OK")


def test_OpenCVRenderer_outputs_rgba():
    r = SubtitleRenderer(backend_name="opencv")
    rgba = r.render("Hello", bbox=(50, 50, 200, 100), frame_size=(200, 300))
    assert rgba.shape == (200, 300, 4)
    assert rgba.dtype == np.uint8
    # bbox 外区域 alpha 应为 0
    assert int(rgba[10, 10, 3]) == 0
    # bbox 内至少有部分 alpha > 0
    assert int(rgba[:, :, 3].max()) > 0
    print(f"✓ OpenCVRenderer RGBA: shape={rgba.shape}, max alpha={int(rgba[:, :, 3].max())}")


# =====================================================================
# L7.5 Compositor
# =====================================================================
def test_CompositorBackend_protocol_conformance():
    assert isinstance(FFmpegCompositor(), CompositorBackend)
    print("✓ CompositorBackend protocol: FFmpegCompositor OK")


def test_FFmpegCompositor_writes_mp4():
    """帧序列 + 静音 → mp4。"""
    import shutil
    import subprocess
    import tempfile
    c = Compositor(backend_name="ffmpeg")
    assert c.is_available, "ffmpeg not in PATH"
    frames = [
        np.full((60, 80, 3), (i * 5) % 255, dtype=np.uint8)
        for i in range(10)
    ]
    audio = np.zeros(24000, dtype=np.float32)  # 1s @ 24kHz
    with tempfile.TemporaryDirectory(prefix="vle_mp4_") as tmp:
        out = Path(tmp) / "out.mp4"
        c.composite(frames, audio, sample_rate=24000, fps=10.0, output_path=str(out))
        assert out.exists()
        size = out.stat().st_size
        ffprobe = shutil.which("ffprobe")
        probe_info = "(ffprobe not available, skipped probe)"
        if ffprobe:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_read_packets,nb_frames,duration",
                 "-of", "default=nw=1", str(out)],
                capture_output=True, text=True,
            )
            probe_info = r.stdout.strip() or "(empty probe output)"
    assert size > 0
    print(f"✓ FFmpegCompositor mp4: {size} bytes, ffprobe:\n{probe_info}")


# =====================================================================
# 端到端
# =====================================================================
def _run_pipeline(fixture_path: Path, frame_stride: int = 2):
    cfg = PipelineConfig(
        frame_stride=frame_stride,
        enable_inpaint=True,
        mask_dilation_px=2,
        inpaint_radius=3,
        debug_sample_rate=10,
        target_locale="en",
        localize_output_dir=f"/tmp/vle_debug/{fixture_path.stem}",
    )
    pipe = VideoLocalizationPipeline(str(fixture_path), cfg)
    try:
        track = pipe.run_localize(target_locale="en")
        return pipe, track
    finally:
        pipe.close()


def test_localizer_full_pipeline_landscape():
    pipe, track = _run_pipeline(LANDSCAPE, frame_stride=2)
    assert len(track.subtitles) >= 1
    # Mock 翻译应加 [en] 前缀
    for s in track.subtitles[:3]:
        assert s.target_text.startswith("[en]"), f"got {s.target_text!r}"
    assert track.rendered_video_path is not None
    out = Path(track.rendered_video_path)
    assert out.exists() and out.stat().st_size > 0
    print(f"✓ landscape localize: {len(track.subtitles)} subs, mp4={out.stat().st_size}B")


def test_localizer_full_pipeline_news():
    pipe, track = _run_pipeline(NEWS, frame_stride=2)
    assert len(track.subtitles) >= 1
    assert track.rendered_video_path is not None
    assert Path(track.rendered_video_path).exists()
    print(f"✓ news localize: {len(track.subtitles)} subs, mp4 exists")


def test_localizer_full_pipeline_portrait():
    pipe, track = _run_pipeline(PORTRAIT, frame_stride=2)
    assert len(track.subtitles) >= 1
    assert track.rendered_video_path is not None
    assert Path(track.rendered_video_path).exists()
    print(f"✓ portrait localize: {len(track.subtitles)} subs, mp4 exists")


def test_run_localize_from_vle_skip_l1_l6():
    """复用 Phase D 的 track.vle.json。"""
    vle = Path("/tmp/vle_debug/landscape_person/track.vle.json")
    if not vle.exists():
        print("~ skip: Phase D track.vle.json not found (Phase D must run first)")
        return
    cfg = PipelineConfig(
        frame_stride=2,
        target_locale="ja",
        localize_output_dir="/tmp/vle_debug/landscape_person",
    )
    track = VideoLocalizationPipeline.run_localize_from_vle(
        vle_path=str(vle), video_path=str(LANDSCAPE),
        config=cfg, target_locale="ja",
    )
    assert len(track.subtitles) >= 1
    # Mock 翻译应加 [ja] 前缀
    for s in track.subtitles[:3]:
        assert s.target_text.startswith("[ja]"), f"got {s.target_text!r}"
    assert track.rendered_video_path is not None
    print(f"✓ run_localize_from_vle: {len(track.subtitles)} subs, ja translated")


def test_localizer_outputs_debug_artifacts():
    """Phase E 的 debug artifacts 落到 /tmp/vle_debug/<fixture>/。"""
    _run_pipeline(LANDSCAPE, frame_stride=2)
    out_dir = Path("/tmp/vle_debug/landscape_person")
    for fn in ("translated_subtitles.png", "final_video_preview.png",
               "localized_audio.wav", "full_pipeline.vle.json"):
        p = out_dir / fn
        assert p.exists(), f"missing {p}"
    # mp4 文件名 = {video_stem}_{target_locale}.mp4
    expected_mp4 = out_dir / "landscape_person_en.mp4"
    assert expected_mp4.exists(), f"missing {expected_mp4}"
    # 验证 localized_audio.wav 可读
    import wave
    with wave.open(str(out_dir / "localized_audio.wav"), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 24000
        assert wf.getnframes() > 0
    # 验证 full_pipeline.vle.json
    import json
    data = json.loads((out_dir / "full_pipeline.vle.json").read_text())
    assert data["version"] == "0.2.0"
    assert data["target_locale"] == "en"
    assert len(data["subtitles"]) >= 1
    print(f"✓ debug artifacts: 5 files in {out_dir.name}/, "
          f"vle version={data['version']}")


# =====================================================================
# 不变量
# =====================================================================
def test_no_business_imports_of_concrete_apis():
    """localizer/backends/ 之外的 localizer/ 业务代码禁止 import 具体翻译/TTS API。"""
    import re
    base = Path(__file__).resolve().parents[1] / "localizer"
    forbidden = [
        r"import\s+edge_tts",
        r"from\s+edge_tts",
        r"import\s+deepl",
        r"from\s+deepl",
        r"import\s+elevenlabs",
        r"from\s+elevenlabs",
        r"import\s+googletrans",
        r"from\s+googletrans",
        r"import\s+coqui",
        r"from\s+coqui",
        r"import\s+TTS",
        r"from\s+TTS\b",
    ]
    for py in base.rglob("*.py"):
        if py.name == "__pycache__" or "__pycache__" in str(py):
            continue
        # backends/ 允许 (因为 backend 本身就是具体实现)
        if "/backends/" in str(py):
            continue
        for ln, line in enumerate(py.read_text().splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            for pat in forbidden:
                if re.search(pat, line):
                    raise AssertionError(
                        f"business code {py.relative_to(base)}:{ln} imports concrete API: {line!r}"
                    )
    print("✓ no business-layer imports of edge_tts / deepl / elevenlabs / coqui")


def test_renderer_params_exposed():
    """OpenCVRenderer 的所有视觉参数都通过构造参数暴露,不在内部 hard-code。"""
    r1 = OpenCVRenderer(font_scale=2.0, color=(0, 255, 0), stroke_color=(0, 0, 0),
                        thickness=3, stroke_thickness=5)
    rgba1 = r1.render("X", bbox=(0, 0, 100, 50), frame_size=(50, 100))
    r2 = OpenCVRenderer(font_scale=0.5, color=(255, 0, 0), stroke_color=(255, 255, 255),
                        thickness=1, stroke_thickness=2)
    rgba2 = r2.render("X", bbox=(0, 0, 100, 50), frame_size=(50, 100))
    # 字体大小不同 → alpha 覆盖像素数应不同
    alpha1 = int((rgba1[:, :, 3] > 0).sum())
    alpha2 = int((rgba2[:, :, 3] > 0).sum())
    assert alpha1 != alpha2, "font_scale change should affect rendered pixels"
    print(f"✓ Renderer params exposed: scale=2.0 → {alpha1}px, scale=0.5 → {alpha2}px")


def test_backends_registered():
    """默认 4 个 backend 已注册。"""
    assert "mock" in TranslatorRegistry.available()
    assert "mock" in TtsRegistry.available()
    assert "opencv" in RendererRegistry.available()
    assert "ffmpeg" in CompositorRegistry.available()
    print(f"✓ translator/tts/renderer/compositor: "
          f"{TranslatorRegistry.available()}/{TtsRegistry.available()}/"
          f"{RendererRegistry.available()}/{CompositorRegistry.available()}")


def main():
    tests = [
        # L7.1
        test_TranslatorBackend_protocol_conformance,
        test_Translator_mock_prefix,
        # L7.2
        test_TtsBackend_protocol_conformance,
        test_Tts_mock_silence,
        # L7.3 TimingAligner
        test_TimingAligner_pad_silence,
        test_TimingAligner_speed_up,
        test_TimingAligner_truncate,
        # L7.4
        test_RendererBackend_protocol_conformance,
        test_OpenCVRenderer_outputs_rgba,
        # L7.5
        test_CompositorBackend_protocol_conformance,
        test_FFmpegCompositor_writes_mp4,
        # 端到端
        test_localizer_full_pipeline_landscape,
        test_localizer_full_pipeline_news,
        test_localizer_full_pipeline_portrait,
        test_run_localize_from_vle_skip_l1_l6,
        test_localizer_outputs_debug_artifacts,
        # 不变量
        test_no_business_imports_of_concrete_apis,
        test_renderer_params_exposed,
        test_backends_registered,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} Phase E tests passed.")


if __name__ == "__main__":
    main()
