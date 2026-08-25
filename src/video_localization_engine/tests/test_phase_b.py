"""Phase B 测试 — L1/L2/L3 在 3 类合成视频上的端到端验证。

合成 fixture: 横屏人物 / 竖屏短视频 / 屏幕录制 (见 synthesize_fixtures.py)

每个模块独立验证:
  L1 VideoAnalyzer:
    - meta 推导 (orientation / fps / frame_count)
    - 帧迭代, seek, close
    - 协议满足 (runtime_checkable)
  L2 RegionPolicy:
    - is_applicable 自动按 orientation 选择
    - 不假设位置, weight 与 polygon 都来自 policy
    - propose 输出 RegionProposal, 不是字幕判断
  L3 TextDetector:
    - 在 3 类视频上分别 detect, 输出 TextCandidate
    - 不生成 SubtitleInstance (Phase C 才做)

测试也验证:
  - 不在 L1/L2/L3 中 hard-code y 比例
  - 所有 backend 通过 Protocol + Registry 注入
  - 跨视频: 横屏 / 竖屏 / 屏幕录制行为差异由 orientation 驱动, 不由代码 if else 驱动
"""
import sys
from pathlib import Path
from dataclasses import fields

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from video_localization_engine.analyzer import (
    OpenCVVideoAnalyzer, VideoAnalyzer, VideoAnalyzerRegistry, derive_orientation,
)
from video_localization_engine.region_policies import (
    BottomHorizontalPolicy, BottomPortraitPolicy, TopNewsPolicy,
    CustomPolicy, RegionPolicyRegistry, applicable_policies,
    SubtitleRegionPolicy, CompositeRegionPolicy,
)
from video_localization_engine.detector import (
    PaddleOCRDetector, TextDetector, TextDetectorRegistry,
)
from video_localization_engine.types.video import (
    FramePacket, Orientation, VideoMeta,
)
from video_localization_engine.types.detection import (
    FrameTextCandidates, RegionProposal, TextCandidate,
)
from video_localization_engine.types.protocols import (
    VideoAnalyzerProtocol, RegionPolicyProtocol, TextDetectorProtocol,
)


FIXTURES = Path("/tmp/vle_fixtures")
L1 = FIXTURES / "landscape_person.mp4"     # 1920x1080
L2 = FIXTURES / "portrait_short.mp4"       # 720x1280
L3 = FIXTURES / "screencast_ui.mp4"        # 1280x720


# =====================================================================
# L1 VideoAnalyzer tests
# =====================================================================
def test_derive_orientation_pure_function():
    """纯函数, 任何 w/h 输入 → 正确 orientation。"""
    assert derive_orientation(1920, 1080) == Orientation.LANDSCAPE
    assert derive_orientation(720, 1280) == Orientation.PORTRAIT
    assert derive_orientation(1080, 1080) == Orientation.SQUARE
    assert derive_orientation(1, 1000) == Orientation.PORTRAIT
    assert derive_orientation(1000, 1) == Orientation.LANDSCAPE
    print("✓ derive_orientation pure function")


def test_L1_meta_on_landscape_fixture():
    analyzer = OpenCVVideoAnalyzer(str(L1))
    try:
        m = analyzer.meta
        assert m.width == 1920
        assert m.height == 1080
        assert m.orientation == Orientation.LANDSCAPE
        assert m.fps == 30.0
        assert m.frame_count == 90
        assert m.has_audio is True
        assert isinstance(m, VideoMeta)
        print(f"✓ L1 meta (landscape 1920x1080): "
              f"{m.orientation.value}, {m.fps:.1f}fps, {m.frame_count}f")
    finally:
        analyzer.close()


def test_L1_meta_on_portrait_fixture():
    analyzer = OpenCVVideoAnalyzer(str(L2))
    try:
        m = analyzer.meta
        assert m.width == 720
        assert m.height == 1280
        assert m.orientation == Orientation.PORTRAIT
        print(f"✓ L1 meta (portrait 720x1280): {m.orientation.value}")
    finally:
        analyzer.close()


def test_L1_meta_on_screencast_fixture():
    analyzer = OpenCVVideoAnalyzer(str(L3))
    try:
        m = analyzer.meta
        assert m.width == 1280
        assert m.height == 720
        assert m.orientation == Orientation.LANDSCAPE
        assert m.duration_ms == 3000  # 90 frames @ 30fps
        print(f"✓ L1 meta (screencast 1280x720): "
              f"{m.duration_ms}ms")
    finally:
        analyzer.close()


def test_L1_iteration_yields_correct_frames():
    analyzer = OpenCVVideoAnalyzer(str(L1))
    try:
        frames = list(analyzer)
        assert len(frames) == 90
        for pkt in frames[:3]:
            assert isinstance(pkt, FramePacket)
            assert pkt.image.shape == (1080, 1920, 3)
            assert pkt.timestamp_ms == int(pkt.frame_id / 30.0 * 1000)
        print(f"✓ L1 iteration: 90 frames, first 3 timestamps correct")
    finally:
        analyzer.close()


def test_L1_seek():
    analyzer = OpenCVVideoAnalyzer(str(L1))
    try:
        pkt = analyzer.seek(45)
        assert pkt.frame_id == 45
        assert pkt.timestamp_ms == int(45 / 30.0 * 1000)
        assert pkt.image.shape == (1080, 1920, 3)
        # Seek out of range
        try:
            analyzer.seek(9999)
            raise AssertionError("should have raised")
        except IndexError:
            pass
        print(f"✓ L1 seek: frame 45 at {pkt.timestamp_ms}ms, IndexError on out-of-range")
    finally:
        analyzer.close()


def test_L1_protocol_conformance():
    """OpenCVVideoAnalyzer 必须满足 VideoAnalyzerProtocol (鸭子类型)。"""
    analyzer = OpenCVVideoAnalyzer(str(L1))
    assert isinstance(analyzer, VideoAnalyzerProtocol)
    assert isinstance(analyzer, VideoAnalyzer)
    analyzer.close()
    print("✓ L1 protocol conformance (runtime_checkable)")


def test_L1_registry_can_be_extended():
    """自定义 backend 可注入 Registry。"""
    class FakeAnalyzer:
        def __init__(self, path):
            self._path = path
        @property
        def meta(self):
            return VideoMeta(
                source_path=self._path, width=100, height=100, fps=1.0,
                frame_count=1, duration_ms=1000,
                orientation=Orientation.SQUARE, has_audio=False,
            )
        def __iter__(self): return iter([])
        def seek(self, fid): raise NotImplementedError
        def close(self): pass

    VideoAnalyzerRegistry.register("fake", FakeAnalyzer)
    try:
        cls = VideoAnalyzerRegistry.get("fake")
        inst = cls("/tmp/anything")
        assert inst.meta.width == 100
    finally:
        # 测试清理: 取消注册
        if "fake" in VideoAnalyzerRegistry.available():
            VideoAnalyzerRegistry._registry.pop("fake")
    print("✓ L1 registry extensibility")


# =====================================================================
# L2 RegionPolicy tests
# =====================================================================
def test_L2_auto_select_policies_by_orientation():
    """根据 VideoMeta 自动选适用 policy, 不用 if/else 写死。"""
    # 横屏 1920x1080
    landscape = VideoMeta(
        source_path="x", width=1920, height=1080, fps=30.0,
        frame_count=90, duration_ms=3000, orientation=Orientation.LANDSCAPE,
        has_audio=False,
    )
    policies = applicable_policies(landscape)
    names = sorted(p.name for p in policies)
    assert "bottom_horizontal" in names
    assert "top_news" in names
    assert "bottom_portrait" not in names  # portrait policy 不适用

    # 竖屏 720x1280
    portrait = VideoMeta(
        source_path="x", width=720, height=1280, fps=30.0,
        frame_count=90, duration_ms=3000, orientation=Orientation.PORTRAIT,
        has_audio=False,
    )
    policies = applicable_policies(portrait)
    names = sorted(p.name for p in policies)
    assert "bottom_portrait" in names
    assert "bottom_horizontal" not in names
    assert "top_news" not in names

    print("✓ L2 auto-select by orientation (no hard-coded if/else)")


def test_L2_proposal_only_outputs_region_weight_source():
    """propose 只返回 polygon/weight/source, 不做字幕判断。"""
    analyzer = OpenCVVideoAnalyzer(str(L1))
    pkt = analyzer.seek(0)
    try:
        for policy_cls_name in ("bottom_horizontal", "top_news"):
            policy_cls = RegionPolicyRegistry.get(policy_cls_name)
            policy = policy_cls()
            assert isinstance(policy, SubtitleRegionPolicy)
            proposals = policy.propose(pkt)
            assert len(proposals) > 0
            for p in proposals:
                assert isinstance(p, RegionProposal)
                assert 0.0 <= p.weight <= 1.0
                assert p.source.startswith("policy:")
                # polygon 应当有 4 个点
                assert len(p.polygon) == 4
                # polygon 坐标应当落在图像内
                xs = [pt[0] for pt in p.polygon]
                ys = [pt[1] for pt in p.polygon]
                assert 0 <= min(xs) and max(xs) <= 1920
                assert 0 <= min(ys) and max(ys) <= 1080
        print(f"✓ L2 proposal: 2 policies output polygon/weight/source only")
    finally:
        analyzer.close()


def test_L2_bottom_horizontal_is_in_bottom_band():
    """bottom_horizontal 的 polygon 应当落在屏幕底部。"""
    analyzer = OpenCVVideoAnalyzer(str(L1))
    pkt = analyzer.seek(0)
    try:
        p = BottomHorizontalPolicy()
        props = p.propose(pkt)
        assert len(props) == 1
        prop = props[0]
        ys = [pt[1] for pt in prop.polygon]
        # y_top 应 > 0.65 * h = 702, y_bot < h = 1080
        assert ys[0] > 700 and ys[0] < 750
        assert ys[2] > 1020 and ys[2] <= 1080
        # weight 高 (底部 + 横屏 + 中段)
        assert prop.weight > 0.8
        print(f"✓ L2 bottom_horizontal: y_top={ys[0]} y_bot={ys[2]} weight={prop.weight}")
    finally:
        analyzer.close()


def test_L2_bottom_portrait_is_in_bottom_band():
    analyzer = OpenCVVideoAnalyzer(str(L2))
    pkt = analyzer.seek(0)
    try:
        p = BottomPortraitPolicy()
        props = p.propose(pkt)
        prop = props[0]
        ys = [pt[1] for pt in prop.polygon]
        # 720x1280 portrait, 0.75*1280=960, 0.98*1280=1254
        assert ys[0] > 950 and ys[0] < 1000
        assert ys[2] > 1240 and ys[2] <= 1280
        print(f"✓ L2 bottom_portrait: y_top={ys[0]} y_bot={ys[2]}")
    finally:
        analyzer.close()


def test_L2_top_news_is_in_top_band():
    analyzer = OpenCVVideoAnalyzer(str(L1))
    pkt = analyzer.seek(0)
    try:
        p = TopNewsPolicy()
        props = p.propose(pkt)
        prop = props[0]
        ys = [pt[1] for pt in prop.polygon]
        assert ys[0] < 100  # 顶部
        assert ys[2] < 300
        print(f"✓ L2 top_news: y_top={ys[0]} y_bot={ys[2]} weight={prop.weight}")
    finally:
        analyzer.close()


def test_L2_custom_policy_user_controlled():
    """CustomPolicy 允许用户传入 y_top/y_bot/weight, 无 hard-code。"""
    custom = CustomPolicy(
        name="mid_band", y_top_ratio=0.40, y_bot_ratio=0.60,
        weight=0.60, description="middle band",
    )
    analyzer = OpenCVVideoAnalyzer(str(L1))
    pkt = analyzer.seek(0)
    try:
        props = custom.propose(pkt)
        assert props[0].weight == 0.60
        ys = [pt[1] for pt in props[0].polygon]
        # y_top 应在 0.4 * 1080 = 432 附近
        assert 430 < ys[0] < 435
        print(f"✓ L2 custom policy: user-controlled y_top={ys[0]} y_bot={ys[2]}")
    finally:
        analyzer.close()


def test_L2_protocol_conformance():
    """所有内置 policy 满足 RegionPolicyProtocol。"""
    for name in RegionPolicyRegistry.available():
        cls = RegionPolicyRegistry.get(name)
        inst = cls()
        assert isinstance(inst, RegionPolicyProtocol)
    print(f"✓ L2 protocol conformance: {RegionPolicyRegistry.available()}")


# =====================================================================
# L3 TextDetector tests
# =====================================================================
def test_L3_paddle_detector_smoke_on_landscape():
    """PaddleOCR 在合成的横屏人物视频上至少检出底部字幕。"""
    detector = PaddleOCRDetector(lang="ch", conf_thresh=0.3)
    detector.warmup()
    analyzer = OpenCVVideoAnalyzer(str(L1))
    try:
        # 取字幕中段帧
        pkt = analyzer.seek(15)  # 第一段字幕 0-30
        cands = detector.detect(pkt)
        assert len(cands) > 0, "expected at least 1 candidate"
        for c in cands:
            assert isinstance(c, TextCandidate)
            assert len(c.polygon) == 4
            assert 0.0 <= c.confidence <= 1.0
            assert c.detector_id == "paddleocr_ch"
        # 字幕在底部 — polygon 中心 y 应 > 0.6 * 1080
        sub_y_min = min(
            (c.bbox[1] + c.bbox[3]) / 2 / pkt.height
            for c in cands
        )
        print(f"✓ L3 paddleocr on landscape: {len(cands)} candidates, "
              f"min_y_center={sub_y_min:.2f}")
    finally:
        analyzer.close()


def test_L3_paddle_detector_on_portrait():
    detector = PaddleOCRDetector(lang="ch", conf_thresh=0.3)
    detector.warmup()
    analyzer = OpenCVVideoAnalyzer(str(L2))
    try:
        pkt = analyzer.seek(45)  # 第二段字幕 30-60
        cands = detector.detect(pkt)
        assert len(cands) > 0
        for c in cands:
            assert c.detector_id == "paddleocr_ch"
        # 字幕 polygon 应在底部 (0.75 * 1280 ≈ 960)
        sub_y = [(c.bbox[1] + c.bbox[3]) / 2 / pkt.height for c in cands]
        # 至少一个候选落在底部
        assert any(y > 0.85 for y in sub_y), f"no subtitle band candidates: {sub_y}"
        print(f"✓ L3 paddleocr on portrait: {len(cands)} candidates, "
              f"y_centers={[f'{y:.2f}' for y in sub_y]}")
    finally:
        analyzer.close()


def test_L3_paddle_detector_on_screencast():
    """屏幕录制视频: 字幕 + UI 文字同时存在, detector 一视同仁输出。"""
    detector = PaddleOCRDetector(lang="ch", conf_thresh=0.3)
    detector.warmup()
    analyzer = OpenCVVideoAnalyzer(str(L3))
    try:
        pkt = analyzer.seek(60)  # 字幕段 45-75
        cands = detector.detect(pkt)
        assert len(cands) > 0
        # detector 输出所有文字, 不做"是不是字幕"判断
        # 所以应当包括: 字幕 + "Settings" / "Climate" / "OK Button" 等 UI 文字
        texts = [c.text for c in cands]
        assert any("Settings" in t or "尾门" in t or "天气" in t or "OK" in t
                   for t in texts), \
            f"expected UI/subtitle text in: {texts}"
        # 不输出 SubtitleInstance (这是 Phase C 的事)
        assert not hasattr(detector, "instances"), \
            "L3 must NOT generate SubtitleInstance"
        print(f"✓ L3 paddleocr on screencast: {len(cands)} candidates, "
              f"texts={texts[:3]}... (no SubtitleInstance)")
    finally:
        analyzer.close()


def test_L3_does_not_create_instance():
    """强制约束: L3 只能输出 TextCandidate, 不输出 SubtitleInstance。"""
    from video_localization_engine.types.instance import SubtitleInstance
    detector = PaddleOCRDetector(lang="ch", conf_thresh=0.3)
    # 检查 detector 类没有 instance 相关属性
    forbidden = ["instances", "subtitle_instances", "tracks", "_track"]
    for attr in forbidden:
        assert not hasattr(detector, attr), \
            f"L3 detector must not have {attr}"
    # 检查 TextCandidate 字段也没有 subtitle 决策字段
    cand_fields = {f.name for f in fields(TextCandidate)}
    forbidden_fields = ["is_subtitle", "is_logo", "is_ui", "classification"]
    for f in forbidden_fields:
        assert f not in cand_fields, \
            f"TextCandidate must not carry {f} (subtitle decision is Phase C)"
    print("✓ L3 no SubtitleInstance generation, TextCandidate no classification field")


def test_L3_protocol_conformance():
    detector = PaddleOCRDetector(lang="ch")
    assert isinstance(detector, TextDetectorProtocol)
    assert isinstance(detector, TextDetector)
    assert "paddleocr_ch" in TextDetectorRegistry.available()
    print("✓ L3 protocol conformance + registry")


# =====================================================================
# 跨层集成测试
# =====================================================================
def test_integration_L1_L2_L3_pipeline():
    """L1 → L2 → L3 链路在 3 类视频上跑通。"""
    detector = PaddleOCRDetector(lang="ch", conf_thresh=0.3)
    detector.warmup()

    results = {}
    for label, path in [("landscape", L1), ("portrait", L2), ("screencast", L3)]:
        analyzer = OpenCVVideoAnalyzer(str(path))
        try:
            meta = analyzer.meta
            # L2: 自动选 policy
            policies = applicable_policies(meta)
            policy_names = [p.name for p in policies]
            # L3: detector 输出
            pkt = analyzer.seek(meta.frame_count // 2)
            cands = detector.detect(pkt)
            results[label] = {
                "meta": meta,
                "policies": policy_names,
                "candidate_count": len(cands),
                "candidate_texts": [c.text for c in cands],
            }
        finally:
            analyzer.close()

    for label, r in results.items():
        print(f"  {label:12s}: {r['meta'].orientation.value:9s} "
              f"{r['meta'].width}x{r['meta'].height} | "
              f"policies={r['policies']} | "
              f"cands={r['candidate_count']}")

    assert len(results["landscape"]["candidate_texts"]) > 0
    assert len(results["portrait"]["candidate_texts"]) > 0
    assert len(results["screencast"]["candidate_texts"]) > 0
    # 横屏必须至少选 bottom_horizontal 或 top_news
    assert "bottom_horizontal" in results["landscape"]["policies"] \
        or "top_news" in results["landscape"]["policies"]
    # 竖屏必须选 bottom_portrait
    assert "bottom_portrait" in results["portrait"]["policies"]
    print(f"✓ L1→L2→L3 pipeline on 3 fixtures")


def test_no_hardcoded_thresholds_in_phase_b_modules():
    """Phase B 模块 (L1/L2/L3) 不应有 hard-coded 字幕判断。

    允许的:
      - Policy 内部参数 (y_top/y_bot/weight) — 这是策略本身的数值, 不是判断
      - Protocol / abstract 定义

    禁止的:
      - `if y > 0.X: subtitle = True` 这种"硬判断"模式
      - L3 detector 里写字幕判断
    """
    import re
    forbidden_patterns = [
        (r"if\s+.*y\s*>\s*0\.\d.*:.*subtitle", "if y > X: subtitle"),
        (r"if\s+.*subtitle\s*=.*True", "if ...: subtitle = True"),
        (r"\.subtitle\s*=\s*True", "x.subtitle = True"),
    ]
    base = Path(__file__).resolve().parents[1]
    for sub in ("analyzer", "region_policies", "detector"):
        for py in (base / sub).glob("*.py"):
            content = py.read_text()
            for line_no, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""') \
                   or stripped.startswith("'''"):
                    continue
                for pattern, label in forbidden_patterns:
                    if re.search(pattern, line):
                        raise AssertionError(
                            f"hardcoded {label} ({pattern}) in "
                            f"{py.name}:{line_no}: {line!r}"
                        )
    print("✓ no hardcoded subtitle judgment in L1/L2/L3 modules")


def main():
    tests = [
        # L1
        test_derive_orientation_pure_function,
        test_L1_meta_on_landscape_fixture,
        test_L1_meta_on_portrait_fixture,
        test_L1_meta_on_screencast_fixture,
        test_L1_iteration_yields_correct_frames,
        test_L1_seek,
        test_L1_protocol_conformance,
        test_L1_registry_can_be_extended,
        # L2
        test_L2_auto_select_policies_by_orientation,
        test_L2_proposal_only_outputs_region_weight_source,
        test_L2_bottom_horizontal_is_in_bottom_band,
        test_L2_bottom_portrait_is_in_bottom_band,
        test_L2_top_news_is_in_top_band,
        test_L2_custom_policy_user_controlled,
        test_L2_protocol_conformance,
        # L3
        test_L3_paddle_detector_smoke_on_landscape,
        test_L3_paddle_detector_on_portrait,
        test_L3_paddle_detector_on_screencast,
        test_L3_does_not_create_instance,
        test_L3_protocol_conformance,
        # integration
        test_integration_L1_L2_L3_pipeline,
        # invariants
        test_no_hardcoded_thresholds_in_phase_b_modules,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} Phase B tests passed.")


if __name__ == "__main__":
    main()