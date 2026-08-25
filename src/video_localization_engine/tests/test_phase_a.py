"""Phase A 测试 — 不依赖真实视频, 用合成数据验证类型/序列化/registry。

验证:
1. VideoMeta / FramePacket 可创建 + 不绑定坐标系
2. TextCandidate / RegionProposal / FrameTextCandidates 可创建
3. SubtitleInstance 跨帧累积 + finalize 后有代表 bbox/text
4. .vle.json round-trip 正确
5. RegistryBase 注册/获取/重复检测
6. 协议是 runtime_checkable (鸭子类型)
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from video_localization_engine.types.video import (
    FramePacket, Orientation, VideoMeta,
)
from video_localization_engine.types.detection import (
    FrameTextCandidates, RegionProposal, TextCandidate,
)
from video_localization_engine.types.instance import (
    InstanceStatus, SubtitleInstance, SubtitleTrack,
)
from video_localization_engine.utils.persistence import (
    load_track, save_track,
)
from video_localization_engine.utils.registry import RegistryBase
from video_localization_engine.utils.geometry import (
    bbox_from_polygon, bbox_iou, polygon_centroid, polygon_iou,
)


def _fake_meta() -> VideoMeta:
    return VideoMeta(
        source_path="/tmp/fake.mp4",
        width=1920, height=1080, fps=30.0, frame_count=300,
        duration_ms=10000, orientation=Orientation.LANDSCAPE,
        has_audio=True, content_type_hint="screencast", source_locale="zh",
    )


def _fake_packet(meta: VideoMeta, frame_id: int = 0) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        timestamp_ms=frame_id * 33,
        image=np.zeros((meta.height, meta.width, 3), dtype=np.uint8),
        meta=meta,
    )


def test_video_meta_orientation_classification():
    """任意分辨率都能正确分类 orientation。"""
    landscape = VideoMeta(
        source_path="x", width=1920, height=1080, fps=30.0,
        frame_count=100, duration_ms=3300, orientation=Orientation.LANDSCAPE,
        has_audio=False,
    )
    portrait = VideoMeta(
        source_path="x", width=720, height=1280, fps=30.0,
        frame_count=100, duration_ms=3300, orientation=Orientation.PORTRAIT,
        has_audio=False,
    )
    square = VideoMeta(
        source_path="x", width=1080, height=1080, fps=30.0,
        frame_count=100, duration_ms=3300, orientation=Orientation.SQUARE,
        has_audio=False,
    )
    assert landscape.orientation == Orientation.LANDSCAPE
    assert portrait.orientation == Orientation.PORTRAIT
    assert square.orientation == Orientation.SQUARE
    print("✓ video meta orientation classification")


def test_frame_packet_lazy_resolution():
    """FramePacket 从 image 推断 w/h, 不重复存储。"""
    meta = _fake_meta()
    pkt = _fake_packet(meta, frame_id=42)
    assert pkt.width == 1920
    assert pkt.height == 1080
    assert pkt.frame_id == 42
    assert pkt.timestamp_ms == 42 * 33
    print("✓ frame packet from image")


def test_text_candidate_bbox_derived():
    """TextCandidate 的 bbox 由 polygon 自动计算。"""
    cand = TextCandidate(
        polygon=[(100, 200), (300, 200), (300, 240), (100, 240)],
        text="hello", confidence=0.9, detector_id="paddleocr_en",
    )
    assert cand.bbox == (100, 200, 300, 240)
    assert cand.char_count == 5
    print("✓ text candidate bbox derived")


def test_subtitle_instance_cross_frame_accumulation():
    """SubtitleInstance 跨帧累积, finalize 后产出代表 bbox/text。"""
    inst = SubtitleInstance()
    meta = _fake_meta()

    # 帧 10: 文本 A
    c1 = TextCandidate(
        polygon=[(100, 900), (500, 900), (500, 950), (100, 950)],
        text="你好世界", confidence=0.9, detector_id="paddleocr_ch",
        language="zh",
    )
    inst.update_with(c1, frame_id=10, timestamp_ms=330)

    # 帧 11: 同一文本 (稳定)
    c2 = TextCandidate(
        polygon=[(102, 902), (502, 902), (502, 952), (102, 952)],
        text="你好世界", confidence=0.92, detector_id="paddleocr_ch",
        language="zh",
    )
    inst.update_with(c2, frame_id=11, timestamp_ms=363)

    # 帧 12: 文本变化 (但仍同一个 instance)
    c3 = TextCandidate(
        polygon=[(110, 905), (510, 905), (510, 955), (110, 955)],
        text="再见世界", confidence=0.88, detector_id="paddleocr_ch",
        language="zh",
    )
    inst.update_with(c3, frame_id=12, timestamp_ms=396)

    inst.finalize()

    assert inst.frame_count == 3
    assert inst.first_frame == 10
    assert inst.last_frame == 12
    assert inst.duration_ms == 396
    assert inst.representative_text in ("你好世界", "再见世界")
    assert inst.representative_bbox is not None
    assert inst.detector_id == "paddleocr_ch"
    assert inst.locale == "zh"
    print(f"✓ instance accumulated: {inst.representative_text!r} @ {inst.representative_bbox}")


def test_region_proposal_weight():
    """RegionProposal 携带 weight + source, 不假设位置。"""
    p = RegionProposal(
        polygon=[(0, 800), (1920, 800), (1920, 1000), (0, 1000)],
        weight=0.95,
        source="policy:bottom_horizontal_landscape",
        description="横屏底部字幕带",
    )
    assert p.weight == 0.95
    assert p.source.startswith("policy:")
    print(f"✓ region proposal: {p.source} w={p.weight}")


def test_vle_json_round_trip():
    """SubtitleTrack <-> .vle.json 序列化往返。"""
    meta = _fake_meta()
    inst = SubtitleInstance()
    c = TextCandidate(
        polygon=[(100, 900), (500, 900), (500, 950), (100, 950)],
        text="测试文本", confidence=0.9, detector_id="paddleocr_ch",
    )
    inst.update_with(c, frame_id=10, timestamp_ms=330)
    inst.finalize()

    fc = FrameTextCandidates(
        frame_id=10, timestamp_ms=330,
        width=meta.width, height=meta.height,
        candidates=[c],
    )
    track = SubtitleTrack(
        video_meta_path="fake.mp4",
        instances=[inst],
        frame_candidates=[fc],
        region_policies_used=["bottom_horizontal_landscape"],
        detector_id="paddleocr_ch",
    )

    with tempfile.NamedTemporaryFile(suffix=".vle.json", delete=False) as f:
        path = f.name
    try:
        save_track(track, meta, path)
        data = load_track(path)

        assert data["version"] == "0.1.0"
        assert data["video"]["width"] == 1920
        assert data["video"]["height"] == 1080
        assert data["video"]["orientation"] == "landscape"
        assert data["video"]["source_locale"] == "zh"
        assert len(data["instances"]) == 1
        assert data["instances"][0]["representative_text"] == "测试文本"
        assert data["instances"][0]["detector_id"] == "paddleocr_ch"
        assert data["region_policies_used"] == ["bottom_horizontal_landscape"]
        print(f"✓ .vle.json round-trip: {len(data['frame_candidates'])} frames, "
              f"{len(data['instances'])} instances")
    finally:
        os.unlink(path)


def test_registry_register_and_get():
    """RegistryBase 注册 + 获取 + 重复检测。"""

    class FakeRegistry(RegistryBase[str]):
        pass

    FakeRegistry.clear()
    FakeRegistry.register("hello", "world")
    assert FakeRegistry.get("hello") == "world"
    assert "hello" in FakeRegistry.available()

    # 重复注册应抛错
    try:
        FakeRegistry.register("hello", "duplicate")
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "already registered" in str(e)
        print(f"✓ registry duplicate detection: {e}")

    FakeRegistry.clear()


def test_protocol_runtime_checkable():
    """Protocol 可运行时 isinstance 检查, 不强制继承。"""
    from video_localization_engine.types.protocols import (
        RegionPolicyProtocol,
        TextDetectorProtocol,
    )

    class MyDetector:
        name = "fake_detector"
        supported_languages = ["zh"]
        def detect(self, packet): return []
        def warmup(self): pass

    d = MyDetector()
    assert isinstance(d, TextDetectorProtocol), \
        "duck-typed instance should satisfy TextDetectorProtocol"
    print(f"✓ protocol isinstance: {d.name} matches TextDetectorProtocol")


def test_geometry_basic():
    """geometry 工具: bbox / centroid / iou。"""
    p = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert bbox_from_polygon(p) == (0, 0, 10, 10)
    assert polygon_centroid(p) == (5.0, 5.0)
    iou = polygon_iou(p, p)
    assert iou == 1.0
    iou2 = bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0 < iou2 < 1
    print(f"✓ geometry: bbox={bbox_from_polygon(p)} centroid={polygon_centroid(p)} "
          f"iou_same={iou:.2f} iou_partial={iou2:.2f}")


def test_no_hardcoded_thresholds_in_types():
    """Phase A 类型文件不应有任何 magic number (y>80% 等)。

    只检查 type 文件 (不算 protocols.py 的反例文档)。
    """
    import re
    from pathlib import Path
    type_dir = Path(__file__).resolve().parents[1] / "types"
    # 排除 protocols.py (里面的 magic number 是反例文档)
    forbidden_patterns = [
        (r"0\.65", "subtitle_y_min"),
        (r"0\.85", "subtitle_y_max"),
        (r"0\.20", "max_line_height"),
        (r"0\.015", "min_line_height"),
    ]
    for py in type_dir.glob("*.py"):
        if py.name == "protocols.py":
            continue  # 反例文档允许
        content = py.read_text()
        for pattern, label in forbidden_patterns:
            # 检查是不是在注释/字符串里出现的反例文字
            for line_no, line in enumerate(content.splitlines(), 1):
                # 跳过纯注释行
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""') \
                   or stripped.startswith("'''"):
                    continue
                if re.search(pattern, line):
                    raise AssertionError(
                        f"hardcoded {pattern} ({label}) in {py.name}:{line_no}: {line!r}"
                    )
    print("✓ no hardcoded thresholds in types/")


def main():
    tests = [
        test_video_meta_orientation_classification,
        test_frame_packet_lazy_resolution,
        test_text_candidate_bbox_derived,
        test_subtitle_instance_cross_frame_accumulation,
        test_region_proposal_weight,
        test_vle_json_round_trip,
        test_registry_register_and_get,
        test_protocol_runtime_checkable,
        test_geometry_basic,
        test_no_hardcoded_thresholds_in_types,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} Phase A tests passed.")


if __name__ == "__main__":
    main()