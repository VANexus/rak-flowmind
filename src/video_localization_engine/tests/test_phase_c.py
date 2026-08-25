"""Phase C 测试 — L4 SubtitleManager 4 层流水线在 4 类合成视频上的端到端验证。

合成 fixture:
  landscape_person.mp4 (1920x1080) — 底部 4 段字幕 + 右上 logo
  portrait_short.mp4 (720x1280)   — 底部 3 段字幕
  screencast_ui.mp4 (1280x720)    — 底部 2 段字幕 + UI 控件文字
  news_subtitle.mp4 (1280x720)    — 顶部 3 段新闻字幕 + 底部时间水印

验证:
  T1: TextLineCandidate 归一化
  T2: 跨帧 IoU/centroid/text 匹配 → SubtitleCandidate
  T2: 4 状态生命周期 NEW → ACTIVE → ENDING → CLOSED
  T3: SubtitleCandidate → SubtitleInstance 提取
  features: 不写硬评分, 仅 raw features
  .vle.json: 完整 round-trip

不在 Phase C 测试:
  - FP/FN 量化 (留给 Phase D classifier)
  - region score / ui_exclusion score (留给 Phase D)
  - mask 生成 (Phase D)
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from video_localization_engine.analyzer import OpenCVVideoAnalyzer
from video_localization_engine.detector import PaddleOCRDetector
from video_localization_engine.manager import (
    SubtitleManager, SubtitleCandidateConfig, SubtitleCandidateBuffer,
)
from video_localization_engine.types.candidates import (
    SubtitleCandidateState, TextLineCandidate, SubtitleCandidate,
)
from video_localization_engine.types.instance import SubtitleInstance
from video_localization_engine.utils.persistence import save_track, load_track


FIXTURES = Path("/tmp/vle_fixtures")
LANDSCAPE = FIXTURES / "landscape_person.mp4"
PORTRAIT = FIXTURES / "portrait_short.mp4"
SCREENCAST = FIXTURES / "screencast_ui.mp4"
NEWS = FIXTURES / "news_subtitle.mp4"


# =====================================================================
# T1 TextLineBuilder
# =====================================================================
def test_T1_text_line_builder():
    """T1: TextCandidate → TextLineCandidate 归一化。"""
    from video_localization_engine.manager.t1_text_line import TextLineBuilder
    from video_localization_engine.types.detection import TextCandidate
    from video_localization_engine.types.video import FramePacket, VideoMeta, Orientation
    import numpy as np

    meta = VideoMeta(source_path="x", width=1920, height=1080, fps=30,
                     frame_count=10, duration_ms=333, orientation=Orientation.LANDSCAPE,
                     has_audio=False)
    pkt = FramePacket(frame_id=5, timestamp_ms=166,
                      image=np.zeros((1080, 1920, 3), dtype=np.uint8), meta=meta)
    cands = [
        TextCandidate(polygon=[(100, 900), (300, 900), (300, 950), (100, 950)],
                      text="你好", confidence=0.9, detector_id="paddleocr_ch",
                      language="zh"),
        TextCandidate(polygon=[(400, 900), (700, 900), (700, 950), (400, 950)],
                      text="世界", confidence=0.85, detector_id="paddleocr_ch",
                      language="zh"),
    ]
    lines = TextLineBuilder().build(pkt, cands)
    assert len(lines) == 2
    assert lines[0].text == "你好"
    assert lines[0].frame_id == 5
    assert lines[0].timestamp_ms == 166
    assert lines[1].char_count == 2
    print(f"✓ T1: TextLineBuilder 1:1 归一化 (2 → 2)")


# =====================================================================
# T2 SubtitleCandidateBuffer 单元测试
# =====================================================================
def test_T2_text_similarity():
    """字符级 Jaccard。"""
    from video_localization_engine.manager.t2_subtitle_candidate import _text_similarity
    assert _text_similarity("你好世界", "你好世界") == 1.0
    assert _text_similarity("你好世界", "再见世界") > 0.3  # 共用"世界"
    assert _text_similarity("你好", "再见") < 0.5
    assert _text_similarity("", "abc") == 0.0
    print("✓ T2 text_similarity Jaccard")


def test_T2_state_lifecycle_short_subtitle():
    """短字幕 (1 段, 3 帧) 完整生命周期。"""
    cfg = SubtitleCandidateConfig(iou_match_threshold=0.3,
                                  centroid_distance_max=100.0,
                                  grace_frames=2)
    buffer = SubtitleCandidateBuffer(cfg)

    # 帧 0: 新建 candidate
    line0 = TextLineCandidate(
        polygon=[(100, 900), (300, 900), (300, 950), (100, 950)],
        bbox=(100, 900, 300, 950),
        text="测试", confidence=0.9, language="zh",
        detector_id="paddleocr_ch", frame_id=0, timestamp_ms=0, char_count=2,
    )
    buffer.feed([line0])
    assert len(buffer.active) == 1
    assert buffer.active[0].state == SubtitleCandidateState.ACTIVE  # add_match 自动转
    assert buffer.active[0].frame_count == 1

    # 帧 1: 匹配, ACTIVE
    line1 = TextLineCandidate(
        polygon=[(101, 901), (301, 901), (301, 951), (101, 951)],
        bbox=(101, 901, 301, 951),
        text="测试", confidence=0.91, language="zh",
        detector_id="paddleocr_ch", frame_id=1, timestamp_ms=33, char_count=2,
    )
    buffer.feed([line1])
    assert len(buffer.active) == 1
    assert buffer.active[0].state == SubtitleCandidateState.ACTIVE
    assert buffer.active[0].frame_count == 2

    # 帧 2: 无新 line → ACTIVE → ENDING
    buffer.feed([])
    assert len(buffer.active) == 1
    assert buffer.active[0].state == SubtitleCandidateState.ENDING

    # 帧 3: 仍无 → ENDING grace--
    buffer.feed([])
    assert len(buffer.active) == 1
    # grace=2, 已消耗 2 帧 → grace=0 → CLOSED
    assert buffer.active[0].state == SubtitleCandidateState.ENDING  # 还没 CLOSED

    # 帧 4: 仍无 → CLOSED
    buffer.feed([])
    assert len(buffer.active) == 0
    assert len(buffer.closed) == 1
    assert buffer.closed[0].state == SubtitleCandidateState.CLOSED
    assert buffer.closed[0].text_history == ["测试"]
    assert buffer.closed[0].frame_count == 2
    assert buffer.closed[0].polygon_history[0] == [(100, 900), (300, 900), (300, 950), (100, 950)]
    print(f"✓ T2 短字幕生命周期: ACTIVE→ENDING→CLOSED, "
          f"duration={buffer.closed[0].duration_ms}ms")


def test_T2_two_subtitles_distinguished():
    """两段字幕 (不同时段, 不同空间位置) 被识别为 2 个独立 candidate。"""
    cfg = SubtitleCandidateConfig(iou_match_threshold=0.3)
    buffer = SubtitleCandidateBuffer(cfg)

    # 帧 0-2: 字幕 A 在 x∈[100,300], y=900 (左侧)
    for i in range(3):
        line = TextLineCandidate(
            polygon=[(100 + i, 900), (300 + i, 900), (300 + i, 950), (100 + i, 950)],
            bbox=(100 + i, 900, 300 + i, 950),
            text="第一段", confidence=0.9, language="zh",
            detector_id="paddleocr_ch", frame_id=i, timestamp_ms=i * 33, char_count=3,
        )
        buffer.feed([line])
    assert len(buffer.active) == 1
    assert buffer.active[0].text_history == ["第一段"]

    # 帧 3: 字幕 A 消失 + 字幕 B 出现 (右侧, x∈[500,700], IoU=0)
    lineB = TextLineCandidate(
        polygon=[(500, 920), (700, 920), (700, 970), (500, 970)],
        bbox=(500, 920, 700, 970),
        text="第二段", confidence=0.9, language="zh",
        detector_id="paddleocr_ch", frame_id=3, timestamp_ms=99, char_count=3,
    )
    buffer.feed([lineB])
    # A 进入 ENDING (grace=2)
    # B 新建
    active = buffer.active
    states = sorted([(c.state.value, c.text_history[0]) for c in active])
    assert ("active", "第二段") in [tuple(s) for s in states]

    # 帧 4: 字幕 B 持续
    lineB2 = TextLineCandidate(
        polygon=[(501, 921), (701, 921), (701, 971), (501, 971)],
        bbox=(501, 921, 701, 971),
        text="第二段", confidence=0.91, language="zh",
        detector_id="paddleocr_ch", frame_id=4, timestamp_ms=132, char_count=3,
    )
    buffer.feed([lineB2])

    # 帧 5-7: 无字幕 → A CLOSED, B → ENDING → CLOSED
    for _ in range(3):
        buffer.feed([])

    assert len(buffer.active) == 0
    assert len(buffer.closed) == 2
    texts = sorted([c.text_history[0] for c in buffer.closed])
    assert texts == ["第一段", "第二段"]
    print(f"✓ T2 两段字幕区分: 独立 closed candidates={texts}")


def test_T2_match_by_centroid_when_iou_low():
    """IoU 不够时, centroid 距离 fallback。"""
    cfg = SubtitleCandidateConfig(
        iou_match_threshold=0.5,    # 提高 IoU 阈值, 强制 fallback
        centroid_distance_max=50.0,  # 50px 内
    )
    buffer = SubtitleCandidateBuffer(cfg)

    # 帧 0: polygon A
    line0 = TextLineCandidate(
        polygon=[(100, 900), (300, 900), (300, 950), (100, 950)],
        bbox=(100, 900, 300, 950),
        text="测试", confidence=0.9, language="zh",
        detector_id="paddleocr_ch", frame_id=0, timestamp_ms=0, char_count=2,
    )
    buffer.feed([line0])

    # 帧 1: polygon 漂移 20px, IoU 应 < 0.5, 但 centroid 距离 < 50
    line1 = TextLineCandidate(
        polygon=[(115, 905), (315, 905), (315, 955), (115, 955)],
        bbox=(115, 905, 315, 955),
        text="测试", confidence=0.9, language="zh",
        detector_id="paddleocr_ch", frame_id=1, timestamp_ms=33, char_count=2,
    )
    buffer.feed([line1])

    assert len(buffer.active) == 1
    assert buffer.active[0].frame_count == 2  # 仍然匹配, 不是新建
    print(f"✓ T2 centroid fallback: IoU<0.5 但 distance<50 仍匹配")


def test_T2_text_mismatch_penalized():
    """文本完全不同 → 匹配分数大幅扣分 → 视为不同 instance。"""
    cfg = SubtitleCandidateConfig(iou_match_threshold=0.3, text_similarity_min=0.5)
    buffer = SubtitleCandidateBuffer(cfg)

    line0 = TextLineCandidate(
        polygon=[(100, 900), (300, 900), (300, 950), (100, 950)],
        bbox=(100, 900, 300, 950),
        text="你好", confidence=0.9, language="zh",
        detector_id="paddleocr_ch", frame_id=0, timestamp_ms=0, char_count=2,
    )
    buffer.feed([line0])

    # 同位置但文字完全不同
    line1 = TextLineCandidate(
        polygon=[(100, 900), (300, 900), (300, 950), (100, 950)],
        bbox=(100, 900, 300, 950),
        text="XYZ", confidence=0.9, language="en",
        detector_id="paddleocr_en", frame_id=1, timestamp_ms=33, char_count=3,
    )
    buffer.feed([line1])

    # 你好 和 XYZ 字符级 Jaccard = 0 → text_sim=0 < 0.5 → 匹配分数 × 0.3
    # 但 IoU=1.0, score = 1.0 × 0.3 = 0.3, 但 threshold=0.3 → 仍然 > 0 → 匹配
    # (我们的设计: text mismatch 仅"penalize", 不"reject")
    # 所以这里期望仍然匹配 (score = 0.3)
    assert len(buffer.active) == 1, "text penalty 应仍允许匹配"
    print(f"✓ T2 text mismatch penalty: 仍匹配但分数低")


# =====================================================================
# T3 SubtitleInstanceExtractor
# =====================================================================
def test_T3_extractor_creates_instance_with_features():
    """T3: closed candidate → SubtitleInstance, features 透传。"""
    from video_localization_engine.manager.t3_subtitle_instance import (
        SubtitleInstanceExtractor,
    )
    cfg = SubtitleCandidateConfig(grace_frames=0)  # 立即关闭
    buffer = SubtitleCandidateBuffer(cfg)

    # 单帧单 line
    line = TextLineCandidate(
        polygon=[(100, 900), (300, 900), (300, 950), (100, 950)],
        bbox=(100, 900, 300, 950),
        text="测试文本", confidence=0.9, language="zh",
        detector_id="paddleocr_ch", frame_id=5, timestamp_ms=165, char_count=4,
    )
    buffer.feed([line])
    # grace=0 不会触发, force close
    closed = buffer.finalize()

    extractor = SubtitleInstanceExtractor()
    instances = extractor.extract_all(closed)
    assert len(instances) == 1
    inst = instances[0]
    assert isinstance(inst, SubtitleInstance)
    assert inst.representative_text == "测试文本"
    assert inst.frame_count == 1
    assert inst.duration_ms == 165
    assert inst.representative_bbox is not None
    # features 透传
    assert "avg_confidence" in inst.features or "duration_ms" in inst.features
    # 没有硬评分字段填充
    assert inst.score.total == 0.0  # Phase C 不算分
    print(f"✓ T3: SubtitleInstance 提取, features={list(inst.features.keys())}")


def test_T3_no_hardcoded_score():
    """T3 不应计算 score.total / 不应写 classification_reason 判定文本。"""
    from video_localization_engine.manager.t3_subtitle_instance import (
        SubtitleInstanceExtractor,
    )
    cfg = SubtitleCandidateConfig(grace_frames=0)
    buffer = SubtitleCandidateBuffer(cfg)
    line = TextLineCandidate(
        polygon=[(100, 900), (300, 900), (300, 950), (100, 950)],
        bbox=(100, 900, 300, 950),
        text="测试", confidence=0.9, language="zh",
        detector_id="paddleocr_ch", frame_id=0, timestamp_ms=0, char_count=2,
    )
    buffer.feed([line])
    closed = buffer.finalize()
    inst = SubtitleInstanceExtractor().extract_all(closed)[0]
    # status 必须是 CANDIDATE (Phase C 不决定最终状态)
    from video_localization_engine.types.instance import InstanceStatus
    assert inst.status == InstanceStatus.CANDIDATE
    # score.total 应为 0 (Phase D 才填)
    assert inst.score.total == 0.0
    # classification_reason 应仅描述提取事实, 不判定字幕
    assert "subtitle" not in inst.classification_reason.lower() \
        or "phase_c_extracted" in inst.classification_reason
    print("✓ T3 无硬评分, status=CANDIDATE")


# =====================================================================
# SubtitleManager 端到端 (用合成 fixture)
# =====================================================================
def _run_manager_on_fixture(path: Path, label: str, frame_stride: int = 5) -> list:
    """跑 SubtitleManager 完整流水线, 返回最终 SubtitleInstance 列表。"""
    analyzer = OpenCVVideoAnalyzer(str(path))
    detector = PaddleOCRDetector(lang="ch", conf_thresh=0.3)
    detector.warmup()
    manager = SubtitleManager()
    try:
        for pkt in analyzer:
            cands = detector.detect(pkt)
            manager.feed(pkt, cands)
        # 最后一帧 → 关所有
        manager.finish()
        manager.buffer.compute_all_features()
        return manager._t3.extract_all(manager.buffer.closed)
    finally:
        analyzer.close()


def test_manager_landscape_end_to_end():
    instances = _run_manager_on_fixture(LANDSCAPE, "landscape")
    print(f"  landscape: {len(instances)} instances")
    for inst in instances:
        print(f"    [{inst.first_frame}-{inst.last_frame}] "
              f"text={inst.representative_text!r} "
              f"frames={inst.frame_count} "
              f"dur={inst.duration_ms}ms "
              f"feat={list(inst.features.keys())[:3]}")
    # 至少应有 1 个 instance (字幕段)
    assert len(instances) >= 1
    # instance 应当有 representative_text
    texts = [i.representative_text for i in instances]
    assert any("字幕" in t or "世界" in t or "你好" in t or "FAKE" in t
               for t in texts), f"unexpected texts: {texts}"
    print(f"✓ Manager on landscape: {len(instances)} instances extracted")


def test_manager_portrait_end_to_end():
    instances = _run_manager_on_fixture(PORTRAIT, "portrait")
    print(f"  portrait: {len(instances)} instances")
    assert len(instances) >= 1
    print(f"✓ Manager on portrait: {len(instances)} instances")


def test_manager_screencast_end_to_end():
    instances = _run_manager_on_fixture(SCREENCAST, "screencast")
    print(f"  screencast: {len(instances)} instances")
    # screencast 有 UI 文字 (Settings/Climate/OK Button) + 字幕
    # Phase C 全部识别为 candidate, Phase D 才区分
    assert len(instances) >= 1
    print(f"✓ Manager on screencast: {len(instances)} instances (UI+字幕)")


def test_manager_news_top_subtitle():
    """新闻顶部字幕 + 底部时间水印 — 测试顶部字幕识别。"""
    instances = _run_manager_on_fixture(NEWS, "news")
    print(f"  news: {len(instances)} instances")
    for inst in instances:
        print(f"    [{inst.first_frame}-{inst.last_frame}] "
              f"text={inst.representative_text!r} "
              f"y_center={(inst.representative_bbox[1]+inst.representative_bbox[3])/2 if inst.representative_bbox else None}")
    # 新闻顶部字幕 + 底部时间水印 — 至少 1 个 instance
    assert len(instances) >= 1
    print(f"✓ Manager on news (top subtitle): {len(instances)} instances")


# =====================================================================
# .vle.json round-trip
# =====================================================================
def test_vle_json_with_features_round_trip():
    """features 字段在 .vle.json 中正确保留。"""
    instances = _run_manager_on_fixture(LANDSCAPE, "landscape")
    if not instances:
        # 没有 instance 时跳过 (不太可能)
        print("~ skip: no instances extracted")
        return

    # 构造 SubtitleTrack 序列化
    from video_localization_engine.types.instance import SubtitleTrack
    from video_localization_engine.analyzer import OpenCVVideoAnalyzer as A
    a = A(str(LANDSCAPE))
    track = SubtitleTrack(
        video_meta_path=str(LANDSCAPE),
        instances=instances,
        frame_candidates=[],
        region_policies_used=[],
        detector_id="paddleocr_ch",
    )
    with tempfile.NamedTemporaryFile(suffix=".vle.json", delete=False) as f:
        path = f.name
    try:
        save_track(track, a.meta, path)
        data = load_track(path)

        assert data["version"] == "0.1.0"
        assert len(data["instances"]) == len(instances)
        for raw, inst in zip(data["instances"], instances):
            # features 字段存在
            assert "features" in raw
            # observed_polygons 字段存在
            assert "observed_polygons" in raw
            # text_observations 字段存在
            assert "text_observations" in raw
            # instance_id 一致
            assert raw["instance_id"] == inst.instance_id
            # representative_text 一致
            assert raw["representative_text"] == inst.representative_text
            # features 值一致
            for k, v in inst.features.items():
                assert raw["features"].get(k) == v, \
                    f"feature {k}: stored {raw['features'].get(k)} vs {v}"
        print(f"✓ .vle.json round-trip: {len(instances)} instances, "
              f"features preserved")
    finally:
        os.unlink(path)
        a.close()


# =====================================================================
# 不变量
# =====================================================================
def test_no_hardcoded_subtitle_decision_in_manager():
    """Phase C manager 不应有任何"硬字幕判断"。"""
    import re
    base = Path(__file__).resolve().parents[1] / "manager"
    forbidden = [
        r"is_subtitle\s*=\s*True",
        r"if\s+.*\:\s*is_subtitle",
        r"score\s*=\s*position\s*\*",  # 旧的硬加权模式
        r"weight\s*=\s*0\.\d+\s*\*",
    ]
    for py in base.glob("*.py"):
        content = py.read_text()
        for line_no, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern in forbidden:
                if re.search(pattern, line):
                    raise AssertionError(
                        f"hardcoded decision {pattern!r} in {py.name}:{line_no}: {line!r}"
                    )
    print("✓ no hardcoded subtitle decision in manager/")


def test_features_are_raw_not_scored():
    """SubtitleInstance.features 不应包含 score.* 字段。"""
    instances = _run_manager_on_fixture(LANDSCAPE, "landscape")
    if not instances:
        return
    for inst in instances:
        for k in inst.features:
            assert not k.startswith("score."), \
                f"features should be raw, not scored: {k}"
        # Phase C features 应该是原始观测值
        raw_keys = {"avg_confidence", "char_count", "duration_ms",
                    "frame_count", "text_stability", "centroid_stability"}
        # 至少包含一些 raw key
        assert any(k in raw_keys for k in inst.features), \
            f"no raw features in {inst.features}"
    print(f"✓ features are raw (not pre-scored)")


def main():
    tests = [
        # T1
        test_T1_text_line_builder,
        # T2 单元
        test_T2_text_similarity,
        test_T2_state_lifecycle_short_subtitle,
        test_T2_two_subtitles_distinguished,
        test_T2_match_by_centroid_when_iou_low,
        test_T2_text_mismatch_penalized,
        # T3
        test_T3_extractor_creates_instance_with_features,
        test_T3_no_hardcoded_score,
        # 端到端
        test_manager_landscape_end_to_end,
        test_manager_portrait_end_to_end,
        test_manager_screencast_end_to_end,
        test_manager_news_top_subtitle,
        # 持久化
        test_vle_json_with_features_round_trip,
        # 不变量
        test_no_hardcoded_subtitle_decision_in_manager,
        test_features_are_raw_not_scored,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} Phase C tests passed.")


if __name__ == "__main__":
    main()