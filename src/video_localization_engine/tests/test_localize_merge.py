"""P10 step 3 测试 — _merge_overlapping_instances_by_fps 行为。

覆盖:
  - 同时间窗 (overlap ≥ 50%) 的 instance 合并为一个 union instance
  - bbox union (外接矩形), 文本按 y 排序后空格 join
  - 不同时间窗不合并 (overlap < 50%) — 各 instance 独立保留
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from video_localization_engine.orchestrator import VideoLocalizationPipeline
from video_localization_engine.types.instance import SubtitleInstance


def _make_inst(iid: str, first_frame: int, duration_ms: int,
               bbox, text: str, locale: str = "zh") -> SubtitleInstance:
    """构造测试 instance — 不走 update_with / finalize, 直接 set 字段。"""
    inst = SubtitleInstance(instance_id=iid)
    inst.first_frame = first_frame
    inst.last_frame = first_frame + int(duration_ms / 1000 * 30)
    inst.frame_count = 1
    inst.duration_ms = duration_ms
    inst.representative_text = text
    inst.representative_bbox = bbox
    inst.locale = locale
    return inst


def test_localize_merge_overlapping_instances():
    """fps=30, 同时间窗 (start=0, dur=2000ms) 的 2 个 instance 应该合并为 1 个。"""
    # 2 个 instance 都在 [0, 2000ms); 100% 重叠
    a = _make_inst("a", first_frame=0, duration_ms=2000,
                   bbox=(50, 100, 350, 140), text="代步工具")
    b = _make_inst("b", first_frame=0, duration_ms=2000,
                   bbox=(60, 200, 360, 240), text="通勤")

    # 调用 static 路径 — 实例化 pipeline 麻烦; 我们直接构造 stub 跑 method
    class _Stub:
        _merge_overlapping_instances_by_fps = (
            VideoLocalizationPipeline._merge_overlapping_instances_by_fps
        )
    merged = _Stub()._merge_overlapping_instances_by_fps([a, b], fps=30.0)

    assert len(merged) == 1, f"应合并为 1 个 instance, got {len(merged)}"
    m = merged[0]
    # bbox union: x1=min(50,60)=50, y1=min(100,200)=100
    #             x2=max(350,360)=360, y2=max(140,240)=240
    assert m.representative_bbox == (50, 100, 360, 240), (
        f"bbox union 错: got {m.representative_bbox}"
    )
    # text 按 y 排序: a.y=100 < b.y=200 → a 在上 → "代步工具 通勤"
    assert m.representative_text == "代步工具 通勤", (
        f"text join 错 (按 y 排序): got {m.representative_text!r}"
    )
    assert m.instance_id.startswith("merge-"), (
        f"应标注 merge- 前缀: got {m.instance_id}"
    )
    print(
        f"✓ merge overlapping: bbox={m.representative_bbox}, text={m.representative_text!r}"
    )


def test_localize_preserve_separate_scenes():
    """两个不重叠时间窗的 instance 应该各自保留, 不合并。"""
    # a: [0, 2000ms) (frames 0~60), b: [3000ms, 5000ms) (frames 90~150)
    # overlap = 0, ratio = 0/2000 = 0 < 0.5 → 不合并
    a = _make_inst("a", first_frame=0, duration_ms=2000,
                   bbox=(50, 100, 350, 140), text="字幕A")
    b = _make_inst("b", first_frame=90, duration_ms=2000,
                   bbox=(50, 100, 350, 140), text="字幕B")

    class _Stub:
        _merge_overlapping_instances_by_fps = (
            VideoLocalizationPipeline._merge_overlapping_instances_by_fps
        )
    merged = _Stub()._merge_overlapping_instances_by_fps([a, b], fps=30.0)

    assert len(merged) == 2, f"应保留 2 个独立 instance, got {len(merged)}"
    texts = sorted([m.representative_text for m in merged])
    assert texts == ["字幕A", "字幕B"], f"text 应原样保留: got {texts}"
    # 不应有 merge- 前缀
    for m in merged:
        assert not m.instance_id.startswith("merge-"), (
            f"不应合并: {m.instance_id}"
        )
    print(f"✓ preserve separate scenes: {len(merged)} instances kept")


def test_localize_merge_partial_overlap_above_threshold():
    """overlap = min 50% 应触发合并 (greedy cluster 跟 cluster 末尾比)。"""
    # a: [0, 2000ms), b: [1000, 3000ms) → overlap = 1000ms / 2000 = 50% → 合并
    a = _make_inst("a", first_frame=0, duration_ms=2000,
                   bbox=(50, 100, 350, 140), text="上")
    b = _make_inst("b", first_frame=30, duration_ms=2000,
                   bbox=(50, 200, 350, 240), text="下")

    class _Stub:
        _merge_overlapping_instances_by_fps = (
            VideoLocalizationPipeline._merge_overlapping_instances_by_fps
        )
    merged = _Stub()._merge_overlapping_instances_by_fps([a, b], fps=30.0)

    assert len(merged) == 1, f"50% overlap 应合并: got {len(merged)}"
    m = merged[0]
    assert m.representative_text == "上 下"
    print(f"✓ merge ≥50% overlap: text={m.representative_text!r}")


def test_localize_merge_empty():
    """空输入应返回空列表。"""
    class _Stub:
        _merge_overlapping_instances_by_fps = (
            VideoLocalizationPipeline._merge_overlapping_instances_by_fps
        )
    assert _Stub()._merge_overlapping_instances_by_fps([], fps=30.0) == []
    print("✓ merge empty: []")