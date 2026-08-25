"""Phase D 测试 — L5/L6 mask + inpaint + 端到端 orchestrator + debug artifacts。

覆盖:
  - MaskArtifact / PolygonMaskBackend (单测, 含 rasterize / dilation / fallback / 多 instance)
  - OpenCVInpaintBackend / InpaintingEngine (单测, 含 backend swap)
  - 端到端: landscape + news (full inpaint pipeline + debug artifacts)
  - 部分流程: portrait + screencast (mask only, no inpaint)
  - 不变量: cv2.inpaint 只在 backend 出现
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from video_localization_engine.inpainting import (
    InpaintingBackendRegistry,
    InpaintingEngine,
)
from video_localization_engine.inpainting.base import InpaintingBackend
from video_localization_engine.inpainting.backends.opencv_backend import OpenCVInpaintBackend
from video_localization_engine.mask import (
    MaskArtifact,
    MaskBackend,
    MaskBackendRegistry,
    MaskGenerator,
    MaskType,
)
from video_localization_engine.mask.backends.polygon_backend import PolygonMaskBackend
from video_localization_engine.orchestrator import (
    PipelineConfig,
    VideoLocalizationPipeline,
    write_debug_artifacts,
)
from video_localization_engine.types.instance import SubtitleInstance, SubtitleTrack
from video_localization_engine.types.video import FramePacket, Orientation, VideoMeta


FIXTURES = Path("/tmp/vle_fixtures")
LANDSCAPE = FIXTURES / "landscape_person.mp4"
PORTRAIT = FIXTURES / "portrait_short.mp4"
SCREENCAST = FIXTURES / "screencast_ui.mp4"
NEWS = FIXTURES / "news_subtitle.mp4"


def _make_meta(w=200, h=100):
    return VideoMeta(
        source_path="x", width=w, height=h, fps=30, frame_count=10, duration_ms=333,
        orientation=Orientation.LANDSCAPE if w >= h else Orientation.PORTRAIT,
        has_audio=False,
    )


def _make_packet(frame_id=5, ts=165, w=200, h=100, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(50, 200, (h, w, 3), dtype=np.uint8)
    return FramePacket(
        frame_id=frame_id, timestamp_ms=ts, image=img, meta=_make_meta(w, h),
    )


def _make_instance(instance_id="i1", frame_ids=None, polys=None):
    frame_ids = frame_ids or [0, 30, 60]
    polys = polys or [
        [(100, 80), (150, 80), (150, 95), (100, 95)],
        [(101, 81), (151, 81), (151, 96), (101, 96)],
        [(102, 82), (152, 82), (152, 97), (102, 97)],
    ]
    inst = SubtitleInstance(instance_id=instance_id)
    for fid, poly in zip(frame_ids, polys):
        inst.observed_frame_ids.append(fid)
        inst.observed_polygons.append(poly)
    inst.first_frame = frame_ids[0]
    inst.last_frame = frame_ids[-1]
    inst.features["avg_confidence"] = 0.9
    return inst


# =====================================================================
# L5 MaskArtifact + PolygonMaskBackend 单测
# =====================================================================
def test_MaskArtifact_polygon_materialize():
    a = MaskArtifact(
        frame_id=0, timestamp_ms=0, mask_type=MaskType.POLYGON,
        polygons=[[(10, 10), (60, 10), (60, 40), (10, 40)]],
        width=100, height=80,
    )
    m = a.materialize()
    assert m.shape == (80, 100)
    assert m.dtype == np.uint8
    # 矩形区域 50x30 = 1500 (cv2.fillPoly 实际值约 1581 due to anti-alias 边界)
    nonzero = int((m > 0).sum())
    assert 1450 <= nonzero <= 1600, f"expected ~1500 px, got {nonzero}"
    print(f"✓ MaskArtifact.polygon materialize: {nonzero} px filled")


def test_PolygonMaskBackend_rasterization_dtype():
    pkt = _make_packet(w=200, h=100)
    inst = _make_instance()
    be = PolygonMaskBackend()
    arts = be.build([inst], pkt, dilation_px=0)
    assert len(arts) == 1
    m = arts[0].materialize()
    assert m.shape == (100, 200)
    assert m.dtype == np.uint8
    assert m.max() in (0, 1, 255)
    assert m.min() == 0
    print(f"✓ PolygonMaskBackend dtype: {m.dtype}, shape={m.shape}, max={m.max()}")


def test_PolygonMaskBackend_picks_polygon_for_frame():
    """instance.observed_frame_ids=[0, 30, 60] → frame 30 用 polygons[1]。"""
    pkt = _make_packet(frame_id=30)
    inst = _make_instance()
    be = PolygonMaskBackend()
    arts = be.build([inst], pkt)
    m = arts[0].materialize()
    # polygons[1] 是 (101..151, 81..96) — 50x15 ≈ 750 (cv2 边界 ~816)
    nonzero = int((m > 0).sum())
    assert 700 <= nonzero <= 850, f"expected ~750, got {nonzero}"
    print(f"✓ PolygonMaskBackend frame-keyed polygon: frame=30 → polygons[1] ({nonzero} px)")


def test_PolygonMaskBackend_falls_back_to_last_polygon():
    """frame=45 不在 observed_frame_ids, 但在 [0, 60] 内, fallback 到最近一次 (frame=30)。"""
    pkt = _make_packet(frame_id=45, ts=1500)
    inst = _make_instance()
    be = PolygonMaskBackend()
    arts = be.build([inst], pkt)
    m = arts[0].materialize()
    nonzero = int((m > 0).sum())
    assert 700 <= nonzero <= 850  # polygons[1] fallback
    print(f"✓ PolygonMaskBackend fallback: frame=45 → nearest last seen (frame 30) ({nonzero} px)")


def test_PolygonMaskBackend_skips_when_outside_window():
    """frame=200 不在 instance 寿命内 → 空 mask。"""
    pkt = _make_packet(frame_id=200, ts=6666)
    inst = _make_instance()
    be = PolygonMaskBackend()
    arts = be.build([inst], pkt)
    m = arts[0].materialize()
    assert int(m.sum()) == 0
    print(f"✓ PolygonMaskBackend out-of-window: empty mask")


def test_PolygonMaskBackend_dilates():
    """dilation_px=5 → mask 边缘外扩。"""
    pkt = _make_packet(w=200, h=100)
    inst = _make_instance()
    be = PolygonMaskBackend()
    m0 = be.build([inst], pkt, dilation_px=0)[0].materialize()
    m5 = be.build([inst], pkt, dilation_px=5)[0].materialize()
    s0 = int((m0 > 0).sum())
    s5 = int((m5 > 0).sum())
    assert s5 > s0, f"dilation should grow mask: {s0} → {s5}"
    print(f"✓ PolygonMaskBackend dilation: 0px={s0} < 5px={s5}")


def test_PolygonMaskBackend_multiple_instances_independent():
    """多个 instance → 各自独立 artifact (orchestrator 决定 union)。"""
    pkt = _make_packet()
    insts = [_make_instance("i1"), _make_instance("i2"), _make_instance("i3")]
    be = PolygonMaskBackend()
    arts = be.build(insts, pkt)
    assert len(arts) == 3
    union = MaskGenerator.union(arts)
    assert union is not None
    # union 含所有 3 个 instance 的 polygons
    assert union.polygons is not None
    assert len(union.polygons) == 3
    print(f"✓ PolygonMaskBackend multi-instance: 3 artifacts, union polygons={len(union.polygons)}")


# =====================================================================
# L6 InpaintingEngine + OpenCVInpaintBackend 单测
# =====================================================================
def test_OpenCVInpaintBackend_changes_pixels():
    """带白方块的合成帧 + 精确 mask → inpaint 后像素有变化。"""
    meta = _make_meta(w=80, h=80)
    img = np.full((80, 80, 3), 100, dtype=np.uint8)
    img[20:60, 20:60] = (255, 255, 255)  # 白方块
    pkt = FramePacket(frame_id=0, timestamp_ms=0, image=img, meta=meta)
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[20:60, 20:60] = 255
    be = OpenCVInpaintBackend(algorithm="telea", radius=3)
    out = be.inpaint(pkt, mask)
    diff = (img != out).any(axis=-1)
    assert diff.sum() > 0
    # 白方块区域内像素不全再是 255
    region = out[20:60, 20:60]
    assert not np.all(np.all(region == 255, axis=-1))
    print(f"✓ OpenCVInpaintBackend pixels changed: {diff.sum()} px modified")


def test_InpaintingEngine_swap_backend():
    """构造 mock backend, 通过 registry swap。"""
    class MockBackend(InpaintingBackend):
        @property
        def name(self): return "mock"
        def is_available(self): return True
        def inpaint(self, packet, mask): return packet.image.copy()

    InpaintingBackendRegistry.register("mock", MockBackend)
    try:
        eng = InpaintingEngine(backend_name="mock")
        pkt = _make_packet()
        from video_localization_engine.mask.artifact import MaskArtifact
        ma = MaskArtifact(frame_id=5, timestamp_ms=165, mask_type=MaskType.POLYGON,
                          polygons=[], width=200, height=100)
        out = eng.inpaint_frame(pkt, ma)
        assert out.shape == pkt.image.shape
        assert np.array_equal(out, pkt.image)
        print("✓ InpaintingEngine swap backend: mock used, output unchanged")
    finally:
        InpaintingBackendRegistry.clear()
        # 重新注册 opencv (clear 会清空所有, 包括默认)
        InpaintingBackendRegistry.register("opencv", OpenCVInpaintBackend)


# =====================================================================
# 端到端 (landscape + news)
# =====================================================================
def _run_pipeline_and_write(fixture_path: Path, enable_inpaint: bool, frame_stride: int = 2):
    cfg = PipelineConfig(
        frame_stride=frame_stride,
        enable_inpaint=enable_inpaint,
        mask_dilation_px=2,
        inpaint_radius=3,
        debug_sample_rate=10,
    )
    pipe = VideoLocalizationPipeline(str(fixture_path), cfg)
    try:
        results = pipe.run_full()
        out_dir = f"/tmp/vle_debug/{fixture_path.stem}"
        if cfg.debug_output_dir:
            out_dir = cfg.debug_output_dir
        write_debug_artifacts(
            track=SubtitleTrack(
                video_meta_path=pipe.meta.source_path,
                instances=list(pipe._all_instances),
                frame_candidates=list(pipe._frame_candidates_buf),
                region_policies_used=[p.name for p in pipe._policies],
                detector_id=cfg.detector_backend,
            ),
            results=results,
            meta=pipe.meta,
            out_dir=out_dir,
            enable_inpaint=enable_inpaint,
            sample_rate=cfg.debug_sample_rate,
        )
        return pipe, results, out_dir
    finally:
        pipe.close()


def test_pipeline_landscape_end_to_end():
    pipe, results, out_dir = _run_pipeline_and_write(LANDSCAPE, enable_inpaint=True,
                                                     frame_stride=2)
    out = Path(out_dir)
    assert out.exists(), f"missing {out_dir}"
    for fn in ("track.vle.json", "subtitles_vis.png", "mask_overlay.png",
               "before.png", "after.png", "side_by_side.png"):
        assert (out / fn).exists(), f"missing {out_dir}/{fn}"
    # track.vle.json 可读
    import json
    data = json.loads((out / "track.vle.json").read_text())
    assert data["version"] == "0.1.0"
    assert len(data["instances"]) == len(pipe._all_instances)
    assert len(pipe._all_instances) >= 1
    # after.png 与 before.png 有差异 (inpaint 改了像素)
    import cv2
    before = cv2.imread(str(out / "before.png"))
    after = cv2.imread(str(out / "after.png"))
    diff = (before != after).any(axis=-1)
    assert diff.sum() > 0, "after.png should differ from before.png"
    print(f"✓ landscape end-to-end: {len(pipe._all_instances)} instances, "
          f"{diff.sum()} px inpaint changed")


def test_pipeline_news_end_to_end():
    pipe, results, out_dir = _run_pipeline_and_write(NEWS, enable_inpaint=True,
                                                     frame_stride=2)
    out = Path(out_dir)
    for fn in ("track.vle.json", "subtitles_vis.png", "mask_overlay.png",
               "before.png", "after.png", "side_by_side.png"):
        assert (out / fn).exists(), f"missing {out_dir}/{fn}"
    assert len(pipe._all_instances) >= 1
    # 新闻 fixture 含顶部字幕, instance 应有 y 较小的 bbox
    top_inst = None
    for inst in pipe._all_instances:
        if inst.representative_bbox:
            y_center = (inst.representative_bbox[1] + inst.representative_bbox[3]) / 2
            if y_center < 200:  # 顶部区域
                top_inst = inst
                break
    assert top_inst is not None, "news fixture should have a top-subtitle instance"
    print(f"✓ news end-to-end: {len(pipe._all_instances)} instances, "
          f"top instance y_center={y_center if top_inst else 'n/a'}")


# =====================================================================
# 部分流程 (portrait + screencast, 仅 mask 不 inpaint)
# =====================================================================
def test_pipeline_portrait_mask_only():
    pipe, results, out_dir = _run_pipeline_and_write(PORTRAIT, enable_inpaint=False,
                                                     frame_stride=2)
    out = Path(out_dir)
    assert (out / "subtitles_vis.png").exists()
    assert (out / "mask_overlay.png").exists()
    assert not (out / "after.png").exists(), "no inpaint → no after.png"
    assert not (out / "side_by_side.png").exists()
    assert len(pipe._all_instances) >= 1
    print(f"✓ portrait mask-only: {len(pipe._all_instances)} instances, after.png absent")


def test_pipeline_screencast_mask_only():
    """screencast 含 UI 文字, Phase D 不分类, 全部 mask。"""
    pipe, results, out_dir = _run_pipeline_and_write(SCREENCAST, enable_inpaint=False,
                                                     frame_stride=2)
    out = Path(out_dir)
    assert (out / "mask_overlay.png").exists()
    assert not (out / "after.png").exists()
    print(f"✓ screencast mask-only: {len(pipe._all_instances)} instances "
          f"(UI 文字一并 mask)")


# =====================================================================
# 不变量
# =====================================================================
def test_no_cv2_inpaint_outside_backend():
    """cv2.inpaint 只允许出现在 inpainting/backends/opencv_backend.py。

    检测的是'实际调用'(行首去除空白后以 cv2.inpaint 开头, 或作为函数实参),
    docstring 中的纯文字说明放过。
    """
    import re
    base = Path(__file__).resolve().parents[1]
    allowed = base / "inpainting" / "backends" / "opencv_backend.py"
    # 匹配 "cv2.inpaint" 后面紧跟 "(" — 实际调用; 常量 (cv2.INPAINT_TELEA) 不会匹配
    pattern = re.compile(r"cv2\.inpaint\s*\(")
    for py in base.rglob("*.py"):
        if py.name == "__pycache__" or "__pycache__" in str(py):
            continue
        content = py.read_text()
        for ln, line in enumerate(content.splitlines(), 1):
            stripped = line.lstrip()
            # 跳过 docstring (三引号 + 注释行)
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if pattern.search(line):
                if py.resolve() == allowed.resolve():
                    continue
                raise AssertionError(
                    f"cv2.inpaint in business code {py.relative_to(base)}:{ln}: {line!r}"
                )
    print("✓ cv2.inpaint only in inpainting/backends/opencv_backend.py")


def test_no_hardcoded_thresholds_in_mask_or_inpainting():
    """mask/ + inpainting/ + orchestrator/ 不含硬字幕阈值或硬加权。"""
    import re
    base = Path(__file__).resolve().parents[1]
    forbidden = [
        r"is_subtitle\s*=\s*True",
        r"if\s+.*\:\s*is_subtitle",
        r"score\s*=\s*position\s*\*",
        r"weight\s*=\s*0\.\d+\s*\*",
        r"if\s+y\s*>\s*0\.\d+:",
    ]
    for sub in ("mask", "inpainting", "orchestrator"):
        for py in (base / sub).rglob("*.py"):
            if py.name == "__pycache__" or "__pycache__" in str(py):
                continue
            content = py.read_text()
            for ln, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for pat in forbidden:
                    if re.search(pat, line):
                        raise AssertionError(
                            f"hardcoded {pat!r} in {py.relative_to(base)}:{ln}: {line!r}"
                        )
    print("✓ no hardcoded thresholds/weights in mask/+inpainting/+orchestrator/")


def test_backends_registered():
    """默认 backend 已注册。"""
    assert "polygon" in MaskBackendRegistry.available()
    assert "opencv" in InpaintingBackendRegistry.available()
    print(f"✓ mask backends: {MaskBackendRegistry.available()}, "
          f"inpaint backends: {InpaintingBackendRegistry.available()}")


def main():
    tests = [
        # L5 单测
        test_MaskArtifact_polygon_materialize,
        test_PolygonMaskBackend_rasterization_dtype,
        test_PolygonMaskBackend_picks_polygon_for_frame,
        test_PolygonMaskBackend_falls_back_to_last_polygon,
        test_PolygonMaskBackend_skips_when_outside_window,
        test_PolygonMaskBackend_dilates,
        test_PolygonMaskBackend_multiple_instances_independent,
        # L6 单测
        test_OpenCVInpaintBackend_changes_pixels,
        test_InpaintingEngine_swap_backend,
        # 端到端
        test_pipeline_landscape_end_to_end,
        test_pipeline_news_end_to_end,
        # 部分流程
        test_pipeline_portrait_mask_only,
        test_pipeline_screencast_mask_only,
        # 不变量
        test_no_cv2_inpaint_outside_backend,
        test_no_hardcoded_thresholds_in_mask_or_inpainting,
        test_backends_registered,
        # P10 (mask x/y 独立 dilation)
        test_mask_dilation_xy_separate,
        test_mask_dilation_y_backward_compat,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} Phase D tests passed.")


# ============================ P10 step 1 ============================

def test_mask_dilation_xy_separate():
    """x/y 独立 dilation: 横向扩 6, 纵向扩 18, mask 上下边缘比左右更宽。"""
    pkt = _make_packet(w=200, h=100, frame_id=0)  # exact match (instance observed_frame_ids[0]=0)
    inst = _make_instance(frame_ids=[0])
    polys = [[(100, 40), (150, 40), (150, 60), (100, 60)]]
    inst = _make_instance(frame_ids=[0], polys=polys)
    be = PolygonMaskBackend()
    art = be.build([inst], pkt, dilation_xy=(6, 18))[0]
    m = art.materialize()
    rows = np.where(m.any(axis=1))[0]
    cols = np.where(m.any(axis=0))[0]
    assert len(rows) > 0 and len(cols) > 0, "mask 应非空"
    # polygon 起点 (100,40)~(150,60)
    x1, x2 = 100, 150
    y1, y2 = 40, 60
    row_top_extra = y1 - rows.min()
    row_bot_extra = rows.max() - y2
    col_left_extra = x1 - cols.min()
    col_right_extra = cols.max() - x2
    assert row_top_extra >= 17, f"y 上扩应 ≥ 18 (实际 {row_top_extra})"
    assert row_bot_extra >= 17, f"y 下扩应 ≥ 18 (实际 {row_bot_extra})"
    assert col_left_extra >= 5 and col_left_extra <= 8, (
        f"x 左扩应 ≈6 (实际 {col_left_extra})"
    )
    assert col_right_extra >= 5 and col_right_extra <= 8, (
        f"x 右扩应 ≈6 (实际 {col_right_extra})"
    )
    # 关键: y 扩张必须比 x 扩张 ≥ 8 (12-6=6, 加误差)
    assert (row_top_extra + row_bot_extra) - (col_left_extra + col_right_extra) >= 8, (
        f"y sum - x sum 应 ≥ 8, got {row_top_extra + row_bot_extra} - {col_left_extra + col_right_extra}"
    )
    print(
        f"✓ xy separate: rows y±{int(row_top_extra)}/{int(row_bot_extra)}, "
        f"cols x±{int(col_left_extra)}/{int(col_right_extra)}"
    )


def test_mask_dilation_y_backward_compat():
    """dilation_xy=None → 回退到 dilation_px=int, 等价 (n, n)."""
    from video_localization_engine.mask.artifact import MaskArtifact
    pkt = _make_packet(w=200, h=100)
    inst = _make_instance()
    be = PolygonMaskBackend()
    # 旧 API: 只传 dilation_px
    art_old = be.build([inst], pkt, dilation_px=8)[0]
    # 新 API 等价
    art_new = be.build([inst], pkt, dilation_xy=(8, 8))[0]
    m_old = art_old.materialize()
    m_new = art_new.materialize()
    # 像素总和应当相等 (允许 ±2 像素差, morph close kernel 计算舍入)
    s_old = int((m_old > 0).sum())
    s_new = int((m_new > 0).sum())
    assert abs(s_old - s_new) <= 4, (
        f"old API (n=8) 与 new API ((8,8)) 像素数应一致, "
        f"got {s_old} vs {s_new}"
    )
    # dilation_px property 应该能拿到 8 (x==y 时)
    assert art_old.dilation_px == 8, (
        f"向后兼容 dilation_px property, got {art_old.dilation_px}"
    )
    # artifact.dilation_xy 直接可读
    assert art_new.dilation_xy == (8, 8)
    print(f"✓ backward compat: old API pixel={s_old}, new API pixel={s_new}")


if __name__ == "__main__":
    main()
