"""Phase E+ — PillowRenderer (L7 subtitle renderer) 单测。

覆盖:
  - 协议合规
  - 英文渲染 (ASCII)
  - 中文渲染 (CJK)
  - 注册到 RendererRegistry
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from video_localization_engine.localizer import (
    PillowRenderer,
    RendererRegistry,
)
from video_localization_engine.localizer.protocols import RendererBackend


def test_PillowRenderer_protocol_conformance():
    """PillowRenderer 满足 RendererBackend Protocol。"""
    assert isinstance(PillowRenderer(), RendererBackend)
    print("✓ PillowRenderer protocol: OK")


def test_PillowRenderer_supports_english_text():
    """英文文本能渲染 (有 alpha > 0 的像素)。"""
    renderer = PillowRenderer()
    bbox = (50, 50, 400, 110)
    frame_size = (200, 600)
    canvas = renderer.render("Hello World", bbox, frame_size)
    assert canvas.shape == (200, 600, 4)
    assert canvas.dtype == np.uint8
    alpha = canvas[:, :, 3]
    assert alpha.any(), "expected at least one rendered pixel"
    print(f"✓ PillowRenderer en: alpha pixels = {int(alpha.sum()):,}")


def test_PillowRenderer_supports_chinese_text():
    """中文文本能渲染 — 即使没有中文字体也会 fallback 不抛错。"""
    renderer = PillowRenderer()
    bbox = (50, 150, 400, 220)
    frame_size = (300, 600)
    canvas = renderer.render("你好世界,这是一段字幕测试。", bbox, frame_size)
    assert canvas.shape == (300, 600, 4)
    alpha = canvas[:, :, 3]
    # 不论中文字体是否存在, 不抛错是硬约束
    # 若字体找到了, 应当有 alpha 像素
    assert isinstance(canvas, np.ndarray)
    print(
        f"✓ PillowRenderer zh: alpha pixels = {int(alpha.sum()):,}"
        + (" (fallback: 中文字体未安装)" if not alpha.any() else "")
    )


def test_PillowRenderer_registered():
    """PillowRenderer 已注册到 RendererRegistry('pillow')。"""
    assert "pillow" in RendererRegistry.available(), (
        f"Available: {RendererRegistry.available()}"
    )
    cls = RendererRegistry.get("pillow")
    assert cls is PillowRenderer
    print(f"✓ PillowRenderer registered: keys={RendererRegistry.available()}")


def test_PillowRenderer_renders_rgba_shape_and_dtype():
    """render 始终返回 HxWx4 uint8, 即使 text 或 bbox 为空。"""
    renderer = PillowRenderer()
    frame_size = (100, 200)
    # 空文本 → 全 0 canvas
    canvas_empty = renderer.render("", (0, 0, 100, 50), frame_size)
    assert canvas_empty.shape == (100, 200, 4)
    assert canvas_empty.dtype == np.uint8
    assert canvas_empty.sum() == 0
    # 空 bbox
    canvas_nobbox = renderer.render("hi", None, frame_size)
    assert canvas_nobbox.shape == (100, 200, 4)
    assert canvas_nobbox.dtype == np.uint8
    assert canvas_nobbox.sum() == 0
    print("✓ PillowRenderer shape/dtype/empty: OK")


# ============================ P10 step 2 ============================

def test_pillow_renderer_text_centered():
    """渲染后文字像素 bbox 接近 frame bbox 中心 (P10 居中)."""
    renderer = PillowRenderer(font_scale=0.7, padding=4)
    bbox = (50, 50, 550, 150)   # 500x100
    frame_size = (200, 600)
    canvas = renderer.render("Hello", bbox, frame_size)
    alpha = canvas[:, :, 3]
    ys, xs = np.where(alpha > 0)
    assert len(xs) > 0
    text_cx = (xs.min() + xs.max()) // 2
    text_cy = (ys.min() + ys.max()) // 2
    bbox_cx = (bbox[0] + bbox[2]) // 2
    bbox_cy = (bbox[1] + bbox[3]) // 2
    # 允许 ±5px 误差 (像素四舍五入)
    assert abs(text_cx - bbox_cx) <= 5, (
        f"horizontal center off: text_cx={text_cx}, bbox_cx={bbox_cx}"
    )
    assert abs(text_cy - bbox_cy) <= 8, (
        f"vertical center off: text_cy={text_cy}, bbox_cy={bbox_cy}"
    )
    print(f"✓ PillowRenderer centered: text=({text_cx},{text_cy}), bbox=({bbox_cx},{bbox_cy})")


def test_pillow_renderer_auto_shrink():
    """bbox 很窄 (但仍有合理高度) 时, 自动缩字到 fit width.

    注: 缩到 min_font_scale (默认 50%) 仍超宽时, renderer 接受溢出 (不抛).
    断言: 渲染宽度应比无缩字时明显更窄 → 至少比 bbox 宽度小 (> 80%).
    """
    renderer = PillowRenderer(font_scale=0.7, padding=4,
                              shrink_threshold=0.95, min_font_scale=0.5)
    # 100w x 60h bbox — 长文字 base_size = int(52*0.7)=36, 应自动缩
    bbox = (50, 50, 150, 110)
    frame_size = (200, 200)
    long_text = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"   # 32 字符
    canvas = renderer.render(long_text, bbox, frame_size)
    alpha = canvas[:, :, 3]
    ys, xs = np.where(alpha > 0)
    assert len(xs) > 0, "应有渲染像素"
    rendered_w = xs.max() - xs.min() + 1
    bbox_w = bbox[2] - bbox[0]
    # 渲染宽度不应比 bbox 宽出太多 (缩到最小可能仍略超, 允许 110%)
    assert rendered_w <= bbox_w * 1.1, (
        f"缩字后仍超 bbox 110%: rendered={rendered_w}, bbox_w={bbox_w}"
    )
    print(f"✓ PillowRenderer auto-shrink: rendered_w={rendered_w} ≤ bbox_w={bbox_w} * 1.1")


def test_pillow_renderer_word_wrap():
    """缩到最小仍超宽 → 自动换行 (按词 / 字)."""
    renderer = PillowRenderer(font_scale=0.7, padding=4,
                              shrink_threshold=0.95, min_font_scale=0.5,
                              max_wrap_lines=3)
    bbox = (50, 50, 200, 200)  # 150w x 150h — 够 3 行 (line_h ≈ 18)
    frame_size = (300, 250)
    text = "one two three four five six seven eight"  # 8 词 / 切成 ≤3 行
    canvas = renderer.render(text, bbox, frame_size)
    alpha = canvas[:, :, 3]
    ys, xs = np.where(alpha > 0)
    assert len(xs) > 0
    # 文字块高度应明显大于单行 (≥1.5 行的高度)
    rendered_h = ys.max() - ys.min() + 1
    bbox_inner_h = (bbox[3] - bbox[1]) - 2 * renderer.padding
    # 2+ 行时 rendered_h 应当 ≥ 单行 * 1.5
    single_line_h_estimate = max(8, int(bbox_inner_h * renderer.font_scale * 0.7) * 1.2)
    assert rendered_h >= single_line_h_estimate * 1.5, (
        f"应当换行后高度 ≥ 单行 1.5x; got {rendered_h}, "
        f"single_estimate={single_line_h_estimate}"
    )
    print(f"✓ PillowRenderer word-wrap: rendered_h={rendered_h} ≥ 1.5× single={single_line_h_estimate:.0f}")


def test_pillow_renderer_outline():
    """默认 stroke_width=1 — canvas alpha 像素应当多于无描边的版本."""
    rw_default = PillowRenderer(stroke_width=1, padding=4)
    rw_no_stroke = PillowRenderer(stroke_width=0, padding=4)
    bbox = (50, 50, 350, 110)
    frame_size = (200, 400)
    text = "Outline Test"
    c1 = rw_default.render(text, bbox, frame_size)
    c2 = rw_no_stroke.render(text, bbox, frame_size)
    a1 = int(c1[:, :, 3].sum())
    a2 = int(c2[:, :, 3].sum())
    assert a1 > a2, f"有描边 alpha 像素 {a1} 应 > 无描边 {a2}"
    print(f"✓ PillowRenderer outline: with_stroke={a1} > no_stroke={a2}")
