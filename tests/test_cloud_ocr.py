"""_cloud_ocr 测试：qwen3.5-ocr 字幕定位（rotate_rect 解析 + 聚合 + 先验过滤）。

坐标系（实测校准）：rotate_rect=[cx,cy,w,h,angle]，各量按图片对应维度归一化 0-1000。
"""

from __future__ import annotations

import pytest

from flowmind.skills import _cloud_ocr
from flowmind.skills._cloud_ocr import OCRError, _rect_to_pixels


# ── rotate_rect → 像素 ──


def test_rect_string_form():
    """'<498><880><488><86><0>' @640x360 → 实测校准值。"""
    px = _rect_to_pixels({"rotate_rect": "<498><880><488><86><0>"}, 640, 360)
    assert px is not None
    x1, y1, x2, y2 = px
    # cx=498→318.7 cy=880→316.8 w=488→312.3 h=86→31.0
    assert x1 == pytest.approx(163, abs=1)
    assert y1 == pytest.approx(301, abs=1)
    assert x2 == pytest.approx(475, abs=1)
    assert y2 == pytest.approx(332, abs=1)


def test_rect_list_form():
    """校准图实测：[469, 872, 642, 101, 0] @1280x720 → 真值 x[200..1000] y[600..660]。"""
    px = _rect_to_pixels({"rotate_rect": [469, 872, 642, 101, 0]}, 1280, 720)
    x1, y1, x2, y2 = px
    assert x1 == pytest.approx(189, abs=3)   # (600.32 - 410.88)
    assert y1 == pytest.approx(591, abs=3)
    assert x2 == pytest.approx(1011, abs=3)
    assert y2 == pytest.approx(664, abs=3)


def test_rect_invalid_shapes():
    assert _rect_to_pixels({"rotate_rect": "<abc>"}, 640, 360) is None
    assert _rect_to_pixels({"rotate_rect": [1, 2]}, 640, 360) is None
    assert _rect_to_pixels({}, 640, 360) is None


# ── 聚合与先验 ──


def _stub_frames(monkeypatch, entries_per_frame: list[list[dict]], tmp_path):
    frames = []
    for i in range(len(entries_per_frame)):
        p = tmp_path / f"f{i}.png"
        p.write_bytes(b"png")
        frames.append(str(p))
    it = iter(entries_per_frame)
    monkeypatch.setattr(_cloud_ocr, "_ocr_frame", lambda frame, key: next(it))
    return frames


def test_locate_union_of_boxes(monkeypatch, tmp_path):
    frames = _stub_frames(
        monkeypatch,
        [[{"rotate_rect": [400, 875, 390, 65, 0]}],   # ≈x[160..415]
         [{"rotate_rect": [395, 878, 385, 70, 0]}]],  # ≈x[150..420]
        tmp_path)
    region = _cloud_ocr.locate_subtitle_region(frames, api_key="k",
                                               frame_width=640, frame_height=360)
    assert region is not None
    assert region["w"] > 250 and region["h"] > 30


def test_locate_none_when_all_no_subtitle(monkeypatch, tmp_path):
    frames = _stub_frames(monkeypatch, [[], []], tmp_path)
    assert _cloud_ocr.locate_subtitle_region(frames, api_key="k",
                                             frame_width=640, frame_height=360) is None


def test_locate_filters_top_region(monkeypatch, tmp_path):
    """y_center 在画面上部（<60%）的框是台标不是字幕，剔除。"""
    # 台标: cy=100/1000*1080=108 < 648；真字幕: cy≈920/1000*1080=993
    frames = _stub_frames(
        monkeypatch,
        [[{"rotate_rect": [300, 100, 300, 40, 0]},
          {"rotate_rect": [300, 850, 300, 40, 0]}]],
        tmp_path)
    region = _cloud_ocr.locate_subtitle_region(frames, api_key="k",
                                               frame_width=1280, frame_height=1080)
    assert region is not None
    assert region["y"] > 1080 * 0.6, "只应保留底部框"


def test_no_key_raises(tmp_path):
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        _cloud_ocr.locate_subtitle_region([str(tmp_path / "x.png")], api_key="")


def test_http_error_classification(monkeypatch, tmp_path):
    p = tmp_path / "f.png"
    p.write_bytes(b"x")

    def boom(url, api_key, body):
        raise OCRError("OCR HTTP 503", category="transient", retriable=True)

    monkeypatch.setattr(_cloud_ocr, "_ocr_request", boom)
    with pytest.raises(OCRError) as ei:
        _cloud_ocr.locate_subtitle_region([str(p)], api_key="k")
    assert ei.value.retriable is True
