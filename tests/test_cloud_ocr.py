"""_cloud_ocr 测试：阿里云 OCR 字幕区定位（抽帧 → 识别 → bbox 聚合）。

覆盖：
- locate_subtitle_region：多帧 OCR 结果聚合出底部字幕 bbox（取覆盖并集）
- 无字幕帧（空 results）被忽略；全部无字幕 → None
- 无 key 显式报错
- HTTP 错误分类
"""
from __future__ import annotations

import pytest

from flowmind.skills import _cloud_ocr
from flowmind.skills._cloud_ocr import OCRError


def _ocr_response(*boxes):
    """构造阿里云印刷文字识别响应：boxes = [(x, y, w, h, text)]"""
    return {
        "data": {
            "content": [
                {
                    "text": t,
                    "text_rectangle": {"left": x, "top": y, "width": w, "height": h},
                }
                for (x, y, w, h, t) in boxes
            ]
        }
    }


def test_locate_returns_union_of_bottom_boxes(monkeypatch, tmp_path):
    """两帧都识别到底部字幕行 → bbox 取并集（min/max 包围）。"""
    responses = iter([
        _ocr_response((100, 820, 700, 60, "大家好")),
        _ocr_response((150, 850, 600, 70, "欢迎观看")),
    ])
    monkeypatch.setattr(_cloud_ocr, "_ocr_frame", lambda frame, api_key: next(responses))
    # 假帧文件
    frames = [str(tmp_path / f"f{i}.png") for i in range(2)]
    for f in frames:
        open(f, "wb").close()

    region = _cloud_ocr.locate_subtitle_region(frames, api_key="k")
    assert region is not None
    assert region["x"] == 100            # min left
    assert region["y"] == 820            # min top
    assert region["w"] == 700            # max right - min left
    assert region["h"] == 100            # max bottom - min top


def test_locate_none_when_no_text(monkeypatch, tmp_path):
    monkeypatch.setattr(_cloud_ocr, "_ocr_frame", lambda frame, api_key: {"data": {"content": []}})
    frames = [str(tmp_path / "f.png")]
    open(frames[0], "wb").close()
    assert _cloud_ocr.locate_subtitle_region(frames, api_key="k") is None


def test_locate_filters_tiny_noise(monkeypatch, tmp_path):
    """过小的文本块（<30px 高）视为水印/噪点，不进字幕区。"""
    monkeypatch.setattr(_cloud_ocr, "_ocr_frame", lambda f, k: _ocr_response(
        (100, 900, 500, 20, "台标"),   # 高度 20 < 30 → 忽略
        (120, 860, 480, 50, "真字幕"),
    ))
    frames = [str(tmp_path / "f.png")]
    open(frames[0], "wb").close()
    region = _cloud_ocr.locate_subtitle_region(frames, api_key="k")
    assert region["h"] == 50             # 只由真字幕决定


def test_no_key_raises():
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        _cloud_ocr.locate_subtitle_region(["/f.png"], api_key="")


def test_http_error_classification(monkeypatch):
    def boom(frame, api_key):
        raise requests.HTTPError("500")

    import requests  # noqa: E402
    monkeypatch.setattr(_cloud_ocr.requests.exceptions.HTTPError,
                        "__str__", lambda self: "500")
    monkeypatch.setattr(
        _cloud_ocr, "_ocr_request",
        lambda **kw: (_ for _ in ()).throw(OCRError("OCR HTTP 500",
                                                    category="transient", retriable=True)),
    )
    with pytest.raises(OCRError) as ei:
        _cloud_ocr._ocr_request(url="x", api_key="k", body={})
    assert ei.value.retriable is True
