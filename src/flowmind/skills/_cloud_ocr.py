"""阿里云 OCR 字幕定位：对抽样帧调云印刷文字识别，聚合出字幕 bbox。

云优先原则：OCR 全走云端（原 VLE 的本地 PaddleOCR 已删）。
策略：抽 N 帧分别 OCR → 过滤小块噪点/台标 → 只取画面下部 40% 区域的文本块
（字幕位置先验）→ 多帧取包围盒并集，作为 ffmpeg delogo 擦除区域。
全部帧无字幕 → 返回 None（走"新增字幕"路径，不擦除）。
"""
from __future__ import annotations

import requests

DEFAULT_BASE = "https://ocr-api.cn-hangzhou.aliyuncs.com"
MIN_TEXT_HEIGHT_PX = 30      # 低于此高度视为台标/水印噪点
BOTTOM_FRACTION = 0.4        # 只认画面底部 40% 区域内的文本为候选字幕


class OCRError(Exception):
    """OCR 调用失败。category/retriable 语义与 errors.py 对齐。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def _ocr_request(*, url: str, api_key: str, body: dict) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30.0)
    except requests.exceptions.RequestException as exc:
        raise OCRError("OCR 网络错误", category="environment") from exc
    if resp.status_code >= 500:
        raise OCRError(f"OCR HTTP {resp.status_code}", category="transient", retriable=True)
    if resp.status_code >= 400:
        raise OCRError(f"OCR HTTP {resp.status_code}（检查图片可访问性）", category="video")
    return resp.json()


def _ocr_frame(frame_path: str, api_key: str) -> dict:
    """单帧识别。生产实现把本地图转 base64 内嵌 body；测试替换 _ocr_request。"""
    import base64

    with open(frame_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return _ocr_request(
        url=f"{DEFAULT_BASE}/v2/ocr/recognizeprintedtext",
        api_key=api_key,
        body={"Image": {"Data": b64}, "Features": []},
    )


def locate_subtitle_region(
    frame_paths: list[str], *, api_key: str,
    frame_width: int | None = None, frame_height: int | None = None,
) -> dict | None:
    """多帧聚合出字幕区 bbox {x,y,w,h}；无字幕返回 None。"""
    if not api_key:
        raise ValueError(
            "收到空 API key。请检查环境变量 DASHSCOPE_API_KEY 是否设置。"
            "云优先原则：OCR 必须走云端，不做本地降级。"
        )
    boxes: list[tuple[int, int, int, int]] = []
    for path in frame_paths:
        try:
            data = _ocr_frame(path, api_key)
        except OCRError:
            raise
        for item in (data.get("data") or {}).get("content") or []:
            rect = item.get("text_rectangle") or {}
            h = int(rect.get("height", 0))
            y = int(rect.get("top", 0))
            # 噪点过滤 + 底部区域先验
            if h < MIN_TEXT_HEIGHT_PX:
                continue
            if frame_height is not None and y < frame_height * (1 - BOTTOM_FRACTION):
                continue
            x = int(rect.get("left", 0))
            w = int(rect.get("width", 0))
            boxes.append((x, y, w, h))
    if not boxes:
        return None
    min_x = min(b[0] for b in boxes)
    min_y = min(b[1] for b in boxes)
    max_r = max(b[0] + b[2] for b in boxes)
    max_b = max(b[1] + b[3] for b in boxes)
    return {"x": min_x, "y": min_y, "w": max_r - min_x, "h": max_b - min_y}
