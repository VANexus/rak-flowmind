"""字幕区定位：百炼 qwen3.5-ocr（云优先，零本地模型）。

策略：对抽样帧调 qwen3.5-ocr 检测底部字幕条 bbox；多帧聚合取包围盒
并集（外扩 padding），作为 ffmpeg delogo 擦除区域。

坐标系（实测校准）：qwen3.5-ocr 返回 rotate_rect=[cx, cy, w, h, angle]，
四个几何量各自按图片对应维度归一化到 0-1000：
    px_cx = cx / 1000 * img_w   px_cy = cy / 1000 * img_h
    px_w  = w  / 1000 * img_w   px_h  = h  / 1000 * img_h
"""
from __future__ import annotations

import base64
import json
import re
import time

import requests

DEFAULT_BASE = "https://dashscope.aliyuncs.com/api/v1"
OCR_MODEL = "qwen3.5-ocr"


class OCRError(Exception):
    """字幕定位调用失败。category/retriable 语义与 errors.py 对齐。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


_PROMPT = "检测图中底部字幕条文本的边界框，输出JSON。"


def _ocr_frame(frame_path: str, api_key: str) -> list[dict]:
    """单帧检测。返回 rotate_rect 原始条目列表
    （元素形如 {"rotate_rect": [cx,cy,w,h,angle], "text": "..."}）。
    测试替换 _ocr_request。
    """
    with open(frame_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    data = _ocr_request(
        url=f"{DEFAULT_BASE}/services/aigc/multimodal-generation/generation",
        api_key=api_key,
        body={
            "model": OCR_MODEL,
            "input": {"messages": [{"role": "user", "content": [
                {"image": f"data:image/png;base64,{b64}"},
                {"text": _PROMPT},
            ]}]},
        },
    )
    raw = data.get("raw_text", "")
    # 回复是 ```json 包裹的数组：[{"rotate_rect": "<498><880><488><86><0>" 或 [n,n,n,n,n], ...}]
    body_text = (raw.strip()
                 .removeprefix("```json").removeprefix("```")
                 .removesuffix("```").strip())
    try:
        items = json.loads(body_text)
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []
    return [it for it in items if isinstance(it, dict) and "rotate_rect" in it]


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _rect_to_pixels(entry: dict, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """rotate_rect → 像素 (x1,y1,x2,y2)。兼容 '<n><n>..' 字符串与原生 list 两种形态。"""
    rr = entry.get("rotate_rect")
    if isinstance(rr, str):
        nums = [float(x) for x in _NUM_RE.findall(rr)]
    elif isinstance(rr, (list, tuple)):
        nums = [float(x) for x in rr]
    else:
        return None
    if len(nums) < 4:
        return None
    cx, cy, w, h = nums[0], nums[1], nums[2], nums[3]
    pcx, pcy = cx / 1000.0 * img_w, cy / 1000.0 * img_h
    pw, ph = w / 1000.0 * img_w, h / 1000.0 * img_h
    x1, y1, x2, y2 = pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2
    if x2 <= x1 or y2 <= y1:
        return None
    return round(x1), round(y1), round(x2), round(y2)


def _ocr_request(*, url: str, api_key: str, body: dict) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_err: Exception | None = None
    for attempt in range(3):  # 服务端偶发 5xx/空响应，轻退避重试
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=60.0)
        except requests.exceptions.RequestException as exc:
            raise OCRError("OCR 网络错误", category="environment") from exc
        if resp.status_code >= 500:
            last_err = OCRError(f"OCR HTTP {resp.status_code}",
                                category="transient", retriable=True)
            time.sleep(1.0 * (attempt + 1))
            continue
        if resp.status_code >= 400:
            raise OCRError(f"OCR HTTP {resp.status_code}", category="video")
        data = resp.json()
        content = ((data.get("output") or {}).get("choices") or [{}])[0]
        msg = content.get("message") or {}
        blocks = msg.get("content") or []
        text = next((b.get("text") for b in blocks if isinstance(b, dict) and b.get("text")),
                    "") or ""
        if not text:
            last_err = OCRError("OCR 响应结构异常")
            time.sleep(1.0 * (attempt + 1))
            continue
        return {"raw_text": text}
    raise last_err or OCRError("OCR 失败")


def locate_subtitle_region(
    frame_paths: list[str], *, api_key: str,
    frame_width: int | None = None, frame_height: int | None = None,
) -> dict | None:
    """多帧聚合出字幕区 bbox {x,y,w,h}（外扩 padding 防描边残留）；
    全部帧无字幕返回 None。"""
    if not api_key:
        raise ValueError("收到空 API key。请检查 DASHSCOPE_API_KEY 是否设置（项目 .env）。")

    boxes: list[tuple[int, int, int, int]] = []
    for path in frame_paths:
        for entry in _ocr_frame(path, api_key):
            if frame_width is None or frame_height is None:
                continue
            px = _rect_to_pixels(entry, frame_width, frame_height)
            if px is None:
                continue
            x1, y1, x2, y2 = px
            # 底部区域先验：字幕应落在画面下部 40%（若已知高度）
            if frame_height is not None and y1 < frame_height * 0.6:
                continue
            # 尺寸合理性先验：bbox 需占画面宽度至少 10%
            if frame_width is not None and (x2 - x1) < frame_width * 0.1:
                continue
            boxes.append((x1, y1, x2, y2))

    if not boxes:
        return None

    # 擦除区外扩 padding：字幕描边/抗锯齿常溢出标框，留边防残留
    pad = 8
    min_x = max(0, min(b[0] for b in boxes) - pad)
    min_y = max(0, min(b[1] for b in boxes) - pad)
    max_x = max(b[2] for b in boxes) + pad
    max_y = max(b[3] for b in boxes) + pad
    if frame_width is not None:
        max_x = min(max_x, frame_width)
    if frame_height is not None:
        max_y = min(max_y, frame_height)
    return {"x": min_x, "y": min_y, "w": max_x - min_x, "h": max_y - min_y}
