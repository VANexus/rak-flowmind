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


def _bbox(boxes: list[tuple[int, int, int, int]]) -> dict:
    """一组像素框 → 外扩 padding 的擦除区 {x,y,w,h}。"""
    pad = 8
    return {
        "x": max(0, min(b[0] for b in boxes) - pad),
        "y": max(0, min(b[1] for b in boxes) - pad),
        "w": max(b[2] for b in boxes) - min(b[0] for b in boxes) + pad * 2,
        "h": max(b[3] for b in boxes) - min(b[1] for b in boxes) + pad * 2,
    }


def _aggregate_regions(
    boxes: list[tuple[int, int, int, int]],
    frame_width: int | None,
    frame_height: int | None,
) -> list[dict]:
    """像素框列表 → 擦除区列表 [{x,y,w,h}, ...]（外扩 padding 防描边残留）。

    返回多个独立区域：竖排视频里标题（顶部）与歌词（底部）可能落在同一
    垂直线上但 y 位置分开，需分别擦除，避免全屏高的一条竖带误伤画面主体。
    空 boxes 返回空列表。云/本地 OCR 后端共用本聚合逻辑。
    """
    if not boxes:
        return []

    # 保留全部横向簇（不再只选画面中央簇）：输入框已过"底部 40%"先验，
    # 边缘簇几乎总是字幕的行首/行尾碎片而非水印；只留中央簇会漏擦边缘
    # 残字（鬼影）。底部水印被顺带擦除可接受（LaMa/修复本就为去内容）。
    all_boxes: list[tuple[int, int, int, int]] = []
    prev_cx: float | None = None
    for b in sorted(boxes, key=lambda b: (b[0] + b[2]) / 2):
        cur_cx = (b[0] + b[2]) / 2
        if prev_cx is not None and cur_cx - prev_cx >= (frame_width or 0) * 0.35:
            continue  # 仅丢弃远离字幕主体的孤立簇（真正边缘水印）
        all_boxes.append(b)
        prev_cx = cur_cx

    # 按 y 中心聚类成"文本行"：每行一个紧贴框（x 取行内最小/最大），
    # 擦除区不再是全宽大横带，LaMa/修复只处理真实文字覆盖的范围。
    y_gap = frame_height * 0.06 if frame_height else 0
    y_sorted = sorted(all_boxes, key=lambda b: (b[1] + b[3]) / 2)
    y_clusters: list[list[tuple[int, int, int, int]]] = [[y_sorted[0]]]
    for b in y_sorted[1:]:
        prev_cy = (y_clusters[-1][-1][1] + y_clusters[-1][-1][3]) / 2
        cur_cy = (b[1] + b[3]) / 2
        if cur_cy - prev_cy < y_gap:
            y_clusters[-1].append(b)
        else:
            y_clusters.append([b])

    regions = [_bbox(c) for c in y_clusters if len(c) >= 1]
    if frame_width is not None:
        for r in regions:
            r["w"] = min(r["w"], frame_width - r["x"])
    if frame_height is not None:
        for r in regions:
            r["h"] = min(r["h"], frame_height - r["y"])
    return regions


def _is_white_subtitle(frame_path: str, box: tuple[int, int, int, int]) -> bool:
    """字幕外观先验：候选框必须是「浅色笔画 + 暗 背景/描边」形态。

    只看"尺寸 + 底部 40%"会漏杀一种场景：字幕压在文字密集的亮底背景上
    （网页/文档截图），页面正文行整体落在画面下部 40% 且宽度过尺寸先验，
    被误当字幕擦除 → LaMa 把正文抹花。白字字幕与白底黑字正文的灰度
    分布差异显著（实测 1920x1080 真帧）：
        网页正文行  white 0.64-0.75  dark ≤0.06
        白字大字幕  white ≈0.46      dark ≈0.47
    判读失败（读图异常/空框）宁可放行也不误杀真字幕。
    """
    import numpy as np
    from PIL import Image

    x1, y1, x2, y2 = box
    try:
        arr = np.asarray(Image.open(frame_path).convert("L").crop((x1, y1, x2, y2)))
    except Exception:
        return True
    if arr.size == 0:
        return False
    white = float((arr >= 200).mean())
    dark = float((arr <= 80).mean())
    # 白底黑字（正文/文档）：大面积亮底 + 几乎无暗部 → 不是白字字幕
    return not (white > 0.60 and dark < 0.10)


def locate_subtitle_region(
    frame_paths: list[str], *, api_key: str,
    frame_width: int | None = None, frame_height: int | None = None,
) -> list[dict]:
    """多帧聚合出字幕擦除区列表 [{x,y,w,h}, ...]（外扩 padding 防描边残留）。

    云路径：抽样帧 → qwen3.5-ocr 检测底部字幕条 bbox → 白字外观先验 →
    共享聚合逻辑。全部帧无字幕返回空列表。
    """
    if not api_key:
        raise ValueError("收到空 API key。请检查 AI_SPEECH_API_KEY 是否设置（项目 .env）。")

    candidates: list[tuple[str, tuple[int, int, int, int]]] = []
    for path in frame_paths:
        for entry in _ocr_frame(path, api_key):
            if frame_width is None or frame_height is None:
                continue
            px = _rect_to_pixels(entry, frame_width, frame_height)
            if px is None:
                continue
            x1, y1, x2, y2 = px
            bw, bh = x2 - x1, y2 - y1
            # 尺寸先验：取长边（兼容竖排字幕——宽 23px 但高 477px）。
            # 短边可能是旋转 90° 的细高文字，不应因"窄"被丢弃。
            # 0.04：容忍句尾短碎片（原 0.1 会漏掉边缘残字，擦除后留鬼影）
            if frame_width is not None and max(bw, bh) < frame_width * 0.04:
                continue
            # 底部区域先验：bbox 底边应进入画面下部 40%（用 y2 而非 y1，
            # 否则顶部起始的竖排标题 y1=760 刚好落在阈值 768 之上被误杀）。
            if frame_height is not None and y2 < frame_height * 0.6:
                continue
            candidates.append((path, (x1, y1, x2, y2)))

    passed = [box for path, box in candidates if _is_white_subtitle(path, box)]
    # 全部候选被外观先验拒绝 → 回落不过滤（彩色字/特殊样式不被误杀）
    boxes = passed or [box for _, box in candidates]
    return _aggregate_regions(boxes, frame_width, frame_height)
