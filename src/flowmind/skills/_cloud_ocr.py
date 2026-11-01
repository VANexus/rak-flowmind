"""字幕区定位：百炼 qwen3.5-ocr（云优先，零本地模型）。

策略：对抽样帧调 qwen3.5-ocr 检测叠加文字（字幕/水印/角标）bbox；
多帧聚合取包围盒并集（外扩 padding），作为擦除区域。

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


_PROMPT = "检测图中所有叠加文字（字幕、水印、角标）的边界框，输出JSON。"

# 同一文本行的纵向合并判据：两框中心距 ≤ 文字高度 × 本系数。取 <1 使
# 纵向相邻但相互独立的两行（或"字幕 + 游走水印"）不被并成一个擦除区/一个
# 行高样本。见 _aggregate_regions 注释中的实测数据。
_LINE_MERGE_RATIO = 0.8


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

    此处不再按横向距离丢弃"孤立簇"：候选去留已由 _passes_region_prior
    （底部带/顶部带/贴边缘）与 _is_white_subtitle 决定，横向再丢弃会把
    贴右边缘的账号水印（与居中字幕 cx 相差 >0.35W）整块漏擦——正是要擦的
    目标（水印逐帧漂移但始终 hug 边缘）。
    """
    if not boxes:
        return []

    # 按 y 聚类成"文本行"：仅当两框垂直间距小于文字自身高度（_LINE_MERGE_RATIO）
    # 才算同一行。绝不能只按"中心距 < 固定比例画高"——漂移水印会纵向游走到
    # 字幕附近（实测参考片水印 cy 1189/1237 vs 字幕 cy 1328，间距 91px < 115px
    # 阈值），一旦并成一行，该区高度会从"单行 ≈103px"膨胀到 215px，下游
    # _match_original_style 据此推出的字号会虚大（215×1.16→被钳到画高 9%）。
    y_sorted = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
    y_clusters: list[list[tuple[int, int, int, int]]] = [[y_sorted[0]]]
    for b in y_sorted[1:]:
        prev = y_clusters[-1][-1]
        prev_cy = (prev[1] + prev[3]) / 2
        cur_cy = (b[1] + b[3]) / 2
        ref_h = min(prev[3] - prev[1], b[3] - b[1])
        if cur_cy - prev_cy <= ref_h * _LINE_MERGE_RATIO:
            y_clusters[-1].append(b)
        else:
            y_clusters.append([b])

    # 行内再按横向连续性切分：同一 y 上相距过远的两段文字（左角标 + 右角标，
    # 或画面两侧的招牌）若并成一个横跨全宽的擦除区，会把中间的画质一起抹掉。
    # 实测反例：两个同 y 的角标框（x≈0 与 x≈1079）曾并出 (0,715,1080,93) 全宽带。
    # 门槛取 0.15 画宽——字幕内部词间距远小于此，不会把一行字幕切开。
    x_gap_max = (frame_width or 0) * 0.15
    rows: list[list[tuple[int, int, int, int]]] = []
    for cluster in y_clusters:
        cluster.sort(key=lambda b: b[0])
        run = [cluster[0]]
        for b in cluster[1:]:
            if b[0] - max(x[2] for x in run) <= x_gap_max:
                run.append(b)
            else:
                rows.append(run)
                run = [b]
        rows.append(run)

    regions = [_bbox(c) for c in rows]
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


def _passes_region_prior(
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> bool:
    """候选框位置先验：叠加文字常落的三类区域，命中任一即保留。

    尺寸先验（长边 ≥4% 画宽）由调用方先行过滤。三类区域：
    - 底部带（y2 ≥ 0.6H）：常规底部字幕条（旧实现唯一保留的区域）；
    - 顶部带（y1 ≤ 0.12H）：顶部免责声明/台标文字（如"此视频仅以娱乐为
      目的，无不良引导"）——旧实现只留底部带会整体漏擦；
    - 贴左右边缘（x1 ≤ 0.06W 或 x2 ≥ 0.94W）：账号水印/角标。该类文字
      逐帧漂移（y 变化大）但水平方向始终 hug 画面边缘；场景内实体 logo
      不贴边，据此可区分，避免误擦画面主体。

    实测依据（1080×1920 参考片）：免责声明 y29-63/x644-1047（命中顶部带）、
    水印 x2=1022~1079（命中贴右缘）、场景 MANGO logo x250-377/y747-820
    （三类都不命中 → 正确排除）。
    """
    x1, y1, x2, y2 = box
    if y2 >= frame_height * 0.6:
        return True
    if y1 <= frame_height * 0.12:
        return True
    return x1 <= frame_width * 0.06 or x2 >= frame_width * 0.94


def locate_subtitle_region(
    frame_paths: list[str], *, api_key: str,
    frame_width: int | None = None, frame_height: int | None = None,
) -> list[dict]:
    """多帧聚合出字幕擦除区列表 [{x,y,w,h}, ...]（外扩 padding 防描边残留）。

    云路径：抽样帧 → qwen3.5-ocr 检测叠加文字 bbox → 位置先验 →
    白字外观先验 → 共享聚合逻辑。全部帧无文字返回空列表。
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
            if max(bw, bh) < frame_width * 0.04:
                continue
            # 位置先验：底部带 ∪ 顶部带 ∪ 贴左右边缘（见 _passes_region_prior）。
            if not _passes_region_prior((x1, y1, x2, y2), frame_width, frame_height):
                continue
            candidates.append((path, (x1, y1, x2, y2)))

    passed = [box for path, box in candidates if _is_white_subtitle(path, box)]
    # 全部候选被外观先验拒绝 → 回落不过滤（彩色字/特殊样式不被误杀）
    boxes = passed or [box for _, box in candidates]
    return _aggregate_regions(boxes, frame_width, frame_height)
