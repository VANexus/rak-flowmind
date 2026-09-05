"""本地 OCR：RapidOCR（onnxruntime），字幕条定位本地推理。

架构升级（GPU 化）：OCR 从「云优先」改为「本地优先」。抽样帧定位字幕条是
轻量场景（每视频仅 cfg.ocr_frame_count 帧），CPU 推理即可，无需 GPU，
规避 onnxruntime-gpu 的 cuDNN/Pascal 兼容负担。云路径（_cloud_ocr）保留。

接口与 _cloud_ocr.locate_subtitle_region 对齐（去 api_key）：
    输入抽样帧路径列表 → 输出 [{x,y,w,h}, ...] 擦除区列表。
多帧聚合（尺寸/底部先验 + 水平/垂直聚类 + padding）复用 _cloud_ocr
的 _aggregate_regions，保证两条后端行为一致。

lazy import：rapidocr_onnxruntime 未安装 → OCRError(category="environment")。
模型实例进程内单例缓存。
"""
from __future__ import annotations

from flowmind.skills._cloud_ocr import OCRError, _aggregate_regions, _is_white_subtitle
from flowmind.tasks.gpu import model_cache_guard


def available() -> bool:
    """rapidocr_onnxruntime 是否可导入（供 backend=auto 判断；测试可替换）。"""
    try:
        import rapidocr_onnxruntime  # noqa: F401  # 导入探测本身即目的，勿删
        return True
    except Exception:
        return False


# 进程内单例
_ocr_engine: object | None = None


def _get_engine():
    """进程内单例 + 双检锁懒加载（并发首调不重复初始化，懒加载语义不变）。"""
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine
    with model_cache_guard():
        if _ocr_engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise OCRError(
                    "未安装 rapidocr-onnxruntime（environment.yml 已含，conda env update 后可用）",
                    category="environment",
                ) from exc
            try:
                _ocr_engine = RapidOCR()
            except Exception as exc:
                raise OCRError(
                    f"本地 OCR 引擎初始化失败: {type(exc).__name__}", category="environment",
                ) from exc
    return _ocr_engine


def _detect_frame(frame_path: str) -> list[tuple[int, int, int, int]]:
    """单帧检测 → 像素框列表 [(x1,y1,x2,y2), ...]。

    RapidOCR 返回 [[box(4 点), text, score], ...]；低置信度（<0.5）丢弃。
    """
    engine = _get_engine()
    try:
        result, _elapsed = engine(frame_path)
    except Exception as exc:
        raise OCRError(
            f"本地 OCR 推理异常: {type(exc).__name__}", category="video",
        ) from exc
    boxes: list[tuple[int, int, int, int]] = []
    for item in result or []:
        try:
            quad, _text, score = item[0], item[1], float(item[2])
        except (IndexError, TypeError, ValueError):
            continue
        if score < 0.5:
            continue
        xs = [float(p[0]) for p in quad]
        ys = [float(p[1]) for p in quad]
        x1, y1, x2, y2 = round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def locate_subtitle_region(
    frame_paths: list[str], *,
    frame_width: int | None = None, frame_height: int | None = None,
) -> list[dict]:
    """多帧聚合出字幕擦除区列表 [{x,y,w,h}, ...]。

    与 _cloud_ocr.locate_subtitle_region 输出契约一致（免 api_key）。
    先验过滤与多帧聚合逻辑完全复用云路径实现。
    """
    candidates: list[tuple[str, tuple[int, int, int, int]]] = []
    for path in frame_paths:
        for x1, y1, x2, y2 in _detect_frame(path):
            if frame_width is None or frame_height is None:
                continue
            bw, bh = x2 - x1, y2 - y1
            # 尺寸先验：取长边（兼容竖排字幕），与云路径一致。
            # 0.04：容忍句尾短碎片（原 0.1 会漏掉边缘残字，擦除后留鬼影）
            if max(bw, bh) < frame_width * 0.04:
                continue
            # 底部区域先验：bbox 底边应进入画面下部 40%，与云路径一致。
            if y2 < frame_height * 0.6:
                continue
            candidates.append((path, (x1, y1, x2, y2)))

    passed = [box for path, box in candidates if _is_white_subtitle(path, box)]
    # 全部候选被外观先验拒绝 → 回落不过滤（彩色字/特殊样式不被误杀）
    boxes = passed or [box for _, box in candidates]
    return _aggregate_regions(boxes, frame_width, frame_height)
