"""本地 OCR：RapidOCR（onnxruntime），字幕条定位本地推理。

架构升级（GPU 化）：OCR 从「云优先」改为「本地优先」。抽样帧定位字幕条是
轻量场景（每视频仅 cfg.ocr_frame_count 帧），CPU 推理即可，无需 GPU，
规避 onnxruntime-gpu 的 cuDNN/Pascal 兼容负担。云路径（_cloud_ocr）保留。

接口与 _cloud_ocr.locate_subtitle_region 对齐（去 api_key）：
    输入抽样帧路径列表 → 输出 [{x,y,w,h}, ...] 擦除区列表。
多帧聚合（尺寸/底部先验 + 水平/垂直聚类 + padding）复用 _cloud_ocr
的 _aggregate_regions，保证两条后端行为一致。

lazy import：rapidocr（新包 v3）与 rapidocr_onnxruntime（旧包 v1）都不在 → OCRError(category="environment")。
模型实例进程内单例缓存。
"""
from __future__ import annotations

from flowmind.skills._cloud_ocr import OCRError, _aggregate_regions, _is_white_subtitle
from flowmind.tasks.gpu import model_cache_guard


def available() -> bool:
    """本机 OCR（**新旧两个包名任一可导入**）是否可用（供 backend=auto 判断；测试可替换）。

    两个包名都要认（2026-10-16 修漂移）：``rapidocr``（**新包 v3**，本机装的是它）与
    ``rapidocr_onnxruntime``（旧包 v1，`environment.yml` 历史钉的那个）。此前只探旧包名
    ⇒ 在新包环境下它恒为 ``False``，于是 ``backend=auto`` 静默落到云 OCR——**「本地优先」失效**，
    而表面看不出任何错（这正是「只登记不改」会留下的坑）。
    """
    return _engine_flavor() != ""


def _engine_flavor() -> str:
    """可用的引擎形态：``"v3"``（新包）/ ``"v1"``（旧包）/ ``""``（都没有）。**新包优先**。"""
    for flavor, module in (("v3", "rapidocr"), ("v1", "rapidocr_onnxruntime")):
        try:
            __import__(module)
            return flavor
        except Exception:
            continue
    return ""


# 进程内单例
_ocr_engine: object | None = None


def _get_engine():
    """进程内单例 + 双检锁懒加载（并发首调不重复初始化，懒加载语义不变）。"""
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine
    with model_cache_guard():
        if _ocr_engine is None:
            flavor = _engine_flavor()
            if not flavor:
                raise OCRError(
                    "未安装本机 OCR（pip install rapidocr；旧包名 rapidocr_onnxruntime 亦可）",
                    category="environment",
                )
            module = "rapidocr" if flavor == "v3" else "rapidocr_onnxruntime"
            try:
                RapidOCR = __import__(module, fromlist=["RapidOCR"]).RapidOCR
            except (ImportError, AttributeError) as exc:
                raise OCRError(
                    f"本机 OCR 包不可用（{module}）: {type(exc).__name__}", category="environment",
                ) from exc
            try:
                _ocr_engine = RapidOCR()
            except Exception as exc:
                raise OCRError(
                    f"本地 OCR 引擎初始化失败: {type(exc).__name__}", category="environment",
                ) from exc
    return _ocr_engine


def _infer_items(engine, frame_path: str) -> list[tuple[object, str, float]]:
    """跑一次推理 → 归一成 ``[(quad, text, score), ...]``（**抹平新旧两包的返回形状**）。

    为什么要有这一层：**两包的返回形状不同**——旧包 ``engine(path)`` 回 ``(items, elapsed)``，
    每项是 ``[box, text, score]``；新包回一个 ``RapidOCROutput`` 对象（``.boxes/.txts/.scores``，
    且这些字段可能是 **numpy 数组**）。此前代码只按旧形状解包 ⇒ 装上 **新包**（没有旧包）时
    会在**推理期**抛 ``TypeError`` 并被误分类成 ``video`` 类错误——换了 `environment.yml`
    却不换代码就会踩这个坑。
    """
    try:
        out = engine(frame_path)
    except Exception as exc:
        raise OCRError(
            f"本地 OCR 推理异常: {type(exc).__name__}", category="video",
        ) from exc
    # 新包：对象形状
    if hasattr(out, "boxes") or hasattr(out, "txts"):
        boxes = list(getattr(out, "boxes", None) if getattr(out, "boxes", None) is not None else [])
        texts = list(getattr(out, "txts", None) if getattr(out, "txts", None) is not None else [])
        scores = list(getattr(out, "scores", None) if getattr(out, "scores", None) is not None else [])
        items: list[tuple[object, str, float]] = []
        for index, text in enumerate(texts):
            score = scores[index] if index < len(scores) else 0.0
            quad = boxes[index] if index < len(boxes) else None
            items.append((quad, str(text), float(score)))
        return items
    # 旧包：`(items, elapsed)`
    result = out[0] if isinstance(out, tuple) and out else out
    items = []
    for item in result or []:
        try:
            items.append((item[0], str(item[1]), float(item[2])))
        except (IndexError, TypeError, ValueError):
            continue
    return items


def _detect_frame(frame_path: str) -> list[tuple[int, int, int, int]]:
    """单帧检测 → 像素框列表 [(x1,y1,x2,y2), ...]。

    引擎返回逐行四点框；低置信度（<0.5）丢弃（**这是字幕定位这条路的自己的口径**，
    与 dsp 文字核查那条路无关——那边零过滤）。
    """
    boxes: list[tuple[int, int, int, int]] = []
    for quad, _text, score in _infer_items(_get_engine(), frame_path):
        if score < 0.5 or quad is None:
            continue
        try:
            xs = [float(p[0]) for p in quad]
            ys = [float(p[1]) for p in quad]
        except (TypeError, IndexError, ValueError):
            continue
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
