"""本地字幕区擦除：LaMa inpainting（simple-lama-inpainting 封装）。

delogo 横带的升级替代：LaMa 对字幕区做图像级修复，复杂背景
（网页/渐变/纹理）不再出现竖向拉丝。逐帧流程：

    ffmpeg 抽帧 → 逐帧掩码（ROI 并集内白字笔画精修，前景不误伤）
    → LaMa ROI 修复（裁字幕邻域→修复→贴回，绝不整帧喂模型）
    → ffmpeg 按源帧率重编码（无声视频，配音在后级拼接）

性能设计（Pascal 8GB 实测口径）：
- ROI 裁剪：LaMa 输入从整帧 1080×1920 缩到字幕邻域（外扩 64px），
  像素量降 ~10×，单帧推理降同量级；crop 内坐标与整帧掩码逻辑一致
- 分区裁剪（关键）：LaMa 的 FFC 主干保留全分辨率特征图，显存与耗时随
  **输入面积**线性上升，故纵向远离的擦除区（底部字幕 + 角标水印）绝不
  合成一个并集框——框内空白是纯浪费。实测竖屏 1080×1920 两区：
  并集框 900×723（65 万像素）535ms/帧、峰值 7.2GB；分区两次共 34 万像素
  247ms/帧、峰值 3.0GB。画质等价（每框仍留 64px 上下文）
- CPU/GPU 重叠：loader 线程做 PNG 解码 + 裁剪 + 掩码（纯 CPU），
  与主线程 LaMa GPU 推理流水线化（CUDA 调用释放 GIL），有界队列背压
- 产物 JPEG quality=95（4:4:4）：重编码 x264 crf18 本身有损，
  JPEG 存盘比 PNG 快数倍；下游对帧只做一次有损编码，视觉无差
- 批量推理不做：显存随面积×batch 线性上升，并集框单样本已占 7.2GB，
  batch≥2 直接 OOM（实测）；且 simple-lama API 单图，ROI 分区是主收益

模型 big-lama（~200MB）首次使用经 torch hub 自动下载到
HF 缓存（/srv/data/models，走代理）；懒加载，8GB 显存安全。
失败显式抛 InpaintError（category/retriable 与 errors.py 对齐），
绝不静默回落 delogo——回落由调用方按 backend 配置决策。
"""
from __future__ import annotations

import threading
from pathlib import Path
from queue import Queue

from flowmind.skills import _media
from flowmind.tasks.gpu import model_cache_guard


class InpaintError(Exception):
    """LaMa 修复失败。category/retriable 语义与 errors.py 对齐。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def available() -> bool:
    """simple-lama-inpainting 可导入即视为可用（模型首次用时才下载）。"""
    try:
        import simple_lama_inpainting  # noqa: F401
        return True
    except ImportError:
        return False


# SimpleLama 进程内单例（双检锁懒加载）。
# 修复历史缺陷：原先逐帧修复循环外每次调用 SimpleLama()，都会把 big-lama
# （~200MB）重新拉进显存——既慢又放大显存峰值；单例化后全进程共享一次加载。
_lama_singleton: object | None = None


def _get_lama():
    global _lama_singleton
    if _lama_singleton is not None:
        return _lama_singleton
    with model_cache_guard():
        if _lama_singleton is None:
            try:
                from simple_lama_inpainting import SimpleLama
            except ImportError as exc:
                raise InpaintError(
                    "未安装 simple-lama-inpainting（environment.yml 刻意不含——"
                    "stale metadata 见文件内注释；按 README/CONTRIBUTING 步骤安装："
                    "conda run -n flowmind pip install --no-deps "
                    "\"simple-lama-inpainting>=0.1.2\"）",
                    category="environment",
                ) from exc
            _lama_singleton = SimpleLama()
    return _lama_singleton


def erase_regions(
    src: str,
    regions: list[dict],
    out_path: str,
    workdir: str,
    solid_regions: list[dict] | None = None,
) -> str:
    """逐帧 LaMa 修复字幕框，并可对固定水印框执行整框修复。

    经 _inpaint_frames_adapter 薄适配层执行，测试可 monkeypatch 替换。
    """
    if not regions and not solid_regions:
        raise InpaintError("擦除区列表为空（应先走 OCR 定位）", category="video")
    return _inpaint_frames_adapter(
        src, regions, out_path, workdir, solid_regions=solid_regions,
    )


_CROP_MARGIN = 64          # 裁剪框外扩量：LaMa 修复所需的上文/下文像素
# 合并判据：合并后多出的"空白面积"不超过一次调用的固定开销（以像素折算）。
# 实测 P104 竖屏：面积系数 ≈846 ms/Mpx，单次调用固定开销 ≈32ms → 折算 3.8 万像素。
# 即"合并省下的一次调用"值不值这块多喂进去的空白。
_MERGE_OVERHEAD_PX = 40000


def _box_of(regions: list[dict], width: int, height: int,
            margin: int = _CROP_MARGIN) -> tuple[int, int, int, int]:
    """一组 region 矩形并集外扩 margin 的裁剪框（clamp 到画面内）。"""
    x0 = min(int(r["x"]) for r in regions)
    y0 = min(int(r["y"]) for r in regions)
    x1 = max(int(r["x"] + r["w"]) for r in regions)
    y1 = max(int(r["y"] + r["h"]) for r in regions)
    return (max(0, x0 - margin), max(0, y0 - margin),
            min(width, x1 + margin), min(height, y1 + margin))


def _area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _group_regions(regions: list[dict], width: int, height: int,
                   margin: int = _CROP_MARGIN) -> list[list[dict]]:
    """擦除区按纵向邻近性分组，每组一个独立裁剪框。

    LaMa（FFC 全分辨率特征图）的显存/耗时随**输入面积**线性上升，而
    并集框里区域之间的空白是纯浪费。故按 y 排序后贪心合并：仅当并集框
    比两框各自面积之和多出的部分不超过 ``_MERGE_OVERHEAD_PX``（≈一次调用的
    固定开销）时才合并——紧邻的同排碎框合并划算，纵向远离的框
    （底部字幕 vs 角标水印）各自独立。每框仍留 margin 上下文，掩码逻辑
    不变 → 画质等价。
    """
    groups: list[list[dict]] = []
    for r in sorted(regions, key=lambda r: int(r["y"])):
        if groups:
            merged_area = _area(_box_of(groups[-1] + [r], width, height, margin))
            cur_area = _area(_box_of(groups[-1], width, height, margin))
            new_area = _area(_box_of([r], width, height, margin))
            if merged_area - cur_area - new_area <= _MERGE_OVERHEAD_PX:
                groups[-1].append(r)
                continue
        groups.append([r])
    return groups


def _text_line_mask(img, roi_np, roi, roi_w, solid_np=None):
    """逐帧掩码：白字笔画检出 → 整行矩形外扩，限制在 ROI 内。

    img 为 LaMa 输入视图（crop 或整帧）；roi_np/roi 已同坐标系。
    solid_np 为实心块（水印等）掩码数组，非空时无条件并入结果（整块必擦）。

    - 白字判定：灰度 >=200（典型字幕白字）
    - 整行矩形外扩：很多字幕带黑描边/半透明底块（颜色与背景混合，
      灰度阈值不可靠），且底块可在白字外侧超出 OCR 框十余像素——
      只修白笔画会让 LaMa 从底块取色 → 白字变黑字残影。故掩码
      垂直方向覆盖整个 ROI 带（ROI 本身即紧贴字幕的外扩区域），
      水平取白字 bbox 外扩 10px，整块交给 LaMa 上下文重建
    - 实心块过滤：填充率(area/bbox_area)>0.5 且 bbox 宽 >80% ROI 宽的
      组件是前景实体（白墙/产品面），不是文字笔画——删除；
      亮背景下文字与背景粘连成高填充大组件被删后 sum≈0，
      自然触发整框回落（该场景下正确行为）
    - 帧内无可辨识白字（彩色字/白底白字）→ 回落整块 ROI
    """
    import numpy as np
    from PIL import Image
    from scipy import ndimage

    gray = np.asarray(img.convert("L"))
    text = (gray >= 200) & roi_np
    lab, n = ndimage.label(text)
    if n:
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            bh = sl[0].stop - sl[0].start
            bw = sl[1].stop - sl[1].start
            comp = lab[sl] == i
            fill = comp.sum() / (bh * bw)
            if fill > 0.5 and bw > roi_w * 0.8:
                text[sl][comp] = False
    if int(text.sum()) < gray.size * 0.0005:
        keep = roi_np  # 整框回落
    else:
        _, xs = np.nonzero(text)
        x0t, x1t = int(xs.min()), int(xs.max())
        line = np.zeros_like(text)
        line[:, max(0, x0t - 10):x1t + 11] = True  # 垂直全覆盖 ROI 带
        keep = line & roi_np
    if solid_np is not None:
        keep = keep | solid_np  # 实心块整块必擦，不参与白字检测
    return Image.fromarray((keep * 255).astype("uint8"))


def _inpaint_frames_adapter(
    src: str,
    regions: list[dict],
    out_path: str,
    workdir: str,
    solid_regions: list[dict] | None = None,
) -> str:
    """逐帧修复主体：抽帧 → 掩码 → LaMa（ROI）→ 重编码。

    CPU（解码/裁剪/掩码，loader 线程）与 GPU（LaMa，主线程）流水线化；
    有界队列背压，任一环节异常汇总后统一转 InpaintError。
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise InpaintError(
            "未安装 simple-lama-inpainting（conda env update -f environment.yml）",
            category="environment",
        ) from exc

    duration_s, width, height = _media.probe_media(src)
    fps = _probe_fps(src)
    if not (width and height):
        raise InpaintError(f"探测视频尺寸失败: {src}", category="video")

    frames_dir = Path(workdir) / "frames"
    out_dir = Path(workdir) / "inpainted_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 全量抽帧（-vsync 0 逐帧精确对应，帧率由重编码阶段还原）
    rc, _, err = _media.run_ffmpeg([
        "ffmpeg", "-y", "-i", src, "-vsync", "0",
        str(frames_dir / "f_%06d.png"),
    ])
    if rc != 0:
        raise InpaintError(f"抽帧失败: {err[-200:]}", category="video")

    # 2) ROI 基准（矩形并集，调用方 _prep_erase_regions 已外扩）+ 分区裁剪框。
    #    掩码与 LaMa 全部在各自 crop 坐标系内工作（裁剪不缩放，像素坐标一致）。
    import numpy as np
    from PIL import Image, ImageDraw
    # 字幕 ROI（白字检测范围）与实心块 ROI（水印等，整块必擦）分开画
    roi_full = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(roi_full)
    for r in regions:
        draw.rectangle([
            max(0, int(r["x"]) - 4), max(0, int(r["y"]) - 4),
            min(width, int(r["x"] + r["w"]) + 4),
            min(height, int(r["y"] + r["h"]) + 4),
        ], fill=255)
    solid_full = Image.new("L", (width, height), 0)
    solid_draw = ImageDraw.Draw(solid_full)
    for r in solid_regions or []:
        solid_draw.rectangle([
            max(0, int(r["x"]) - 4), max(0, int(r["y"]) - 4),
            min(width, int(r["x"] + r["w"]) + 4),
            min(height, int(r["y"] + r["h"]) + 4),
        ], fill=255)
    # 每个分区一个 (裁剪框, 字幕ROI图, 字幕ROI数组, 实心块数组, 该组最宽
    # region) —— 逐帧复用。字幕与实心块一起分组：紧邻的并入同一裁剪框，
    # 省一次 LaMa 调用；远离的（底部字幕 vs 角标水印）各自独立
    plans = []
    for group in _group_regions(list(regions) + list(solid_regions or []),
                                width, height):
        box = _box_of(group, width, height)
        plans.append((box, roi_full.crop(box),
                      np.asarray(roi_full.crop(box)) > 0,
                      np.asarray(solid_full.crop(box)) > 0,
                      max((int(r["w"]) for r in group), default=0)))

    # 3) LaMa 修复（loader 线程与 GPU 流水线化；产物 JPEG q95 省盘省时）
    frame_paths = sorted(frames_dir.glob("f_*.png"))
    if not frame_paths:
        raise InpaintError(f"抽帧产物为空: {frames_dir}", category="video")

    errors: list[Exception] = []
    q: Queue = Queue(maxsize=8)  # 背压：loader 快于 GPU 时阻塞在 put

    def _loader() -> None:
        try:
            for fp in frame_paths:
                img_full = Image.open(fp).convert("RGB")
                items = []
                for box, roi, roi_np, solid_np, roi_w in plans:
                    crop = img_full.crop(box)
                    items.append((box, crop,
                                  _text_line_mask(crop, roi_np, roi, roi_w,
                                                  solid_np)))
                q.put((fp, img_full, items))
        except Exception as exc:  # noqa: BLE001  统一转 InpaintError
            errors.append(exc)
        finally:
            q.put(None)

    loader_t = threading.Thread(target=_loader, name="inpaint-loader", daemon=True)
    loader_t.start()
    completed = False  # 正常消费完（含 None 哨兵）才置位
    try:
        lama = _get_lama()
        for fp, img_full, items in iter(q.get, None):
            for box, crop, mask in items:  # 逐分区修复后贴回（分区独立，互不重叠）
                result = lama(crop, mask)
                if result.size != crop.size:
                    # simple-lama 对输入做 /8 symmetric 补边， jit 模型可能
                    # 原样返回补边尺寸——裁回左上角原尺寸再贴回
                    result = result.crop((0, 0, crop.size[0], crop.size[1]))
                img_full.paste(result, box[:2])
            img_full.save(str(out_dir / f"{fp.stem}.jpg"), quality=95, subsampling=0)
        completed = True
    except Exception as exc:  # noqa: BLE001  统一转 InpaintError
        errors.append(exc)
    finally:
        if not completed:
            # 异常路径排空队列：loader 可能阻塞在 q.put，不排空 join 永等
            while q.get() is not None:
                pass
        loader_t.join()
    if errors:
        exc = errors[0]
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda" in msg and "error" in msg:
            raise InpaintError(
                f"LaMa GPU 修复失败（显存不足?）: {type(exc).__name__}",
                category="transient", retriable=True,
            ) from exc
        if "connection" in msg or "download" in msg or "fetch" in msg:
            raise InpaintError(
                f"big-lama 模型下载失败（需代理）: {type(exc).__name__}",
                category="environment",
            ) from exc
        raise InpaintError(
            f"LaMa 修复失败: {type(exc).__name__}", category="transient",
        ) from exc

    # 4) 按源帧率重编码（无声；配音/混音在后级；产物帧为 JPEG）
    rc, _, err = _media.run_ffmpeg([
        "ffmpeg", "-y", "-framerate", fps,
        "-i", str(out_dir / "f_%06d.jpg"),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", out_path,
    ])
    if rc != 0:
        raise InpaintError(f"重编码失败: {err[-200:]}", category="video")
    # 抽帧中间目录很大（1080p 每秒约 40MB），成功后即清理；失败保留供排查
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    return out_path


def _probe_fps(src: str) -> str:
    """探测源视频帧率（返回 "num/den" 原始形式，供 -framerate 使用）。"""
    rc, out, err = _media.run_ffprobe([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", src,
    ])
    if rc != 0:
        raise InpaintError(f"探测帧率失败: {err[-200:]}", category="video")
    import json
    try:
        data = json.loads(out)
        rate = data["streams"][0]["r_frame_rate"]
        if not rate or rate == "0/0":
            raise ValueError(rate)
        return rate
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
        raise InpaintError(f"帧率解析失败: {exc}", category="video") from exc
