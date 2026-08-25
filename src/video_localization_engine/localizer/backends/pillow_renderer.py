"""PillowRenderer — PIL/Pillow 字幕渲染，支持中文 + 英文。

cv2.putText 只支持 Hershey (英文 ASCII) 字体,渲染中文变成方框。
PillowRenderer 自动选字体:
  - 优先: WenQuanYi / NotoSans / SourceHan (中文)
  - 退化: DejaVuSans (英文)
  - 都找不到则: 用 PIL 默认字体 (字符渲染不了也不报错)

P10 改进:
  - 自动缩字: text 渲染宽度 > bbox 宽度 * shrink_threshold 时按比例缩字 (最多 50%)
  - 自动换行: 缩到最小仍超 → 按词 (EN 按空格, CJK 按字) 切 max_wrap_lines 行
  - 居中: 水平 + 垂直都居中 (多行垂直等距居中)
  - 描边: 默认 1px 黑色描边 (提升可读性)

构造参数:
  color:            RGB tuple, 字色
  stroke_color:     RGB tuple, 描边色
  stroke_width:     int, 描边像素 (0 = 不描边)
  font_scale:       float (0.0~1.0), 字号占 bbox 高度的比例
  padding:          int, bbox 内边距
  shrink_threshold: text 宽度超过 bbox.width * shrink_threshold 时启动缩字 (默认 0.95)
  min_font_scale:   字号最小缩到 font_scale * min_font_scale (默认 0.5)
  max_wrap_lines:   最多换几行 (默认 3)

环境变量覆盖字体搜索:
  VLE_PIL_FONT_CN : 中文字体 .ttf / .otf / .ttc 路径 (优先于系统查找)
  VLE_PIL_FONT_EN : 英文字体 .ttf 路径
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from video_localization_engine.localizer.protocols import RendererBackend
from video_localization_engine.types.detection import BBox


_log = logging.getLogger(__name__)


# --------------------------------- font discovery ---------------------------------

_CN_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/source-han-sans/SourceHanSansSC-Regular.otf",
    "/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]
_EN_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False


@lru_cache(maxsize=1)
def _pick_font_path(*, want_cjk: bool) -> Optional[str]:
    """返回第一个可用的字体路径;找不到返回 None。"""
    override_env = ("VLE_PIL_FONT_CN" if want_cjk else "VLE_PIL_FONT_EN")
    override = os.environ.get(override_env, "").strip()
    if override and _exists(override):
        return override
    pool = _CN_CANDIDATES if want_cjk else _EN_CANDIDATES
    for p in pool:
        if _exists(p):
            return p
    return None


def _has_cjk(text: str) -> bool:
    """粗略判断: text 是否含 CJK 字符。"""
    for ch in text:
        cp = ord(ch)
        if (
            0x3000 <= cp <= 0x303F
            or 0x3400 <= cp <= 0x4DBF
            or 0x4E00 <= cp <= 0x9FFF
            or 0xF900 <= cp <= 0xFAFF
            or 0xFF00 <= cp <= 0xFFEF
            or 0xAC00 <= cp <= 0xD7AF
        ):
            return True
    return False


def _measure(draw: ImageDraw.ImageDraw, text: str,
             font: ImageFont.ImageFont,
             stroke_width: int) -> Tuple[int, int, int, int]:
    """PIL measure bbox. 返回 (l, t, r, b)."""
    try:
        return draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    except AttributeError:
        tw, th = font.getsize(text)
        return (0, 0, tw, th)


def _wrap_text(text: str, max_lines: int = 3) -> List[str]:
    """把 text 切到 ≤ max_lines 行.

    EN / has-space: 按词平衡切
    CJK / no-space: 按字符平均切
    """
    if max_lines <= 0:
        return [text]
    has_space = " " in text
    if has_space:
        words = text.split(" ")
        per_line = max(1, (len(words) + max_lines - 1) // max_lines)
        lines: List[str] = []
        i = 0
        while i < len(words):
            if len(lines) >= max_lines:
                # 溢出追加到末行
                rest = " ".join(words[i:])
                lines[-1] = ((lines[-1] + " " + rest).strip()
                             if lines[-1] else rest)
                break
            chunk = words[i:i + per_line]
            lines.append(" ".join(chunk))
            i += per_line
        return lines if lines else [text]
    # CJK / no-space: 按字
    n = len(text)
    if n <= max_lines:
        return list(text) if n > 1 else [text]
    per = (n + max_lines - 1) // max_lines
    lines = []
    i = 0
    while i < n:
        if len(lines) >= max_lines:
            lines[-1] += text[i:]
            break
        chunk = text[i:i + per]
        if chunk:
            lines.append(chunk)
        i += per
    return lines if lines else [text]


# --------------------------------- the backend ---------------------------------


class PillowRenderer(RendererBackend):
    """PIL/Pillow 字幕渲染 — 自动选字体、CJK 友好、自动缩字/换行/居中/描边。"""

    def __init__(self,
                 color: Tuple[int, int, int] = (255, 255, 255),
                 stroke_color: Tuple[int, int, int] = (0, 0, 0),
                 stroke_width: int = 1,
                 font_scale: float = 0.7,
                 padding: int = 6,
                 shrink_threshold: float = 0.95,
                 min_font_scale: float = 0.5,
                 max_wrap_lines: int = 3):
        self.color = tuple(color)
        self.stroke_color = tuple(stroke_color)
        self.stroke_width = int(stroke_width)
        self.font_scale = float(font_scale)
        self.padding = int(padding)
        self.shrink_threshold = float(shrink_threshold)
        self.min_font_scale = float(min_font_scale)
        self.max_wrap_lines = max(1, int(max_wrap_lines))
        self._font_cache: dict = {}

    @property
    def name(self) -> str:
        return "pillow"

    def _get_font(self, text: str, font_size: int) -> ImageFont.ImageFont:
        want_cjk = _has_cjk(text)
        path = _pick_font_path(want_cjk=want_cjk)
        font_size = max(8, int(font_size))
        key = (path, want_cjk, font_size)
        if key not in self._font_cache:
            if path is not None:
                try:
                    self._font_cache[key] = ImageFont.truetype(path, font_size)
                except Exception:
                    self._font_cache[key] = ImageFont.load_default()
            else:
                self._font_cache[key] = ImageFont.load_default()
        return self._font_cache[key]

    def render(self, text: str, bbox: BBox,
               frame_size: Tuple[int, int], **kwargs) -> np.ndarray:
        """绘制 (text, bbox) 到 RGBA canvas (HxWx4 uint8)."""
        h, w = frame_size
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        if not bbox or not text or not text.strip():
            return canvas
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return canvas

        inner_w = max(1, bw - 2 * self.padding)
        inner_h = max(1, bh - 2 * self.padding)
        limit_w = inner_w * self.shrink_threshold

        # 1. 选基准字号 (PIL fontsize 整数)
        base_font_size = max(8, int(inner_h * self.font_scale))
        min_size = max(8, int(base_font_size * self.min_font_scale))

        # 2. 自动缩字循环 — 直到 ≤ limit_w 或到 min_size
        tmp = Image.new("RGBA", (inner_w, inner_h), (0, 0, 0, 0))
        tmp_draw = ImageDraw.Draw(tmp)
        cur_size = base_font_size
        cur_text = text
        # 第一轮: 实测基准
        font = self._get_font(cur_text, cur_size)
        l, t, r, b = _measure(tmp_draw, cur_text, font, self.stroke_width)
        tw, th = r - l, b - t
        shrink_iters = 0
        while tw > limit_w and cur_size > min_size and shrink_iters < 6:
            scale = limit_w / max(tw, 1)
            new_size = max(min_size, int(cur_size * scale))
            if new_size >= cur_size:
                break
            cur_size = new_size
            font = self._get_font(cur_text, cur_size)
            l, t, r, b = _measure(tmp_draw, cur_text, font, self.stroke_width)
            tw, th = r - l, b - t
            shrink_iters += 1

        # 3. 若仍超宽 → 自动换行
        lines: List[str] = [cur_text]
        if tw > limit_w and self.max_wrap_lines > 1:
            lines = _wrap_text(cur_text, max_lines=self.max_wrap_lines)

        # 4. 画 region (RGBA)
        region = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(region)

        # 实测每行 metrics
        line_metrics: List[Tuple[int, int, int, int]] = []
        for line in lines:
            l_, t_, r_, b_ = _measure(draw, line, font, self.stroke_width)
            line_metrics.append((l_, t_, r_, b_))
        if not line_metrics:
            return canvas
        tops = [m[1] for m in line_metrics]
        bots = [m[3] for m in line_metrics]
        total_h = max(bots) - min(tops)
        total_w = max((m[2] - m[0]) for m in line_metrics)
        if total_w <= 0 or total_h <= 0:
            return canvas

        # 5. 多行垂直布局 — 每行 line_h = 字号 * 1.2, 中间有 gap
        line_gap = max(2, int(cur_size * 0.15))
        line_h = max(8, int(cur_size * 1.2))
        block_h = line_h * len(lines) - line_gap  # 最后一行无 gap
        # 整体 bbox 内垂直居中
        block_top = max(0, (bh - block_h) // 2)
        first_top = min(tops)

        for idx, (line, m) in enumerate(zip(lines, line_metrics)):
            l_, t_, r_, b_ = m
            lw = r_ - l_
            # 水平居中
            tx = (bw - lw) // 2 - l_
            # 垂直: 行起点 + (行内 top - first_top) 偏移, 让首行 top 对齐 block_top
            ty = block_top + idx * line_h + (t_ - first_top)
            if self.stroke_width > 0:
                draw.text((tx, ty), line, font=font,
                          fill=(*self.color, 255),
                          stroke_width=self.stroke_width,
                          stroke_fill=(*self.stroke_color, 255))
            else:
                draw.text((tx, ty), line, font=font,
                          fill=(*self.color, 255))

        # 6. 把 region 贴回画布
        canvas_arr = np.array(region)
        canvas[y1:y2, x1:x2, :] = canvas_arr
        return canvas
