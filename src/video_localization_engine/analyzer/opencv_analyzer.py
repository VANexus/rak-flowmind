"""OpenCV 实现 VideoAnalyzer。

依赖: opencv-python (cv2)。

设计:
- meta 一次性从 cap metadata 推 (fps, frame_count), Orientation 推导
- 帧迭代器按需 cv2.read(), 不缓存
- seek 用 cap.set(cv2.CAP_PROP_POS_FRAMES) (有 1-2 帧解码开销, 已知 OpenCV 限制)
"""
from __future__ import annotations

import os
from typing import Iterator

import cv2
import numpy as np

from video_localization_engine.analyzer.base import VideoAnalyzer, derive_orientation
from video_localization_engine.types.video import FramePacket, VideoMeta


class OpenCVVideoAnalyzer(VideoAnalyzer):
    """OpenCV 后端 — 默认 backend。"""

    def __init__(self, path: str, source_locale: str | None = None,
                 content_type_hint: str | None = None):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self._path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"failed to open video: {path}")

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = int((frame_count / fps) * 1000) if fps else 0
        # audio: OpenCV 不可知, 暂估为 True (如果 fps > 0 且有帧数)
        has_audio = frame_count > 0  # 简化: 默认 True, 不影响字幕逻辑

        self._meta = VideoMeta(
            source_path=os.path.abspath(path),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_ms=duration_ms,
            orientation=derive_orientation(width, height),
            has_audio=has_audio,
            content_type_hint=content_type_hint,
            source_locale=source_locale,
        )

    @property
    def meta(self) -> VideoMeta:
        return self._meta

    def __iter__(self) -> Iterator[FramePacket]:
        # 每次重新从 0 开始
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, img = self._cap.read()
            if not ret:
                break
            frame_id = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            timestamp_ms = int((frame_id / self._meta.fps) * 1000)
            yield FramePacket(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                image=img,
                meta=self._meta,
            )

    def seek(self, frame_id: int) -> FramePacket:
        if frame_id < 0 or frame_id >= self._meta.frame_count:
            raise IndexError(f"frame_id {frame_id} out of [0, {self._meta.frame_count})")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, img = self._cap.read()
        if not ret:
            raise IOError(f"failed to read frame {frame_id}")
        # OpenCV CAP_PROP_POS_FRAMES 反映"下一帧要读的位置", 已读+1
        # 用我们请求的 frame_id 算 timestamp, 不依赖 OpenCV 的回读
        timestamp_ms = int((frame_id / self._meta.fps) * 1000)
        return FramePacket(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            image=img,
            meta=self._meta,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None