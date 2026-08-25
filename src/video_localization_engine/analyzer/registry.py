"""L1 VideoAnalyzer Registry — backend 可替换。"""
from __future__ import annotations

from video_localization_engine.analyzer.base import VideoAnalyzer
from video_localization_engine.utils.registry import RegistryBase


class VideoAnalyzerRegistry(RegistryBase[type[VideoAnalyzer]]):
    """注册 VideoAnalyzer 后端类 (不是实例)。"""
    pass


# 默认注册 OpenCV
from video_localization_engine.analyzer.opencv_analyzer import OpenCVVideoAnalyzer
VideoAnalyzerRegistry.register("opencv", OpenCVVideoAnalyzer)