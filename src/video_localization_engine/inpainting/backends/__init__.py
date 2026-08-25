"""inpainting backends — 各 backend 实现。"""
from video_localization_engine.inpainting.backends.lama_backend import LamaInpaintingBackend
from video_localization_engine.inpainting.backends.opencv_backend import OpenCVInpaintBackend

__all__ = ["LamaInpaintingBackend", "OpenCVInpaintBackend"]
