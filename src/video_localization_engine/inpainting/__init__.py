"""L6: InpaintingEngine — backend-swappable image inpainting."""
from video_localization_engine.inpainting.base import InpaintingBackend
from video_localization_engine.inpainting.inpainting_engine import InpaintingEngine
from video_localization_engine.inpainting.registry import InpaintingBackendRegistry

# 默认注册 — orchestrator 无需手动 register
from video_localization_engine.inpainting.backends import LamaInpaintingBackend, OpenCVInpaintBackend  # noqa: F401
InpaintingBackendRegistry.register("opencv", OpenCVInpaintBackend)
InpaintingBackendRegistry.register("lama", LamaInpaintingBackend)

__all__ = [
    "InpaintingBackend", "InpaintingEngine", "InpaintingBackendRegistry",
    "OpenCVInpaintBackend", "LamaInpaintingBackend",
]
