"""InpaintingBackend Registry — backend 可替换。"""
from __future__ import annotations

from video_localization_engine.inpainting.base import InpaintingBackend
from video_localization_engine.utils.registry import RegistryBase


class InpaintingBackendRegistry(RegistryBase[type[InpaintingBackend]]):
    """注册 InpaintingBackend 后端类。"""
    pass
