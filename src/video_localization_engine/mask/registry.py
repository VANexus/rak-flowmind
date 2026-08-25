"""MaskBackend Registry — backend 可替换。"""
from __future__ import annotations

from video_localization_engine.mask.base import MaskBackend
from video_localization_engine.utils.registry import RegistryBase


class MaskBackendRegistry(RegistryBase[type[MaskBackend]]):
    """注册 MaskBackend 后端类。"""
    pass
