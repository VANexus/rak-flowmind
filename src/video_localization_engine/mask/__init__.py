"""L5: MaskGenerator — backend-agnostic subtitle mask artifacts."""
from video_localization_engine.mask.artifact import MaskArtifact, MaskType
from video_localization_engine.mask.base import MaskBackend
from video_localization_engine.mask.mask_generator import MaskGenerator
from video_localization_engine.mask.registry import MaskBackendRegistry

# 默认注册 — orchestrator 无需手动 register
from video_localization_engine.mask.backends.polygon_backend import PolygonMaskBackend
MaskBackendRegistry.register("polygon", PolygonMaskBackend)

__all__ = [
    "MaskType", "MaskArtifact", "MaskBackend", "MaskGenerator",
    "MaskBackendRegistry", "PolygonMaskBackend",
]
