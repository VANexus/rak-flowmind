"""VideoLocalizationPipeline — wires L1→L2→L3→L4→L5→L6 end-to-end."""
from video_localization_engine.orchestrator.debug_writer import write_debug_artifacts
from video_localization_engine.orchestrator.pipeline import (
    PipelineConfig,
    PipelineFrameResult,
    VideoLocalizationPipeline,
)

__all__ = [
    "PipelineConfig", "PipelineFrameResult", "VideoLocalizationPipeline",
    "write_debug_artifacts",
]
