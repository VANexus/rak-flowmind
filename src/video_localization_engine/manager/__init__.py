"""L4: SubtitleManager — T1 → T2 → T3 流水线."""
from video_localization_engine.manager.t1_text_line import TextLineBuilder
from video_localization_engine.manager.t2_subtitle_candidate import (
    SubtitleCandidateBuffer,
    SubtitleCandidateConfig,
)
from video_localization_engine.manager.t3_subtitle_instance import (
    SubtitleInstanceExtractor,
)
from video_localization_engine.manager.pipeline import SubtitleManager

__all__ = [
    "TextLineBuilder",
    "SubtitleCandidateBuffer",
    "SubtitleCandidateConfig",
    "SubtitleInstanceExtractor",
    "SubtitleManager",
]
