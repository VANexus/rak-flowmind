"""T1: TextCandidate → TextLineCandidate。

单帧内的归一化。当前实现是 1 TextCandidate = 1 TextLineCandidate (PaddleOCR 输出粒度)。
未来可加 multi-word 合并 (e.g. 同行多个 char polygon → 一个 line)。
"""
from __future__ import annotations

from typing import List

from video_localization_engine.types.candidates import TextLineCandidate
from video_localization_engine.types.detection import TextCandidate
from video_localization_engine.types.video import FramePacket


class TextLineBuilder:
    """T1 builder: 单帧 → 单行候选列表。"""

    def build(self, packet: FramePacket, candidates: List[TextCandidate]) -> List[TextLineCandidate]:
        return [
            TextLineCandidate.from_text_candidate(c, packet.frame_id, packet.timestamp_ms)
            for c in candidates
        ]