"""L4 SubtitleManager 顶层: 串联 T1 → T2 → T3。

对外接口:
  feed(packet, candidates)  → 每帧推
  finish()                  → 返回 List[SubtitleInstance]

内部状态:
  T1 TextLineBuilder
  T2 SubtitleCandidateBuffer
  T3 SubtitleInstanceExtractor
"""
from __future__ import annotations

from typing import List, Optional

from video_localization_engine.manager.t1_text_line import TextLineBuilder
from video_localization_engine.manager.t2_subtitle_candidate import (
    SubtitleCandidateBuffer,
    SubtitleCandidateConfig,
)
from video_localization_engine.manager.t3_subtitle_instance import (
    SubtitleInstanceExtractor,
)
from video_localization_engine.types.detection import TextCandidate
from video_localization_engine.types.instance import SubtitleInstance
from video_localization_engine.types.video import FramePacket


class SubtitleManager:
    """L4 顶层: T1 → T2 → T3 串联。"""

    def __init__(self, config: Optional[SubtitleCandidateConfig] = None):
        self._t1 = TextLineBuilder()
        self._t2 = SubtitleCandidateBuffer(config or SubtitleCandidateConfig())
        self._t3 = SubtitleInstanceExtractor()
        # 内部累积 frame_candidates, 供 .vle.json 完整保存
        self._frame_candidates_buffer: List = []

    def feed(self, packet: FramePacket,
             candidates: List[TextCandidate]) -> List[SubtitleInstance]:
        """输入一帧 + OCR candidates, 返回该帧新提取的 instance。

        通常 instance 在 CLOSED 后才返回 (寿命结束的字幕)。
        """
        # 缓存 raw candidates, 供 .vle.json
        from video_localization_engine.types.detection import FrameTextCandidates
        self._frame_candidates_buffer.append(
            FrameTextCandidates(
                frame_id=packet.frame_id,
                timestamp_ms=packet.timestamp_ms,
                width=packet.width,
                height=packet.height,
                candidates=candidates,
            )
        )

        # T1: 单帧归一化
        lines = self._t1.build(packet, candidates)
        # T2: 跨帧聚合
        closed_cands = self._t2.feed(lines)
        # T3: 提取 instance (仅 closed 的)
        instances = self._t3.extract_all(closed_cands)
        return instances

    def finish(self) -> List[SubtitleInstance]:
        """所有帧结束: 强制关闭剩余 active candidates 并提取 instance。"""
        remaining = self._t2.finalize()
        self._t2.compute_all_features()
        instances = self._t3.extract_all(remaining)
        return instances

    def get_frame_candidates_snapshot(self):
        """返回已 buffer 的所有 frame candidates (供 .vle.json 保存)。"""
        return list(self._frame_candidates_buffer)

    @property
    def buffer(self) -> SubtitleCandidateBuffer:
        """暴露给 Phase D 的 classifier 用于查看 active 状态。"""
        return self._t2