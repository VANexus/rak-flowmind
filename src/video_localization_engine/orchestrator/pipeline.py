"""VideoLocalizationPipeline — L1→L2→L3→L4→L5→L6 + L7 Localizer 端到端。

设计:
- Analyze (L1) → Detect (L3) → SubtitleManager (L4) 走完一遍累积所有 instance
- 然后二次 iterate video, 对每帧: 找出覆盖该帧的 active instance → mask → inpaint
- L7 Localizer: Translator → TTS → TimingAligner → Renderer → Compositor → mp4
- Debug artifacts 通过 debug_writer 落盘

不假设任何视频类型 / region / 字幕位置;所有 backend 通过 Registry 注入。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from video_localization_engine.analyzer.registry import VideoAnalyzerRegistry
from video_localization_engine.detector.registry import TextDetectorRegistry
from video_localization_engine.inpainting.inpainting_engine import InpaintingEngine
from video_localization_engine.localizer import (
    Compositor,
    LocalizedSubtitle,
    LocalizedTrack,
    SubtitleRenderer,
    TimingAligner,
    TtsEngine,
    Translator,
    write_localize_artifacts,
)
from video_localization_engine.localizer.types import AudioSegment
from video_localization_engine.localizer.vle_loader import deserialize_track
from video_localization_engine.manager.pipeline import SubtitleManager
from video_localization_engine.mask.artifact import MaskArtifact
from video_localization_engine.mask.mask_generator import MaskGenerator
from video_localization_engine.region_policies.base import SubtitleRegionPolicy
from video_localization_engine.region_policies.registry import applicable_policies
from video_localization_engine.types.detection import (
    FrameTextCandidates,
    RegionProposal,
)
from video_localization_engine.types.instance import SubtitleInstance, SubtitleTrack
from video_localization_engine.types.video import FramePacket, VideoMeta
from video_localization_engine.utils.persistence import load_track


@dataclass
class PipelineConfig:
    """Pipeline 配置 — 全 backend 可替换, 全参数显式。"""
    analyzer_backend: str = "opencv"
    detector_backend: str = "paddleocr_ch"
    region_policies: Optional[List[str]] = None
    mask_backend: str = "polygon"
    mask_dilation_px: int = 12
    # (P10) x/y 独立 dilation: y 方向比 x 宽, 用于覆盖字幕下边沿阴影/字幕条底色
    # 默认 None → 回退到 mask_dilation_px (x/y 同值)
    mask_dilation_y: Optional[int] = 20
    inpaint_backend: str = "opencv"
    inpaint_algorithm: str = "telea"
    inpaint_radius: int = 3
    frame_stride: int = 1
    enable_inpaint: bool = True
    debug_output_dir: Optional[str] = None
    debug_sample_rate: int = 1

    # ---- L7 Localizer 字段 (Phase E 新增) ----
    target_locale: str = "en"
    translator_backend: str = "mock"
    tts_backend: str = "mock"
    renderer_backend: str = "opencv"
    compositor_backend: str = "ffmpeg"
    tts_sample_rate: int = 24000
    renderer_font_scale: float = 1.0
    localize_output_dir: Optional[str] = None  # 默认复用 debug_output_dir


@dataclass
class PipelineFrameResult:
    """每帧的 orchestrator 输出。"""
    frame_id: int
    timestamp_ms: int
    before_image: np.ndarray
    after_image: np.ndarray
    instances_active: List[SubtitleInstance] = field(default_factory=list)
    mask_artifacts: List[MaskArtifact] = field(default_factory=list)


class VideoLocalizationPipeline:
    """L1-L7 端到端流水线。"""

    def __init__(self, video_path: str, config: Optional[PipelineConfig] = None):
        self.video_path = video_path
        self.config = config or PipelineConfig()

        analyzer_cls = VideoAnalyzerRegistry.get(self.config.analyzer_backend)
        self.analyzer = analyzer_cls(video_path)
        self.meta: VideoMeta = self.analyzer.meta

        detector_cls_or_factory = TextDetectorRegistry.get(self.config.detector_backend)
        try:
            self.detector = detector_cls_or_factory()
        except TypeError:
            self.detector = detector_cls_or_factory
        if hasattr(self.detector, "warmup"):
            self.detector.warmup()

        self.manager = SubtitleManager()
        self.mask_gen = MaskGenerator(
            backend_name=self.config.mask_backend,
            dilation_px=self.config.mask_dilation_px,
            dilation_y=self.config.mask_dilation_y,
        )
        if self.config.enable_inpaint:
            # 字幕复检 OCR — 用同一 detector (避免每次新 PaddleOCR 实例的 30s 预热)
            ocr_checker = self._build_ocr_checker(self.config.detector_backend)
            self.inpaint_eng = InpaintingEngine(
                backend_name=self.config.inpaint_backend,
                ocr_checker=ocr_checker,
                algorithm=self.config.inpaint_algorithm,
                radius=self.config.inpaint_radius,
            )
        else:
            self.inpaint_eng = None

        if self.config.region_policies is None:
            self._policies: List[SubtitleRegionPolicy] = applicable_policies(self.meta)
        else:
            from video_localization_engine.region_policies.registry import RegionPolicyRegistry
            self._policies = []
            for name in self.config.region_policies:
                cls = RegionPolicyRegistry.get(name)
                self._policies.append(cls())

        self._all_instances: List[SubtitleInstance] = []
        self._frame_candidates_buf: List[FrameTextCandidates] = []
        self._active_by_frame: Dict[int, List[SubtitleInstance]] = {}
        self._proposals_by_frame: Dict[int, List[RegionProposal]] = {}

    # ---- L1-L4 ----
    def run_to_track(self) -> SubtitleTrack:
        for pkt in self.analyzer:
            if pkt.frame_id % self.config.frame_stride != 0:
                continue
            proposals: List[RegionProposal] = []
            for pol in self._policies:
                proposals.extend(pol.propose(pkt))
            self._proposals_by_frame[pkt.frame_id] = proposals
            candidates = self.detector.detect(pkt)
            self._frame_candidates_buf.append(FrameTextCandidates(
                frame_id=pkt.frame_id,
                timestamp_ms=pkt.timestamp_ms,
                width=pkt.width,
                height=pkt.height,
                candidates=candidates,
            ))
            closed = self.manager.feed(pkt, candidates)
            self._all_instances.extend(closed)
        remaining = self.manager.finish()
        self._all_instances.extend(remaining)
        for inst in self._all_instances:
            for fid in inst.observed_frame_ids:
                self._active_by_frame.setdefault(fid, []).append(inst)
        return SubtitleTrack(
            video_meta_path=self.meta.source_path,
            instances=list(self._all_instances),
            frame_candidates=list(self._frame_candidates_buf),
            region_policies_used=[p.name for p in self._policies],
            detector_id=self.config.detector_backend,
            pipeline_version="0.1.0",
        )

    # ---- L5-L6 ----
    def run_full(self) -> List[PipelineFrameResult]:
        self.run_to_track()
        results: List[PipelineFrameResult] = []
        for pkt in self.analyzer:
            if pkt.frame_id % self.config.frame_stride != 0:
                continue
            active = self._active_by_frame.get(pkt.frame_id, [])
            artifacts = self.mask_gen.generate_for_frame(active, pkt)
            union = MaskGenerator.union(artifacts)
            if self.config.enable_inpaint and self.inpaint_eng is not None and union is not None:
                after = self.inpaint_eng.inpaint_frame(pkt, union, retest_ocr=True)
            else:
                after = pkt.image.copy()
            results.append(PipelineFrameResult(
                frame_id=pkt.frame_id,
                timestamp_ms=pkt.timestamp_ms,
                before_image=pkt.image,
                after_image=after,
                instances_active=list(active),
                mask_artifacts=artifacts,
            ))
        return results

    # ---- L7 Localizer 入口 ----
    def run_localize(self, target_locale: Optional[str] = None,
                     output_path: Optional[str] = None) -> LocalizedTrack:
        """L1-L6 + L7 全跑。"""
        results = self.run_full()
        return self._localize(self._all_instances, results, self.meta,
                              target_locale or self.config.target_locale, output_path)

    @classmethod
    def run_localize_from_vle(
        cls, vle_path: str, video_path: str, config: PipelineConfig,
        target_locale: str, output_path: Optional[str] = None,
    ) -> LocalizedTrack:
        """复用 .vle.json, 跳过 L1-L4, 仅重跑 L5-L6 + L7。"""
        raw = load_track(vle_path)
        meta, instances, _ = deserialize_track(raw)
        pipe = cls(video_path, config)
        # 替换 meta (因 vle 的 meta 不含 source_locale 全字段也没事; 用 vle 优先)
        pipe.meta = meta
        pipe._all_instances = list(instances)
        pipe._active_by_frame = {}
        for inst in instances:
            for fid in inst.observed_frame_ids:
                pipe._active_by_frame.setdefault(fid, []).append(inst)
        # L5-L6
        results: List[PipelineFrameResult] = []
        for pkt in pipe.analyzer:
            if pkt.frame_id % pipe.config.frame_stride != 0:
                continue
            active = pipe._active_by_frame.get(pkt.frame_id, [])
            artifacts = pipe.mask_gen.generate_for_frame(active, pkt)
            union = MaskGenerator.union(artifacts)
            if pipe.config.enable_inpaint and pipe.inpaint_eng is not None and union is not None:
                after = pipe.inpaint_eng.inpaint_frame(pkt, union, retest_ocr=True)
            else:
                after = pkt.image.copy()
            results.append(PipelineFrameResult(
                frame_id=pkt.frame_id,
                timestamp_ms=pkt.timestamp_ms,
                before_image=pkt.image,
                after_image=after,
                instances_active=list(active),
                mask_artifacts=artifacts,
            ))
        track = pipe._localize(instances, results, meta, target_locale, output_path)
        pipe.close()
        return track

    # ---- L7 子流水线 ----
    def _localize(self, instances: List[SubtitleInstance],
                  results: List[PipelineFrameResult], meta: VideoMeta,
                  target_locale: str, output_path: Optional[str]) -> LocalizedTrack:
        # 0. P10 step 3 — 合并同窗 instance (按时间窗重叠 ≥ 50%)
        fps = self.meta.fps if self.meta and self.meta.fps else 30.0
        merged_instances = self._merge_overlapping_instances_by_fps(
            instances, fps=fps, overlap_ratio=0.5,
        )

        # 1. 翻译 (一次 translate() 处理合并后的整段文本, 不再逐 instance)
        translator = Translator(backend_name=self.config.translator_backend)
        localized_subs: List[LocalizedSubtitle] = []
        for inst in merged_instances:
            src_text = inst.representative_text
            src_locale = inst.locale or "zh"
            tgt_text = translator.translate(src_text, src_locale, target_locale)
            # SubtitleInstance 没有 timestamp_history; start_ms 用 first_frame × (1000/fps)
            if inst.first_frame and self.meta.fps:
                start_ms = int(inst.first_frame / self.meta.fps * 1000)
            else:
                start_ms = 0
            localized_subs.append(LocalizedSubtitle(
                instance_id=inst.instance_id,
                source_text=src_text,
                target_text=tgt_text,
                source_locale=src_locale,
                target_locale=target_locale,
                start_ms=start_ms,
                end_ms=inst.duration_ms,
                bbox=inst.representative_bbox,
                polygon=inst.representative_polygon,
            ))

        # 2. TTS + TimingAligner + 拼 full_audio
        # 全局 timeline 对齐:
        #   1) 收集所有字幕的目标时间窗 [sub.start_ms, sub.end_ms)
        #   2) 对每句做 TTS, 把累计的偏移 cue 写到 sub.audio.start_ms
        #      (而非简单按 start_ms 嵌入), 避免句间累积漂移
        #   3) 长句不调速 — 慢就慢, 累计时间超过 target_end_ms 时裁掉
        #      多余部分, 后续句铺到下一个 cue 起点。这样比逐句调速更稳。
        tts = TtsEngine(backend_name=self.config.tts_backend)
        sr = self.config.tts_sample_rate
        full_audio = np.zeros(0, dtype=np.float32)
        for sub in localized_subs:
            samples = tts.synth(sub.target_text, target_locale, sample_rate=sr)
            sub.audio = AudioSegment(
                instance_id=sub.instance_id,
                start_ms=sub.start_ms, end_ms=sub.end_ms,
                sample_rate=sr, samples=samples,
                translated_text=sub.target_text,
            )

        # 全局铺轨 — 按原字幕目标时间窗逐句嵌入, 累计写回 full_audio
        full_audio = _render_audio_track(localized_subs, sr)

        # 3. 渲染 + alpha blend 帧序列
        renderer = SubtitleRenderer(
            backend_name=self.config.renderer_backend,
            font_scale=self.config.renderer_font_scale,
        )
        frame_size = (meta.height, meta.width)
        rendered_frames: List[np.ndarray] = []
        for fr in results:
            img = fr.after_image.copy()
            for sub in localized_subs:
                if sub.start_ms <= fr.timestamp_ms <= sub.end_ms and sub.bbox:
                    rgba = renderer.render(sub.target_text, sub.bbox, frame_size)
                    if rgba.shape[2] == 4 and rgba[:, :, 3].any():
                        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
                        img = (rgba[:, :, :3] * alpha + img * (1 - alpha)).astype(np.uint8)
            rendered_frames.append(img)

        # 4. mp4 输出
        rendered_video_path: Optional[str] = None
        if output_path is None:
            out_dir = self.config.localize_output_dir or self.config.debug_output_dir
            if out_dir:
                stem = Path(meta.source_path).stem
                output_path = str(Path(out_dir) / f"{stem}_{target_locale}.mp4")
        if output_path and rendered_frames:
            compositor = Compositor(backend_name=self.config.compositor_backend)
            compositor.composite(rendered_frames, full_audio if len(full_audio) > 0 else None,
                                 sr, meta.fps, output_path)
            rendered_video_path = output_path

        # 5. LocalizedTrack
        track = LocalizedTrack(
            video_meta=meta, target_locale=target_locale,
            subtitles=localized_subs,
            rendered_video_path=rendered_video_path,
            audio_path=None,
        )

        # 6. debug artifacts
        out_dir = self.config.localize_output_dir or self.config.debug_output_dir
        if out_dir:
            write_localize_artifacts(
                track=track, results=results, rendered_frames=rendered_frames,
                full_audio=full_audio, sample_rate=sr, out_dir=out_dir,
            )
        return track

    def close(self) -> None:
        try:
            self.analyzer.close()
        except Exception:
            pass

    # ---- P10 step 3: 同窗 instance 合并 ----

    def _merge_overlapping_instances_by_fps(
        self, instances: List["SubtitleInstance"], fps: float,
        overlap_ratio: float = 0.5,
    ) -> List["SubtitleInstance"]:
        """按 fps + 时间窗重叠 ≥ overlap_ratio 合并相邻 instance.

        合并规则:
          1. 按 start_ms (= first_frame/fps × 1000) 升序排
          2. greedy 维护 clusters; 对每个新 instance 算与 cluster 的时间重叠:
                 overlap_ms / min(this_len, cluster_len) ≥ overlap_ratio → 归 cluster
          3. 合并后 instance:
             - bbox = 所有成员的 union (外接矩形)
             - text = 按 bbox y 排序后用空格 join (上方在上)
             - start_ms = min, end_ms = max, duration_ms = end - start
             - first_frame/last_frame 同步重算
             - instance_id = "merge-" + "+".join([id[:8] for id in members])
        """
        if not instances:
            return []
        if fps <= 0:
            fps = 30.0
        # 1. 给每个 instance 算 start_ms / end_ms
        spans: List[Tuple[int, int, "SubtitleInstance"]] = []
        for inst in instances:
            if inst.first_frame:
                start_ms = int(inst.first_frame / fps * 1000)
            else:
                start_ms = 0
            end_ms = start_ms + max(int(inst.duration_ms or 0), 1)
            spans.append((start_ms, end_ms, inst))
        spans.sort(key=lambda s: s[0])
        # 2. greedy 合并
        clusters: List[List[Tuple[int, int, "SubtitleInstance"]]] = []
        for start, end, inst in spans:
            merged = False
            if clusters:
                cluster_start = clusters[-1][0][0]
                cluster_end = max(s[1] for s in clusters[-1])
                inter_s = max(start, cluster_start)
                inter_e = min(end, cluster_end)
                overlap_ms = max(0, inter_e - inter_s)
                this_len = end - start
                cluster_len = cluster_end - cluster_start
                if min(this_len, cluster_len) > 0:
                    ratio = overlap_ms / min(this_len, cluster_len)
                else:
                    ratio = 0.0
                if ratio >= overlap_ratio:
                    clusters[-1].append((start, end, inst))
                    merged = True
            if not merged:
                clusters.append([(start, end, inst)])
        # 3. 合成 merged instances
        merged_list: List[SubtitleInstance] = []
        for cluster in clusters:
            if len(cluster) == 1:
                merged_list.append(cluster[0][2])
                continue
            members = [c[2] for c in cluster]
            start_ms = min(c[0] for c in cluster)
            end_ms = max(c[1] for c in cluster)
            xs1 = [inst.representative_bbox[0]
                   for inst in members if inst.representative_bbox]
            ys1 = [inst.representative_bbox[1]
                   for inst in members if inst.representative_bbox]
            xs2 = [inst.representative_bbox[2]
                   for inst in members if inst.representative_bbox]
            ys2 = [inst.representative_bbox[3]
                   for inst in members if inst.representative_bbox]
            union_bbox = (min(xs1), min(ys1), max(xs2), max(ys2)) if xs1 else None
            ordered = sorted(
                members,
                key=lambda x: x.representative_bbox[1] if x.representative_bbox else 0,
            )
            merged_text = " ".join(
                (inst.representative_text or "").strip()
                for inst in ordered
            )
            base = members[0]
            from dataclasses import replace
            new_inst = replace(
                base,
                instance_id="merge-" + "+".join(
                    inst.instance_id[:8] for inst in members
                ),
                representative_text=merged_text,
                representative_bbox=union_bbox,
                first_frame=int(start_ms / 1000 * fps),
                last_frame=int(end_ms / 1000 * fps),
                duration_ms=end_ms - start_ms,
            )
            merged_list.append(new_inst)
        return merged_list

    def _build_ocr_checker(self, detector_backend: str):
        """构造一个 (image, mask) -> [(x,y,w,h), ...] callable 给 InpaintingEngine 复检字幕残余。

        设计: 复用 pipeline.__init__ 时已经 warmup 完的 self.detector 实例,
        避免每次复检都新建 PaddleOCR (首次预热 30s)。

        Args:
            detector_backend: TextDetectorRegistry 注册名 (e.g. 'paddleocr_ch')
        """
        from video_localization_engine.detector.registry import TextDetectorRegistry
        from video_localization_engine.types.video import FramePacket

        detector = self.detector  # 复用已 warmup 的实例

        def _checker(crop_img, _crop_mask):
            # TextDetector.detect 需要 FramePacket — 用哑 meta 套上就行
            pkt = FramePacket(
                frame_id=0,
                timestamp_ms=0,
                image=crop_img,
                meta=self.meta,
            )
            try:
                cands = detector.detect(pkt)
            except Exception:
                return []
            bboxes = []
            for c in cands or []:
                if not c.text or not c.text.strip():
                    continue
                if c.confidence < 0.3:
                    continue
                if c.bbox is None:
                    continue
                x0, y0, x1, y1 = c.bbox
                bboxes.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
            return bboxes

        return _checker


def _render_audio_track(localized_subs, sample_rate: int) -> np.ndarray:
    """全局 timeline 铺轨 — 把每句合成音按原字幕目标时间窗嵌入, 得到完整音轨。

    设计要点 (与逐句拼接的旧实现对比):
      - 旧: 用 TimingAligner 把每句压缩/调速/补静音到 sub 的 [start_ms, end_ms) 窗口
            → 长句被强制调速 20% 内, 破音; 短句被补静音延长, 累计往后飘
      - 新: 保留原句合成节奏 (不调速), 全局上把每句的 samples 铺到它在原时间轴上的
            真实起点; 落后于目标 end_ms 时末尾裁掉, 领先时尾部多保留 (叠成轻微重叠)
            → 累积误差被「下一句以 sub.start_ms 为锚」消除, 不会把后面全数飘掉

    Args:
        localized_subs: 已经跑过 TTS 的 localized 字幕 (含 sub.audio.samples)
        sample_rate: 单声道采样率 (Hz)

    Returns:
        全局 mono float32 音轨; 长度 = max(sub.end_ms for sub...) 对应的样本数
        不重叠 / 无声时也是全量 audio, 避免下游 ffmpeg 抱怨短音频。
    """
    if not localized_subs:
        return np.zeros(0, dtype=np.float32)
    sr = sample_rate
    ms_to_n = lambda ms: max(0, int(ms / 1000 * sr))

    end_ms_max = max(sub.end_ms for sub in localized_subs)
    total_n = ms_to_n(end_ms_max) + ms_to_n(localized_subs[-1].end_ms - localized_subs[-1].start_ms)
    if total_n <= 0:
        return np.zeros(0, dtype=np.float32)
    track = np.zeros(total_n, dtype=np.float32)

    for sub in localized_subs:
        if sub.audio is None or len(sub.audio.samples) == 0:
            continue
        sub_dur_ms = sub.end_ms - sub.start_ms
        cursor_n = ms_to_n(sub.start_ms)
        if cursor_n >= len(track):
            continue
        chunk = sub.audio.samples
        # 如果这一句超出目标时长, 在尾端裁掉 (不让它"跨越"到下一句的开始)
        target_n = min(len(chunk), ms_to_n(sub_dur_ms) if sub_dur_ms > 0 else len(chunk))
        if target_n > 0:
            end_n = min(cursor_n + target_n, len(track))
            write_n = end_n - cursor_n
            if write_n > 0:
                track[cursor_n:end_n] = chunk[:write_n]
    return track
