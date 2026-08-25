"""FFmpegCompositor — ffmpeg subprocess 合成 mp4。

流程:
  1. 帧序列 → 临时目录 PNG (frame_0001.png ...)
  2. 音频 → 临时 wav (mono float32 → int16 PCM)
  3. ffmpeg -framerate fps -i frame_%04d.png -i audio.wav
         -c:v libx264 -pix_fmt yuv420p -c:a aac out.mp4
  4. 返回 output_path
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from video_localization_engine.localizer.protocols import CompositorBackend


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _write_wav(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    """mono float32 [-1, 1] → int16 PCM wav。"""
    # 限幅 + 转 int16
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


class FFmpegCompositor(CompositorBackend):
    """ffmpeg 后端 — 默认 Compositor。"""

    def __init__(self, video_codec: str = "libx264",
                 audio_codec: str = "aac",
                 pix_fmt: str = "yuv420p",
                 preset: str = "ultrafast",
                 crf: int = 23):
        self.video_codec = video_codec
        self.audio_codec = audio_codec
        self.pix_fmt = pix_fmt
        self.preset = preset
        self.crf = crf

    @property
    def name(self) -> str:
        return "ffmpeg"

    def is_available(self) -> bool:
        return _have_ffmpeg()

    def composite(self, frames: List[np.ndarray], audio: Optional[np.ndarray],
                  sample_rate: int, fps: float, output_path: str) -> str:
        if not frames:
            raise ValueError("frames is empty")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="vle_compose_") as tmp:
            tmp_dir = Path(tmp)
            # 1. 帧序列 → PNG (BGR cv2.imwrite 与 ffmpeg 兼容)
            frame_paths: List[str] = []
            for i, fr in enumerate(frames):
                p = tmp_dir / f"frame_{i + 1:04d}.png"
                cv2.imwrite(str(p), fr)
                frame_paths.append(p.name)
            # 2. 音频 → wav
            wav_path: Optional[Path] = None
            if audio is not None and len(audio) > 0:
                wav_path = tmp_dir / "audio.wav"
                _write_wav(audio, sample_rate, wav_path)
            # 3. ffmpeg 拼 mp4
            fr_pattern = str(tmp_dir / "frame_%04d.png")
            cmd = ["ffmpeg", "-y", "-framerate", str(fps),
                   "-i", fr_pattern]
            if wav_path is not None:
                cmd += ["-i", str(wav_path)]
            cmd += [
                "-c:v", self.video_codec, "-pix_fmt", self.pix_fmt,
                "-preset", self.preset, "-crf", str(self.crf),
                "-threads", "1",  # 限线程, WSL2 内存峰值更低 (避免 OOM)
            ]
            if wav_path is not None:
                cmd += ["-c:a", self.audio_codec, "-shortest", "-aac_pns", "0"]
            cmd += [str(out)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed (rc={r.returncode}):\n"
                    f"  cmd: {' '.join(cmd)}\n"
                    f"  stderr: {r.stderr[-1000:]}"
                )
        return str(out)
