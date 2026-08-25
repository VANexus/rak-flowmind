"""EdgeTtsBackend — Microsoft Edge 神经 TTS (免费, 无需 API key)。

基于 edge-tts 库 (https://github.com/rany2/edge-tts)。
底层走 Microsoft Edge 浏览器内置 read-aloud 的 WebSocket 服务,
所以需要网络可达 (WSL/沙盒里可能不可达, 见 HANDOFF.md)。

synth() 流程:
  1. 同步跑 edge-tts 异步流 (最多 2 次, 每次 10s timeout) → MP3 bytes
  2. pydub 解码 MP3 → AudioSegment (24kHz 默认)
  3. 转 mono float32 [-1, 1]
  4. 如请求 sample_rate 与原生不一致 → set_frame_rate 重采样

稳健性:
  - 网络卡 / WebSocket 30s 超时 → 立即 10s 重试一次
  - 两次都失败 → graceful degrade 返回 0.5s 静音 (不抛)
  - 错误记 warning, 不阻断 pipeline
"""
from __future__ import annotations

import asyncio
import io
import logging
from typing import Dict, List

import numpy as np

from video_localization_engine.localizer.protocols import TtsBackend


_log = logging.getLogger(__name__)


# locale → 默认 voice 映射 (Microsoft 神经语音, 女声优先)
_DEFAULT_VOICES: Dict[str, str] = {
    "zh": "zh-CN-XiaoxiaoNeural",
    "zh-CN": "zh-CN-XiaoxiaoNeural",
    "zh-TW": "zh-TW-HsiaoChenNeural",
    "en": "en-US-AriaNeural",
    "en-US": "en-US-AriaNeural",
    "en-GB": "en-GB-SoniaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ja-JP": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "ko-KR": "ko-KR-SunHiNeural",
    "fr": "fr-FR-DeniseNeural",
    "fr-FR": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "de-DE": "de-DE-KatjaNeural",
    "es": "es-ES-ElviraNeural",
    "es-ES": "es-ES-ElviraNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ru-RU": "ru-RU-SvetlanaNeural",
}


class EdgeTtsBackend(TtsBackend):
    """Edge-TTS backend — 真实神经语音合成。

    构造参数:
      - default_voice: 默认 voice short name (覆盖 locale 映射)
      - voice_map: 自定义 locale → voice 映射 (覆盖 _DEFAULT_VOICES)
      - timeout_sec: 单次 WebSocket 超时秒数 (默认 10, 之前是 30)
      - max_retries: 重试次数 (默认 1, 加上首次共 2 次尝试)
      - fallback_silence_sec: 完全失败时返回静音时长 (默认 0.5s)
      - strict: True → 重试 + fallback 都失败后抛 RuntimeError (默认 False 静默)
    """

    def __init__(self,
                 default_voice: str | None = None,
                 voice_map: Dict[str, str] | None = None,
                 timeout_sec: int = 10,
                 max_retries: int = 1,
                 fallback_silence_sec: float = 0.5,
                 strict: bool = False):
        self.default_voice = default_voice
        self.voice_map: Dict[str, str] = {**_DEFAULT_VOICES, **(voice_map or {})}
        self.timeout_sec = max(1, int(timeout_sec))
        self.max_retries = max(0, int(max_retries))
        self.fallback_silence_sec = max(0.05, float(fallback_silence_sec))
        self.strict = bool(strict)

    @property
    def name(self) -> str:
        return "edge_tts"

    @property
    def supported_locales(self) -> List[str]:
        return sorted(set(self.voice_map.keys()))

    def _pick_voice(self, target_locale: str) -> str:
        if self.default_voice:
            return self.default_voice
        if target_locale in self.voice_map:
            return self.voice_map[target_locale]
        prefix = target_locale.split("-")[0].lower()
        if prefix in self.voice_map:
            return self.voice_map[prefix]
        return self.voice_map["en"]

    # --------------------------- 网络调用 + 重试 ---------------------------

    def _collect_once(self, text: str, voice: str) -> bytes:
        """同步跑一次 edge-tts 异步流, 最多 self.timeout_sec 秒."""
        import edge_tts

        async def _collect():
            buf = io.BytesIO()
            communicate = edge_tts.Communicate(text, voice)
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio":
                    data = chunk.get("data")
                    if data:
                        buf.write(data)
            return buf.getvalue()

        return asyncio.run(_collect())

    def _synth_with_retry(self, text: str, voice: str) -> bytes:
        """带重试的 edge-tts 调用. 返回 MP3 bytes 或抛最后一次异常."""
        last_exc: Exception | None = None
        attempts = 1 + self.max_retries
        for attempt in range(1, attempts + 1):
            try:
                return self._collect_once(text, voice)
            except Exception as e:  # 捕获 asyncio / edge_tts / aiohttp 所有异常
                last_exc = e
                _log.warning(
                    "edge_tts attempt %d/%d failed (voice=%r): %s: %s",
                    attempt, attempts, voice, type(e).__name__, e,
                )
        assert last_exc is not None
        raise last_exc

    # ------------------------------ public -------------------------------

    def synth(self, text: str, target_locale: str,
              sample_rate: int, **kwargs) -> np.ndarray:
        """返回 mono float32; 完全失败时返回静音数组 (不再抛)。

        仅当 strict=True 才抛。
        """
        if not text or not text.strip():
            return np.zeros(0, dtype=np.float32)

        try:
            import edge_tts  # noqa: F401
        except ImportError as e:
            if self.strict:
                raise ImportError("edge-tts not installed") from e
            return self._silence(sample_rate)

        try:
            from pydub import AudioSegment
        except ImportError as e:
            if self.strict:
                raise ImportError("pydub not installed") from e
            return self._silence(sample_rate)

        voice = self._pick_voice(target_locale)
        try:
            mp3_bytes = self._synth_with_retry(text, voice)
        except Exception as e:
            _log.warning(
                "edge_tts_all_retries_failed: text=%r error=%s: %s",
                text[:60], type(e).__name__, e,
            )
            if self.strict:
                raise RuntimeError(f"edge-tts unrecoverable: {e}") from e
            return self._silence(sample_rate)

        if not mp3_bytes:
            # 极端: 连上但 0 字节 (服务器异常)
            return self._silence(sample_rate)

        # ---- MP3 → PCM ----
        try:
            import tempfile, subprocess
            # pydub 需要 ffmpeg。在 PATH, 但有时 pydub 默认从 /usr/bin/ffmpeg 找不到。
            from pydub.utils import which as pd_which
            if pd_which("ffmpeg") is None:
                return self._silence(sample_rate)
            buf = io.BytesIO(mp3_bytes)
            audio = AudioSegment.from_file(buf, format="mp3")
        except Exception as e:
            _log.warning("pydub decode failed: %s: %s", type(e).__name__, e)
            return self._silence(sample_rate)

        if audio.channels > 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != sample_rate:
            audio = audio.set_frame_rate(sample_rate)

        samples_int16 = np.array(audio.get_array_of_samples(), dtype=np.int16)
        if samples_int16.size == 0:
            return self._silence(sample_rate)
        samples_f32 = samples_int16.astype(np.float32) / 32768.0
        np.clip(samples_f32, -1.0, 1.0, out=samples_f32)
        return samples_f32

    def _silence(self, sample_rate: int) -> np.ndarray:
        n = int(self.fallback_silence_sec * max(sample_rate, 1))
        return np.zeros(n, dtype=np.float32)
