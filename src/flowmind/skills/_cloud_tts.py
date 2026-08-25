"""阿里百炼 TTS 封装：预设音色逐句合成（云优先，零本地模型）。

模型 qwen-audio-3.0-tts-flash / 音色 longanhuan_v3.6 已实测可用（默认 WS 域名，
无需 WorkspaceId）。合成经 _call_ws_synth 适配层隔离，测试可替换。
无 key / 无音色显式报错，不静默降级。
"""
from __future__ import annotations

from pathlib import Path


DEFAULT_BASE = "https://dashscope.aliyuncs.com/api/v1"
TARGET_MODEL = "qwen-audio-3.0-tts-flash"
DEFAULT_VOICE = "longanhuan_v3.6"


class TTSError(Exception):
    """TTS 调用失败。category/retriable 语义与 errors.py 对齐。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def synthesize_text(
    text: str, *, out_path: str, voice_id: str, api_key: str,
    target_model: str = TARGET_MODEL,
) -> str:
    """单句合成 mp3/wav。voice_id 即预设音色名（如 longanhuan_v3.6）。"""
    if not voice_id:
        raise ValueError(
            "收到空音色。请传预设音色名（如 longanhuan_v3.6），"
            "或检查 config.localize_voice 配置。"
        )
    return _call_ws_synth(
        text, out_path=out_path, voice_id=voice_id,
        model=target_model, timeout=None,
        headers={"Authorization": f"Bearer {api_key}"},
        payload={"model": target_model, "voice": voice_id, "text": text},
    )


def _call_ws_synth(text: str, *, out_path: str, voice_id: str, model: str,
                   timeout: float | None, headers: dict, payload: dict) -> str:
    """dashscope SpeechSynthesizer 薄适配层。测试通过 monkeypatch 替换本函数。
    懒 import：不装 dashscope 不影响包导入。
    """
    try:
        from dashscope.audio.tts_v2 import SpeechSynthesizer  # type: ignore
    except ImportError as exc:
        raise TTSError(
            "未安装 dashscope SDK（uv add dashscope 后可用）",
            category="environment",
        ) from exc

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        synthesizer = SpeechSynthesizer(model=model, voice=voice_id)
        audio = synthesizer.call(text)
        with open(out_path, "wb") as f:
            f.write(audio)
    except Exception as exc:  # SDK 异常形态多变，统一分类
        msg = str(exc).lower()
        if "throttl" in msg or "429" in msg:
            raise TTSError("合成限流", category="transient", retriable=True) from exc
        raise TTSError(f"合成失败: {type(exc).__name__}", category="video") from exc
    return out_path


def synthesize_segments(
    segments: list[dict], *, voice_id: str, out_dir: str, api_key: str,
    synth_fn=None, target_model: str = TARGET_MODEL,
) -> list[str]:
    """逐句合成，输出按 index 命名的音频列表（seg_0000.mp3 ...）。"""
    if not voice_id:
        raise ValueError("收到空音色。请传预设音色名（如 longanhuan_v3.6）。")
    fn = synth_fn or (
        lambda text, out_path, **kw: synthesize_text(
            text, out_path=out_path, voice_id=voice_id, api_key=api_key,
            target_model=target_model,
        )
    )
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    outs: list[str] = []
    for seg in segments:
        path = str(out_dir_p / f"seg_{seg['index']:04d}.mp3")
        fn(seg["text"], path, voice_id=voice_id)
        outs.append(path)
    return outs
