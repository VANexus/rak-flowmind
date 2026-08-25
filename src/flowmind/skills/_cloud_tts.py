"""阿里百炼 CosyVoice 封装：声音复刻注册（REST）+ 逐句合成（WS 薄适配）。

云优先原则：TTS/声音克隆全走云端；无 key / 无 voice_id 显式报错。

音色注册（参考百炼 voice-enrollment API）：
- POST {base}/services/audio/tts/customization
- body: {model: "voice-enrollment", input: {action, target_model, prefix, url}}
- 返回 output.voice_id；音色与 target_model 绑定（本模块常量统一为
  cosyvoice-v3.5-plus，注册与合成必须一致）

合成走 dashscope SDK 的 WebSocket SpeechSynthesizer；为可测试性，
实际调用隔离在 _call_ws_synth 里，测试注入 synth_fn 替换。
"""
from __future__ import annotations

from pathlib import Path

import requests

DEFAULT_BASE = "https://dashscope.aliyuncs.com/api/v1"
TARGET_MODEL = "cosyvoice-v3.5-plus"


class TTSError(Exception):
    """TTS 调用失败。category/retriable 语义与 errors.py 对齐。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


# ── 音色管理（REST） ──


def _customization(action: str, extra_input: dict, *, api_key: str,
                   api_base: str = DEFAULT_BASE) -> dict:
    """voice-enrollment 统一入口。"""
    if not api_key:
        raise ValueError(
            "收到空 API key。请检查环境变量 DASHSCOPE_API_KEY 是否设置。"
        )
    url = f"{api_base.rstrip('/')}/services/audio/tts/customization"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": "voice-enrollment", "input": {"action": action, **extra_input}}
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30.0)
    except requests.exceptions.RequestException as exc:
        raise TTSError("音色管理网络错误", category="environment") from exc
    if resp.status_code >= 500:
        raise TTSError(f"音色接口 HTTP {resp.status_code}",
                       category="transient", retriable=True)
    if resp.status_code >= 400:
        raise TTSError(f"音色接口 HTTP {resp.status_code}", category="video")
    return resp.json()


def create_voice(
    sample_audio_url: str, *, prefix: str, api_key: str,
    api_base: str = DEFAULT_BASE, target_model: str = TARGET_MODEL,
) -> str:
    """上传样音 URL 注册克隆音色，返回 voice_id。样音要求 10~20s、≥16kHz、清晰人声。"""
    data = _customization(
        "create_voice",
        {"target_model": target_model, "prefix": prefix, "url": sample_audio_url},
        api_key=api_key, api_base=api_base,
    )
    voice_id = (data.get("output") or {}).get("voice_id")
    if not voice_id:
        raise TTSError("create_voice 响应缺 voice_id")
    return str(voice_id)


def list_voices(*, api_key: str, api_base: str = DEFAULT_BASE) -> list[dict]:
    data = _customization("list_voices", {"prefix": ""}, api_key=api_key, api_base=api_base)
    return list((data.get("output") or {}).get("voices") or [])


def delete_voice(voice_id: str, *, api_key: str,
                 api_base: str = DEFAULT_BASE) -> None:
    _customization("delete_voice", {"voice_id": voice_id},
                   api_key=api_key, api_base=api_base)


# ── 合成 ──


def synthesize_text(
    text: str, *, out_path: str, voice_id: str, api_key: str,
    target_model: str = TARGET_MODEL,
) -> str:
    """单句合成 wav。经 _call_ws_synth（dashscope WS），测试可替换。"""
    if not voice_id:
        raise ValueError(
            "收到空 voice_id。先用 voice_clone_enroll 技能注册音色，"
            "或显式传入已注册的音色 ID。"
        )
    return _call_ws_synth(
        text, out_path=out_path, voice_id=voice_id,
        model=target_model, timeout=None,
        headers={"Authorization": f"Bearer {api_key}"},
        payload={"model": target_model, "voice": voice_id, "text": text},
    )


def _call_ws_synth(text: str, *, out_path: str, voice_id: str, model: str,
                   timeout: float | None, headers: dict, payload: dict) -> str:
    """dashscope WS 合成薄适配层。生产实现用 dashscope.SpeechSynthesizer；
    测试通过 monkeypatch 替换本函数。懒 import：不装 dashscope 时不影响包导入。
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
        msg = str(exc)
        if "throttl" in msg.lower() or "429" in msg:
            raise TTSError("合成限流", category="transient", retriable=True) from exc
        raise TTSError(f"合成失败: {type(exc).__name__}", category="video") from exc
    return out_path


def synthesize_segments(
    segments: list[dict], *, voice_id: str, out_dir: str, api_key: str,
    synth_fn=None, target_model: str = TARGET_MODEL,
) -> list[str]:
    """逐句合成，输出按 index 命名的 wav 列表（seg_0000.wav ...）。"""
    if not voice_id:
        raise ValueError("收到空 voice_id。请先注册克隆音色或传预设音色 ID。")
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
        path = str(out_dir_p / f"seg_{seg['index']:04d}.wav")
        fn(seg["text"], path, voice_id=voice_id)
        outs.append(path)
    return outs
