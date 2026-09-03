"""本地 TTS：Qwen3-TTS-12Hz-0.6B-Base（零样本声音克隆，Pascal 6.1 适配）。

云优先原则的本地例外：用户显式启用 tts_backend=local 或 auto 且本地栈可用时，
配音走本地模型（原片人声克隆 / 预设音色），失败显式报错，不静默回落云端。

- Pascal 6.1 约束：fp32 + sdpa（bf16/flash-attn 不可用）；显存约 4GB
- 模型：Qwen/Qwen3-TTS-12Hz-0.6B-Base（HF 缓存 /srv/data/models，2.5GB）
- 克隆：ref_audio（wav 路径/数组）+ ref_text（参考音频转写）零样本克隆，
  支持跨语种（参考中文、输出英文）
- 依赖系统 sox（sudo apt install sox）与 torchaudio==2.5.1+cu121
  （qwen-tts 安装会把 torchaudio 升到不匹配版本，见 environment.yml 注释）
"""
from __future__ import annotations

from pathlib import Path

_LANG_MAP = {
    "zh": "Chinese", "en": "English", "de": "German", "it": "Italian",
    "pt": "Portuguese", "es": "Spanish", "ja": "Japanese", "ko": "Korean",
    "fr": "French", "ru": "Russian",
}

_SHARED: dict = {"model": None}
_PROMPT_CACHE: dict = {}


class LocalTTSError(Exception):
    """本地 TTS 失败。category/retriable 语义与 errors.py 对齐。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def map_language(target_lang: str) -> str:
    """ISO 639-1 → 模型语言名；未知语言原样大写返回（让模型显式报错）。"""
    return _LANG_MAP.get((target_lang or "").lower(), (target_lang or "").capitalize())


def available() -> bool:
    """qwen-tts + torch 可导入即视为可用（模型首次用时才从 HF 缓存加载）。"""
    try:
        import qwen_tts  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _get_model():
    """进程内单例。加载失败显式报错（环境问题），绝不静默。"""
    if _SHARED["model"] is None:
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise LocalTTSError(
                "未安装 qwen-tts（conda env update -f environment.yml）",
                category="environment",
            ) from exc
        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            _SHARED["model"] = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                device_map=device,
                dtype=torch.float32,            # Pascal 6.1 不支持 bf16
                attn_implementation="sdpa",     # flash-attn 仅 fp16/bf16，不可用
            )
        except Exception as exc:
            raise LocalTTSError(
                f"Qwen3-TTS 模型加载失败: {type(exc).__name__}: {exc}",
                category="environment",
            ) from exc
    return _SHARED["model"]


def _get_prompt(ref_audio: str, ref_text: str | None):
    """音色提示缓存：同一参考音频只提取一次 VoiceClonePromptItem，全程复用。

    逐句重复 create_voice_clone_prompt 会让每句独立抽取声纹，采样波动下
    三句话可能出现两种音色；复用同一 prompt 保证整段视频音色一致（且省时）。
    """
    key = (str(Path(ref_audio).resolve()), ref_text or "")
    if key not in _PROMPT_CACHE:
        model = _get_model()
        try:
            _PROMPT_CACHE[key] = model.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text or None,
            )[0]
        except Exception as exc:
            raise LocalTTSError(
                f"音色提取失败: {type(exc).__name__}: {exc}", category="video",
            ) from exc
    return _PROMPT_CACHE[key]


def _generate(text: str, *, language: str,
              ref_audio: str | None, ref_text: str | None) -> tuple:
    """克隆合成入口。Base 变体只支持 voice_clone（预设音色是 Speech 变体能力），
    无 ref_audio 直接报错。返回 (wavs, sr)。"""
    if not ref_audio:
        raise LocalTTSError(
            "本地 TTS（Base 变体）仅支持声音克隆：必须提供 ref_audio"
            "（原片人声自动克隆或显式样本）",
            category="video",
        )
    model = _get_model()
    prompt = _get_prompt(ref_audio, ref_text)
    try:
        # 签名要求 List[VoiceClonePromptItem] 或 Dict，单个对象会 TypeError
        return model.generate_voice_clone(
            text=text, language=language, voice_clone_prompt=[prompt],
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or ("cuda" in msg and "error" in msg):
            raise LocalTTSError(
                f"本地合成 GPU 失败（显存不足?）: {type(exc).__name__}",
                category="transient", retriable=True,
            ) from exc
        raise LocalTTSError(
            f"本地合成失败: {type(exc).__name__}: {exc or '(无错误信息)'}",
            category="video",
        ) from exc


def synthesize(
    text: str, *, out_path: str, voice: str | None = None,
    ref_audio: str | None = None, ref_text: str | None = None,
    target_lang: str = "zh",
) -> str:
    """克隆合成一句到 out_path（wav，模型原生采样率）。

    ref_audio = 参考音频（wav 路径）；ref_text = 其转写（None 走
    x-vector 模式）。voice 参数仅为兼容签名保留，克隆模式忽略。
    """
    if not text or not text.strip():
        raise LocalTTSError("合成文本为空", category="video")
    if ref_audio and not Path(ref_audio).exists():
        raise LocalTTSError(
            f"克隆参考音频不存在: {ref_audio}", category="video",
        )
    try:
        import soundfile as sf
    except ImportError as exc:
        raise LocalTTSError("未安装 soundfile", category="environment") from exc

    wavs, sr = _generate(
        text, language=map_language(target_lang),
        ref_audio=ref_audio, ref_text=ref_text,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, wavs[0], sr)
    return out_path


def synthesize_segments(
    segments: list[dict], *, out_dir: str,
    ref_audio: str | None = None, ref_text: str | None = None,
    voice: str | None = None, target_lang: str = "zh",
    synth_fn=None,
) -> list[str]:
    """逐句合成，输出按 index 命名的 wav 列表（seg_0000.wav ...）。

    与 _cloud_tts.synthesize_segments 返回结构对齐（流水线按顺序
    与 translated 时间戳对齐），便于 _build_timed_audio 无差别消费。
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    fn = synth_fn or (
        lambda text, out_path, **kw: synthesize(
            text, out_path=out_path, voice=voice,
            ref_audio=ref_audio, ref_text=ref_text,
            target_lang=target_lang,
        )
    )
    outs: list[str] = []
    for seg in segments:
        path = str(out_dir_p / f"seg_{seg['index']:04d}.wav")
        fn(seg["text"], path)
        outs.append(path)
    return outs
