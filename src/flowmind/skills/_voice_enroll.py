"""阿里百炼声音复刻（voice enrollment）封装。

10~20 秒样本音频 → create_voice 返回复刻音色 ID，可直接作为
localize_video 的 voice_id 传入（复刻音色与预设音色走同一
SpeechSynthesizer 合成路径，流水线零改动）。

约束（百炼 voice-enrollment 接口）：
- target_model 必须与后续合成模型一致，否则合成失败（默认
  qwen-audio-3.0-tts-flash，与 config.localize_tts_model 对齐）。
- 样本仅支持公网可访问 URL，不支持本地文件直传。
- prefix 仅允许字母数字且 ≤10 字符（生成音色名 {target_model}-{prefix}-{id}）。

无 key / 非 URL / SDK 缺失显式报错，不静默降级。
"""
from __future__ import annotations

TARGET_MODEL = "qwen-audio-3.0-tts-flash"
DEFAULT_PREFIX = "flwm"


class EnrollError(Exception):
    """音色复刻失败。category/retriable 语义与 errors.py 对齐。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def create_voice(
    sample_url: str, *, prefix: str, api_key: str,
    target_model: str = TARGET_MODEL, language_hint: str | None = None,
) -> str:
    """提交样本音频创建复刻音色，返回 voice_id。

    经 _create_voice_adapter 薄适配层调用 SDK，测试可 monkeypatch 替换。
    """
    if not sample_url.startswith(("http://", "https://")):
        raise EnrollError(
            f"样本音频必须是公网可访问 URL（收到: {sample_url[:80]}）。"
            "复刻接口不支持本地文件直传，请先上传到 OSS 等可公网访问的地址。",
            category="video",
        )
    return _create_voice_adapter(
        sample_url, prefix=prefix, api_key=api_key,
        target_model=target_model, language_hint=language_hint,
    )


def _create_voice_adapter(
    sample_url: str, *, prefix: str, api_key: str,
    target_model: str, language_hint: str | None,
) -> str:
    """dashscope VoiceEnrollmentService 薄适配层。测试通过 monkeypatch 替换。
    懒 import：不装 dashscope 不影响包导入。
    """
    try:
        from dashscope.audio.tts_v2 import VoiceEnrollmentService  # type: ignore
    except ImportError as exc:
        raise EnrollError(
            "未安装 dashscope SDK（pip install dashscope 后可用）",
            category="environment",
        ) from exc

    try:
        svc = VoiceEnrollmentService(api_key=api_key)
        voice_id = svc.create_voice(
            target_model=target_model,
            prefix=prefix,
            url=sample_url,
            language_hints=[language_hint] if language_hint else None,
        )
    except Exception as exc:  # SDK 异常形态多变，统一分类
        msg = str(exc).lower()
        if "throttl" in msg or "429" in msg:
            raise EnrollError("复刻限流", category="transient", retriable=True) from exc
        raise EnrollError(f"音色复刻失败: {type(exc).__name__}", category="transient") from exc
    if not voice_id:
        raise EnrollError("复刻接口未返回 voice_id", category="transient")
    return voice_id
