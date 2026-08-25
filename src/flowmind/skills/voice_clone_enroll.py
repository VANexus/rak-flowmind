"""voice_clone_enroll 技能：CosyVoice 克隆音色管理（注册 / 列表 / 删除）。

云优先原则：声音克隆全走云端；无 key 显式 degraded（environment），不静默降级。
"""
from __future__ import annotations


from pydantic import BaseModel, Field, model_validator

from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.skill import skill
from flowmind.skills._cloud_tts import TTSError
from flowmind.skills._image_backend import resolve_api_key  # noqa: F401 复用 env 读取

_VERSION = "0.1.0"

KEY_ENV = "DASHSCOPE_API_KEY"


class EnrollInput(BaseModel):
    """音色管理入参。action 决定必填字段。"""

    action: str = Field(..., description="create | list | delete")
    sample_audio_url: str | None = Field(
        default=None,
        description="样音 URL（10~20s、≥16kHz、清晰人声 WAV/MP3）；create 必填",
    )
    prefix: str | None = Field(default=None, description="音色名前缀；create 必填")
    voice_id: str | None = Field(default=None, description="音色 ID；delete 必填")

    @model_validator(mode="after")
    def _check_action_fields(self) -> "EnrollInput":
        if self.action == "create" and not (self.sample_audio_url or "").strip():
            raise ValueError("action=create 需要 sample_audio_url（样音 URL）")
        if self.action == "delete" and not (self.voice_id or "").strip():
            raise ValueError("action=delete 需要 voice_id")
        return self


class EnrollReport(BaseModel):
    """音色管理业务载荷。"""

    action: str
    voice_id: str | None = None
    voices: list[dict] = []
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


@skill(id="voice_clone_enroll", name="克隆音色管理", version=_VERSION)
def voice_clone_enroll(inp: EnrollInput) -> SkillOutput[EnrollReport]:
    """注册/列出/删除 CosyVoice 克隆音色。

    create：上传样音 → 返回 voice_id（供 localize_video 的 voice_id 入参使用）。
    注册免费；样音要求 10~20 秒连续清晰朗读、≥16kHz。
    """
    api_key = resolve_api_key(KEY_ENV)

    # 无 key：显式 degraded（云优先，不静默降级）
    if not api_key:
        return _fail(inp.action, f"未设置环境变量 {KEY_ENV}", "environment")

    try:
        from flowmind.skills import _cloud_tts

        if inp.action == "create":
            voice_id = _do_create(inp.sample_audio_url or "", inp.prefix or "flowmind",
                                  api_key)
            report = EnrollReport(action="create", voice_id=voice_id)
            chain = ReasoningChain(
                conclusion=f"已注册克隆音色 {voice_id}",
                triggered_rules=[], evidence=[],
                causal_analysis="样音上传百炼 voice-enrollment，target_model=cosyvoice-v3.5-plus",
                risk_note="音色与模型绑定；1 年未使用会被系统回收。",
            )
            return SkillOutput(data=report, reasoning=[chain], confidence=1.0, sample_size=1)
        if inp.action == "list":
            voices = _cloud_tts.list_voices(api_key=api_key)
            chain = ReasoningChain(
                conclusion=f"共 {len(voices)} 个已注册音色",
                triggered_rules=[], evidence=[],
                causal_analysis="查询百炼 voice-enrollment 列表接口",
                risk_note="voice_id 供 localize_video 配音使用。",
            )
            return SkillOutput(
                data=EnrollReport(action="list", voices=voices),
                reasoning=[chain], confidence=1.0, sample_size=len(voices),
            )
        if inp.action == "delete":
            _cloud_tts.delete_voice(inp.voice_id or "", api_key=api_key)
            chain = ReasoningChain(
                conclusion=f"已删除音色 {inp.voice_id}",
                triggered_rules=[], evidence=[],
                causal_analysis="调百炼 delete_voice 接口",
                risk_note="删除后不可恢复。",
            )
            return SkillOutput(
                data=EnrollReport(action="delete", voice_id=inp.voice_id),
                reasoning=[chain], confidence=1.0, sample_size=1,
            )
        raise TTSError(f"未知 action: {inp.action}", category="video")
    except TTSError as exc:
        return _fail(inp.action, str(exc), exc.category, retriable=exc.retriable)


def _do_create(sample_url: str, prefix: str, api_key: str) -> str:
    """create 薄封装（测试可替换）。"""
    from flowmind.skills import _cloud_tts

    return _cloud_tts.create_voice(sample_url, prefix=prefix, api_key=api_key)


def _fail(action: str, warning: str, category: str,
          *, retriable: bool = False) -> SkillOutput[EnrollReport]:
    report = EnrollReport(action=action, failure_category=category,
                          retriable=retriable, warning=warning)
    chain = ReasoningChain(
        conclusion=f"音色 {action} 失败（{category}）",
        triggered_rules=[], evidence=[],
        causal_analysis=f"{type(Warning).__name__}: 见 warning 字段",
        risk_note="按 failure_category 决策：environment 先配 key，transient 可重试。",
    )
    return SkillOutput(
        data=report, reasoning=[chain], confidence=0.0, sample_size=0,
        degraded=True, degradation_reason=category,
    )
