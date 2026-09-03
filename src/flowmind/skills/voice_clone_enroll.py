"""voice_clone_enroll 技能：声音复刻（样本音频 → 复刻音色 ID）。

调百炼 voice-enrollment：10~20 秒公网可访问样本 → 返回 voice_id，
可直接传给 localize_video 的 voice_id 参数做克隆配音（合成模型与
config.localize_tts_model 强制一致，避免复刻音色与合成模型错配）。
失败契约遵循 HTTP 依赖类：r.ok=True + metrics.degraded + data.failure_category。
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

from flowmind.config import load_config
from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.errors import is_retriable
from flowmind.skill import skill
from flowmind.skills import _voice_enroll
from flowmind.skills._secrets import get_api_key

_VERSION = "0.1.0"

KEY_SPEECH = "AI_SPEECH_API_KEY"

_PREFIX_RE = re.compile(r"^[A-Za-z0-9]{1,10}$")


class VoiceCloneEnrollInput(BaseModel):
    """声音复刻入参。"""

    sample_url: str = Field(
        ..., min_length=1,
        description="样本音频的公网可访问 URL（10~20 秒、干净人声、无背景音乐）",
    )
    prefix: str | None = Field(
        default=None,
        description="音色名前缀（仅字母数字，≤10 字符）；None=读 config.voice_clone_prefix",
    )
    language_hint: str | None = Field(
        default=None,
        description="样本语种（如 zh/en/ja）；None=自动检测",
    )
    target_model: str | None = Field(
        default=None,
        description="绑定合成模型；None=读 config.localize_tts_model（保证与配音一致）",
    )

    @field_validator("sample_url")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("sample_url 不能为空")
        return v.strip()

    @field_validator("prefix")
    @classmethod
    def _prefix_ok(cls, v: str | None) -> str | None:
        if v is not None and not _PREFIX_RE.fullmatch(v):
            raise ValueError("prefix 仅允许字母数字且不超过 10 字符")
        return v


class VoiceCloneEnrollReport(BaseModel):
    """声音复刻业务载荷。"""

    voice_id: str                        # 复刻音色 ID，直接传 localize_video.voice_id
    target_model: str                    # 绑定的合成模型（复刻/合成必须一致）
    failure_category: str | None = None
    retriable: bool = False
    warning: str | None = None


@skill(id="voice_clone_enroll", name="声音复刻（样本→音色ID）", version=_VERSION)
def voice_clone_enroll(inp: VoiceCloneEnrollInput) -> SkillOutput[VoiceCloneEnrollReport]:
    """样本音频创建复刻音色，返回可复用的 voice_id。"""
    cfg = load_config().localizer
    speech_key = get_api_key(KEY_SPEECH)
    if not speech_key:
        return _fail(inp, f"未设置环境变量 {KEY_SPEECH}（复刻需要）", "environment")

    target_model = inp.target_model or cfg.localize_tts_model
    prefix = inp.prefix or getattr(cfg, "voice_clone_prefix", _voice_enroll.DEFAULT_PREFIX)

    try:
        voice_id = _voice_enroll.create_voice(
            inp.sample_url, prefix=prefix, api_key=speech_key,
            target_model=target_model, language_hint=inp.language_hint,
        )
    except _voice_enroll.EnrollError as exc:
        return _fail(inp, str(exc), exc.category, retriable=exc.retriable)

    chain = ReasoningChain(
        conclusion=f"复刻音色创建成功：{voice_id}",
        triggered_rules=[], evidence=[],
        causal_analysis=(
            f"voice-enrollment 提交样本（prefix={prefix}）→ 绑定模型 {target_model} "
            "→ 返回 voice_id；复刻音色与预设音色走同一合成路径"
        ),
        risk_note="voice_id 请持久保存以便复用；删除音色后此 ID 失效。",
    )
    return SkillOutput(
        data=VoiceCloneEnrollReport(voice_id=voice_id, target_model=target_model),
        reasoning=[chain], confidence=0.95, sample_size=1,
    )


def _fail(inp, warning: str, category: str, *,
          retriable: bool = False) -> SkillOutput[VoiceCloneEnrollReport]:
    """统一 degraded 返回：脱敏（warning 不带 key/host）。"""
    report = VoiceCloneEnrollReport(
        voice_id="", target_model="",
        failure_category=category,
        retriable=retriable or is_retriable(category),
        warning=f"{warning}（{category}）",
    )
    chain = ReasoningChain(
        conclusion=f"声音复刻失败（{category}）",
        triggered_rules=[], evidence=[],
        causal_analysis="预检或复刻接口调用失败，见 warning 字段",
        risk_note=(
            "按 failure_category 决策：environment 先检查网络/key；"
            "video 修样本 URL（需公网可访问）；transient 可重试。"
        ),
    )
    return SkillOutput(
        data=report, reasoning=[chain], confidence=0.0, sample_size=0,
        degraded=True, degradation_reason=category,
    )
