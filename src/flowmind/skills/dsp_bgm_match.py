"""bgm_match 技能：背景音乐——**占位降级**（契约 §3.3 明确规定的当前状态）。

契约 §3.3：
- 依赖 ``timeline``（按**锁定总时长**选曲，与有没有配音无关）；降级 **degrade**；
  受「背景音乐」开关控制。
- 出参 ``{bgmKey, ducking:{enabled,db}, durationMs}``；无合适曲目 → ``degraded=true``
  + 原因（成片**空轨继续**）。
- **⏸ 占位（2026-10-12）**：**选曲能力与接口均未给**（《豆包语音》只提供 TTS；
  MediaKit 只有音量 / 闪避参数，无曲库）。本节点**保留在 DAG 与前端开关里**，
  后端未实现期间**一律**返回 ``degraded=true`` + 「BGM 选曲能力未接通」→ 成片空轨继续。

因此本技能当前是**如实降级**契约的实现，不是偷懒：拿到选曲接口后只需在本文件内
替换 ``_pick_track`` 一处，其余（DAG / 开关 / 降级语义 / 出参形状）不变。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills import _dsp_common as common
from flowmind.skills._dsp_common import DspBaseInput

_VERSION = "0.1.0"

_UNAVAILABLE = "BGM 选曲能力未接通"
_DEFAULT_DUCKING_DB = -12


class BgmMatchInput(DspBaseInput):
    """BGM 入参（公共上下文 + 可选情绪/风格提示）。"""
    mood: str = Field(default="", description="情绪 / 风格（可空；选曲接通后由它驱动）")


class Ducking(BaseModel):
    """闪避参数（配音起时压低 BGM）。"""
    enabled: bool = True
    db: int = _DEFAULT_DUCKING_DB


class BgmMatchData(BaseModel):
    """出参：BGM key（未接通时为空）+ 闪避参数 + 时长。"""
    bgmKey: str = Field(default="", description="BGM 的云存储 key；未接通时为空（成片空轨）")
    ducking: Ducking = Field(default_factory=Ducking)
    durationMs: int = 0
    warnings: list[str] = Field(default_factory=list)


def _pick_track(inp: BgmMatchInput, total_ms: int) -> str:
    """选曲（**待接通**）：当前无曲库 / 选曲接口 → 一律返回空。

    接通后在此返回 BGM 的云存储 key（按 ``total_ms`` 与 ``inp.mood`` 选曲）。
    """
    return ""


@skill(id="bgm_match", name="背景音乐", version=_VERSION,
       category="抖音成片", tags=["dsp", "bgm"])
def bgm_match(inp: BgmMatchInput) -> SkillOutput[BgmMatchData]:
    """按锁定总时长 / 情绪选曲 + 闪避参数；当前无选曲接口 → 如实降级为空轨。"""
    edl = common.dep_edl(inp.deps)
    total_ms = edl.totalMs if edl else 0
    bgm_key = _pick_track(inp, total_ms)

    if not bgm_key:
        return SkillOutput(
            data=BgmMatchData(bgmKey="", ducking=Ducking(), durationMs=total_ms,
                              warnings=[_UNAVAILABLE]),
            reasoning=[common.chain(
                f"背景音乐未匹配：{_UNAVAILABLE}（成片空轨继续）",
                causal_analysis=(
                    f"参考时长 = 锁定时间线 totalMs={total_ms}ms；"
                    "选曲能力与接口均未给（MediaKit 只有音量/闪避参数，无曲库）。"
                ),
                risk_note=(
                    "本节点属 degrade：失败不阻塞成片。拿到选曲接口后在 _pick_track 一处接通即可，"
                    "DAG / 开关 / 出参形状均不变。"
                ),
            )],
            confidence=0.0, degraded=True, degradation_reason=_UNAVAILABLE,
        )

    return SkillOutput(
        data=BgmMatchData(bgmKey=bgm_key, ducking=Ducking(), durationMs=total_ms),
        reasoning=[common.chain(
            f"背景音乐匹配完成：{bgm_key}（{total_ms}ms）",
            causal_analysis=f"按锁定总时长 {total_ms}ms 与情绪「{inp.mood or '未指定'}」选曲。",
            risk_note="闪避参数在 render_compose 侧按配音轨生效。",
        )],
        confidence=0.9, sample_size=1,
    )
