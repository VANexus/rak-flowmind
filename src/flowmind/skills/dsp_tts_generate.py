"""tts_generate 技能：口播配音（火山 · 豆包语音「音频生成」）。

契约 §3.1（**2026-10-13 改判，ADR-069**）：``tts`` 是 **DAG 起点**（``dependsOn: []``）——
它只需要 run args 里的 ``shots[].copy``，**不需要任何上游产物**；时间线由 ``timeline`` 在它之后
按**本步返回的逐句实测时间戳**锁定（``tts → timeline``）。**旧「甲案·时间线前移」已推翻**。

- 入参：``shots[].copy``（逐镜口播文案）。
- 文本：**纯文案**（``_dsp_tts.build_text_prompt``）——旧版那句「时间轴控制」自然语言**已删**
  （那是求模型配合、厂商不保证精度，且现在不需要）。
- **时长真源**：``enable_subtitle=true`` 的 ``subtitle.sentences``（句级 / 词级时间戳，**不估算、不 ASR**）
  + 顶层 ``duration``/``original_duration``。它们随 ``voSegments``/``durationMs`` 透出，
  供 ``timeline`` 的 ``_dsp_align`` 把句子归属到镜头——**这就是成片时长的唯一真源**。
- 音轨落「音频夹」``{cachePrefix}/{runId}/audio/voice.mp3``（契约 §2.4）。

上限：文档「单次最长 120s」。**无法再事先知道实测时长**（配音已排在时间线之前），故改成
**按镜头估算值之和预检**（估算 > 120s 即拒绝，绝不送一个必然超限的请求）；实测超限则**如实告警**。

降级 **abort**：失败即抛错（``invoke()`` 兜成 ``ok=False``），运行器据此中止整单——
成片必需步不静默降级（错误永不静默）。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.config import load_config
from flowmind.contracts import SkillOutput
from flowmind.skill import skill
from flowmind.skills import _dsp_common as common
from flowmind.skills import _dsp_storage as storage
from flowmind.skills import _dsp_tts as tts_client
from flowmind.skills._dsp_common import DspBaseInput

_VERSION = "0.2.0"


class TtsGenerateInput(DspBaseInput):
    """配音入参（公共上下文 + 可选音色 / 语速覆盖）。"""
    speaker: str = Field(default="", description="音色 id；缺省用 DspConfig.tts_default_speaker")
    speechRate: int = Field(default=0, description="语速 [-50, 100]")


class VoWord(BaseModel):
    """**词级**（token）实测时间戳（ms，距音频开始）——比句**更细**的实测边界。

    透出给 ``timeline``：某镜实测超长、而其内部没有句边界可切时，``_dsp_align._split_point``
    靠它把切点**下沉到词**（词边界同属实测 token，画面与配音仍逐帧对齐）。
    来自豆包 ``subtitle.sentences[].words[]``；厂商未返回即**空表**（不臆造）。
    """
    atMs: int
    durationMs: int
    text: str


class VoSegment(BaseModel):
    """逐句实测时间戳（**时间线真源**：`_dsp_align` 靠它把句子归属到镜头；字幕也靠它）。

    ``words`` = 该句下的**词级** token（同级实测，供**句内切分下沉**）——可空。
    """
    id: str
    atMs: int
    durationMs: int
    text: str
    words: list[VoWord] = Field(default_factory=list)


class TtsGenerateData(BaseModel):
    """出参：音轨 key + **实测总时长** + 逐句实测时间戳 + 音色。"""
    audioKey: str
    durationMs: int = Field(..., description="音频**实测**总时长（厂商返回；时间线以它收口）")
    voSegments: list[VoSegment] = Field(default_factory=list)
    voice: str = ""
    warnings: list[str] = Field(default_factory=list)


def _shot_copies(inp: TtsGenerateInput) -> list[dict]:
    """逐镜文案（DAG 起点：不读任何上游 EDL —— 时间线由 ``timeline`` 在本步之后锁）。"""
    return [{"id": s.id, "copy": s.copy_text} for s in inp.shots]


@skill(id="tts_generate", name="口播配音", version=_VERSION,
       category="抖音成片", tags=["dsp", "tts", "voice"])
def tts_generate(inp: TtsGenerateInput) -> SkillOutput[TtsGenerateData]:
    """按逐镜文案出音轨 + **逐句实测时间戳**（时间线真源）；超长 / 无句子 / 配额即抛错（abort）。"""
    cfg = load_config().dsp

    if not inp.shots:
        raise ValueError(
            "tts_generate 需要 shots[]（配音文本来自逐镜文案）；无镜头即无文本可合成。"
        )

    # 预检：实测时长此刻不可知，故按「镜头估算值之和」拦下必然超限的请求（如实标注为估算）
    est_ms = sum(s.durationMs for s in inp.shots)
    if est_ms > tts_client.MAX_SINGLE_MS:
        raise tts_client.DspTtsError(
            f"预计配音总时长 {est_ms}ms（按 shots[].durationMs 估算之和）超过单次合成长度上限 "
            f"{tts_client.MAX_SINGLE_MS}ms（文档：单次最长 120s，当前未实现分片）。请精简文案或拆分镜头。",
            category="video",
        )

    prompt = tts_client.build_text_prompt(_shot_copies(inp), max_chars=cfg.tts_max_chars)
    speaker = inp.speaker.strip() or cfg.tts_default_speaker

    audio = tts_client.synthesize(
        cfg, text_prompt=prompt, speaker=speaker, speech_rate=inp.speechRate,
        request_id=inp.idempotencyKey or f"{inp.runId}:{inp.stepId}",
    )

    key = common.audio_key(inp.cachePrefix, inp.runId, ext=cfg.tts_format)
    storage.put_bytes(key, audio.audio_bytes, content_type=audio.content_type)

    vo_segments = [
        VoSegment(
            id=f"s{idx}", atMs=s.start_ms, durationMs=s.duration_ms, text=s.text,
            words=[VoWord(atMs=w.start_ms, durationMs=w.duration_ms, text=w.text) for w in s.words],
        )
        for idx, s in enumerate(audio.sentences, start=1)
    ]
    warnings: list[str] = []
    if not speaker:
        warnings.append("未指定音色（speaker）：本次用模型默认音色；音色属 L1 人工确认项，建议显式指定")
    if audio.duration_ms > tts_client.MAX_SINGLE_MS:
        warnings.append(
            f"配音实测时长 {audio.duration_ms}ms 超过文档的单次上限 {tts_client.MAX_SINGLE_MS}ms"
            "——请核对厂商计费与产物完整性（时间线仍以实测为准）。"
        )
    # 估算与实际相差悬殊时如实提示（不纠正——真源是实测，估算只是参考）
    if est_ms and abs(audio.duration_ms - est_ms) > max(3000, est_ms * 0.5):
        warnings.append(
            f"配音实测 {audio.duration_ms}ms 与镜头估算之和 {est_ms}ms 相差较大"
            "（估算按字数、实测含停顿与语速差异）——成片时长以实测为准。"
        )

    return SkillOutput(
        data=TtsGenerateData(
            audioKey=key, durationMs=audio.duration_ms, voSegments=vo_segments,
            voice=audio.voice, warnings=warnings,
        ),
        reasoning=[common.chain(
            f"配音完成：{len(vo_segments)} 句 / {audio.duration_ms}ms（落 {key}）",
            causal_analysis=(
                f"按逐镜文案出音轨（{len(inp.shots)} 镜，纯文案无时间轴控制）；"
                "逐句时间戳取自 enable_subtitle 的 subtitle.sentences（含**词级 token**），"
                "透出给 timeline 的 _dsp_align 作**时长真源**（长句内部无句边界时可下沉到词切分）。"
            ),
            risk_note=(
                "配音结果有以上告警，请核对（音色 / 超限 / 估算偏差）。"
                if warnings else
                "音轨与逐句时间戳已就绪；时间线将由 timeline 按本次实测锁定，字幕直接用它（无需 ASR 对齐）。"
            ),
        )],
        confidence=0.9 if not warnings else 0.6,
        sample_size=len(vo_segments),
    )
