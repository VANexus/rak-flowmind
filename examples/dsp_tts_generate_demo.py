"""dsp_tts_generate 技能演示 —— 口播配音（豆包语音；出音轨 + **逐句实测时间戳**）。

运行：PYTHONPATH=<repo>/src conda run -n flowmind python examples/dsp_tts_generate_demo.py

**ADR-069 改判后**，`tts` 是 **DAG 起点**（`dependsOn: []`）：不读任何上游产物、不需要锁定时间线；
它的**逐句实测时间戳**反过来成为 `timeline` 的时长真源（`tts → timeline`）。

展示：
1. discover() 自动字段发现
2. happy path：**无上游依赖**也跑通；`text_prompt` 是**纯文案**（旧「时间轴控制」已删）；
   出音轨 + 逐句实测时间戳（含**词级 token**）+ 实测总时长
3. error：预计配音超单次合成上限 120s → 明确报错（需分片，当前未实现）
4. error：无 shots（无文本可合成）→ abort

mock 方式：替换 ``_dsp_tts.synthesize``（模块级薄适配层）与 ``_dsp_storage.put_bytes``，
不需要网络 / 密钥 / 对象存储。
"""

from __future__ import annotations

import flowmind.skills  # noqa: F401  触发 @skill 注册
from flowmind.discover import field_names
from flowmind.skill import invoke
from flowmind.skills import _dsp_storage as storage
from flowmind.skills import _dsp_tts as tts_client

_WRITTEN: dict[str, str] = {}
_CALLS: list[dict] = []


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _install_fakes() -> None:
    def synthesize(cfg, *, text_prompt, speaker, speech_rate=0, request_id=""):
        _CALLS.append({"text_prompt": text_prompt, "speaker": speaker,
                       "speech_rate": speech_rate, "request_id": request_id})
        return tts_client.TtsAudio(
            audio_bytes=b"FAKE-MP3-BYTES", content_type="audio/mpeg",
            sentences=[
                tts_client.VoSentence(
                    start_ms=0, end_ms=2900, text="开门的瞬间",
                    words=[
                        tts_client.VoWord(start_ms=0, end_ms=1400, text="开门"),
                        tts_client.VoWord(start_ms=1400, end_ms=2900, text="的瞬间"),
                    ],
                ),
                tts_client.VoSentence(
                    start_ms=3000, end_ms=6000, text="保温 12 小时",
                    words=[
                        tts_client.VoWord(start_ms=3000, end_ms=4500, text="保温"),
                        tts_client.VoWord(start_ms=4500, end_ms=6000, text="12 小时"),
                    ],
                ),
            ],
            duration_ms=6000, voice=speaker,
        )

    def put_bytes(key: str, data: bytes, content_type: str | None = None) -> str:
        _WRITTEN[key] = f"<{len(data)}B {content_type}>"
        return key

    tts_client.synthesize = synthesize
    storage.put_bytes = put_bytes


def _ctx(deps: dict | None = None, shots: list[dict] | None = None) -> dict:
    return {
        "mode": "content", "platform": "douyin",
        "cachePrefix": "dsp/scratch/t1", "outputPrefix": "dsp/output/t1",
        "runId": "run-demo-2", "stepId": "tts", "idempotencyKey": "run-demo-2:tts",
        "speaker": "demo-voice",
        "shots": shots if shots is not None else [
            {"id": "s1", "durationMs": 3000, "weight": 3, "copy": "开门的瞬间"},
            {"id": "s2", "durationMs": 3000, "weight": 6, "copy": "保温 12 小时"},
        ],
        "materials": [], "deps": deps or {},
    }


def main() -> None:
    section("0) discover('tts_generate') —— Agent 自查字段")
    for path, names in field_names("tts_generate").items():
        print(f"  {path}: {names}")

    section("1) Happy path：**无上游依赖**（DAG 起点）→ 音轨 + 逐句实测时间戳")
    _install_fakes()
    _WRITTEN.clear()
    _CALLS.clear()
    r = invoke("tts_generate", _ctx())   # deps 为空——不再需要 deps.timeline.edl
    check(r.ok, "happy path 应 ok（tts 是 DAG 起点，不需上游）")
    d = r.data
    print(f"  audioKey   : {d.audioKey} → {_WRITTEN.get(d.audioKey)}")
    print(f"  durationMs : {d.durationMs}（**实测**总长）/ voice={d.voice}")
    for seg in d.voSegments:
        print(f"    • {seg.id}: {seg.atMs}–{seg.atMs + seg.durationMs}ms 「{seg.text}」")
    print(f"  text_prompt（发往厂商）: {_CALLS[0]['text_prompt']!r}")
    check(d.audioKey == "dsp/scratch/t1/run-demo-2/audio/voice.mp3", "音轨须落「音频夹」")
    check(d.durationMs == 6000, "durationMs = 厂商实测总长（时间线的收口依据）")
    check(len(d.voSegments) == 2, "两句 → 两个 voSegment")
    check(d.voSegments[0].words and d.voSegments[0].words[0].text == "开门",
          "voSegment 须带**词级 token**（供 timeline 的句内下沉切分）")
    check(
        [(w.atMs, w.durationMs, w.text) for w in d.voSegments[1].words]
        == [(3000, 1500, "保温"), (4500, 1500, "12 小时")],
        "词级 token 的 atMs/durationMs/text 逐条透出",
    )
    check(
        _CALLS[0]["text_prompt"] == "开门的瞬间\n保温 12 小时",
        "text_prompt 必须是**纯文案**（各镜文案按 \\n 拼接）",
    )
    check("时间轴控制" not in _CALLS[0]["text_prompt"],
          "旧「时间轴控制」自然语言段已删（那是求模型配合，厂商不保证精度）")
    check(_CALLS[0]["request_id"] == "run-demo-2:tts", "幂等键须透传给厂商（防重复计费）")
    check(not any("偏差" in w for w in d.warnings), "不再有 drift 告警（时长即真源，无偏差可言）")

    section("2) error：预计配音超 120s 单次上限 → 明确报错（需分片）")
    r = invoke("tts_generate", _ctx(shots=[{"id": "s1", "durationMs": 130_000, "copy": "很长的文案"}]))
    print(f"  ok        : {r.ok}")
    print(f"  error.msg : {(r.error.message if r.error else '')[:100]}")
    check(r.ok is False and "120" in (r.error.message if r.error else ""),
          "超上限须报明 120s 分片要求（按镜头估算之和预检）")

    section("3) error：无 shots（无文本可合成）→ abort")
    r = invoke("tts_generate", _ctx(shots=[]))
    print(f"  ok        : {r.ok}")
    print(f"  error.msg : {(r.error.message if r.error else '')[:80]}")
    check(r.ok is False, "无镜头即无文本：必须失败（不静默降级）")

    print("\n✅ dsp_tts_generate_demo 全部通过")


if __name__ == "__main__":
    main()
