"""
voice_clone_enroll 技能演示 —— 声音复刻（样本音频 → 复刻音色 ID）。

运行：conda run -n flowmind python examples/voice_clone_enroll_demo.py

流程：百炼 voice-enrollment 提交 10~20 秒公网样本 → 返回 voice_id，
可直接传给 localize_video 的 voice_id 做克隆配音。

本 demo mock 复刻接口调用，展示编排与输出形状；
真打需 export AI_SPEECH_API_KEY 并提供公网可访问样本音频 URL。
"""

from __future__ import annotations

import flowmind.skills  # noqa: F401  触发 @skill 注册
import flowmind.skills.voice_clone_enroll as vce
from flowmind.discover import field_names
from flowmind.skill import invoke


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def main() -> None:
    section("0) discover('voice_clone_enroll') —— Agent 自查字段")
    for p, names in field_names("voice_clone_enroll").items():
        print(f"  {p}: {names}")

    section("1) Happy path：样本 URL → voice_id（mock 复刻接口）")
    vce.get_api_key = lambda env: "mock-key"
    vce._voice_enroll._create_voice_adapter = (
        lambda url, *, prefix, api_key, target_model, language_hint:
        f"{target_model}-{prefix}-demo123456"
    )
    r = invoke("voice_clone_enroll", {
        "sample_url": "https://cdn.example.com/samples/speaker_a.wav",
        "language_hint": "zh",
    })
    print(f"  ok              : {r.ok}")
    print(f"  voice_id        : {r.data.voice_id}")
    print(f"  绑定合成模型    : {r.data.target_model}")
    print(f"  trace_id        : {r.trace.trace_id[:8]}...")
    print(f"  推理结论        : {r.reasoning[0].conclusion}")
    print(f"  → 用法: invoke('localize_video', {{...,'voice_id': '{r.data.voice_id}'}})")

    section("2) 无 key：显式 degraded，不静默降级")
    vce.get_api_key = lambda env: None
    r = invoke("voice_clone_enroll", {
        "sample_url": "https://cdn.example.com/samples/speaker_a.wav"})
    print(f"  degraded        : {r.metrics.degraded}")
    print(f"  failure_category: {r.data.failure_category}（environment → 先配 key）")
    print(f"  warning         : {r.data.warning}")

    section("3) 非 URL 样本（本地路径）：video 类（修输入）")
    vce.get_api_key = lambda env: "k"
    r = invoke("voice_clone_enroll", {"sample_url": "/tmp/local_voice.wav"})
    print(f"  failure_category: {r.data.failure_category}（video）")
    print(f"  warning         : {r.data.warning}")

    section("4) prefix 非法：VALIDATION（pydantic 校验拦截）")
    r = invoke("voice_clone_enroll", {
        "sample_url": "https://cdn.example.com/s.wav", "prefix": "中文前缀!"})
    print(f"  ok              : {r.ok}")
    print(f"  error.code      : {r.error.code}")
    print(f"  error.message   : {r.error.message}")


if __name__ == "__main__":
    main()
