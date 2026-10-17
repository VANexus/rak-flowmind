"""dsp_subtitle_build 技能演示 —— 字幕 SRT（依配音逐句时间戳，只出参数不入画）。

运行：PYTHONPATH=<repo>/src conda run -n flowmind python examples/dsp_subtitle_build_demo.py

展示：
1. discover() 自动字段发现
2. happy path：deps.tts.voSegments 逐句 → SRT 文件 + 逐条字幕
2b. **词级 SRT**（ADR-073）：厂商带 `words[]` → **另出**一份词级 SRT（供导出/二次剪辑，不参与烧制）
3. 时间点越界 → **如实报出、不夹取**（warnings）
4. 无配音时间戳 → 如实降级（不伪造字幕）

mock 方式：替换 ``_dsp_storage.put_text`` 为内存实现，不需要对象存储。
"""

from __future__ import annotations

import flowmind.skills  # noqa: F401  触发 @skill 注册
from flowmind.discover import field_names
from flowmind.skill import invoke
from flowmind.skills import _dsp_storage as storage

_WRITTEN: dict[str, str] = {}


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _install_storage() -> None:
    def put_text(key: str, text: str, content_type: str | None = None) -> str:
        _WRITTEN[key] = text
        return key

    storage.put_text = put_text


def _ctx(deps: dict) -> dict:
    return {
        "mode": "content", "platform": "douyin",
        "cachePrefix": "dsp/scratch/t1", "outputPrefix": "dsp/output/t1",
        "runId": "run-demo-4", "stepId": "subtitle", "idempotencyKey": "run-demo-4:subtitle",
        "shots": [], "materials": [], "deps": deps,
    }


def _edl(total_ms: int) -> dict:
    return {"timeline": {"edl": {"version": 1, "totalMs": total_ms}}}


def main() -> None:
    section("0) discover('subtitle_build') —— Agent 自查字段")
    for path, names in field_names("subtitle_build").items():
        print(f"  {path}: {names}")

    section("1) Happy path：配音逐句时间戳 → SRT")
    _install_storage()
    _WRITTEN.clear()
    deps = _edl(9000) | {"tts": {"voSegments": [
        {"id": "s1", "atMs": 0, "durationMs": 2900, "text": "开门的瞬间"},
        {"id": "s2", "atMs": 3000, "durationMs": 3000, "text": "保温 12 小时"},
    ]}}
    r = invoke("subtitle_build", _ctx(deps))
    check(r.ok, "happy path 应 ok")
    d = r.data
    print(f"  srtKey     : {d.srtKey}")
    print(f"  字幕条数   : {len(d.subtitles)}")
    print("  SRT 片段   :")
    for line in _WRITTEN[d.srtKey].strip().splitlines()[:4]:
        print(f"    | {line}")
    check(d.srtKey in _WRITTEN, "SRT 须落对象存储")
    check(len(d.subtitles) == 2, "两条字幕")
    check("00:00:02,900 --> 00:00:03,000" not in _WRITTEN[d.srtKey], "时间戳格式应正确")
    check("00:00:00,000 --> 00:00:02,900" in _WRITTEN[d.srtKey], "第一条 0→2.9s")
    check(d.warnings == [], "正常时间点不应有越界告警")
    check(d.wordsSrtKey == "", "厂商未给词级 token ⇒ 不出词级 SRT（不拿句级冒充，ADR-073）")

    section("2b) 词级 SRT（ADR-073）：带 words[] → 另出一份词级字幕（不参与烧制）")
    _WRITTEN.clear()
    r = invoke("subtitle_build", _ctx(_edl(9000) | {"tts": {"voSegments": [
        {"id": "s1", "atMs": 0, "durationMs": 2000, "text": "开门的瞬间",
         "words": [{"atMs": 0, "durationMs": 900, "text": "开门"},
                   {"atMs": 900, "durationMs": 1100, "text": "的瞬间"}]},
    ]}}))
    d = r.data
    print(f"  srtKey      : {d.srtKey}")
    print(f"  wordsSrtKey : {d.wordsSrtKey}")
    print("  词级 SRT 片段：")
    for line in _WRITTEN[d.wordsSrtKey].strip().splitlines()[:4]:
        print(f"    | {line}")
    check(d.srtKey in _WRITTEN and d.wordsSrtKey in _WRITTEN, "两份 SRT 都须落对象存储")
    check(d.wordsSrtKey != d.srtKey, "词级与句级是两个文件（烧制口径不变）")
    check(len(d.subtitles) == 1, "句级仍是 1 条（烧制口径不受影响）")
    check("00:00:00,900 --> 00:00:02,000" in _WRITTEN[d.wordsSrtKey], "词级第 2 条 0.9→2.0s")
    check("开门" in _WRITTEN[d.wordsSrtKey] and "的瞬间" in _WRITTEN[d.wordsSrtKey], "词级逐词成条")

    section("2) 时间点越界 → 如实报出、不夹取")
    r = invoke("subtitle_build", _ctx(_edl(5000) | {"tts": {"voSegments": [
        {"id": "s1", "atMs": 4000, "durationMs": 3000, "text": "超出锁定时间线的句子"},
    ]}}))
    print(f"  warnings   : {r.data.warnings}")
    check(r.ok and r.data.warnings, "越界须如实写入 warnings（不静默夹取）")

    section("3) 无配音时间戳（关配音 / 未跑 tts）→ 如实降级")
    r = invoke("subtitle_build", _ctx(_edl(9000)))
    print(f"  metrics.degraded   : {r.metrics.degraded}")
    print(f"  degradation_reason : {r.metrics.degradation_reason}")
    print(f"  srtKey             : {r.data.srtKey!r}")
    check(r.metrics.degraded is True and r.data.srtKey == "", "无逐句时间戳即不产出 SRT")

    print("\n✅ dsp_subtitle_build_demo 全部通过")


if __name__ == "__main__":
    main()
