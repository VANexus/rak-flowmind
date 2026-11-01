"""说话人分离 helper 演示 —— _diarize 的轮次归属与逐说话人参考音频产出。

运行：PYTHONPATH=$PWD/src conda run -n flowmind python examples/diarize_demo.py

三段式展示：
1. happy 路径：双说话人轮次 → assign_speakers 打说话人标签 + build_speaker_refs 产出参考音频
2. 默认路径：available() 的后端语义（本地栈不可用时 diarize 抛 environment）
3. error 路径：某说话人音频过短 → low_confidence=True（不报错）；wav 不存在 → video

本 demo **不下载模型、不依赖网络、不占 GPU**：
- diarize() 被 monkeypatch 为合成轮次（离线、确定性，不加载 CAM++ 模型），
  其余调用的 assign_speakers / build_speaker_refs 均为真实实现；
- 音频用 ffmpeg 现场合成 6s 正弦波，写入 tempfile.mkdtemp() 临时目录，不读用户文件。
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import flowmind.skills._diarize as dz


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _make_wav(path: Path, seconds: float = 6.0) -> None:
    """用 ffmpeg 合成一段 16k 单声道正弦波 wav（离线，不依赖任何素材）。"""
    proc = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"sine=frequency=440:duration={seconds}",
         "-ac", "1", "-ar", "16000", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 合成 wav 失败: {proc.stderr[-200:]}")


def section_happy(wav: Path, workdir: Path) -> None:
    section("1) Happy path：双说话人分离 + 参考音频产出 + 标签归属")
    print(f"  合成音频 : {wav.name}（16k 单声道 6.0s，ffmpeg sine）")

    # monkeypatch：离线确定性轮次（不加载 VAD / CAM++ 模型，不联网）
    turns = [
        {"begin": 0.0, "end": 3.0, "speaker": "spk0"},
        {"begin": 3.0, "end": 6.0, "speaker": "spk1"},
    ]
    segments = [
        {"index": 0, "begin": 0.0, "end": 1.5, "text": "你好，我是第一个说话人"},
        {"index": 1, "begin": 1.5, "end": 3.0, "text": "我是第二个说话人"},
        {"index": 2, "begin": 3.0, "end": 6.0, "text": "我们继续聊第二个话题"},
    ]

    original_diarize = dz.diarize
    dz.diarize = lambda wav_path, **kw: turns
    try:
        got_turns = dz.diarize(str(wav))
    finally:
        dz.diarize = original_diarize
    for t in got_turns:
        print(f"  diarize  : begin={t['begin']:.1f} end={t['end']:.1f} speaker={t['speaker']}")

    # 真实纯函数：按「重叠最大者」归属说话人
    tagged = dz.assign_speakers(segments, got_turns)
    print("  assign_speakers（纯函数，重叠最大者归属）:")
    for seg in tagged:
        print(f"    index={seg['index']} begin={seg['begin']:.1f} end={seg['end']:.1f}"
              f" speaker={seg['speaker']:5s} text={seg['text']}")
    check(all(set(s) == {"index", "begin", "end", "text", "speaker"} for s in tagged),
          "应仅新增 speaker 键，其余键原样保留")
    check(all("speaker" not in s for s in segments), "入参不应被原地修改")
    check([s["speaker"] for s in tagged] == ["spk0", "spk0", "spk1"],
          "按重叠最大者归属应为 spk0/spk0/spk1")

    # 真实裁切拼接：逐说话人产出参考音频（ffmpeg atrim + concat）
    refs = dz.build_speaker_refs(
        tagged, str(wav), str(workdir / "refs"),
        min_seconds=3.0, target_seconds=10.0, max_seconds=15.0,
    )
    print("  build_speaker_refs（真实裁切拼接）:")
    for speaker, info in refs.items():
        print(f"    {speaker}: ref_audio={Path(info['ref_audio']).name}"
              f" seconds={info['seconds']} low_confidence={info['low_confidence']}")
        print(f"        ref_text={info['ref_text']}")
    check(set(refs) == {"spk0", "spk1"}, "两个说话人各应产出一份参考音频")
    check(all(Path(i["ref_audio"]).exists() for i in refs.values()),
          "ref_spk0.wav / ref_spk1.wav 应真实存在")
    check(all(i["low_confidence"] is False for i in refs.values()),
          "两说话人累计音频均 ≥3.0s，不应低置信")


def section_default(wav: Path) -> None:
    section("2) 默认路径：available() 为假时的后端语义")
    print(f"  available()（真实值，纯 import 探测）: {dz.available()}")

    original_available = dz.available
    dz.available = lambda: False
    raised = None
    try:
        dz.diarize(str(wav))
    except dz.DiarizeError as exc:
        raised = exc
    finally:
        dz.available = original_available

    check(raised is not None, "available()=False 时 diarize 应抛 DiarizeError")
    print(f"  抛异常   : {type(raised).__name__}")
    print(f"  category : {raised.category}")
    print(f"  message  : {raised}")
    check(raised.category == "environment", "本地栈不可用应为 environment 分类")
    print("  语义     : auto 模式 → 流水线捕获后跳过分离、回落整片单音色克隆；"
          "local 模式 → 直接向上报错（错误永不静默）")


def section_error(wav: Path, workdir: Path) -> None:
    section("3) error 路径：某说话人音频过短 → low_confidence=True（不报错）")
    segments = [
        {"index": 0, "begin": 0.0, "end": 1.2, "text": "很短的一句", "speaker": "spk0"},
        {"index": 1, "begin": 1.2, "end": 4.5, "text": "这段够长累计超过三秒", "speaker": "spk1"},
    ]
    refs = dz.build_speaker_refs(
        segments, str(wav), str(workdir / "refs_short"), min_seconds=3.0,
    )
    for speaker, info in refs.items():
        print(f"    {speaker}: seconds={info['seconds']} low_confidence={info['low_confidence']}")
    check(refs["spk0"]["low_confidence"] is True, "累计 1.2s < 3.0s 应低置信")
    check(refs["spk1"]["low_confidence"] is False, "累计 3.3s 应正常产出")
    check(Path(refs["spk0"]["ref_audio"]).exists(), "低置信仍应产出 wav（不报错）")

    print("\n  再补：wav 路径不存在 → DiarizeError(category='video')")
    raised = None
    try:
        dz.build_speaker_refs(segments, str(workdir / "no_such.wav"), str(workdir / "refs_x"))
    except dz.DiarizeError as exc:
        raised = exc
    check(raised is not None, "不存在的 wav 应抛 DiarizeError")
    print(f"  {type(raised).__name__}: category={raised.category} msg={raised}")
    check(raised.category == "video", "输入媒体问题应为 video 分类")


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="diarize_demo_"))
    try:
        wav = workdir / "dialog.wav"
        _make_wav(wav, 6.0)
        section_happy(wav, workdir)
        section_default(wav)
        section_error(wav, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("\n✅ diarize_demo 全部通过")


if __name__ == "__main__":
    main()
