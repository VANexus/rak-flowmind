"""
翻译本土化 + 时长对齐质量自检 演示 —— 三段式冒烟（不联网、不加载模型、不占 GPU）。

运行：PYTHONPATH=$PWD/src conda run -n flowmind python examples/localize_quality_demo.py

本 demo 全程零网络请求、不加载 Qwen3-TTS / LaMa / funasr 模型、不占 GPU 槽：
  1) 术语表命中锁定译法：真实 load_glossary + 真实 translate_segments（注入假 httpx
     client 拦截请求）→ 验证术语/文化规则确实拼进系统提示；
  2) 长度预算超限：真实 estimate_length_budget + 真实 translate_segments（假 client）
     → 验证返回句段的 over_budget 标志；
  3) 配音溢出统计与 quality 块形状：真实 _mix_timed_audio（现场 ffmpeg 生成 wav）
     → 核对 compressed/overflow/max_overflow_ratio/worst_index，再 mock 全链路跑
     invoke("localize_video") 打印 r.data.quality 完整形状。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import flowmind.skills  # noqa: F401  触发 @skill 注册
import flowmind.skills._llm_translate as lt
import flowmind.skills._media as media
import flowmind.skills.localize_video as lv
from flowmind.skill import invoke


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


class FakeResponse:
    """假 httpx 响应：只暴露 translate_segments 用到的 status_code / json()。"""

    def __init__(self, status_code: int, data: dict):
        self.status_code = status_code
        self._data = data

    def json(self) -> dict:
        return self._data


class FakeClient:
    """假 httpx.Client：记录请求体，并按预设译文构造 Anthropic 形状响应（零网络）。

    replies 以「批内局部序号」为键（与 translate_segments 发出的 user payload 一致），
    取不到时回落原文本，保证回复条数与原文一致。
    """

    def __init__(self, replies: dict[int, str]):
        self.replies = replies
        self.calls: list[dict] = []

    def post(self, url: str, headers: dict | None = None,
             **kwargs) -> FakeResponse:
        # httpx 以关键字 json=<body> 传请求体；用 **kwargs 收，避免遮蔽 json 模块
        body = kwargs["json"]
        self.calls.append({"url": url, "body": body})
        payload = json.loads(body["messages"][0]["content"])
        items = [
            {"index": s["index"], "text": self.replies.get(s["index"], s["text"])}
            for s in payload["segments"]
        ]
        text = json.dumps(items, ensure_ascii=False)
        return FakeResponse(200, {"content": [{"type": "text", "text": text}]})


def demo_glossary(tmp: Path) -> None:
    """1) 术语表命中锁定译法（真实 load_glossary + 真实 translate_segments）。"""
    section("1) 术语表命中锁定译法")
    glossary_path = tmp / "glossary.json"
    glossary_path.write_text(json.dumps({
        "terms": {"老干妈": "Lao Gan Ma", "直播带货": "live commerce"},
        "culture_rules": ["货币换算为美元", "称谓改用 first name"],
    }, ensure_ascii=False), encoding="utf-8")

    terms, rules = lt.load_glossary(str(glossary_path))
    print(f"  terms         : {terms}")
    print(f"  culture_rules : {rules}")

    # 真实 translate_segments（假 client 拦截，不发网络请求）
    src = [{"index": 0, "begin": 0.0, "end": 2.0, "text": "老干妈靠直播带货火了"}]
    client = FakeClient({0: "Lao Gan Ma went viral through live commerce"})
    out = lt.translate_segments(
        src, target_lang="en", api_key="mock-key", client=client,
        glossary=terms, culture_rules=rules,
    )
    system_prompt = client.calls[0]["body"]["system"]
    hit = ("Lao Gan Ma" in system_prompt and "live commerce" in system_prompt)
    print(f"  译文          : {out[0]['text']}")
    print(f"  提示词注入术语: {hit}"
          f"（{len(client.calls)} 次请求，URL={client.calls[0]['url']}）")
    print(f"  提示词术语行  : "
          f"{[ln for ln in system_prompt.splitlines() if '必须译为' in ln]}")
    print(f"  提示词文化行  : "
          f"{[ln for ln in system_prompt.splitlines() if '美元' in ln or 'first name' in ln]}")
    assert hit, "术语/文化规则未拼进系统提示"

    # 路径不存在 → LLMTtranslateError(category="environment")
    try:
        lt.load_glossary("/no/such/file.json")
    except lt.LLMTtranslateError as exc:
        print(f"  缺文件        : {type(exc).__name__} / category={exc.category}"
              f" / {exc}")
        assert exc.category == "environment"
    else:
        raise AssertionError("load_glossary 未对不存在的路径报错")


def demo_length_budget() -> None:
    """2) 长度预算超限 → over_budget=True（真实 estimate_length_budget）。"""
    section("2) 长度预算超限 → over_budget=True")
    segments = [
        {"index": 0, "begin": 0.0, "end": 2.0, "text": "欢迎来到直播间"},
        {"index": 1, "begin": 2.0, "end": 2.4, "text": "今天上新"},
    ]
    budget = lt.estimate_length_budget(segments, "en")
    print(f"  LANG_SPEED_FACTOR['en'] = {lt.LANG_SPEED_FACTOR['en']}（每秒字符数经验值）")
    print(f"  estimate_length_budget  = {budget}")

    # 真实 translate_segments（假 client 拦截）：第 0 句故意超长、第 1 句不超
    client = FakeClient({0: "A" * 40, 1: "Hi"})
    out = lt.translate_segments(
        segments, target_lang="en", api_key="mock-key",
        client=client, length_budget=budget,
    )
    for seg in out:
        n = len(seg["text"])
        print(f"  index={seg['index']}  预算={budget[seg['index']]:>2}  "
              f"译文长度={n:>2}  over_budget={seg['over_budget']}  "
              f"text={seg['text']!r}")
    budget_lines = [ln for ln in client.calls[0]["body"]["system"].splitlines()
                    if "个字符" in ln]
    print(f"  提示词预算行  : {budget_lines}")
    assert out[0]["over_budget"] is True and out[1]["over_budget"] is False, \
        "over_budget 标记与预算不符"


def demo_dub_stats(tmp: Path) -> dict:
    """3A) 真实 _mix_timed_audio：三档 tempo 现场核对统计。"""
    section("3A) 配音溢出统计（真实 _mix_timed_audio + 现场 ffmpeg 生成 wav）")
    dubs = []
    for name, seconds, freq in (("d0", 2.5, 440), ("d1", 2.4, 523), ("d2", 3.5, 659)):
        wav = str(tmp / f"{name}.wav")
        rc, _, err = media.run_ffmpeg([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"sine=frequency={freq}:duration={seconds}",
            "-ac", "1", "-ar", "16000", wav,
        ])
        assert rc == 0, f"生成 wav 失败: {err[-120:]}"
        dubs.append(wav)
    # tempo = 配音时长 / 原句时长：1.00 不压 / 1.04 压缩 / 2.50 溢出
    segments = [
        {"index": 0, "begin": 0.0, "end": 2.5, "text": "句 0"},
        {"index": 1, "begin": 2.5, "end": 4.8, "text": "句 1"},
        {"index": 2, "begin": 4.8, "end": 6.2, "text": "句 2"},
    ]
    out_path, stats = lv._mix_timed_audio(
        segments, dubs, 6.2, str(tmp / "mixed.wav"), max_tempo=1.3)
    for seg, dub in zip(segments, dubs):
        seg_dur = seg["end"] - seg["begin"]
        tempo = media.probe_media(dub)[0] / seg_dur
        print(f"  index={seg['index']}  原句={seg_dur:.2f}s  "
              f"配音={media.probe_media(dub)[0]:.2f}s  tempo={tempo:.3f}")
    print(f"  stats         : {stats}")
    print(f"  产物          : {Path(out_path).name}（{Path(out_path).stat().st_size} 字节）")
    assert stats["compressed"] == 1 and stats["overflow"] == 1, "分级变速统计不符"
    assert stats["worst_index"] == 2 and stats["max_overflow_ratio"] == 2.5, \
        "最差溢出句定位不符"
    # 向后兼容：_build_timed_audio 仍返回 str
    legacy = lv._build_timed_audio(
        segments, dubs, 6.2, str(tmp / "mixed_legacy.wav"))
    print(f"  _build_timed_audio 返回类型: {type(legacy).__name__}（=str 向后兼容）")
    assert isinstance(legacy, str)
    return stats


def install_mock_pipeline(dub_stats: dict) -> None:
    """3B) mock 全链路：云调用 / ffmpeg / 本地模型 / 混音统计全部替身。"""
    segs = [
        {"index": 0, "begin": 0.0, "end": 2.1, "text": "大家好，欢迎收看本期节目"},
        {"index": 1, "begin": 2.1, "end": 4.8, "text": "今天我们来聊发动机保养"},
    ]
    translated = [
        {**segs[0], "text": "Hi everyone, welcome", "over_budget": False},
        {**segs[1], "text": "Today we talk about engine maintenance",
         "over_budget": True},   # 故意标超预算，验证 quality.translate_over_budget
    ]
    lv.get_api_key = lambda env: "mock-key"
    lv._media.extract_audio = lambda v, o, **kw: o
    lv._media.probe_media = lambda p: (4.8, 640, 360)
    lv._media.extract_frame = lambda v, t, o: (Path(o).write_bytes(b"") or o)
    lv._media.burn_subtitles = lambda vp, op, ass, erase_regions=None: (
        Path(op).write_bytes(b"x") or op)
    lv._media.mix_audio = lambda vp, d, op, keep_background=False, bg_path=None: op
    lv._inpaint.available = lambda: False     # demo 走 delogo 分支（不跑真 LaMa）
    lv._local_tts.available = lambda: False   # demo 走云 TTS 分支（不加载 4GB 本地模型）
    lv._music_sep.available = lambda: False   # demo 不做真人声分离（无 demucs 依赖）
    lv._diarize.available = lambda: False     # demo 跳过说话人分离（不跑真 funasr 模型）
    lv._local_asr.available = lambda: False   # auto 后端回落到已 mock 的云 ASR
    lv._local_ocr.available = lambda: False   # auto 后端回落到已 mock 的云 OCR
    lv._cloud_asr.transcribe_local = lambda wav, api_key, **kw: segs
    lv._locate_region = lambda src, dur, wd, key, cfg, **kw: [
        {"x": 120, "y": 280, "w": 400, "h": 40}]   # 落在 640x360 画面内，避免被裁边丢弃
    lv._llm_translate.translate_segments = lambda s, **kw: translated
    lv._cloud_tts.synthesize_segments = lambda segments, **kw: [
        f"{kw['out_dir']}/seg_{x['index']:04d}.mp3" for x in segments]
    lv._mix_timed_audio = lambda segs, dubs, dur, out, **kw: (out, dub_stats)
    lv._replace_audio = lambda vp, audio, out: out


def demo_invoke(dub_stats: dict) -> None:
    """3B) invoke("localize_video") → 打印 quality 块完整形状。"""
    section("3B) invoke('localize_video') → r.data.quality 形状")
    Path("/tmp/demo.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")  # 预检用假 mp4
    install_mock_pipeline(dub_stats)
    r = invoke("localize_video", {
        "video_path": "/tmp/demo.mp4",
        "target_lang": "en",
        "voice_id": "cosyvoice-v3.5-plus-demo-abc",
        "output_path": "/tmp/demo_localized.mp4",
    })
    print(f"  ok              : {r.ok}")
    print(f"  diarize_backend : {r.data.diarize_backend}")
    print(f"  quality         : {json.dumps(r.data.quality, ensure_ascii=False)}")
    print(f"  推理结论        : {r.reasoning[0].conclusion}")
    assert r.ok and r.data.quality is not None
    assert r.data.quality["dub"]["overflow"] == 1
    assert r.data.quality["translate_over_budget"] == [1]
    assert r.data.quality["erase_coverage"] == 1


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    demo_glossary(tmp)
    demo_length_budget()
    dub_stats = demo_dub_stats(tmp)
    demo_invoke(dub_stats)
    print(f"\n{'═' * 60}\n  ALL PASS（临时目录：{tmp}）\n{'═' * 60}")


if __name__ == "__main__":
    main()
