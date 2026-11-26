"""edit_* 渲染层离线回归 —— 不调 LLM，纯本机 ffmpeg 验证渲染正确性。

背景：10-28 真实运行里渲染层连修数个 bug（xfade duration=0 截断、Windows
盘符冒号、-vf 位置），修复后长期没有真实渲染验证；kenburns 的
-loop 1 -t D × zoompan d=N 长度爆炸隐患也只有离线渲染能兜住。

运行：PYTHONPATH=<repo>/src python examples/edit_render_check.py
（需要本机 ffmpeg/ffprobe 在 PATH；全程零 API 调用，CPU 约 1-2 分钟）

覆盖路径与断言（成片时长 = 时间线折算时长，容差 ±0.6s）：
1. xfade 链（fade/none 混合转场 + 变速 ×2 + 尾帧定格 + 图片 Ken Burns）
   + 短口播 apad 垫底（口播 1.5s < 成片，验证视频不再被 -shortest 截断）
   + 字幕烧录（真实 SRT 路径含 Windows 盘符冒号）；
2. 全 none 转场 → concat demuxer 路径 + subtitle_srt=None（subtitle=False）；
3. preview 自检帧抽取。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flowmind.editing import media  # noqa: E402
from flowmind.editing.models import TimelineDoc, TimelineShot  # noqa: E402

_TOLERANCE_S = 0.6


def _gen_assets(work: Path) -> dict[str, Path]:
    """ffmpeg lavfi 生成测试素材（视频×2 / 图片 / 短口播）。"""
    def run(cmd: list[str]) -> None:
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    clip1 = work / "clip1.mp4"
    clip2 = work / "clip2.mp4"
    img = work / "img.png"
    vo = work / "vo.m4a"
    bgm = work / "bgm.wav"
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=d=6:s=640x360:r=25",
         *_x264(), str(clip1)])
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "smptebars=d=4:s=640x360:r=25",
         *_x264(), str(clip2)])
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=d=1:s=800x600:r=25",
         "-frames:v", "1", str(img)])
    # 口播 1.5s：短于成片，验证 apad 垫底后视频不被截断
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5",
         "-c:a", "aac", str(vo)])
    # BGM 10s：长于成片，验证 atrim + 淡入淡出
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=220:duration=10", str(bgm)])
    return {"clip1": clip1, "clip2": clip2, "img": img,
            "vo": vo, "bgm": bgm}


def _x264() -> list[str]:
    return ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"]


def _shot(shot_id: str, asset: str, *, source_in: float = 0.0,
          duration_s: float, mode: str = "video_clip", pan: str = "none",
          transition: str = "fade", speed: float = 1.0, freeze_s: float = 0.0,
          subtitle: str = "", subtitle_emphasis: list[str] | None = None,
          badge: str = "", hook: bool = False, zoom_start: float = 1.0,
          zoom_end: float = 1.0,
          source_path: Path = Path("")) -> TimelineShot:
    return TimelineShot(
        id=shot_id, section_id="", source_asset=asset, source_in=source_in,
        duration_s=duration_s, mode=mode, pan=pan, transition=transition,
        speed=speed, freeze_s=freeze_s, subtitle_text=subtitle,
        subtitle_emphasis=subtitle_emphasis or [], badge=badge, hook=hook,
        zoom_start=zoom_start, zoom_end=zoom_end,
        source_path=str(source_path))


def _check(label: str, got: float, want: float) -> None:
    delta = abs(got - want)
    ok = delta <= _TOLERANCE_S
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: 成片 {got:.2f}s / "
          f"时间线 {want:.2f}s（偏差 {delta:.2f}s，容差 {_TOLERANCE_S}s）")
    if not ok:
        raise SystemExit(f"渲染回归失败：{label}")


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="flowmind-render-check-"))
    print(f"工作目录：{work}")
    assets = _gen_assets(work)

    # ── 场景 1：v1.3 设计层全路径 ──
    # 设计卡×2 + 多样转场 + 视频缓推 + kenburns + 变速/定格 + ASS 动态字幕
    # （关键词高亮/角标/钩子居中）+ 短口播 apad + BGM 混音
    from flowmind.editing.models import TitleCard

    shots_a = [
        _shot("card_title", "card", duration_s=1.5, mode="card",
              transition="none", source_path=Path("")),
        _shot("sh1", "a1", source_in=0.5, duration_s=2.0,
              transition="zoomin", subtitle="第一镜：钩子大字",
              subtitle_emphasis=["钩子"], badge="01/04", hook=True,
              zoom_start=1.0, zoom_end=1.08, source_path=assets["clip1"]),
        _shot("sh2", "a1", source_in=3.0, duration_s=1.2, speed=2.0,
              transition="slideleft", subtitle="第二镜：两倍速",
              badge="02/04", zoom_start=1.08, zoom_end=1.0,
              source_path=assets["clip1"]),
        _shot("sh3", "a2", duration_s=2.0, mode="image_kenburns", pan="in",
              transition="circleopen", subtitle="第三镜：图片 Ken Burns",
              badge="03/04", source_path=assets["img"]),
        _shot("sh4", "a1", source_in=1.0, duration_s=1.0, freeze_s=0.6,
              transition="dissolve", subtitle="第四镜：定格收尾",
              badge="04/04", source_path=assets["clip1"]),
        _shot("card_end", "card", duration_s=1.5, mode="card",
              transition="fadeblack", source_path=Path("")),
    ]
    timeline_a = TimelineDoc(
        width=640, height=360, fps=30, shots=shots_a,
        title_card=TitleCard(main="标题卡·钩子", sub="副标题一行",
                             duration_s=1.5),
        end_card=TitleCard(main="点赞关注", sub="CTA 副标题", duration_s=1.5),
        bgm_path=str(assets["bgm"]))
    starts = media.effective_global_starts(
        shots_a, transition_duration_s=0.5, fps=30)
    for shot, start in zip(shots_a, starts):
        shot.global_start_s = start
        shot.global_end_s = round(start + shot.duration_s, 3)
    timeline_a.duration_s = round(starts[-1] + shots_a[-1].duration_s, 3)
    timeline_a.voiceover_path = str(assets["vo"])

    ass = work / "out.ass"   # 真实落盘路径含 Windows 盘符冒号，顺带回归转义
    ass.write_text(media.build_ass(timeline_a), encoding="utf-8")
    assert "Dialogue:" in ass.read_text(encoding="utf-8"), "ASS 无事件"
    out_a = media.render(
        timeline_a, workdir=work / "render_a", subtitle_file=ass,
        transition_duration_s=0.5, subtitle_font="Arial",
        subtitle_font_size=22, x264_preset="veryfast", video_bitrate="2M")
    got_a = media.probe_summary(str(out_a))["duration_s"]
    _check("场景1 设计卡+多样转场+推拉+ASS 字幕+BGM+口播", got_a,
           timeline_a.duration_s)
    assert got_a > 1.6, "短口播把视频截断了（apad 未生效）"
    info_a = media.probe(str(out_a))
    assert any(st.get("codec_type") == "audio" for st in info_a["streams"]), \
        "BGM/口播混音后无音轨"

    # ── 场景 2：全 none → concat 路径 + subtitle_srt=None ──
    shots_b = [
        _shot("sh1", "a1", source_in=0.0, duration_s=1.5,
              source_path=assets["clip1"]),
        _shot("sh2", "a3", source_in=0.0, duration_s=2.0,
              source_path=assets["clip2"]),
    ]
    timeline_b = TimelineDoc(width=640, height=360, fps=30, shots=shots_b)
    starts_b = media.effective_global_starts(
        shots_b, transition_duration_s=0.5, fps=30)
    for shot, start in zip(shots_b, starts_b):
        shot.global_start_s = start
        shot.global_end_s = round(start + shot.duration_s, 3)
    timeline_b.duration_s = round(starts_b[-1] + shots_b[-1].duration_s, 3)
    out_b = media.render(
        timeline_b, workdir=work / "render_b", subtitle_file=None,
        transition_duration_s=0.5, subtitle_font="Arial",
        subtitle_font_size=22, x264_preset="veryfast", video_bitrate="2M")
    got_b = media.probe_summary(str(out_b))["duration_s"]
    _check("场景2 concat 直切+无字幕", got_b, timeline_b.duration_s)

    # ── 场景 3：preview 自检帧 ──
    previews = media.extract_preview(str(out_a), work / "preview", count=4)
    assert len(previews) == 4 and all(p.exists() for p in previews), \
        "preview 自检帧抽取失败"
    print(f"  [PASS] 场景3 自检帧 {len(previews)} 张 → {work / 'preview'}")

    print("\nRENDER CHECK ALL PASS ✅（产物保留在上方工作目录供人工过目）")
    _ = shutil   # 保留 import：人工复核时可 shutil.rmtree(work)


if __name__ == "__main__":
    main()
