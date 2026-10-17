"""dsp 裁切的**真 ffmpeg 复核**（ADR-080 阶段 5）：把 ``_media.clip_to_window`` 的
三条分支 + 失败路径在**真 ffmpeg** 上跑一遍，并用 ffprobe **复核输出时长**。

与 demo 的分工：``dsp_image_to_video_demo`` 要**离线可跑**（CI 无 ffmpeg），故那一版
**注入假 ffmpeg/ffprobe** 逐字断言命令行；本脚本**恰好相反**——它**要求真 ffmpeg**，
用合成片段验「命令真跑下去会得到什么」（`-t` 真裁、`tpad` 真补、`-an` 真丢音轨）。

运行（ffmpeg 从 ``FLOWMIND_MEDIA_BIN_DIR`` 或 ``PATH`` 解析——与生产同一套 `_media` 解析）：

    PYTHONPATH=$PWD/src python examples/dsp_trim_realcheck.py
    PYTHONPATH=$PWD/src python examples/dsp_trim_realcheck.py /path/to/ffmpeg/bin

**找不到 ffmpeg 即自行跳过（exit 0）**——故不进「一键全跑 demo」回归底线（那条线要求
任何机器都绿），是**按需**复核。产物写系统临时目录，跑完自清。

覆盖（源时长由 ffprobe 实测，断言走**关系**而非硬编码数字）：
  A 源 > 窗口 ⇒ 裁掉尾部（`outMs == 窗口` / `shortfall == 0` / 产物真 ≈ 窗口）
  B 源 < 窗口 + freeze_pad ⇒ 末帧定格补齐（产物真 ≈ 窗口 且**比源长**，证 `tpad` 生效）
  C 源 < 窗口 + 关闭补齐 ⇒ 原样返回但**照实报短**（不落派生件）
  D 源 == 窗口 ⇒ 原样返回、零 ffmpeg 调用（不落派生件）
  E 源不是视频 ⇒ 抛 `MediaError`（**不假装裁过**）
  F 质检抽帧（阶段 4 遗留的 ffmpeg 依赖）⇒ 10%/50%/90% 三帧真抽出 PNG / 时长未知只抽首帧 /
    坏文件抛 `MediaError`
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from flowmind.skills import _dsp_qc as qc
from flowmind.skills import _media as media

# 允许显式给 ffmpeg bin 目录：写进 FLOWMIND_MEDIA_BIN_DIR，让脚本走**与生产同一套**解析
# （`_media` 优先从该目录取绝对路径，不要求 ffmpeg 进系统 PATH）。
if len(sys.argv) > 1:
    os.environ["FLOWMIND_MEDIA_BIN_DIR"] = sys.argv[1]


def _ffmpeg_ready() -> bool:
    """ffmpeg 是否可用（走 `_media` 的解析 ⇒ `.env` 里配的 bin 目录同样生效）。"""
    try:
        rc, _, _ = media.run_ffmpeg(["ffmpeg", "-version"])
    except FileNotFoundError:
        return False
    return rc == 0


if not _ffmpeg_ready():
    print("跳过：找不到 ffmpeg/ffprobe（按需复核；可在 .env 配 FLOWMIND_MEDIA_BIN_DIR，"
          "或把 bin 目录当第一个参数传入）")
    sys.exit(0)

_OK = 0
_FAIL = 0


def check(label: str, cond: bool, detail: str) -> None:
    global _OK, _FAIL
    if cond:
        _OK += 1
        print(f"  PASS  {label}  {detail}")
    else:
        _FAIL += 1
        print(f"  FAIL  {label}  {detail}")


def gen_clip(path: Path, seconds: float) -> Path:
    """合成一段纯色片段（真编码，故 ffprobe 能量出真时长）。"""
    rc, _, err = media.run_ffmpeg([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi",
        "-i", f"color=c=red:s=320x240:d={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ])
    if rc != 0:
        raise RuntimeError(f"合成片段失败: {err[-200:]}")
    return path


def dur_ms(path: Path) -> int:
    return int(round(media.probe_media(str(path))[0] * 1000))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="dsp-trim-") as tmp:
        work = Path(tmp)
        print("== 资产（合成片段，ffprobe 实测时长） ==")
        long_clip = gen_clip(work / "long.mp4", 5.2)
        short_clip = gen_clip(work / "short.mp4", 1.5)
        exact_clip = gen_clip(work / "exact.mp4", 3.0)
        long_ms, short_ms, exact_ms = dur_ms(long_clip), dur_ms(short_clip), dur_ms(exact_clip)
        print(f"  long={long_ms}ms  short={short_ms}ms  exact={exact_ms}ms")

        print("\n== A 源 > 窗口 ⇒ 裁掉尾部 ==")
        out_a = work / "a_out.mp4"
        fit = media.clip_to_window(str(long_clip), str(out_a), 3000)
        check("A.outMs==3000", fit.outMs == 3000, f"outMs={fit.outMs}")
        check("A.shortfall==0", fit.shortfallMs == 0, f"shortfall={fit.shortfallMs}")
        check("A.trimmed", fit.trimmed is True, f"trimmed={fit.trimmed}")
        check("A.path==out", fit.path == str(out_a), f"path={Path(fit.path).name}")
        real_a = dur_ms(out_a)
        check("A.产物真≈3.0s", abs(real_a - 3000) <= 80, f"ffprobe={real_a}ms")
        check("A.sourceMs", fit.sourceMs == long_ms, f"sourceMs={fit.sourceMs}")

        print("\n== B 源 < 窗口 + freeze_pad ⇒ 末帧定格补齐 ==")
        out_b = work / "b_out.mp4"
        fit = media.clip_to_window(str(short_clip), str(out_b), 4000)
        check("B.outMs==4000", fit.outMs == 4000, f"outMs={fit.outMs}")
        check("B.shortfall==窗口-源", fit.shortfallMs == 4000 - short_ms,
              f"shortfall={fit.shortfallMs}（期望 {4000 - short_ms}）")
        check("B.trimmed", fit.trimmed is True, f"trimmed={fit.trimmed}")
        real_b = dur_ms(out_b)
        check("B.产物真≈4.0s", abs(real_b - 4000) <= 80, f"ffprobe={real_b}ms")
        check("B.比源长（tpad 真生效）", real_b > short_ms + 1000, f"{real_b}ms > {short_ms}ms")

        print("\n== C 源 < 窗口 + 不补 ⇒ 原样返回但照实报短 ==")
        out_c = work / "c_out.mp4"
        fit = media.clip_to_window(str(short_clip), str(out_c), 4000, freeze_pad=False)
        check("C.path==源", fit.path == str(short_clip), f"path={Path(fit.path).name}")
        check("C.outMs==源", fit.outMs == short_ms, f"outMs={fit.outMs}")
        check("C.shortfall 照实报", fit.shortfallMs == 4000 - short_ms,
              f"shortfall={fit.shortfallMs}")
        check("C.trimmed=False", fit.trimmed is False, f"trimmed={fit.trimmed}")
        check("C.未落派生件", not out_c.exists(), f"out_exists={out_c.exists()}")

        print("\n== D 源 == 窗口 ⇒ 原样返回、零 ffmpeg 调用 ==")
        out_d = work / "d_out.mp4"
        fit = media.clip_to_window(str(exact_clip), str(out_d), exact_ms)
        check("D.path==源", fit.path == str(exact_clip), f"path={Path(fit.path).name}")
        check("D.outMs==源", fit.outMs == exact_ms, f"outMs={fit.outMs}")
        check("D.shortfall==0", fit.shortfallMs == 0, f"shortfall={fit.shortfallMs}")
        check("D.trimmed=False", fit.trimmed is False, f"trimmed={fit.trimmed}")
        check("D.未落派生件", not out_d.exists(), f"out_exists={out_d.exists()}")

        print("\n== E 源不是视频 ⇒ MediaError（不假装裁过） ==")
        bad = work / "bad.mp4"
        bad.write_text("not a video")
        try:
            media.clip_to_window(str(bad), str(work / "e_out.mp4"), 3000)
            check("E.抛 MediaError", False, "居然没抛")
        except media.MediaError as exc:
            check("E.抛 MediaError", True, f"msg={str(exc)[:60]}")

        print("\n== F 质检抽帧（阶段 4 遗留的 ffmpeg 依赖） ==")
        frames_dir = work / "frames"
        frames_dir.mkdir()
        sample = qc.sample_frames(str(long_clip), str(frames_dir), long_ms / 1000.0)
        check("F.抽到 3 帧", len(sample.paths) == 3, f"n={len(sample.paths)}")
        check("F.ratio==10/50/90%", sample.ratios == [0.1, 0.5, 0.9], f"ratios={sample.ratios}")
        sizes = [os.path.getsize(p) for p in sample.paths]
        check("F.三帧都是真 PNG（非空）", all(s > 0 for s in sizes), f"sizes={sizes}")
        check("F.路径落到 out_dir", all(p.startswith(str(frames_dir)) for p in sample.paths),
              f"dir={Path(sample.paths[0]).parent.name}")
        only_first = qc.sample_frames(str(long_clip), str(frames_dir), 0.0)
        check("F.时长未知 ⇒ 只抽首帧", len(only_first.paths) == 1, f"n={len(only_first.paths)}")
        try:
            qc.sample_frames(str(bad), str(frames_dir), 3.0)
            check("F.坏文件抛 MediaError", False, "居然没抛")
        except media.MediaError as exc:
            check("F.坏文件抛 MediaError", True, f"msg={str(exc)[:60]}")

    print(f"\n结果：{_OK} passed / {_FAIL} failed")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
