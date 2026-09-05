"""本地人声分离：Demucs htdemucs（MIT），背景音源净化。

keep_background_audio 的背景源原本是整条原声（人声+BGM 混合），混入后
"背景音乐还有人声"。本模块用 demucs --two-stems=vocals 分离出伴奏轨
（no_vocals），流水线以它为背景音源，BGM 不再带原声人声。

- 子进程隔离执行（demucs 模型加载重，不污染流水线进程）
- 伴奏缓存 workdir/bg_instrumental.wav，重复调用直接命中
- 模型 htdemucs ~160MB 首次经代理下载（TORCH_HOME 指向 /srv/data/models/torch，
  与 HF 缓存约定一致）；CPU 可跑（12s 片 1-2 分钟），GPU 更快
- 失败显式抛 MusicSepError（category/retriable 与 errors 语义对齐），
  绝不静默回落到含人声的原声
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


class MusicSepError(Exception):
    """人声分离失败。category/retriable 语义与 errors.py 对齐。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def available() -> bool:
    """demucs 包可导入即视为可用（模型首次用时才下载）。"""
    return importlib.util.find_spec("demucs") is not None


def separate_vocals(src_wav: str, workdir: str) -> str:
    """分离出伴奏轨（no_vocals），缓存于 workdir/bg_instrumental.wav。"""
    out = Path(workdir) / "bg_instrumental.wav"
    if out.exists():
        return str(out)
    if not Path(src_wav).exists():
        raise MusicSepError(f"待分离音轨不存在: {src_wav}", category="video")

    tmp = Path(workdir) / "demucs_out"
    env = dict(os.environ)
    # 模型缓存约定：与 HF_HOME 同级根目录（/srv/data/models）
    env.setdefault("TORCH_HOME", "/srv/data/models/torch")
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", "htdemucs", "--two-stems=vocals",
        "-o", str(tmp), src_wav,
    ]
    try:
        rc = subprocess.run(
            cmd, capture_output=True, text=True,
            # 显式 UTF-8 解码：父进程 locale 被原生依赖翻成 C 时，demucs/tqdm
            # 输出含非 ASCII 进度字符（'█'），跟随 locale 解码会 UnicodeDecodeError
            encoding="utf-8", errors="replace",
            timeout=1800, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise MusicSepError(
            "人声分离超时（30min）", category="transient", retriable=True,
        ) from exc
    except OSError as exc:
        raise MusicSepError(
            f"demucs 子进程启动失败: {type(exc).__name__}", category="environment",
        ) from exc

    produced = tmp / "htdemucs" / Path(src_wav).stem / "no_vocals.wav"
    if rc.returncode != 0 or not produced.exists():
        err = (rc.stderr or rc.stdout or "")[-300:].strip()
        low = err.lower()
        if "download" in low or "connection" in low or "timed out" in low or "unreachable" in low:
            raise MusicSepError(
                f"htdemucs 模型下载失败（需代理）: {err}", category="environment",
            )
        retriable = "memory" in low or "out of memory" in low
        raise MusicSepError(
            f"人声分离失败: {err or '(无输出)'}",
            category="transient" if retriable else "unknown", retriable=retriable,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), str(out))
    shutil.rmtree(tmp, ignore_errors=True)
    return str(out)
