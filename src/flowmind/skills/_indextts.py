"""IndexTTS2 原生配音时长控制 —— 可行性验证 spike（**未接入主流水线**）。

背景
----
本仓库当前用 ``atempo`` 变速把 TTS 配音压进原句时长
（``localize_video._build_timed_audio``）——属后期拉伸：配音长于原句时整体
加速，长句可能变调。IndexTTS2（bilibili index-tts）宣称支持**原生时长控制**，
本模块是「能否在本机替代 atempo」的 spike 落点（是否接入由结论决定，
``localize_video.py`` 未做任何改动）。

可用性结论（2026-09-24 spike）：**不可用 —— 决策门 1 不通过**
------------------------------------------------------------
上游 ``indextts`` 2.0.0 的 ``pyproject.toml`` 硬钉：

    requires-python = ">=3.10,<3.12"
    torch==2.8.*      torchaudio==2.8.*      （uv 索引指向 cu128）

本机是 Pascal 6.1（P104-100，8GB），仓库为此钉死 ``torch==2.6.0+cu126``
（见 CLAUDE.md「本地模型」段），两者不可共存：

1. ``torch==2.8.*`` 是 ``==`` 硬约束：主环境钉的是 torch 2.6.0，无法满足
   → 门 1 不通过（按 spike 判定规则：「要求 torch ≥ 2.6 即不通过」）；
2. 上游默认索引解析落到 ``torch==2.8.0+cu128``；PyTorch 官方公告
   （dev-discuss 2025-07）：**自 2.8 起 CUDA-12.8 二进制移除
   Maxwell/Pascal 支持**。实测 torch 2.8.0+cu128 的 ``libtorch_cuda.so``
   内嵌 arch 为 sm_70/75/80/86/90/100/120，**无 sm_61**（对照缓存
   torch 2.14.0+cu130 同样无 sm_61）—— cu128 路线上 P104 无可用 kernel；
3. ``requires-python <3.12`` 亦把主环境 Python 3.12 挡在门外。

⚠ **cu126 变数（已核实）**：Pascal 移除**仅限 cu128/12.9/13.x 构建**——
cu126 构建仍保留 Maxwell/Pascal（``sm_50;sm_60``，同 major 内前向兼容
到 sm_61），且 ``download.pytorch.org/whl/cu126`` **存在
``torch==2.8.0+cu126``**。仓库已于 2026-09 完成 cu121→cu126 迁移
（主环境 ``torch 2.6.0+cu126``），但 IndexTTS2 的 ``torch==2.8.*`` 硬约束
仍不被主环境 2.6.0 满足，且 ``requires-python <3.12`` 也把主环境
Python 3.12 挡在门外——门 1/3 依旧不通过。要重跑门 2/3/4（可听音频 /
显存≤4GB / RTF<1.5）须另起**独立 venv + cu126 的 torch 2.8**。
**当前结论：保留 Qwen3-TTS**（``_local_tts.py``），atempo 时长对齐继续沿用。

本模块仅保留 ``available()`` / ``synthesize()`` 骨架，便于将来 GPU 换代
或 cu126 迁移落地后重新评估。当前 ``available()`` 恒为 False，
``synthesize()`` 显式抛 ``IndexTTSError``（``category="environment"``），
**绝不静默降级**（与 ``_local_tts`` / ``_decide`` 同一失败语义）。
"""
from __future__ import annotations

import os
from pathlib import Path

# IndexTTS2 的 torch 硬约束（上游 pyproject.toml：torch==2.8.*）
_MIN_TORCH = (2, 8)
# 无 sm_6x kernel 的构建（cu128+）所需算力下限：最低 Volta(7.0)
_MIN_COMPUTE_CAPABILITY = (7, 0)

# 不可用原因（报错 / 日志复用，避免多处重复文案）
_UNAVAILABLE_REASON = (
    "IndexTTS2 在本机不可用：上游硬钉 torch==2.8.*，而主环境钉的是 "
    "torch==2.6.0+cu126（CLAUDE.md 硬约束）；且上游 requires-python <3.12 "
    "与主环境 Python 3.12 冲突 —— 保留 Qwen3-TTS"
)

# 进程内单例（首次合成时懒加载，避免启动即吃显存）
_ENGINE: dict = {"model": None}


class IndexTTSError(Exception):
    """IndexTTS2 失败。category/retriable 语义与 errors.py 对齐。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def _major_minor(version: str) -> tuple:
    """``'2.8.0+cu126'`` → ``(2, 8)``；解析失败返回 (0, 0)。"""
    try:
        return tuple(int(x) for x in version.split("+")[0].split(".")[:2])
    except ValueError:
        return (0, 0)


def _torch_ok() -> bool:
    """torch 版本 + CUDA 构建是否同时满足 IndexTTS2 与本机 Pascal 的要求。

    需 ``torch>=2.8``；且运行中的 CUDA 构建要保留 Pascal（``sm_6x``）kernel
    —— cu126 及更早保留，cu128+ 已剔除（此时需 Volta/Ampere 及以上）。
    """
    try:
        import torch
    except ImportError:
        return False
    if _major_minor(torch.__version__) < _MIN_TORCH or not torch.cuda.is_available():
        return False
    if any(a.startswith("sm_6") for a in torch.cuda.get_arch_list()):
        return True
    return tuple(torch.cuda.get_device_capability()) >= _MIN_COMPUTE_CAPABILITY


def available() -> bool:
    """IndexTTS2 是否可用：``indextts`` 可导入且 torch/GPU 满足其硬约束。

    主环境（torch 2.6.0+cu126 + Pascal 6.1）恒为 False —— 见模块 docstring。
    """
    try:
        import indextts  # noqa: F401
    except ImportError:
        return False
    return _torch_ok()


def _get_engine():
    """进程内单例懒加载 IndexTTS2；加载失败显式报错（环境问题），绝不静默。"""
    if _ENGINE["model"] is not None:
        return _ENGINE["model"]
    model_dir = os.environ.get("FLOWMIND_INDEXTTS_DIR", "").strip()
    if not model_dir:
        raise IndexTTSError(
            "未配置 IndexTTS2 权重目录（env FLOWMIND_INDEXTTS_DIR）",
            category="environment",
        )
    try:
        from indextts.infer_v2 import IndexTTS2
    except ImportError as exc:
        raise IndexTTSError(_UNAVAILABLE_REASON, category="environment") from exc
    try:
        _ENGINE["model"] = IndexTTS2(
            cfg_path=str(Path(model_dir) / "config.yaml"),
            model_dir=model_dir,
            use_fp16=False,          # Pascal 6.1 无 bf16；fp16 亦非其 kernel 目标
            use_cuda_kernel=False,   # BigVGAN2 的 CUDA 反锯齿 kernel 需新架构
            use_deepspeed=False,
        )
    except Exception as exc:
        raise IndexTTSError(
            f"IndexTTS2 模型加载失败: {type(exc).__name__}: {exc}",
            category="environment",
        ) from exc
    return _ENGINE["model"]


def synthesize(
    text: str, *, out_path: str, ref_audio: str | None = None,
    target_lang: str = "zh",
) -> str:
    """零样本克隆合成一句到 out_path（wav）。签名对齐 ``_local_tts.synthesize``。

    IndexTTS2 仅支持声音克隆：必须提供 ref_audio。``target_lang`` 为
    IndexTTS-2.5（多语种变体）保留，IndexTTS2 v2 由文本自身判定语种。
    spike 结论：本机不可用 —— 显式抛 ``IndexTTSError(category="environment")``。
    """
    if not text or not text.strip():
        raise IndexTTSError("合成文本为空", category="video")
    if not ref_audio:
        raise IndexTTSError(
            "IndexTTS2 仅支持声音克隆：必须提供 ref_audio", category="video",
        )
    if not Path(ref_audio).exists():
        raise IndexTTSError(f"克隆参考音频不存在: {ref_audio}", category="video")
    if not available():
        raise IndexTTSError(_UNAVAILABLE_REASON, category="environment")
    engine = _get_engine()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        engine.infer(spk_audio_prompt=ref_audio, text=text, output_path=out_path)
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            raise IndexTTSError(
                "IndexTTS2 GPU 显存不足", category="transient", retriable=True,
            ) from exc
        raise IndexTTSError(
            f"IndexTTS2 合成失败: {type(exc).__name__}: {exc}", category="video",
        ) from exc
    return out_path
