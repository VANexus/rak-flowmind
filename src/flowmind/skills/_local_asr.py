"""本地 ASR：faster-whisper（CTranslate2），GPU 本地推理。

架构升级（GPU 化）：ASR 从「云优先」改为「本地优先」——P104-100（CC 6.1/Pascal）
上 CTranslate2 官方支持（下限 CC 6.0），int8 量化走 dp4a。
云路径（_cloud_asr，dashscope）保留为回落。

接口与 _cloud_asr.transcribe_local 完全对齐：
    输出 [{index, begin, end, text}]（秒），供翻译与 TTS 逐句对齐。

lazy import：faster_whisper 未安装 → ASRError(category="environment")。
模型实例进程内缓存（按 model/device/compute_type），避免重复加载。
模型缓存/下载已配在 ~/.bashrc：HF_HOME=/srv/data/models + 代理 127.0.0.1:7890，import 即生效。
"""
from __future__ import annotations

from pathlib import Path

from flowmind.skills._cloud_asr import ASRError  # 复用同一异常类型，流水线统一捕获

DEFAULT_MODEL = "small"       # 8GB Pascal：small int8 <1.5GB 显存；可调 medium
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "int8"  # CC 6.1 支持 dp4a，int8 是 Pascal 上推荐档


def available() -> bool:
    """faster_whisper 是否可导入（供 backend=auto 判断；测试可替换）。"""
    try:
        import faster_whisper  # noqa: F401  # 导入探测本身即目的，勿删
        return True
    except Exception:
        return False


# 进程内模型缓存：key=(model, device, compute_type)
_models: dict[tuple, object] = {}


def _ensure_cuda_runtime() -> None:
    """GPU 推理前显式加载 CUDA 运行时库（仅 device=cuda 时需要）。

    torch 的 pip 轮子把 cuBLAS/cudart 装在 site-packages/nvidia/ 下，torch 自己能找到，
    但 CTranslate2 按默认 loader 路径搜索 soname 会失败（libcublas.so.12 not found）。
    这里用 ctypes dlopen 一次（注册进全局命名空间），CTranslate2 即可解析。
    任一库已可加载则直接返回；全部失败也不抛错——交给推理时的原始报错定性。
    """
    import ctypes
    import site

    names = ("libcublasLt.so.12", "libcublas.so.12", "libcudart.so.12")
    try:
        ctypes.CDLL(names[1])
        return
    except OSError:
        pass
    search_dirs: list[Path] = []
    try:
        search_dirs += [Path(d) for d in site.getsitepackages()]
        search_dirs.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    for sp in search_dirs:
        nvidia = sp / "nvidia"
        if not nvidia.is_dir():
            continue
        found = 0
        for name in names:
            for lib in nvidia.rglob(name):
                try:
                    ctypes.CDLL(str(lib))
                    found += 1
                    break
                except OSError:
                    continue
        if found >= 2:  # cublasLt + cublas 齐了即够 cudart 通常随 RPATH 自带
            return


def _get_model(model: str, device: str, compute_type: str):
    key = (model, device, compute_type)
    if key not in _models:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ASRError(
                "未安装 faster-whisper（environment.yml 已含，conda env update 后可用）",
                category="environment",
            ) from exc
        try:
            _models[key] = WhisperModel(model, device=device, compute_type=compute_type)
        except Exception as exc:
            raise ASRError(
                f"本地 ASR 模型加载失败（{model}@{device}）: {type(exc).__name__}",
                category="environment",
            ) from exc
    return _models[key]


def transcribe_local(
    wav_path: str, *,
    model: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    language: str | None = None,
) -> list[dict]:
    """本地 wav → 句段列表 [{index, begin, end, text}]（秒）。

    与 _cloud_asr.transcribe_local 输出契约一致：
    - begin/end 为秒（float），index 为句序号
    - 过滤空文本句
    - language 为源语言（zh/en/th/...）；None = 模型自动检测
    """
    if not Path(wav_path).exists():
        raise ASRError(f"音频文件不存在: {wav_path}", category="video")

    if device.startswith("cuda"):
        _ensure_cuda_runtime()

    whisper_model = _get_model(model, device, compute_type)
    try:
        segments, _info = whisper_model.transcribe(
            wav_path,
            language=language,
            vad_filter=True,        # 过滤静音段，减少幻觉句
            beam_size=5,
        )
        out: list[dict] = []
        for seg in segments:
            text = str(seg.text).strip()
            if not text:
                continue
            out.append({
                "index": len(out),
                "begin": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": text,
            })
        return out
    except ASRError:
        raise
    except Exception as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda" in msg and "error" in msg:
            raise ASRError(
                f"本地 ASR GPU 推理失败: {type(exc).__name__}", category="environment",
            ) from exc
        raise ASRError(
            f"本地 ASR 推理异常: {type(exc).__name__}", category="video",
        ) from exc
