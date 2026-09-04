"""GPU 资源管控：全局推理信号量 + 模型缓存双检锁。

单卡 8GB（P104-100，Pascal 6.1）显存预算：whisper ~1.5GB +
Qwen3-TTS ~4GB + LaMa ~1.5GB ≈ 7GB——**任何并发加载/推理都可能 OOM**。

两道闸：
- ``gpu_lane()``（Semaphore(1)）：GPU 推理阶段互斥。TaskManager
  workers=1 是第一道闸（任务间串行）；本信号量是第二道，覆盖
  sync invoke 直连与任务线程并发的场景。
- ``model_cache_guard()``：本地模型懒加载双检锁外圈，防止并发首调
  重复加载同一模型（_local_asr/_local_ocr/_local_tts/_inpaint 共用）。

云 API 阶段（dashscope/qwen-audio/LongCat）不占 GPU 槽——纯网络 I/O。
"""
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

# GPU 推理槽：单卡串行（绝不放宽为 >1）
gpu_semaphore = threading.Semaphore(1)

# 模型缓存初始化锁：双检锁外圈（懒加载语义不变）
model_init_lock = threading.Lock()


@contextmanager
def gpu_lane() -> Iterator[None]:
    """独占 GPU 推理槽。本地 ASR / LaMa 逐帧擦除 / 本地 TTS 的调用方包一层。"""
    with gpu_semaphore:
        yield


@contextmanager
def model_cache_guard() -> Iterator[None]:
    """模型缓存初始化双检锁外圈，用法（保持懒加载行为）::

        if key not in _models:
            with model_cache_guard():
                if key not in _models:
                    _models[key] = _load_model()   # 慢操作，锁内只发生一次
        return _models[key]
    """
    with model_init_lock:
        yield
