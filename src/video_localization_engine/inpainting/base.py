"""L6 InpaintingBackend 抽象基类。

业务层 (InpaintingEngine) 只持有 InpaintingBackend 引用;具体 inpaint 算法由 backend 实现。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from video_localization_engine.types.video import FramePacket


class InpaintingBackend(ABC):
    """Inpaint backend 抽象。

    业务层禁止直接调用 cv2.inpaint / np inpaint;只允许通过 backend.inpaint。
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool:
        """backend 是否可用 (模型文件存在 / GPU OK 等)。"""
        ...

    @abstractmethod
    def inpaint(self, packet: FramePacket, mask: np.ndarray) -> np.ndarray:
        """输入一帧 + 二值 mask (0/255),返回 inpaint 后图像 (np.ndarray HxWx3 uint8)。"""
        ...
