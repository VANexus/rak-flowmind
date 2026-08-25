"""Backend Registry 基类 — 各层 backend 注册的统一机制。

设计:
- 每个 layer (L2/L3/L5/L6) 各有一个 Registry, 继承 RegistryBase
- register() 装饰器注册, get() 取出
- 默认实现由各 layer 在 __init__ 时注册; 用户可自定义 backend 注入
"""
from __future__ import annotations

from typing import Dict, Generic, Type, TypeVar

T = TypeVar("T")


class RegistryBase(Generic[T]):
    """Registry 基类。子类化时指定 backend 类型 + 注册表名。

    注意: 每个子类通过 __init_subclass__ 拿到自己的 _registry,
    避免 Generic[T] 共享类属性的问题。
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    @classmethod
    def register(cls, name: str, backend: T) -> None:
        if name in cls._registry:
            raise ValueError(f"{cls.__name__}: '{name}' already registered")
        cls._registry[name] = backend

    @classmethod
    def get(cls, name: str) -> T:
        if name not in cls._registry:
            raise KeyError(
                f"{cls.__name__}: '{name}' not registered. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """仅测试用。"""
        cls._registry.clear()