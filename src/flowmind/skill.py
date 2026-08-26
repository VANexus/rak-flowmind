"""技能框架：@skill 装饰器 + 注册表 + invoke。

这是「skills 融合 mcp」的融合点：一次 @skill 定义即登记进注册表，
server 端遍历注册表把每个技能暴露为 MCP 工具。
invoke() 统一为技能套上 SkillResult 信封（trace/计时/错误兜底）。
"""
from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ValidationError

from flowmind.contracts import (
    ReliabilityMetrics,
    SkillError,
    SkillOutput,
    SkillResult,
    TraceContext,
    new_trace,
)


@dataclass
class SkillSpec:
    """一个已注册技能的元数据。"""
    id: str
    name: str
    version: str
    description: str = ""           # 从函数 docstring 第一段提取，供 manifest/discover 用
    func: Callable[[Any], SkillOutput] = None  # type: ignore[assignment]
    input_model: type[BaseModel] | None = None
    output_model: type[BaseModel] | None = None  # 从返回注解 SkillOutput[T] 提取的 T
    category: str = "通用"          # UI 分组（后端声明，前端只渲染，不推断）
    tags: list[str] = field(default_factory=list)  # 意图标签（后端声明）


_REGISTRY: dict[str, SkillSpec] = {}


def _extract_description(func: Callable) -> str:
    """从函数 docstring 提取第一段非空文字作为 description。

    - 没 docstring → 空串
    - 多段 → 用第一段（直到第一个空行）
    - 中文 / 英文都行，按原文返回
    """
    doc = inspect.getdoc(func)
    if not doc:
        return ""
    # 取第一段（空行分隔）
    first_para = doc.split("\n\n", 1)[0].strip()
    # 单段里的首行
    first_line = first_para.split("\n", 1)[0].strip()
    return first_line


def _extract_output_model(func: Callable, hints: dict[str, Any]) -> type[BaseModel] | None:
    """从返回类型注解 `SkillOutput[T]` 提取 T。

    返回 None 当：注解缺失 / 不是 SkillOutput[...] / T 不是 BaseModel 子类。

    注意：Pydantic Generic 的参数化形式 `SkillOutput[X]` 在运行时是一个独立的类，
    元数据存在 `__pydantic_generic_metadata__['args']` 里，`typing.get_origin/get_args`
    看不到。所以两种来源都要查。
    """
    ret = hints.get("return")
    if ret is None:
        return None

    candidates: list[Any] = []

    # 1. 走 typing 标准接口（处理 typing.Generic[SkillOutput, T] 的写法）
    origin = typing.get_origin(ret)
    args = typing.get_args(ret)
    if origin is SkillOutput and args:
        candidates.append(args[0])

    # 2. 走 pydantic 的内部 metadata（处理 Pydantic Generic 的参数化形式）
    pyd_meta = getattr(ret, "__pydantic_generic_metadata__", None)
    if isinstance(pyd_meta, dict):
        for a in pyd_meta.get("args", ()):
            candidates.append(a)

    for c in candidates:
        if isinstance(c, type) and issubclass(c, BaseModel):
            return c
    return None


def skill(
    *,
    id: str,
    name: str,
    version: str,
    category: str = "通用",
    tags: list[str] | None = None,
) -> Callable:
    """把一个业务函数登记为技能。函数签名首参注解即输入模型。

    注解通过 ``typing.get_type_hints`` 解析，因此模块是否启用
    ``from __future__ import annotations``（PEP 563）都不影响：字符串注解
    会按函数所在模块的全局命名空间求值回真实类型。

    返回类型若为 ``SkillOutput[T]``，会自动把 T 记为 output_model，供
    manifest / discover() 暴露给 Agent 做输出字段发现。

    ``category`` 与 ``tags`` 由后端显式声明（source of truth），
    前端只负责渲染，不再硬编码推断。这样接入任意新后端时，
    其技能分组与语义标签由该后端自主决定，前端零改动。
    """
    def deco(func: Callable[[Any], SkillOutput]) -> Callable[[Any], SkillOutput]:
        params = list(inspect.signature(func).parameters.values())
        if not params:
            raise TypeError(f"技能 {id} 必须有一个输入模型参数")
        first_name = params[0].name
        try:
            hints = typing.get_type_hints(func)
        except Exception as exc:
            raise TypeError(f"技能 {id} 的首参注解解析失败：{exc}") from exc
        input_model = hints.get(first_name)
        if not (isinstance(input_model, type) and issubclass(input_model, BaseModel)):
            raise TypeError(f"技能 {id} 的首参注解必须是 pydantic BaseModel 子类")
        if id in _REGISTRY:
            raise ValueError(f"技能 id 已注册，禁止重复：{id}")
        output_model = _extract_output_model(func, hints)
        _REGISTRY[id] = SkillSpec(
            id=id, name=name, version=version,
            description=_extract_description(func),
            func=func, input_model=input_model, output_model=output_model,
            category=category, tags=tags or [],
        )
        return func
    return deco


def registry() -> dict[str, SkillSpec]:
    """返回注册表快照。"""
    return dict(_REGISTRY)


def _fail(skill_id: str, trace: TraceContext, error: SkillError) -> SkillResult:
    """构造失败信封（错误永不静默）。"""
    return SkillResult(
        ok=False,
        skill=skill_id,
        version=_REGISTRY[skill_id].version if skill_id in _REGISTRY else "unknown",
        trace=trace,
        metrics=ReliabilityMetrics(latency_ms=0.0, confidence=0.0, sample_size=0),
        error=error,
    )


def invoke(skill_id: str, raw_args: dict, trace: TraceContext | None = None) -> SkillResult:
    """调用技能并组装对外 SkillResult 信封。任何失败均返回结构化结果。"""
    tr = trace or new_trace()
    spec = _REGISTRY.get(skill_id)
    if spec is None:
        return _fail(skill_id, tr, SkillError(code="NOT_FOUND", message=f"未知技能：{skill_id}"))

    try:
        inp = spec.input_model.model_validate(raw_args)
    except ValidationError as exc:
        return _fail(skill_id, tr, SkillError(code="VALIDATION", message="入参校验失败", details={"errors": exc.errors(include_url=False)}))

    start = perf_counter()
    try:
        out: SkillOutput = spec.func(inp)
    except Exception as exc:  # 兜底：技能内部异常不外泄为崩溃
        return _fail(skill_id, tr, SkillError(code="INTERNAL", message=str(exc)))

    latency_ms = (perf_counter() - start) * 1000.0
    metrics = ReliabilityMetrics(
        latency_ms=latency_ms,
        confidence=out.confidence,
        sample_size=out.sample_size,
        degraded=out.degraded,
        degradation_reason=out.degradation_reason,
    )
    return SkillResult(
        ok=True,
        skill=spec.id,
        version=spec.version,
        trace=tr,
        data=out.data,
        reasoning=out.reasoning,
        metrics=metrics,
    )
