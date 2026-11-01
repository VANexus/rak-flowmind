"""System One 决策模型客户端（TypeSafe Jev 云 / Laya 本地双后端）。

统一契约（两侧同形状，仅传输地址与鉴权不同）：

    POST <base>/decide        （本地 Laya sidecar）
    POST https://api.typesafe.ai/v1/systemone   （云 Jev）
      body: {"state": <str|object|array>, "model": "<name>",
             "questions": {<question_id>: <Question>}}
      → {"model": "...", "answers": {<question_id>: <Answer>},
         "usage": {"input_tokens": n, "output_tokens": n}}

Question 三原语（与 TypeSafe 契约逐字一致，Laya 侧同形状）：
- choice: ``{"type": "choice", "instructions": str,
             "criteria": {<option>: <描述>}}`` —— 至多 255 个选项
- score : ``{"type": "score", "instructions": str,
             "criteria": [<级别描述>, ...]}`` —— 2-10 个有序级别
- noul  : ``{"type": "noul", "instructions": str,
             "criteria": {"true": <描述>, "false": <描述>}}`` —— criteria 可省

Answer（按 type 区分）：
- choice: ``{"type": "choice", "choice": <option>,
             "probabilities": {...}, "confidence": 0-1}``
- score : ``{"type": "score", "score": <float>,
             "probabilities": [...], "confidence": 0-1}``
- noul  : ``{"type": "noul", "noul": 0-1}`` —— 直接是「yes」的概率

后端选择（配置源顺序：env → config.toml → 内置默认）：
- ``FLOWMIND_DECIDE_BACKEND`` / config ``decide.backend`` = ``local|cloud|auto``
- ``auto`` 语义：本地服务地址已配置 → local；否则 TypeSafe key 已配置 → cloud；
  两端均不可用 → 显式抛 DecideError（**不静默降级**，与 _bge_embed 同语义）
- 云端点默认 TypeSafe 官方 ``api.typesafe.ai``；接中转/聚合网关时设
  ``TYPESAFE_BASE_URL``（或 config ``decide.cloud_base_url``）覆盖 host

⚠ **本地后端走独立进程的 HTTP sidecar**（仓库根 ``laya_sidecar.py``），
主环境只走 HTTP、不 import laya——保持模型加载与主流水线进程隔离。实测
``laya`` 0.3.20 依赖仅 ``torch>=2.0`` / ``transformers>=4.48``，与本仓库
torch 2.6.0+cu126 / transformers 4.57.3 相容，故 sidecar 直接跑在主 conda
env（``pip install laya==0.3.20``），隔离目的只是不让模型常驻主进程显存。

失败语义：一律抛 ``DecideError``（category 与 errors.py 语义对齐），绝不吞异常、
绝不返回伪造的低置信答案——调用方自行决定降级或上抛。
"""
from __future__ import annotations

import os
from typing import Any

import requests

from flowmind.config import ensure_env_loaded
from flowmind.skills._secrets import get_api_key

# TypeSafe 云端点与 key 环境变量名（契约真源：https://docs.typesafe.ai/api.md）
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_KEY_ENV = "TYPESAFE_API_KEY"

# Question 原语白名单（与 TypeSafe / Laya 契约一致）
_VALID_TYPES = ("choice", "score", "noul")

# choice 选项数上限（TypeSafe 契约）
_MAX_CHOICE_OPTIONS = 255
# score 级别数上下限（TypeSafe 契约）
_MIN_SCORE_LEVELS = 2
_MAX_SCORE_LEVELS = 10


class DecideError(Exception):
    """决策模型调用失败。category 语义与 errors.py 对齐。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


# ── 配置解析（env → config.toml → 内置默认） ──

def _cfg():
    """延迟导入 config，避免模块级循环依赖。"""
    from flowmind.config import get_config

    return get_config().decide


def _env(name: str) -> str:
    """读取环境变量（先确保 .env 已加载，再取进程环境）。

    必须先 ensure_env_loaded()：.env 未加载时裸 os.environ 会漏掉 .env 里的
    配置项，``auto`` 就会误判「本地未配置」而错路由到云。
    """
    ensure_env_loaded()
    return os.environ.get(name, "").strip()


def _backend_setting() -> str:
    return _env("FLOWMIND_DECIDE_BACKEND").lower() or _cfg().backend.strip().lower()


def _local_base_url() -> str:
    return (_env("FLOWMIND_DECIDE_BASE_URL")
            or _cfg().local_base_url.strip()).rstrip("/")


def _cloud_key() -> str | None:
    return get_api_key(TYPESAFE_KEY_ENV)


def _cloud_url() -> str:
    """云端点解析：env TYPESAFE_BASE_URL → config decide.cloud_base_url → 官方地址。

    中转/聚合网关用同一 OpenAI 风格 key 但不同 host 时，填这个字段即可切换；
    留空恒走 TypeSafe 官方端点。
    """
    base = (_env("TYPESAFE_BASE_URL")
            or _cfg().cloud_base_url.strip()).rstrip("/")
    if not base:
        return TYPESAFE_URL
    # 允许填基址（自动补 /v1/systemone）或完整端点
    return base if base.endswith("/systemone") else f"{base}/v1/systemone"


def _timeout() -> float:
    return float(_cfg().timeout_s)


def resolve_backend() -> str:
    """解析实际使用的后端，返回 ``"local"`` 或 ``"cloud"``。

    两端均不可用 / 开关取值非法时抛 DecideError（显式失败，不静默降级）。
    """
    setting = _backend_setting()
    if setting == "local":
        if not _local_base_url():
            raise DecideError(
                "decide.backend=local 但未配置本地服务地址"
                "（设置 FLOWMIND_DECIDE_BASE_URL 或 config decide.local_base_url）",
                category="environment")
        return "local"
    if setting == "cloud":
        if not _cloud_key():
            raise DecideError(
                f"decide.backend=cloud 但未配置 {TYPESAFE_KEY_ENV}（TypeSafe API key）",
                category="environment")
        return "cloud"
    if setting != "auto":
        raise DecideError(
            f"未知 decide.backend 取值：{setting!r}（应为 local / cloud / auto）",
            category="environment")
    # auto：本地优先，云回落，两端皆无显式报错
    if _local_base_url():
        return "local"
    if _cloud_key():
        return "cloud"
    raise DecideError(
        "决策模型未配置：本地服务地址与 TypeSafe key 均缺失"
        "（设置 FLOWMIND_DECIDE_BASE_URL 或 TYPESAFE_API_KEY 后可用）",
        category="environment")


def is_configured() -> bool:
    """任一端可用即 True（不抛异常，供调用方做能力探测）。"""
    try:
        resolve_backend()
    except DecideError:
        return False
    return True


# ── 入参校验（在本地拦掉契约违规，避免白跑一次远程调用） ──

def _validate_questions(questions: dict[str, Any]) -> None:
    if not isinstance(questions, dict) or not questions:
        raise DecideError(
            "questions 必须是非空映射（question_id → Question）", category="environment")
    for qid, q in questions.items():
        if not isinstance(q, dict):
            raise DecideError(f"问题 {qid!r} 必须是对象", category="environment")
        qtype = q.get("type")
        if qtype not in _VALID_TYPES:
            raise DecideError(
                f"问题 {qid!r} 的 type 非法：{qtype!r}"
                f"（应为 {' / '.join(_VALID_TYPES)}）",
                category="environment")
        if not q.get("instructions"):
            raise DecideError(f"问题 {qid!r} 缺少 instructions", category="environment")
        criteria = q.get("criteria")
        if qtype == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise DecideError(
                    f"choice 问题 {qid!r} 的 criteria 必须是非空映射（选项 → 描述）",
                    category="environment")
            if len(criteria) > _MAX_CHOICE_OPTIONS:
                raise DecideError(
                    f"choice 问题 {qid!r} 选项数 {len(criteria)} 超上限 {_MAX_CHOICE_OPTIONS}",
                    category="environment")
        elif qtype == "score":
            if not isinstance(criteria, list) or not (
                    _MIN_SCORE_LEVELS <= len(criteria) <= _MAX_SCORE_LEVELS):
                raise DecideError(
                    f"score 问题 {qid!r} 的 criteria 必须是 "
                    f"{_MIN_SCORE_LEVELS}-{_MAX_SCORE_LEVELS} 个有序级别描述",
                    category="environment")
        elif qtype == "noul" and criteria is not None and not isinstance(criteria, dict):
            raise DecideError(
                f"noul 问题 {qid!r} 的 criteria 必须是 {{true, false}} 映射或省略",
                category="environment")


# ── 传输 ──

def _post(url: str, payload: dict, headers: dict[str, str]) -> dict:
    """POST 并解析统一响应形状；任何失败抛 DecideError（消息脱敏）。"""
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=_timeout())
    except requests.exceptions.Timeout as exc:
        raise DecideError("决策模型服务超时（environment）",
                          category="environment", retriable=True) from exc
    except requests.exceptions.ConnectionError as exc:
        raise DecideError("决策模型服务不可达（environment）",
                          category="environment") from exc
    if resp.status_code == 401:
        raise DecideError(
            f"决策模型鉴权失败（401），请检查 {TYPESAFE_KEY_ENV}", category="environment")
    if resp.status_code == 429:
        raise DecideError("决策模型限流（429）", category="transient", retriable=True)
    if resp.status_code != 200:
        raise DecideError(
            f"决策模型 HTTP {resp.status_code}",
            category="transient", retriable=resp.status_code >= 500)
    try:
        data = resp.json()
    except ValueError as exc:
        raise DecideError("决策模型响应非 JSON", category="unknown") from exc
    if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
        raise DecideError("决策模型响应形状不符（缺 answers）", category="unknown")
    return data


# ── 入口 ──

def decide(
    state: str | dict | list,
    questions: dict[str, dict],
    *,
    model: str | None = None,
) -> dict:
    """对 ``state`` 求值一组类型化问题，返回统一结果字典。

    参数
    ----
    state : 待判定内容。纯文本用 str；结构化上下文（工单 / 记录 / 应用状态）
        用 dict 或 list，问题内可用 ``反引号路径``（如 ``ticket.messages[0].text``）引用。
    questions : ``question_id → Question`` 映射（三原语见模块 docstring）。
        一次请求内的问题**并行求值、互不可见**，适合放独立维度。
    model : 覆盖模型名。None = 本地用 ``decide.local_model``、
        云用 ``decide.cloud_model``。

    返回
    ----
    ``{"backend": "local"|"cloud", "model": <实际模型名>,
       "answers": {question_id: Answer}, "usage": {...}}``

    异常
    ----
    ``DecideError``：未配置 / 入参违约 / 网络失败 / 响应形状不符。
    """
    _validate_questions(questions)
    backend = resolve_backend()
    if backend == "local":
        model_name = model or _cfg().local_model
        url = f"{_local_base_url()}/decide"
        headers: dict[str, str] = {}
    else:
        model_name = model or _cfg().cloud_model
        url = _cloud_url()
        headers = {"Authorization": f"Bearer {_cloud_key()}"}

    payload = {"state": state, "model": model_name, "questions": questions}
    data = _post(url, payload, headers)
    return {
        "backend": backend,
        "model": str(data.get("model") or model_name),
        "answers": data["answers"],
        "usage": data.get("usage") or {},
    }


def answer_confidence(answer: dict) -> float:
    """从 Answer 提取可用于门控的置信度（choice/score；noul 无 → 1.0）。

    取值优先级（**先取 ``answer_confidence``，再回落 ``confidence``**）：

    1. ``answer_confidence`` —— 模型给出的最大类概率（Laya 侧提供）。它是
       概率本身，不经温度缩放，**不受温度校准影响**，是可靠的门控依据；
    2. ``confidence`` —— TypeSafe 云侧的校准置信度字段。⚠ Laya 侧该字段
       **未校准**：随附 checkpoint 的温度表含越界项，laya 会把越界项钳到
       0.5 并告警「Treat confidence from the affected entries as
       uncalibrated」（``laya/router.py``）——实测 choice 的 confidence 可低到
       0.03 而实际分布明显偏向某一选项。故仅在无 ``answer_confidence`` 时回落。

    用于阈值门控：低置信度应转交更贵的通用 LLM 或人工复核，而不是照单全收。
    注意 noul 恒返回 1.0（该原语没有独立置信度字段，``noul`` 值本身就是 yes
    概率）——它的 0.5 表示「yes/no 概率接近」，不是「中等强度」，判断门控阈值
    时须按原语分别处理，勿统一套用。
    """
    if not isinstance(answer, dict):
        return 0.0
    if answer.get("type") == "noul":
        return 1.0
    for field in ("answer_confidence", "confidence"):
        try:
            return float(answer[field])
        except (KeyError, TypeError, ValueError):
            continue
    return 0.0
