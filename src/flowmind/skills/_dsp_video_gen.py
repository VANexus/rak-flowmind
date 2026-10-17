"""图生视频客户端（晨星 AI · ``api.chenxingai.vip``，契约 §7.5 第 1 步）。

**厂商位置**：这是**独立第三方聚合**（转售 Seedance / MiniMax 等），与火山
MediaKit（渲染）、豆包语音（配音）**不共享端点与密钥**——接口字段按**用户提供的
接口文档**实现（本仓无 PDF，故不登记 ``docs/vendor/`` 错误码表）。

调用面（文档原文）：
- ``POST {base}/videos/generations``（提交）→ 平铺 ``{id, task_id, object, model,
  status, progress, size}``；任务 id 读 ``id``（``task_id`` 同值）
- ``GET  {base}/videos/{task_id}``（轮询）→ 状态 ``queued`` / ``in_progress`` /
  ``processing`` / ``completed`` / ``succeeded`` / ``failed``；建议轮询 5–10 秒
- 完成取 ``metadata.url``（**真实 mp4 直链，无需鉴权**；亦见 ``url`` 字段），
  或带鉴权代理 ``GET /videos/{task_id}/content``
- 鉴权 ``Authorization: Bearer sk-…``（集群 Secret，**密钥只在后端**）

⚠️ **视频临时保存 24 小时**（文档）——故产物**必须转存**，不得把直链当 key（§7.3② 同款纪律）。

⚠️ **``seconds`` 取值「视模型而定」**（文档原话），**不保证连续整数**。本实现照
镜头建议时长**原样下发**；厂商拒绝即抛结构化错误（不偷偷夹取 / 不编造时长）。

⚠️ 聚合商请求字段里**没有**指南首尾帧用的 ``content.role``——本仓把**尾帧**放 ``images``
数组第 2 位当「首尾帧桥」（ADR-074），这是**该接口面上最接近的落法**，语义**待真跑验证**。

⚠️ **运镜（ADR-076）**：本聚合端点**没有任何运动入参**（请求体只有 model/prompt/aspect_ratio/
size/seconds/quality/image/images）。故运镜**只能经提示词语言**表达——`motion_expression()`
是「本仓运镜符号 → 引擎表达」的**单点适配器**，返回值里的 ``params`` 当前**恒空**（不传）；
**引擎将来透出运动参数时，只改这一张表**（与前端 `lib/content/dsp-motion.ts` 的
`DSP_MOTION_ENGINE` **同源同值**）。**效果（模型是否真按提示词动镜头）待真跑验证。**
"""
from __future__ import annotations

import time

import requests

from flowmind.config import DspConfig
from flowmind.skills._secrets import get_api_key

_TERMINAL_OK = frozenset({"completed", "succeeded", "success", "done", "finished"})
_TERMINAL_FAIL = frozenset({"failed", "error", "canceled", "cancelled", "timeout"})

# 错误分类（文档给出的 error.code → 本仓三分类；语义对齐 errors.py）。
# **可重试** = 服务端临时故障 / 限流。
_VIDEOGEN_TRANSIENT_CODES = frozenset({
    "api_error",            # 服务内部错误，建议重试
    "rate_limit_exceeded",  # 请求频率过高
    "server_error", "internal_error", "timeout",
})
# **需查环境** = 鉴权 / 计费问题，别重试（改配置或充值）。
_VIDEOGEN_ENVIRONMENT_CODES = frozenset({
    "authentication_error",   # API Key 无效或未提供
    "insufficient_quota",     # 账户余额不足
    "permission_error", "account_deactivated",
})


class DspVideoGenError(Exception):
    """图生视频调用失败。category/retriable 语义与 errors.py 对齐。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def videogen_configured(cfg: DspConfig) -> bool:
    """是否已配置图生视频密钥。"""
    return bool(get_api_key(cfg.videogen_key_env))


def _classify_videogen_error(code: str) -> tuple[str, bool]:
    """按厂商 ``error.code`` 分类为 ``(category, retriable)``；未登记兜底 ``video``。"""
    key = (code or "").strip()
    if key in _VIDEOGEN_TRANSIENT_CODES:
        return "transient", True
    if key in _VIDEOGEN_ENVIRONMENT_CODES:
        return "environment", False
    return "video", False


def _error_of(raw: object) -> tuple[str, str]:
    """从响应体取 ``(code, message)``（文档：``{"error": {"code", "message"}}``）。"""
    if isinstance(raw, dict):
        err = raw.get("error")
        if isinstance(err, dict) and err:
            return str(err.get("code") or ""), str(err.get("message") or "")
        if isinstance(err, str) and err:
            return err, str(raw.get("message") or "")
    return "", ""


def _headers(cfg: DspConfig, api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _raise_for(cfg: DspConfig, action: str, raw: object, status: int) -> None:
    code, message = _error_of(raw)
    category, retriable = _classify_videogen_error(code)
    raise DspVideoGenError(
        f"图生视频{action}失败（HTTP {status}）：{code or '未知'} {message}".strip(),
        category=category, retriable=retriable,
    )


# ── 运镜适配（单点表，ADR-076）───────────────────────────────────
# 本仓运镜**符号**（与前端 `lib/content/dsp-motion.ts` 的 `DSP_MOTION_TYPES` 同源同值）。
MOTION_TYPES: tuple[str, ...] = (
    "none", "push_in", "pull_out", "pan_left", "pan_right",
    "tilt_up", "tilt_down", "zoom_in", "zoom_out", "orbit",
)
_INTENSITY_WORD: dict[str, str] = {
    "minimal": "极轻、几乎难以察觉", "subtle": "轻微", "standard": "平稳", "strong": "明显",
}
_DIRECTION_WORD: dict[str, str] = {
    "toward_subject": "朝主体", "away_from_subject": "背离主体",
    "left_to_right": "自左向右", "right_to_left": "自右向左", "none": "",
}
# 符号 → `{params, phrase}`。`params` = **引擎的运动入参**：本聚合端点没有该字段 ⇒ 一律空。
# 将来引擎透出运动参数时**只改这里**（不去猜别的模型认什么名字）。
MOTION_ENGINE: dict[str, dict[str, object]] = {
    "none": {"params": {}, "phrase": "摄影机保持固定"},
    "push_in": {"params": {}, "phrase": "摄影机缓慢推近"},
    "pull_out": {"params": {}, "phrase": "摄影机缓慢拉远"},
    "pan_left": {"params": {}, "phrase": "摄影机向左平移"},
    "pan_right": {"params": {}, "phrase": "摄影机向右平移"},
    "tilt_up": {"params": {}, "phrase": "摄影机上摇"},
    "tilt_down": {"params": {}, "phrase": "摄影机下摇"},
    "zoom_in": {"params": {}, "phrase": "镜头缓慢变焦推近"},
    "zoom_out": {"params": {}, "phrase": "镜头缓慢变焦拉远"},
    "orbit": {"params": {}, "phrase": "摄影机环绕主体"},
}

# **运镜关闭**时的固定机位条款（正向后缀）——「关」不等于「不做 I2V」，只是要求别动镜头。
# 与前端 `DSP_MOTION_STATIC_CLAUSE` **同源同值**。「画面动态强度」（连主体动态也不要）是**另一个**
# 控制项，将来单独设，**不并入本开关**——故这里明确**允许**主体/环境的自然动态。
MOTION_STATIC_CLAUSE = (
    "摄影机保持固定（固定机位、无推拉摇移、画面稳定不抖动），允许主体与环境发生自然的动态。"
)


def motion_expression(motion: object | None) -> tuple[str, dict[str, object]]:
    """**引擎适配器**：结构化运镜方案 → 该引擎的 ``(prompt 追加, 参数)``（ADR-076）。

    比照前端 `motionPromptFor` 的口径：
    - ``type == "none"`` ⇒ 出**固定机位条款**（不是空串——「不运镜」也要明确告诉模型别动镜头）；
    - 其余 ⇒ 优先用 Agent 给的 ``positive``（它看过图、能写到具体主体），缺则回落**确定性短语**
      （含幅度 / 方向 / 焦点 + 保真约束）；
    - **词表外的符号 ⇒ 空串 + 空参数**（调用方如实标注，不硬凑运动）；
    - ``params`` 当前**恒空**（本引擎无运动入参），是留给将来引擎透出参数的**接口位**。
    """
    if motion is None:
        return MOTION_STATIC_CLAUSE, {}
    mtype = str(getattr(motion, "type", "") or "").strip()
    entry = MOTION_ENGINE.get(mtype)
    if entry is None:
        return "", {}
    if mtype == "none":
        return MOTION_STATIC_CLAUSE, {}
    params = dict(entry["params"])  # type: ignore[arg-type]
    positive = str(getattr(motion, "positive", "") or "").strip()
    if positive:
        return positive, params
    intensity = _INTENSITY_WORD.get(str(getattr(motion, "intensity", "") or ""), "平稳")
    direction = _DIRECTION_WORD.get(str(getattr(motion, "direction", "") or ""), "")
    target = str(getattr(motion, "target", "") or "").strip()
    phrase = (
        f"{entry['phrase']}（幅度{intensity}）"
        f"{f'，{direction}' if direction else ''}"
        f"{f'，焦点是{target}' if target else ''}"
        "；保持原图主体的形状、颜色与构图不变，光照自然、运动真实"
    )
    return phrase, params


def seconds_str(ms: int) -> str:
    """镜头时长（ms）→ 厂商 ``seconds`` 字符串。

    文档：``seconds``「取值视模型而定（字符串或整数）」——本仓**原样下发**（整秒不带小数，
    否则保留一位），**不夹取到某枚举**（不替厂商猜）。
    """
    value = max(ms, 0) / 1000.0
    return str(int(value)) if value == int(value) else f"{value:.1f}"


def parse_size(size: str) -> tuple[int, int]:
    """``"720x1280"`` → ``(720, 1280)``；形状不对返回 ``(0, 0)``（调用方据此**跳过**降采样）。"""
    parts = (size or "").lower().split("x")
    if len(parts) != 2:
        return 0, 0
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError:
        return 0, 0
    return (w, h) if w > 0 and h > 0 else (0, 0)


def duration_value(seconds: str) -> int:
    """镜头秒数 → 厂商 ``duration``（**整数秒**，至少 1）。

    2026-10-17 真调：该接口认 ``duration``（int），旧请求体里的 ``seconds`` / ``aspect_ratio`` /
    ``quality`` 都不是它认识的字段——``seedance-2.5-global-sp`` 的 ``invalid_request`` 正是由此暴露。
    镜头窗口是 2.7–5.6s ⇒ 四舍五入到整秒（成片侧仍会裁到锁定窗口，ADR-080）。
    """
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        value = 4.0
    return max(1, int(round(value)))


def build_payload(
    cfg: DspConfig, *, prompt: str, image_url: str, seconds: str, tail_image_url: str = "",
    motion_params: dict[str, object] | None = None,
) -> dict:
    """按**当前**接口组装提交体（2026-10-17 真调更正，见 ADR-111）。

    ``image`` = 首帧参考素材；``tail_image_url`` 非空时再放进 ``images`` 第 2 位 =
    **首尾帧桥**（ADR-074）——数组形态在新模型上**尚未真跑验证**，厂商若忽略则退化为单图（无错误画面）。

    ``motion_params`` = **运镜的引擎参数位**（ADR-076）。本聚合端点**没有**运动字段，故
    ``motion_expression()`` 当前恒返回空 dict、这里**一个键都不会并进去**（空即不传，不编造厂商参数）。

    ⚠️ **不注入** ``aspect_ratio`` / ``quality`` / ``seconds``：它们不属于该视频接口的字段，
    真调会诱发上游 ``invalid_request``（ADR-111）。比例由 ``size`` 表达、时长用 ``duration``。
    """
    payload: dict = {
        "model": cfg.videogen_model,
        "prompt": prompt,
        "duration": duration_value(seconds),
        "size": cfg.videogen_size,
    }
    if image_url:
        payload["image"] = image_url
        if tail_image_url and tail_image_url != image_url:
            refs = [image_url, tail_image_url]
            payload["images"] = refs[: max(1, cfg.videogen_max_ref_images)]
    # 运镜参数位（ADR-076）：**空即不传**——本引擎为空，不往请求体里塞不认识的自造字段
    if motion_params:
        payload.update({k: v for k, v in motion_params.items() if k not in payload})
    return payload


def submit(
    cfg: DspConfig, *, prompt: str, image_url: str, seconds: str, tail_image_url: str = "",
    motion_params: dict[str, object] | None = None,
) -> str:
    """提交图生视频任务，返回 ``task_id``（文档：读 ``id`` / ``task_id``）。"""
    api_key = get_api_key(cfg.videogen_key_env)
    if not api_key:
        raise DspVideoGenError(
            f"未配置图生视频密钥 {cfg.videogen_key_env}（dsp 图生视频必需）。",
            category="environment")
    if not prompt.strip():
        raise DspVideoGenError("图生视频需要 prompt（镜头文案为空）。", category="video")

    url = cfg.videogen_api_base.rstrip("/") + cfg.videogen_submit_path
    payload = build_payload(
        cfg, prompt=prompt, image_url=image_url, seconds=seconds,
        tail_image_url=tail_image_url, motion_params=motion_params)
    try:
        resp = requests.post(url, json=payload, headers=_headers(cfg, api_key),
                             timeout=cfg.videogen_timeout_s)
    except requests.exceptions.Timeout as exc:
        raise DspVideoGenError(f"图生视频提交超时（{url}）", category="environment") from exc
    except requests.exceptions.RequestException as exc:
        raise DspVideoGenError(f"图生视频连接失败：{exc}", category="environment") from exc

    try:
        raw = resp.json()
    except ValueError:
        raw = None
    if not resp.ok:
        _raise_for(cfg, "提交", raw, resp.status_code)
    if not isinstance(raw, dict):
        raise DspVideoGenError("图生视频提交返回非 JSON", category="video")

    task_id = raw.get("id") or raw.get("task_id") or raw.get("data", {}).get("id")
    if not task_id:
        raise DspVideoGenError(
            f"图生视频提交响应缺少任务 id：{sorted(raw)[:12]}", category="video")
    return str(task_id)


def _query(cfg: DspConfig, api_key: str, task_id: str) -> dict:
    url = cfg.videogen_api_base.rstrip("/") + cfg.videogen_task_path.format(task_id=task_id)
    try:
        resp = requests.get(url, headers=_headers(cfg, api_key), timeout=cfg.videogen_timeout_s)
    except requests.exceptions.RequestException as exc:
        raise DspVideoGenError(
            f"图生视频查询失败：{exc}", category="transient", retriable=True) from exc
    try:
        raw = resp.json()
    except ValueError:
        raw = None
    if not resp.ok:
        _raise_for(cfg, "查询", raw, resp.status_code)
    if not isinstance(raw, dict):
        raise DspVideoGenError("图生视频查询返回非 JSON", category="video")
    return raw


def result_url(raw: dict) -> str:
    """从完成态响应里取视频直链（文档：``metadata.url``，亦见 ``url``）。"""
    meta = raw.get("metadata")
    if isinstance(meta, dict) and meta.get("url"):
        return str(meta["url"])
    for key in ("url", "video_url"):
        if raw.get(key):
            return str(raw[key])
    data = raw.get("data")
    if isinstance(data, dict):
        inner = data.get("metadata")
        if isinstance(inner, dict) and inner.get("url"):
            return str(inner["url"])
    return ""


def wait(cfg: DspConfig, task_id: str, *, api_key: str = "") -> dict:
    """轮询至终态，返回完成态响应（含 ``metadata.url``）。

    超时**如实报错**，不假装完成；``failed`` 时透出厂商 ``error``（不编造原因）。
    """
    key = api_key or get_api_key(cfg.videogen_key_env)
    if not key:
        raise DspVideoGenError(
            f"未配置图生视频密钥 {cfg.videogen_key_env}。", category="environment")
    deadline = time.monotonic() + cfg.videogen_poll_max_s
    last_status = "unknown"
    while True:
        raw = _query(cfg, key, task_id)
        status = str(raw.get("status") or "").lower()
        if status:
            last_status = status
        if status in _TERMINAL_OK:
            if not result_url(raw):
                raise DspVideoGenError(
                    f"图生视频任务完成但缺少视频直链（metadata.url / url）：{sorted(raw)[:12]}",
                    category="video")
            return raw
        if status in _TERMINAL_FAIL:
            code, message = _error_of(raw)
            category, retriable = _classify_videogen_error(code)
            raise DspVideoGenError(
                f"图生视频任务失败（status={status}）：{code} {message}".strip(),
                category=category, retriable=retriable)
        if time.monotonic() >= deadline:
            raise DspVideoGenError(
                f"图生视频轮询超时（>{cfg.videogen_poll_max_s}s，最后状态 {last_status}）",
                category="transient", retriable=True)
        time.sleep(cfg.videogen_poll_interval_s)


def generate(
    cfg: DspConfig, *, prompt: str, image_url: str, seconds: str, tail_image_url: str = "",
    motion_params: dict[str, object] | None = None,
) -> dict:
    """提交 + 轮询至完成，返回 ``{task_id, url, ...}``（薄适配层：demo 直接替换本函数）。"""
    api_key = get_api_key(cfg.videogen_key_env)
    task_id = submit(
        cfg, prompt=prompt, image_url=image_url, seconds=seconds,
        tail_image_url=tail_image_url, motion_params=motion_params)
    raw = wait(cfg, task_id, api_key=api_key)
    return {"task_id": task_id, "url": result_url(raw), "raw": raw}
