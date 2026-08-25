"""v0.3.7: 视频本地化后端客户端。

封装 video-localizer HTTP 调用，消除 5 个 localize_* 技能里重复的
requests.get/post/delete + 错误分类 + 超时管理。

特性：
- 统一错误分类（environment / video / transient / unknown）
- 统一超时 / 404 / 5xx 处理
- 健康检查 fast-fail
- 直接走模块级 requests（不持连接池），便于测试 monkeypatch / 并发轮询
"""
from __future__ import annotations

import requests

from flowmind.config import LocalizerConfig, get_config
from flowmind.contracts import SkillError
from flowmind.errors import ErrorCode


class VLAPIError(Exception):
    """视频本地化后端调用异常，携带结构化错误分类。"""

    def __init__(self, code: str, message: str, category: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.category = category
        self.details = details or {}


class VLClient:
    """video-localizer 后端的 thin 客户端。

    用法（每个 localize_* 技能实例化一次）：
        cfg = get_config().localizer
        client = VLClient(cfg)
        resp = client.post("/batch", payload)

    直接调用模块级 requests.get/post/delete（不持有连接池），让测试可以用
    模块级 monkeypatch 拦截，也保证 localize_status 的多线程轮询安全。
    """

    def __init__(self, cfg: LocalizerConfig | None = None):
        self.cfg = cfg or get_config().localizer

    @property
    def base_url(self) -> str:
        return f"{self.cfg.api_base.rstrip('/')}{self.cfg.api_prefix}"

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health_check(self) -> None:
        """fast-fail：VL 不通立刻抛带分类的 VLAPIError（environment/transient/video）。"""
        try:
            r = requests.get(self._url("/health"), timeout=self.cfg.health_timeout)
        except requests.RequestException as exc:
            raise VLAPIError(
                code=ErrorCode.INTERNAL,
                message="video-localizer 健康检查失败",
                category="environment",
                details={"url": self.base_url},
            ) from exc
        if r.status_code >= 500:
            raise VLAPIError(ErrorCode.INTERNAL, "健康检查 5xx", "transient", {"status_code": r.status_code})
        if r.status_code >= 400:
            raise VLAPIError(ErrorCode.INTERNAL, "健康检查 4xx", "video", {"status_code": r.status_code})
        r.raise_for_status()

    def post(self, path: str, payload: dict) -> dict:
        """POST 请求。失败抛 VLAPIError（带 category）。"""
        try:
            r = requests.post(
                self._url(path), json=payload, timeout=self.cfg.http_timeout
            )
        except requests.RequestException as exc:
            raise VLAPIError(
                code=ErrorCode.INTERNAL,
                message=f"POST {path} 失败: {exc}",
                category="environment",
            ) from exc
        return self._parse(r, path)

    def get(self, path: str) -> dict:
        """GET 请求。404 → NOT_FOUND；其他 4xx/5xx → INTERNAL。"""
        try:
            r = requests.get(self._url(path), timeout=self.cfg.http_timeout)
        except requests.RequestException as exc:
            raise VLAPIError(
                code=ErrorCode.INTERNAL,
                message=f"GET {path} 失败: {exc}",
                category="environment",
            ) from exc
        if r.status_code == 404:
            raise VLAPIError(
                code=ErrorCode.NOT_FOUND,
                message=f"资源不存在: {path}",
                category="video",
            )
        return self._parse(r, path)

    def delete(self, path: str) -> dict:
        try:
            r = requests.delete(self._url(path), timeout=self.cfg.http_timeout)
        except requests.RequestException as exc:
            raise VLAPIError(
                code=ErrorCode.INTERNAL,
                message=f"DELETE {path} 失败: {exc}",
                category="environment",
            ) from exc
        return self._parse(r, path)

    @staticmethod
    def _parse(r: requests.Response, path: str) -> dict:
        """解析响应：4xx → 错误（一律 video，与 errors.py 的 4xx 分类对齐）；5xx → transient；2xx → JSON。"""
        if 400 <= r.status_code < 500:
            try:
                detail = r.json()
            except Exception:
                detail = str(getattr(r, "text", ""))[:200]
            raise VLAPIError(
                code=ErrorCode.VALIDATION if r.status_code in (400, 422) else ErrorCode.INTERNAL,
                message=f"{r.status_code} {path}: {detail}",
                category="video",
                details={"status_code": r.status_code, "body": detail},
            )
        if r.status_code >= 500:
            raise VLAPIError(
                code=ErrorCode.INTERNAL,
                message=f"5xx {path}: {str(getattr(r, 'text', ''))[:200]}",
                category="transient",
                details={"status_code": r.status_code},
            )
        try:
            return r.json()
        except Exception:
            return {"_raw": str(getattr(r, "text", ""))}


def vlapi_to_skill_error(exc: VLAPIError) -> SkillError:
    """VLAPIError → SkillError 转换（让 invoke() 统一兜底）。

    注意：SkillError 没有 `category` 字段（契约层不变量）。类别信息塞进
    details["category"]，调用方可在 `error.details["category"]` 读到。
    """
    details = dict(exc.details or {})
    details.setdefault("category", exc.category)
    return SkillError(
        code=exc.code,
        message=exc.message,
        retriable=exc.category == "transient",
        details=details,
    )
