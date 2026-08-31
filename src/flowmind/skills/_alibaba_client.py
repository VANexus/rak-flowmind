"""阿里国际站开放 API 客户端（TOP 协议）。

实现阿里国际站（alibaba.icbu.open.product.* / alibaba.product.* 等）HTTP 调用的最小子集：
- TOP 公共参数组装 + 签名（hmac / md5）
- ``call(method, biz_params, session)`` 发起 POST，解析 JSON，错误分类结构化抛出
- ``upload_image`` 对 AI 生成的托管图片 URL 做直传（对应 product_post 的
  ``product_image.image_file_list[].image_file_url`` 字段接受公网 URL）

注意：本客户端是「开放平台免 API」集成的最小实现，部署前需运营在阿里国际站开放平台
注册开发者 App（AppKey/Secret）并完成店铺 OAuth 授权取得 session（进 .env）。
仅支持 JSON 响应（format=json）、hmac 签名。
"""
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

import httpx
import requests  # noqa: F401  保留模块级引用：测试 fixture 经 <mod>.requests 打桩拦截


class AlibabaAPIError(Exception):
    """阿里国际站 API 失败。category/retriable 语义与 errors.py 一致。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def sign_top_params(params: dict[str, str], secret: str, method: str = "hmac") -> str:
    """计算 TOP 协议签名。

    - ``hmac``：HMAC-MD5，key=secret，data=按 key 字典序拼接的 key+value。
    - ``md5``：MD5(secret + 拼接串 + secret)。
    签名统一大写。
    """
    base = "".join(f"{k}{params[k]}" for k in sorted(params))
    if method == "md5":
        raw = f"{secret}{base}{secret}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest().upper()
    return hmac.new(secret.encode("utf-8"), base.encode("utf-8"), hashlib.md5).hexdigest().upper()


class AlibabaClient:
    """TOP 协议客户端：组装公共参数、签名、POST、解析。"""

    def __init__(
        self,
        *,
        api_base: str,
        app_key: str,
        app_secret: str,
        session: str | None = None,
        sign_method: str = "hmac",
        timeout_s: float = 20.0,
        client: httpx.Client | None = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.app_key = app_key
        self.app_secret = app_secret
        self.session = session
        self.sign_method = sign_method
        self.timeout_s = timeout_s
        self._client = client

    def _now_gmt8(self) -> str:
        tz = timezone(timedelta(hours=8))
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    def call(self, method: str, biz_params: dict, session: str | None = None) -> dict:
        if not self.app_key or not self.app_secret:
            raise AlibabaAPIError(
                "未设置 ALIBABA_APP_KEY / ALIBABA_APP_SECRET",
                category="environment", retriable=False,
            )

        params: dict[str, str] = {
            "method": method,
            "app_key": self.app_key,
            "timestamp": self._now_gmt8(),
            "format": "json",
            "v": "2.0",
            "sign_method": self.sign_method,
        }
        sess = session if session is not None else self.session
        if sess:
            params["session"] = sess

        # 业务参数：复合类型（dict/list）JSON 序列化，标量转字符串
        for k, v in biz_params.items():
            if isinstance(v, (dict, list)):
                params[k] = json.dumps(v, ensure_ascii=False)
            else:
                params[k] = str(v)

        params["sign"] = sign_top_params(params, self.app_secret, self.sign_method)

        try:
            if self._client is not None:
                resp = self._client.post(self.api_base, data=params)
            else:
                with httpx.Client(timeout=self.timeout_s) as c:
                    resp = c.post(self.api_base, data=params)
        except requests.exceptions.Timeout as exc:
            raise AlibabaAPIError("阿里国际站请求超时", category="environment", retriable=False) from exc
        except httpx.TimeoutException as exc:
            raise AlibabaAPIError("阿里国际站请求超时", category="environment", retriable=False) from exc
        except httpx.HTTPError as exc:
            raise AlibabaAPIError(f"阿里国际站连接失败：{type(exc).__name__}", category="environment", retriable=False) from exc

        if resp.status_code >= 500:
            raise AlibabaAPIError(f"阿里国际站 HTTP {resp.status_code}", category="transient", retriable=True)
        if resp.status_code >= 400:
            raise AlibabaAPIError(f"阿里国际站 HTTP {resp.status_code}", category="video", retriable=False)

        try:
            data = resp.json()
        except ValueError as exc:
            raise AlibabaAPIError("阿里国际站返回非法 JSON", category="unknown", retriable=False) from exc

        err = data.get("error_response")
        if err:
            code = err.get("code") or err.get("sub_code") or "UNKNOWN"
            msg = err.get("msg") or err.get("sub_msg") or "接口返回错误"
            raise AlibabaAPIError(f"阿里国际站接口错误 {code}：{msg}", category="video", retriable=False)
        return data

    def upload_image(self, image_url: str) -> str:
        """把主图 URL 转为可被 product_post 引用的 URL。

        简化实现：AI 生成图为公网托管 URL，product_post 的
        ``image_file_list[].image_file_url`` 直接接受公网 URL，故原样透传；
        本地文件上传到图片银行（alibaba.product.photobank.upload）留待后续按需扩展。
        """
        if not image_url:
            raise AlibabaAPIError("主图 URL 为空，无法发布", category="unknown", retriable=False)
        return image_url


def new_client_from_config(cfg) -> AlibabaClient:
    """按 config + 环境变量组装 AlibabaClient（key 从 env 读取，绝不进 toml/commit）。

    返回 (client, session)；session 为空表示未授权，由上层走 degraded。
    """
    from flowmind.skills._secrets import get_api_key

    app_key = get_api_key(cfg.app_key_env) or ""
    app_secret = get_api_key(cfg.app_secret_env) or ""
    session = get_api_key(cfg.session_env)
    return AlibabaClient(
        api_base=cfg.api_base,
        app_key=app_key,
        app_secret=app_secret,
        session=session,
        sign_method=cfg.sign_method,
        timeout_s=cfg.timeout_s,
    )