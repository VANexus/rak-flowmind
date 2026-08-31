"""微信公众号 API 客户端：access_token → 草稿箱 → 发布。

协议：微信公众平台 API（https://developers.weixin.qq.com/doc/office_account/）。
流程：
  1. get_access_token(app_id, app_secret) → access_token
  2. upload_thumb_image(access_token, image_url) → media_id（封面图）
  3. add_draft(access_token, articles) → media_id（草稿）
  4. free_publish(access_token, media_id) → publish_id

安全：AppID / AppSecret 由调用方从环境变量读出后传入，本模块不直接读 env。
错误分类：连接失败=environment、5xx=transient(可重试)、4xx=video、业务码非0=video。
"""
from __future__ import annotations

import httpx
import requests  # noqa: F401  保留模块级引用：测试 fixture 经 <mod>.requests 打桩拦截


class WechatAPIError(Exception):
    """微信 API 调用失败。category/retriable 语义与 errors.py 一致。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False, errcode: int | None = None):
        super().__init__(message)
        self.category = category
        self.retriable = retriable
        self.errcode = errcode


def get_access_token(
    *,
    app_id: str,
    app_secret: str,
    api_base: str = "https://api.weixin.qq.com/cgi-bin",
    timeout_s: float = 30.0,
    client: httpx.Client | None = None,
) -> str:
    """GET /token?grant_type=client_credential → access_token。"""
    if not app_id or not app_secret:
        raise ValueError("app_id / app_secret 不能为空。请检查环境变量 WECHAT_APP_ID / WECHAT_APP_SECRET。")

    url = f"{api_base.rstrip('/')}/token"
    params = {
        "grant_type": "client_credential",
        "appid": app_id,
        "secret": app_secret,
    }
    try:
        if client is not None:
            resp = client.get(url, params=params)
        else:
            with httpx.Client(timeout=timeout_s) as c:
                resp = c.get(url, params=params)
    except requests.exceptions.Timeout as exc:
        raise WechatAPIError("获取 access_token 超时", category="environment", retriable=False) from exc
    except httpx.TimeoutException as exc:
        raise WechatAPIError("获取 access_token 超时", category="environment", retriable=False) from exc
    except httpx.HTTPError as exc:
        raise WechatAPIError(f"获取 access_token 连接失败：{type(exc).__name__}", category="environment", retriable=False) from exc

    if resp.status_code >= 500:
        raise WechatAPIError(f"微信 API HTTP {resp.status_code}", category="transient", retriable=True)
    if resp.status_code >= 400:
        raise WechatAPIError(f"微信 API HTTP {resp.status_code}", category="video", retriable=False)

    data = resp.json()
    if "errcode" in data and data["errcode"] != 0:
        raise WechatAPIError(
            f"微信 API 业务错误：{data.get('errmsg', 'unknown')}（errcode={data['errcode']}）",
            category="video", retriable=False, errcode=data["errcode"],
        )
    token = data.get("access_token")
    if not token:
        raise WechatAPIError("微信 API 返回缺少 access_token", category="unknown", retriable=False)
    return token


def upload_thumb_image(
    *,
    access_token: str,
    image_url: str,
    api_base: str = "https://api.weixin.qq.com/cgi-bin",
    timeout_s: float = 30.0,
    client: httpx.Client | None = None,
) -> str:
    """下载封面图 → POST /material/add_material?type=image → media_id（永久素材）。"""
    # 先下载图片
    try:
        if client is not None:
            img_resp = client.get(image_url, timeout=timeout_s)
        else:
            with httpx.Client(timeout=timeout_s) as c:
                img_resp = c.get(image_url, timeout=timeout_s)
    except (httpx.HTTPError, requests.exceptions.RequestException) as exc:
        raise WechatAPIError(f"下载封面图失败：{type(exc).__name__}", category="environment", retriable=False) from exc

    if img_resp.status_code >= 400:
        raise WechatAPIError(f"封面图 URL 返回 HTTP {img_resp.status_code}", category="video", retriable=False)

    # 上传为永久素材
    url = f"{api_base.rstrip('/')}/material/add_material"
    params = {"access_token": access_token, "type": "image"}
    files = {"media": ("thumb.jpg", img_resp.content, "image/jpeg")}

    try:
        if client is not None:
            resp = client.post(url, params=params, files=files)
        else:
            with httpx.Client(timeout=timeout_s) as c:
                resp = c.post(url, params=params, files=files)
    except (httpx.HTTPError, requests.exceptions.RequestException) as exc:
        raise WechatAPIError(f"上传封面图失败：{type(exc).__name__}", category="environment", retriable=False) from exc

    if resp.status_code >= 500:
        raise WechatAPIError(f"微信 API HTTP {resp.status_code}", category="transient", retriable=True)
    if resp.status_code >= 400:
        raise WechatAPIError(f"微信 API HTTP {resp.status_code}", category="video", retriable=False)

    data = resp.json()
    if "errcode" in data and data["errcode"] != 0:
        raise WechatAPIError(
            f"上传封面图业务错误：{data.get('errmsg', 'unknown')}（errcode={data['errcode']}）",
            category="video", retriable=False, errcode=data["errcode"],
        )
    media_id = data.get("media_id")
    if not media_id:
        raise WechatAPIError("上传封面图返回缺少 media_id", category="unknown", retriable=False)
    return media_id


def add_draft(
    *,
    access_token: str,
    title: str,
    content: str,
    thumb_media_id: str,
    summary: str | None = None,
    author: str | None = None,
    api_base: str = "https://api.weixin.qq.com/cgi-bin",
    timeout_s: float = 30.0,
    client: httpx.Client | None = None,
) -> str:
    """POST /draft/add → 创建草稿 → 返回 media_id。"""
    url = f"{api_base.rstrip('/')}/draft/add"
    params = {"access_token": access_token}

    article: dict = {
        "title": title[:64],  # 微信限制标题 64 字节
        "content": content,
        "thumb_media_id": thumb_media_id,
        "need_open_comment": 0,
    }
    if summary:
        article["digest"] = summary[:120]
    if author:
        article["author"] = author[:8]

    body = {"articles": [article]}

    try:
        if client is not None:
            resp = client.post(url, params=params, json=body)
        else:
            with httpx.Client(timeout=timeout_s) as c:
                resp = c.post(url, params=params, json=body)
    except (httpx.HTTPError, requests.exceptions.RequestException) as exc:
        raise WechatAPIError(f"创建草稿失败：{type(exc).__name__}", category="environment", retriable=False) from exc

    if resp.status_code >= 500:
        raise WechatAPIError(f"微信 API HTTP {resp.status_code}", category="transient", retriable=True)
    if resp.status_code >= 400:
        raise WechatAPIError(f"微信 API HTTP {resp.status_code}", category="video", retriable=False)

    data = resp.json()
    if "errcode" in data and data["errcode"] != 0:
        raise WechatAPIError(
            f"创建草稿业务错误：{data.get('errmsg', 'unknown')}（errcode={data['errcode']}）",
            category="video", retriable=False, errcode=data["errcode"],
        )
    media_id = data.get("media_id")
    if not media_id:
        raise WechatAPIError("创建草稿返回缺少 media_id", category="unknown", retriable=False)
    return media_id


def free_publish(
    *,
    access_token: str,
    media_id: str,
    api_base: str = "https://api.weixin.qq.com/cgi-bin",
    timeout_s: float = 30.0,
    client: httpx.Client | None = None,
) -> str:
    """POST /freepublish/submit → 发布草稿 → 返回 publish_id。"""
    url = f"{api_base.rstrip('/')}/freepublish/submit"
    params = {"access_token": access_token}
    body = {"media_id": media_id}

    try:
        if client is not None:
            resp = client.post(url, params=params, json=body)
        else:
            with httpx.Client(timeout=timeout_s) as c:
                resp = c.post(url, params=params, json=body)
    except (httpx.HTTPError, requests.exceptions.RequestException) as exc:
        raise WechatAPIError(f"发布失败：{type(exc).__name__}", category="environment", retriable=False) from exc

    if resp.status_code >= 500:
        raise WechatAPIError(f"微信 API HTTP {resp.status_code}", category="transient", retriable=True)
    if resp.status_code >= 400:
        raise WechatAPIError(f"微信 API HTTP {resp.status_code}", category="video", retriable=False)

    data = resp.json()
    if "errcode" in data and data["errcode"] != 0:
        raise WechatAPIError(
            f"发布业务错误：{data.get('errmsg', 'unknown')}（errcode={data['errcode']}）",
            category="video", retriable=False, errcode=data["errcode"],
        )
    publish_id = data.get("publish_id")
    if not publish_id:
        raise WechatAPIError("发布返回缺少 publish_id", category="unknown", retriable=False)
    return str(publish_id)
