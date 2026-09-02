"""微信公众号 API 客户端：access_token → 草稿箱 → 发布 → 群发。

协议：微信公众平台 API（https://developers.weixin.qq.com/doc/office_account/）。
流程：
  1. get_access_token(app_id, app_secret) → access_token（进程内 TTL 缓存）
  2. upload_thumb_image(access_token, image_url) → media_id（封面图，永久素材）
  3. upload_content_images(access_token, content) → HTML（正文图 uploadimg 转存为 mmbiz URL）
  4. add_draft(access_token, articles) → media_id（草稿）
  5. free_publish(access_token, media_id) → publish_id（发布）
  6. mass_send(access_token, media_id) → msg_id（群发，提交即异步执行）
  7. get_publish_status / get_article_url / get_mass_status → 状态查询

安全：AppID / AppSecret 由调用方从环境变量读出后传入，本模块不直接读 env。
错误分类：连接失败=environment、5xx=transient(可重试)、4xx=video、业务码非0=video。
"""
from __future__ import annotations

import re
import time

import httpx
import requests  # noqa: F401  保留模块级引用：测试 fixture 经 <mod>.requests 打桩拦截


class WechatAPIError(Exception):
    """微信 API 调用失败。category/retriable 语义与 errors.py 一致。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False, errcode: int | None = None):
        super().__init__(message)
        self.category = category
        self.retriable = retriable
        self.errcode = errcode


# ── access_token 进程内缓存（微信每日 2000 次配额；按 app_id 键控、提前 5 分钟过期）──
_TOKEN_CACHE: dict[str, tuple[str, float]] = {}


def _post_json(
    *,
    url: str,
    access_token: str,
    body: dict,
    api_base: str,
    timeout_s: float,
    client: httpx.Client | None,
    what: str,
) -> dict:
    """统一 POST JSON 到带 access_token 的接口，返回业务 JSON。"""
    params = {"access_token": access_token}
    try:
        if client is not None:
            resp = client.post(url, params=params, json=body)
        else:
            with httpx.Client(timeout=timeout_s) as c:
                resp = c.post(url, params=params, json=body)
    except (httpx.HTTPError, requests.exceptions.RequestException) as exc:
        raise WechatAPIError(f"{what}失败：{type(exc).__name__}", category="environment", retriable=False) from exc

    if resp.status_code >= 500:
        raise WechatAPIError(f"微信 API HTTP {resp.status_code}", category="transient", retriable=True)
    if resp.status_code >= 400:
        raise WechatAPIError(f"微信 API HTTP {resp.status_code}", category="video", retriable=False)

    data = resp.json()
    if isinstance(data, dict) and "errcode" in data and data["errcode"] != 0:
        raise WechatAPIError(
            f"{what}业务错误：{data.get('errmsg', 'unknown')}（errcode={data['errcode']}）",
            category="video", retriable=False, errcode=data["errcode"],
        )
    return data


def get_access_token(
    *,
    app_id: str,
    app_secret: str,
    api_base: str = "https://api.weixin.qq.com/cgi-bin",
    timeout_s: float = 30.0,
    client: httpx.Client | None = None,
    use_cache: bool = True,
) -> str:
    """GET /token?grant_type=client_credential → access_token。

    默认启用进程内 TTL 缓存（按 app_id），避免高频调用触达每日配额。
    """
    if not app_id or not app_secret:
        raise ValueError("app_id / app_secret 不能为空。请检查环境变量 WECHAT_APP_ID / WECHAT_APP_SECRET。")

    now = time.time()
    cached = _TOKEN_CACHE.get(app_id)
    if use_cache and cached and cached[1] > now:
        return cached[0]

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
    if use_cache:
        expires_in = int(data.get("expires_in", 7200))
        _TOKEN_CACHE[app_id] = (token, now + expires_in - 300)
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


_IMG_SRC_RE = re.compile(r'(<img\b[^>]*?\bsrc=")([^"]+)(")', re.IGNORECASE)
_MMBIZ_PREFIX = ("https://mmbiz.qpic.cn/", "http://mmbiz.qpic.cn/")


def upload_content_images(
    *,
    access_token: str,
    content: str,
    api_base: str = "https://api.weixin.qq.com/cgi-bin",
    timeout_s: float = 30.0,
    client: httpx.Client | None = None,
) -> tuple[str, list[dict]]:
    """把正文 HTML 里的外链图片全部转存为公众号图片（media/uploadimg → mmbiz URL）。

    返回 (替换后的 HTML, 转存记录 [{src, url, ok, error?}])。
    已是 mmbiz 的图片直接跳过。单张转存失败记入记录但不中断整篇（可后续重试）。
    """
    uploaded: list[dict] = []
    visited: set[str] = set()

    def repl(m: re.Match) -> str:
        url = m.group(2)
        if url.startswith(_MMBIZ_PREFIX) or url in visited:
            return m.group(0)
        visited.add(url)
        try:
            new_url = upload_content_image(
                access_token=access_token, image_url=url,
                api_base=api_base, timeout_s=timeout_s, client=client,
            )
            uploaded.append({"src": url, "url": new_url, "ok": True})
            return m.group(1) + new_url + m.group(3)
        except WechatAPIError as exc:
            uploaded.append({"src": url, "url": "", "ok": False, "error": str(exc)[:200]})
            return m.group(0)  # 保留原图，不阻断整篇

    new_content = _IMG_SRC_RE.sub(repl, content)
    return new_content, uploaded


def upload_content_image(
    *,
    access_token: str,
    image_url: str,
    api_base: str = "https://api.weixin.qq.com/cgi-bin",
    timeout_s: float = 30.0,
    client: httpx.Client | None = None,
) -> str:
    """下载单张图片 → POST /media/uploadimg → mmbiz URL（图文正文专用，不占素材库）。"""
    try:
        if client is not None:
            img_resp = client.get(image_url, timeout=timeout_s)
        else:
            with httpx.Client(timeout=timeout_s) as c:
                img_resp = c.get(image_url, timeout=timeout_s)
    except (httpx.HTTPError, requests.exceptions.RequestException) as exc:
        raise WechatAPIError(f"下载正文图失败：{type(exc).__name__}", category="environment", retriable=False) from exc

    if img_resp.status_code >= 400:
        raise WechatAPIError(f"正文图 URL 返回 HTTP {img_resp.status_code}", category="video", retriable=False)

    url = f"{api_base.rstrip('/')}/media/uploadimg"
    params = {"access_token": access_token}
    files = {"media": ("body.jpg", img_resp.content, "image/jpeg")}
    try:
        if client is not None:
            resp = client.post(url, params=params, files=files)
        else:
            with httpx.Client(timeout=timeout_s) as c:
                resp = c.post(url, params=params, files=files)
    except (httpx.HTTPError, requests.exceptions.RequestException) as exc:
        raise WechatAPIError(f"上传正文图失败：{type(exc).__name__}", category="environment", retriable=False) from exc

    if resp.status_code >= 500:
        raise WechatAPIError(f"微信 API HTTP {resp.status_code}", category="transient", retriable=True)
    if resp.status_code >= 400:
        raise WechatAPIError(f"微信 API HTTP {resp.status_code}", category="video", retriable=False)

    data = resp.json()
    if "errcode" in data and data["errcode"] != 0:
        raise WechatAPIError(
            f"上传正文图业务错误：{data.get('errmsg', 'unknown')}（errcode={data['errcode']}）",
            category="video", retriable=False, errcode=data["errcode"],
        )
    new_url = data.get("url")
    if not new_url:
        raise WechatAPIError("上传正文图返回缺少 url", category="unknown", retriable=False)
    return new_url


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

    data = _post_json(
        url=url, access_token=access_token, body={"articles": [article]},
        api_base=api_base, timeout_s=timeout_s, client=client, what="创建草稿",
    )
    media_id = data.get("media_id")
    if not media_id:
        raise WechatAPIError("创建草稿返回缺少 media_id", category="unknown", retriable=False)
    return media_id


def free_publish(
    *,
    access_token: str,
    media_id: str,
    publish_time: int | None = None,
    api_base: str = "https://api.weixin.qq.com/cgi-bin",
    timeout_s: float = 30.0,
    client: httpx.Client | None = None,
) -> str:
    """POST /freepublish/submit → 发布草稿 → 返回 publish_id。

    publish_time 可选（Unix 秒）：仅当公众号开通「定时发布」权限时微信才接受；
    未开通会返回业务错误，由上层给出明确提示。
    """
    url = f"{api_base.rstrip('/')}/freepublish/submit"
    body: dict = {"media_id": media_id}
    if publish_time:
        body["publish_time"] = int(publish_time)

    data = _post_json(
        url=url, access_token=access_token, body=body,
        api_base=api_base, timeout_s=timeout_s, client=client, what="发布",
    )
    publish_id = data.get("publish_id")
    if not publish_id:
        raise WechatAPIError("发布返回缺少 publish_id", category="unknown", retriable=False)
    return str(publish_id)


def mass_send(
    *,
    access_token: str,
    media_id: str,
    clientmsgid: str | None = None,
    send_ignore_reprint: int = 0,
    api_base: str = "https://api.weixin.qq.com/cgi-bin",
    timeout_s: float = 30.0,
    client: httpx.Client | None = None,
) -> str:
    """POST /message/mass/sendall → 群发图文消息 → 返回 msg_id。

    提交即成功、异步执行；发送后原草稿 media_id 失效（微信自动删除草稿）。
    clientmsgid 用于幂等防重；send_ignore_reprint 控制转载声明校验。
    """
    url = f"{api_base.rstrip('/')}/message/mass/sendall"
    body: dict = {
        "filter": {"is_to_all": True},
        "mpnews": {"media_id": media_id},
        "msgtype": "mpnews",
        "send_ignore_reprint": send_ignore_reprint,
    }
    if clientmsgid:
        body["clientmsgid"] = clientmsgid

    data = _post_json(
        url=url, access_token=access_token, body=body,
        api_base=api_base, timeout_s=timeout_s, client=client, what="群发",
    )
    msg_id = data.get("msg_id")
    if msg_id is None:
        raise WechatAPIError("群发返回缺少 msg_id", category="unknown", retriable=False)
    return str(msg_id)


def get_publish_status(
    *,
    access_token: str,
    publish_id: str,
    api_base: str = "https://api.weixin.qq.com/cgi-bin",
    timeout_s: float = 30.0,
    client: httpx.Client | None = None,
) -> dict:
    """POST /freepublish/get → 发布状态。

    返回：{ publish_id, publish_status:0=成功,1=发布中,2=原草稿审核失败,3=成功且需审核中,
            fail_idx, article_detail:{ count, item:[{idx, article_url, ...}] } }
    """
    url = f"{api_base.rstrip('/')}/freepublish/get"
    return _post_json(
        url=url, access_token=access_token, body={"publish_id": str(publish_id)},
        api_base=api_base, timeout_s=timeout_s, client=client, what="查询发布状态",
    )


def get_article_url(
    *,
    access_token: str,
    publish_id: str,
    api_base: str = "https://api.weixin.qq.com/cgi-bin",
    timeout_s: float = 30.0,
    client: httpx.Client | None = None,
) -> str | None:
    """POST /freepublish/getarticle → 已发布文章的永久链接（第一篇文章）。"""
    url = f"{api_base.rstrip('/')}/freepublish/getarticle"
    data = _post_json(
        url=url, access_token=access_token, body={"publish_id": str(publish_id)},
        api_base=api_base, timeout_s=timeout_s, client=client, what="查询发布文章",
    )
    articles = data.get("article_list") or []
    if articles:
        return articles[0].get("url")
    return None


def get_mass_status(
    *,
    access_token: str,
    msg_id: str,
    api_base: str = "https://api.weixin.qq.com/cgi-bin",
    timeout_s: float = 30.0,
    client: httpx.Client | None = None,
) -> dict:
    """POST /message/mass/get → 群发状态。

    返回：{ msg_id, msg_status:0=群发成功,1=群发中,2=群发失败,3=被封禁,4=触发频控,5=审核中... }
    """
    url = f"{api_base.rstrip('/')}/message/mass/get"
    return _post_json(
        url=url, access_token=access_token, body={"msg_id": str(msg_id)},
        api_base=api_base, timeout_s=timeout_s, client=client, what="查询群发状态",
    )
