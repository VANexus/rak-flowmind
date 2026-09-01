"""TikHub 第三方数据 API 客户端（https://docs.tikhub.io/）。

替代不稳定的自建 Creative Center 爬虫：TikHub 在服务端维护 TikTok Creative
Center 等数据源的抓取，本客户端只负责鉴权、请求与统一错误分类。

已真机验证（2026-09）：
- 鉴权：``Authorization: Bearer <TIKHUB_API_KEY>``；
- 域名：大陆直连加速域名 ``https://api.tikhub.dev``（无需代理），
  海外用 ``https://api.tikhub.io``，路径/参数完全一致；
- 统一响应信封：``{code, message, data, ...}``，业务数据在 ``data``；
- 热门标签榜：``POST /api/v1/tiktok/ads/get_trends_hashtag_list``，
  返回字段与自建 GetHashtagList 完全同构
  （items[hashtagName/vv/publishCnt/popularityCurve/rankIndex/industryIDs]），
  且无需登录 cookie 即给全量（totalCount=100）。

错误分类语义与 errors.py 对齐：
- environment：密钥缺失/无效(401/403)、余额不足(402)、连通性问题——修环境，不盲目重试；
- transient：限流(429)/服务端 5xx/超时——可重试；
- unknown：参数/数据问题(400/404/422) 或结构异常——不重试。
"""
from __future__ import annotations

import httpx

# 热门标签榜单（Creative Center Trends → hashtag）
PATH_TRENDS_HASHTAG_LIST = "/api/v1/tiktok/ads/get_trends_hashtag_list"

# Instagram 话题搜索（关键词 → 话题列表，含 media_count）
PATH_IG_SEARCH_HASHTAGS = "/api/v1/instagram/v2/search_hashtags"

# TikHub 仅接受 7/30/90 三档时间窗
VALID_TIME_RANGES = (7, 30, 90)


class TikHubError(Exception):
    """TikHub API 失败。category/retriable 语义与 errors.py 一致。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def normalize_time_range(period_days: int | None) -> int:
    """把任意周期天数归一到 TikHub 支持的最近档位（7/30/90），平局取小档。"""
    try:
        days = int(period_days) if period_days is not None else 7
    except (TypeError, ValueError):
        return 7
    return min(VALID_TIME_RANGES, key=lambda cand: (abs(cand - days), cand))


class TikHubClient:
    """TikHub HTTP 客户端：Bearer 鉴权 + 信封解析 + 错误分类。"""

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        timeout_s: float = 30.0,
        client: httpx.Client | None = None,
    ):
        self.api_base = (api_base or "https://api.tikhub.dev").rstrip("/")
        self.api_key = (api_key or "").strip()
        self.timeout_s = timeout_s
        self._client = client

    # ── 基础请求 ──────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(self, method: str, path: str, *, json_body: dict | None = None,
                 params: dict | None = None) -> dict:
        if not self.api_key:
            raise TikHubError(
                "未设置 TIKHUB_API_KEY；请在 .env 配置 TikHub API Key 后重试。",
                category="environment", retriable=False,
            )
        url = self.api_base + path
        try:
            if self._client is not None:
                resp = self._client.request(
                    method, url, json=json_body, params=params, headers=self._headers(),
                )
            else:
                with httpx.Client(timeout=self.timeout_s) as c:
                    resp = c.request(
                        method, url, json=json_body, params=params, headers=self._headers(),
                    )
        except httpx.TimeoutException as exc:
            raise TikHubError(
                f"TikHub 请求超时（{self.timeout_s}s）", category="transient", retriable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise TikHubError(
                f"TikHub 连接失败：{type(exc).__name__}（大陆环境请确认 api_base 为 "
                "https://api.tikhub.dev 加速域名）",
                category="environment", retriable=True,
            ) from exc

        self._raise_for_status(resp)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise TikHubError("TikHub 返回非法 JSON", category="unknown", retriable=False) from exc

        # 信封 code 与 HTTP 状态双保险（正常 code=200）
        code = payload.get("code") if isinstance(payload, dict) else None
        if code is not None and int(code) != 200:
            msg = payload.get("message_zh") or payload.get("message") or f"业务码 {code}"
            raise TikHubError(f"TikHub 业务错误 code={code}：{msg}", category="unknown", retriable=False)
        if not isinstance(payload, dict) or "data" not in payload:
            raise TikHubError("TikHub 响应缺少 data 字段（结构可能已变化）",
                              category="unknown", retriable=False)
        return payload

    def _raise_for_status(self, resp: httpx.Response) -> None:
        status = resp.status_code
        if 200 <= status < 300:
            return
        detail = ""
        try:
            body = resp.json()
            detail = body.get("message_zh") or body.get("message") or body.get("detail") or ""
        except Exception:  # noqa: BLE001
            detail = (resp.text or "")[:200]
        if status in (401, 403):
            raise TikHubError(
                f"TikHub 鉴权失败 HTTP {status}：{detail or 'API Key 无效/过期/权限不足'}",
                category="environment", retriable=False,
            )
        if status == 402:
            raise TikHubError(
                "TikHub 余额不足 HTTP 402：请充值后重试（本次请求未成功计费）。",
                category="environment", retriable=False,
            )
        if status == 429:
            raise TikHubError("TikHub 触发限流 HTTP 429：请降速后重试。",
                              category="transient", retriable=True)
        if status >= 500:
            raise TikHubError(f"TikHub 服务端故障 HTTP {status}：{detail}",
                              category="transient", retriable=True)
        raise TikHubError(f"TikHub 请求错误 HTTP {status}：{detail}",
                          category="unknown", retriable=False)

    def post(self, path: str, body: dict) -> dict:
        return self._request("POST", path, json_body=body)

    def get(self, path: str, params: dict | None = None) -> dict:
        return self._request("GET", path, params=params)

    # ── 业务端点 ──────────────────────────────────────────────────────

    def trending_hashtags(
        self,
        *,
        time_range: int = 7,
        country_code: str = "US",
        page: int = 1,
        limit: int = 20,
        industry_id: int | None = None,
        cookie: str | None = None,
    ) -> dict:
        """Creative Center 热门标签榜（趋势）。返回信封内 data：
        ``{items, pagination, BaseResp}``。
        """
        body: dict = {
            "time_range": normalize_time_range(time_range),
            "country_code": country_code or "US",
            "page": max(1, int(page)),
            "limit": max(1, int(limit)),
        }
        if industry_id:
            body["industry_id"] = int(industry_id)
        if cookie:
            body["cookie"] = cookie
        return self.post(PATH_TRENDS_HASHTAG_LIST, body).get("data") or {}

    def instagram_search_hashtags(self, *, keyword: str) -> dict:
        """IG 话题搜索（V2 访客视角，关键词驱动）。返回 ``{count, items}``。

        已真机验证（2026-09）：无需登录 cookie，items 为
        ``[{id, name, media_count, profile_pic_url, allow_following}]``；
        该端点在信封 data 内再包一层 data，这里统一解平。
        """
        data = self.get(PATH_IG_SEARCH_HASHTAGS, params={"keyword": (keyword or "").strip()}).get("data") or {}
        inner = data.get("data") if isinstance(data, dict) and isinstance(data.get("data"), dict) else None
        return inner or data


def new_client_from_config(cfg) -> TikHubClient:
    """按 config + 环境变量组装 TikHubClient（key 从 env/.env 读取，绝不进 toml/commit）。"""
    from flowmind.skills._secrets import get_api_key

    api_key = get_api_key(getattr(cfg, "tikhub_key_env", "TIKHUB_API_KEY")) or ""
    return TikHubClient(
        api_base=getattr(cfg, "tikhub_api_base", "https://api.tikhub.dev"),
        api_key=api_key,
        timeout_s=float(getattr(cfg, "tikhub_timeout_s", 30.0)),
    )
