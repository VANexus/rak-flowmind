"""TikHub 第三方数据 API 客户端（https://docs.tikhub.io/）。

替代不稳定的自建 Creative Center 爬虫：TikHub 在服务端维护 TikTok Creative
Center 等数据源的抓取，本客户端只负责鉴权、请求与统一错误分类。

已真机验证（2026-09）：
- 鉴权：``Authorization: Bearer <AI_TRENDS_API_KEY>``；
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

from flowmind.skills._tikhub_cache import (
    _set_cache_meta,
    get_cache,
    get_last_cache_meta,  # noqa: F401  # re-export：b2b_keyword_trends 从本模块导入
)

# 热门标签榜单（Creative Center Trends → hashtag）
PATH_TRENDS_HASHTAG_LIST = "/api/v1/tiktok/ads/get_trends_hashtag_list"
# 热门标签详情（受众画像/长周期曲线/代表视频）
PATH_TRENDS_HASHTAG_DETAIL = "/api/v1/tiktok/ads/get_trends_hashtag_detail"
# Creative Center 字典：支持国家地区 / 广告筛选项（行业/目标/语言/形式/周期）
PATH_ADS_LOCATION_LIST = "/api/v1/tiktok/ads/get_location_list"
PATH_ADS_TOP_FILTERS = "/api/v1/tiktok/ads/get_top_ads_filters"
# 广告创意库（真实在投广告搜索）
PATH_ADS_SEARCH = "/api/v1/tiktok/ads/search_ads"

# Web 系：每日趋势搜索词
PATH_WEB_TRENDING_SEARCHWORDS = "/api/v1/tiktok/web/fetch_trending_searchwords"

# App V3 系（最稳表面，优先使用）
PATH_APP_VIDEO_SEARCH = "/api/v1/tiktok/app/v3/fetch_video_search_result"
PATH_APP_ONE_VIDEO = "/api/v1/tiktok/app/v3/fetch_one_video"
PATH_APP_USER_PROFILE = "/api/v1/tiktok/app/v3/handler_user_profile"
PATH_APP_USER_COUNTRY = "/api/v1/tiktok/app/v3/fetch_user_country_by_username"
PATH_APP_MUSIC_CHART = "/api/v1/tiktok/app/v3/fetch_music_chart_list"
PATH_APP_CREATOR_INSIGHTS = "/api/v1/tiktok/app/v3/fetch_creator_search_insights"

# Shop Web 系（电商选品，只维护该系列）
PATH_SHOP_CATEGORIES = "/api/v1/tiktok/shop/web/fetch_products_category_list"
PATH_SHOP_SEARCH = "/api/v1/tiktok/shop/web/fetch_search_products_list"
PATH_SHOP_PRODUCT_DETAIL = "/api/v1/tiktok/shop/web/fetch_product_detail_v3"
PATH_SHOP_REVIEWS = "/api/v1/tiktok/shop/web/fetch_product_reviews_v2"
PATH_SHOP_SELLER_PRODUCTS = "/api/v1/tiktok/shop/web/fetch_seller_products_list"
PATH_SHOP_SUGGEST = "/api/v1/tiktok/shop/web/fetch_search_word_suggestion_v2"

# Instagram 话题搜索（关键词 → 话题列表，含 media_count）
PATH_IG_SEARCH_HASHTAGS = "/api/v1/instagram/v2/search_hashtags"
# Instagram 话题下真实帖子
PATH_IG_HASHTAG_POSTS = "/api/v1/instagram/v2/fetch_hashtag_posts"

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
        cache=None,
    ):
        self.api_base = (api_base or "https://api.tikhub.dev").rstrip("/")
        self.api_key = (api_key or "").strip()
        self.timeout_s = timeout_s
        self._client = client
        # 磁盘缓存（TikHubCache）：None = 直连不缓存（测试/显式禁用）
        self._cache = cache

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
                "未设置 AI_TRENDS_API_KEY；请在 .env 配置趋势 API Key 后重试。",
                category="environment", retriable=False,
            )
        cache = self._cache
        if cache is None:
            _set_cache_meta("live", hit=False, age_s=0.0)
            return self._send(method, path, json_body=json_body, params=params)

        key = cache.make_key(method, path, json_body, params)
        # per-key 进程内锁：并发同参请求合并（后者直接复用前者写入的缓存）
        with cache._lock_for(key):
            cached = cache.get(key)
            if cached is not None:
                cached_payload, age_s = cached
                mode = cache.decide(path, age_s)
                if mode == "local":
                    _set_cache_meta("local", hit=True, age_s=age_s)
                    return cached_payload

            learned = cache.learned_window(path)
            expect_free = bool(
                cached is not None
                and learned is not None
                and age_s < learned  # type: ignore[operator]
            )
            try:
                payload, resp_headers = self._send(
                    method, path, json_body=json_body, params=params, with_headers=True,
                )
            except TikHubError:
                # 外呼失败：有旧缓存就回落（宁给旧数据不给空态；空态由上层 skill 语义处理）
                if cached is not None:
                    _set_cache_meta("local_fallback", hit=True, age_s=age_s)  # type: ignore[possibly-undefined]
                    return cached_payload  # type: ignore[possibly-undefined]
                raise

            cache.learn_from_headers(path, resp_headers)
            cache.put(key, path, payload)
            _set_cache_meta("speculative" if expect_free else "live", hit=False, age_s=0.0)
            return payload

    def _send(self, method: str, path: str, *, json_body: dict | None = None,
              params: dict | None = None, with_headers: bool = False):
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
        return (payload, dict(resp.headers)) if with_headers else payload

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

    # ── Ads / Creative Center 情报 ────────────────────────────────────

    def ads_hashtag_detail(self, *, hashtag_id: str, time_range: int = 30,
                          country_code: str = "US", cookie: str | None = None) -> dict:
        """单个热门标签详情（受众年龄/国家分布、长周期曲线、代表视频）。"""
        body: dict = {
            "hashtag_id": str(hashtag_id),
            "time_range": normalize_time_range(time_range),
            "country_code": country_code or "US",
        }
        if cookie:
            body["cookie"] = cookie
        return self.post(PATH_TRENDS_HASHTAG_DETAIL, body).get("data") or {}

    def ads_locations(self) -> dict:
        """Creative Center 支持的国家/地区列表。

        该端点请求体是「cookie 字符串」而非 JSON 对象，无 cookie 时传空字符串。
        返回结构为双层信封 ``{code,msg,data:{country:[{id,value,label}]}}``。
        """
        payload = self._request("POST", PATH_ADS_LOCATION_LIST, json_body="")
        return _inner_data(payload.get("data"))

    def ads_filters(self) -> dict:
        """热门广告筛选项字典：industry/objective/ad_language/pattern_label/period/country。"""
        payload = self._request("POST", PATH_ADS_TOP_FILTERS, json_body="")
        return _inner_data(payload.get("data"))

    def ads_search(self, *, keyword: str, period: int = 180, page: int = 1, limit: int = 20,
                   country_code: str = "US", order_by: str = "for_you", objective: int | None = None,
                   industry: str | None = None, cookie: str | None = None) -> dict:
        """搜索真实在投广告创意库。返回 ``{materials:[...], pagination:{...}}``。"""
        body: dict = {
            "keyword": (keyword or "").strip(),
            "period": int(period),
            "page": max(1, int(page)),
            "limit": max(1, int(limit)),
            "country_code": country_code or "US",
            "order_by": order_by or "for_you",
        }
        if objective is not None:
            body["objective"] = int(objective)
        if industry:
            body["industry"] = str(industry)
        if cookie:
            body["cookie"] = cookie
        return _inner_data(self.post(PATH_ADS_SEARCH, body).get("data"))

    # ── Web：每日趋势搜索词 ───────────────────────────────────────────

    def web_trending_searchwords(self) -> dict:
        """TikTok 站内每日趋势搜索关键词（无参，返回 trending_search_words 列表）。"""
        return self.get(PATH_WEB_TRENDING_SEARCHWORDS).get("data") or {}

    # ── App V3：内容/达人/音乐 ────────────────────────────────────────

    def app_video_search(self, *, keyword: str, count: int = 20, offset: int = 0,
                         sort_type: int = 0, publish_time: int = 0, region: str = "US") -> dict:
        """关键词搜视频（App V3，最稳）。返回含 search_item_list 的原始结构。"""
        return self.get(PATH_APP_VIDEO_SEARCH, params={
            "keyword": (keyword or "").strip(),
            "count": max(1, int(count)),
            "offset": max(0, int(offset)),
            "sort_type": int(sort_type),
            "publish_time": int(publish_time),
            "region": region or "US",
        }).get("data") or {}

    def app_one_video(self, *, aweme_id: str) -> dict:
        """单个视频详情（含无水印下载地址）。返回 aweme_detail。"""
        return self.get(PATH_APP_ONE_VIDEO, params={"aweme_id": str(aweme_id)}).get("data") or {}

    def app_user_profile(self, *, unique_id: str = "", user_id: str = "",
                         sec_user_id: str = "") -> dict:
        """达人账号资料（三选一标识）。返回 user。"""
        params: dict = {}
        if unique_id:
            params["unique_id"] = unique_id
        elif user_id:
            params["user_id"] = str(user_id)
        elif sec_user_id:
            params["sec_user_id"] = str(sec_user_id)
        return self.get(PATH_APP_USER_PROFILE, params=params).get("data") or {}

    def app_user_country(self, *, username: str) -> dict:
        """达人账号注册国家（资料接口不含国家，需专门端点）。"""
        return self.get(PATH_APP_USER_COUNTRY, params={"username": (username or "").strip()}).get("data") or {}

    def app_music_chart(self, *, scene: int = 0, count: int = 50, cursor: int = 0) -> dict:
        """热门音乐榜。返回 music_list/chart_info。"""
        return self.get(PATH_APP_MUSIC_CHART, params={
            "scene": int(scene), "count": max(1, int(count)), "cursor": max(0, int(cursor)),
        }).get("data") or {}

    def app_creator_insights(self, *, limit: int = 20, offset: int = 0, tab: str = "all",
                             language_filters: str = "en", category_filters: str = "") -> dict:
        """创作者搜索洞察（热门创作选题/趋势序列）。返回 inspiration_list。"""
        params: dict = {
            "limit": max(1, int(limit)), "offset": max(0, int(offset)),
            "tab": tab or "all", "language_filters": language_filters or "en",
            "creator_source": "general_search",
        }
        if category_filters:
            params["category_filters"] = category_filters
        return self.get(PATH_APP_CREATOR_INSIGHTS, params=params).get("data") or {}

    # ── Shop：电商选品 ────────────────────────────────────────────────

    def shop_categories(self, *, region: str = "US") -> list:
        """TikTok Shop 商品类目树（一级 + children）。返回列表。"""
        data = self.get(PATH_SHOP_CATEGORIES, params={"region": region or "US"}).get("data")
        return data if isinstance(data, list) else []

    def shop_search_products(self, *, keyword: str, region: str = "US",
                             offset: int = 0, page_token: str = "") -> dict:
        """按关键词搜索 TikTok Shop 商品。返回 ``{products,shops,has_more,load_more_params}``。"""
        params: dict = {"search_word": (keyword or "").strip(), "region": region or "US",
                        "offset": max(0, int(offset))}
        if page_token:
            params["page_token"] = page_token
        return _inner_data(self.get(PATH_SHOP_SEARCH, params=params).get("data"))

    def shop_product_detail(self, *, product_id: str, region: str = "US") -> dict:
        """商品详情 V3（数据完整）。返回 product_data。"""
        return self.get(PATH_SHOP_PRODUCT_DETAIL,
                        params={"product_id": str(product_id), "region": region or "US"}).get("data") or {}

    def shop_product_reviews(self, *, product_id: str, region: str = "US",
                             page_start: int = 1, sort_rule: int = 2) -> dict:
        """商品评论 V2（V1 已不稳定）。返回 ``{product_reviews,review_ratings,total_reviews,has_more}``。"""
        return _inner_data(self.get(PATH_SHOP_REVIEWS, params={
            "product_id": str(product_id), "region": region or "US",
            "page_start": max(1, int(page_start)), "sort_rule": int(sort_rule),
        }).get("data"))

    def shop_seller_products(self, *, seller_id: str, region: str = "US") -> dict:
        """商家在售商品列表。返回 ``{products,...}``。"""
        return _inner_data(self.get(PATH_SHOP_SELLER_PRODUCTS,
                                   params={"seller_id": str(seller_id), "region": region or "US"}).get("data"))

    def shop_search_suggest(self, *, keyword: str, region: str = "US") -> list[str]:
        """Shop 搜索词联想（返回字符串数组）。"""
        data = _inner_data(self.get(PATH_SHOP_SUGGEST, params={
            "search_word": (keyword or "").strip(), "region": region or "US",
        }).get("data"))
        return [str(x) for x in data] if isinstance(data, list) else []

    # ── Instagram 话题帖子 ────────────────────────────────────────────

    def instagram_hashtag_posts(self, *, keyword: str, feed_type: str = "top",
                                pagination_token: str = "") -> dict:
        """IG 话题下真实帖子（V2）。返回 ``{count,items,pagination_token}``。"""
        params: dict = {"keyword": (keyword or "").strip().lstrip("#"),
                        "feed_type": feed_type or "top"}
        if pagination_token:
            params["pagination_token"] = pagination_token
        data = self.get(PATH_IG_HASHTAG_POSTS, params=params).get("data") or {}
        inner = data.get("data") if isinstance(data, dict) and isinstance(data.get("data"), dict) else None
        merged = inner or data
        # 分页 token 与 data.data 平级，补回去
        if isinstance(data, dict) and data.get("pagination_token") and isinstance(merged, dict):
            merged.setdefault("pagination_token", data["pagination_token"])
        return merged


def _inner_data(data):
    """很多 TikHub 端点在统一信封 data 内再包一层 ``{code,message/msg,data}``；
    存在内层 data 时解平，否则原样返回（对列表/空值安全）。"""
    if isinstance(data, dict) and "data" in data and (
        "code" in data or "msg" in data or "message" in data
    ):
        inner = data.get("data")
        return inner if inner is not None else data
    return data


def new_client_from_config(cfg) -> TikHubClient:
    """按 config + 环境变量组装 TikHubClient（key 从 env/.env 读取，绝不进 toml/commit）。

    tikhub_cache_enabled=True（默认）时挂磁盘缓存：soft_ttl 内直回本地，
    过期后真实外呼；若端点从响应头学习到服务端免费缓存窗口则升级 speculative。
    """
    from flowmind.skills._secrets import get_api_key

    api_key = get_api_key(getattr(cfg, "tikhub_key_env", "AI_TRENDS_API_KEY")) or ""
    cache = None
    if getattr(cfg, "tikhub_cache_enabled", True):
        from pathlib import Path

        db_path = (
            getattr(cfg, "tikhub_cache_db_path", "")
            or str(Path.cwd() / ".cache" / "tikhub_cache.db")
        )
        cache = get_cache(
            db_path,
            default_soft_ttl_s=float(getattr(cfg, "tikhub_cache_soft_ttl_s", 1800.0)),
            max_free_window_s=float(getattr(cfg, "tikhub_cache_max_window_s", 21600.0)),
        )
    return TikHubClient(
        api_base=getattr(cfg, "tikhub_api_base", "https://api.tikhub.dev"),
        api_key=api_key,
        timeout_s=float(getattr(cfg, "tikhub_timeout_s", 30.0)),
        cache=cache,
    )
