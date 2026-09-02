"""TikHub 情报数据解析纯函数（真实字段，绝不编造）。

把 Ads / App V3 / Shop Web / Instagram V2 各端点的原始 ``data`` 解析为统一、
精简、可直接序列化给前端的业务结构。所有解析对缺失字段宽容（.get 链 + 类型归一），
但绝不填充任何虚构值：取不到就是 None / 空列表。

已逐端点真机验证字段路径（2026-09，fixture 见 tests/fixtures/tikhub/）。
"""
from __future__ import annotations


def _to_int(v, default: int | None = 0) -> int | None:
    """宽松整数化：字符串数字/浮点都接受，无法解析时返回 default。"""
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _first_url(node) -> str:
    """从 {url_list:[...]} 或直接字符串取第一个 URL。"""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        urls = node.get("url_list")
        if isinstance(urls, list) and urls:
            return str(urls[0])
        for k in ("url", "uri"):
            if node.get(k):
                return str(node[k])
    return ""


# ── Ads 广告创意库 ───────────────────────────────────────────────────

def parse_ad_materials(data: dict) -> list[dict]:
    """search_ads → 统一广告创意行。data 已由 client 解平双层信封。"""
    if not isinstance(data, dict):
        return []
    raw = data.get("materials")
    if not isinstance(raw, list):
        return []
    rows: list[dict] = []
    for i, m in enumerate(raw, start=1):
        if not isinstance(m, dict):
            continue
        vi = m.get("video_info") if isinstance(m.get("video_info"), dict) else {}
        video_urls = vi.get("video_url") if isinstance(vi.get("video_url"), dict) else {}
        rows.append({
            "id": str(m.get("id") or ""),
            "rank": i,
            "title": str(m.get("ad_title") or "").strip(),
            "brand": str(m.get("brand_name") or "").strip(),
            "ctr": float(m["ctr"]) if isinstance(m.get("ctr"), (int, float)) else None,
            "likes": _to_int(m.get("like")),
            "cost": _to_int(m.get("cost")),
            "objective": str(m.get("objective_key") or ""),
            "industry_key": str(m.get("industry_key") or ""),
            "is_search": bool(m.get("is_search")),
            "duration_s": float(vi["duration"]) if isinstance(vi.get("duration"), (int, float)) else None,
            "cover_url": str(vi.get("cover") or ""),
            "video_url": str(video_urls.get("720p") or video_urls.get("default") or ""),
            "width": _to_int(vi.get("width")),
            "height": _to_int(vi.get("height")),
        })
    return rows


def parse_ad_pagination(data: dict) -> dict:
    pg = data.get("pagination") if isinstance(data, dict) else None
    if not isinstance(pg, dict):
        return {"has_more": False, "page": 1, "total": 0}
    return {
        "has_more": bool(pg.get("has_more") or pg.get("hasMore")),
        "page": _to_int(pg.get("page"), 1) or 1,
        "total": _to_int(pg.get("total_count") or pg.get("totalCount"), 0) or 0,
    }


def parse_ad_filters(data: dict) -> dict:
    """top_ads_filters → 精简字典（行业/目标/语言/形式/周期/国家）。"""
    if not isinstance(data, dict):
        return {}
    out: dict[str, list] = {}
    for key in ("industry", "objective", "ad_language", "pattern_label", "period", "country"):
        col = data.get(key)
        if not isinstance(col, list):
            continue
        out[key] = [{
            "id": str(x.get("id")) if x.get("id") is not None else "",
            "label": str(x.get("value") or x.get("label") or ""),
            "parent_id": _to_int(x.get("parent_id")) if x.get("parent_id") is not None else None,
        } for x in col if isinstance(x, dict)]
    return out


def parse_locations(data: dict) -> list[dict]:
    """location_list → [{id,name}]。"""
    col = data.get("country") if isinstance(data, dict) else None
    if not isinstance(col, list):
        return []
    return [{"id": str(x.get("id") or ""), "name": str(x.get("value") or x.get("label") or "")}
            for x in col if isinstance(x, dict)]


# ── 热门标签详情 ─────────────────────────────────────────────────────

def parse_hashtag_detail(data: dict) -> dict:
    """trends_hashtag_detail → 话题深度画像。"""
    if not isinstance(data, dict):
        return {}
    curve = [
        {"timestamp": str(p.get("timestamp") or ""), "value": float(p.get("value") or 0)}
        for p in (data.get("popularityCurve") or []) if isinstance(p, dict)
    ]
    age = [
        {"level": str(a.get("ageLevel") or ""), "percent": _to_float(a.get("vvPercent"))}
        for a in (data.get("ageProfile") or []) if isinstance(a, dict)
    ]
    geo = [
        {"country": str(g.get("countryCode") or ""), "tgi": _to_float(g.get("countryTgiScore"))}
        for g in (data.get("representativeCountryProfile") or []) if isinstance(g, dict)
    ]
    videos = []
    for v in (data.get("videoList") or []):
        if not isinstance(v, dict):
            continue
        vu = v.get("videoURL") if isinstance(v.get("videoURL"), dict) else {}
        videos.append({
            "item_id": str(v.get("itemID") or ""),
            "cover_url": str(v.get("coverURL") or ""),
            "video_url": str(vu.get("default") or ""),
        })
    return {
        "hashtag_id": str(data.get("hashtagID") or ""),
        "name": str(data.get("hashtagName") or ""),
        "vv": _to_int(data.get("vv")),
        "publish_cnt": _to_int(data.get("publishCnt")),
        "time_range": _to_int(data.get("timeRange")),
        "country": str(data.get("countryCode") or ""),
        "industry_ids": [_to_int(x) for x in (data.get("industryIDs") or []) if x is not None],
        "curve": curve,
        "age_profile": age,
        "country_profile": geo,
        "videos": videos,
    }


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── Web 每日搜索热词 ─────────────────────────────────────────────────

def parse_trending_searchwords(data: dict) -> list[dict]:
    raw = data.get("trending_search_words") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    return [
        {"word": str(x.get("trendingSearchWord") or "").strip(),
         "type": str(x.get("trendingSearchWordType") or "")}
        for x in raw if isinstance(x, dict) and str(x.get("trendingSearchWord") or "").strip()
    ]


# ── App V3 视频搜索 ──────────────────────────────────────────────────

def parse_video_search(data: dict) -> list[dict]:
    """fetch_video_search_result → 统一爆款视频行。"""
    raw = data.get("search_item_list") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    rows: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        a = item.get("aweme_info")
        if not isinstance(a, dict):
            continue
        st = a.get("statistics") if isinstance(a.get("statistics"), dict) else {}
        author = a.get("author") if isinstance(a.get("author"), dict) else {}
        video = a.get("video") if isinstance(a.get("video"), dict) else {}
        nw = video.get("download_no_watermark_addr") or video.get("download_addr") or {}
        music = a.get("music") if isinstance(a.get("music"), dict) else {}
        rows.append({
            "aweme_id": str(a.get("aweme_id") or ""),
            "desc": str(a.get("desc") or "").strip(),
            "create_time": _to_int(a.get("create_time")),
            # aweme.video.duration 单位是毫秒，转成秒（ads/music 的 duration 本就是秒，勿混）
            "duration_s": (round(_to_int(video.get("duration")) / 1000, 1)
                           if _to_int(video.get("duration")) else None),
            "play": _to_int(st.get("play_count")),
            "likes": _to_int(st.get("digg_count")),
            "comments": _to_int(st.get("comment_count")),
            "shares": _to_int(st.get("share_count")),
            "collects": _to_int(st.get("collect_count")),
            "author_id": str(author.get("uid") or author.get("unique_id") or ""),
            "author": str(author.get("nickname") or author.get("unique_id") or ""),
            "author_handle": str(author.get("unique_id") or ""),
            "author_followers": _to_int(author.get("follower_count")),
            "cover_url": _first_url(video.get("cover")) or _first_url(video.get("origin_cover")),
            "video_url": _first_url(nw),
            "music_title": str(music.get("title") or ""),
        })
    return rows


def parse_one_video(data: dict) -> dict:
    """fetch_one_video → 单视频（含无水印地址）。"""
    a = data.get("aweme_detail") if isinstance(data, dict) else None
    if not isinstance(a, dict):
        return {}
    wrapped = {"search_item_list": [{"aweme_info": a}]}
    rows = parse_video_search(wrapped)
    return rows[0] if rows else {}


# ── 音乐榜 ───────────────────────────────────────────────────────────

def parse_music_chart(data: dict) -> list[dict]:
    raw = data.get("music_list") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    rows: list[dict] = []
    for i, m in enumerate(raw, start=1):
        if not isinstance(m, dict):
            continue
        mi = m.get("music_info") if isinstance(m.get("music_info"), dict) else {}
        artists = mi.get("artists") if isinstance(mi.get("artists"), list) else []
        rows.append({
            "rank": i,
            "music_id": str(m.get("id") or mi.get("id_str") or mi.get("id") or ""),
            "title": str(mi.get("title") or "").strip(),
            "author": str(mi.get("author") or "").strip(),
            "duration_s": _to_int(mi.get("duration")),
            "user_count": _to_int(mi.get("user_count")),
            "trend": _to_int(m.get("trend")),
            "cover_url": _first_url(mi.get("cover_large")) or _first_url(mi.get("cover_medium")),
            "artists": [str(x.get("nick_name") or x.get("handle") or "") for x in artists
                        if isinstance(x, dict)],
        })
    return rows


# ── 创作者搜索洞察（选题灵感） ───────────────────────────────────────

def parse_creator_insights(data: dict) -> list[dict]:
    raw = data.get("inspiration_list") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    rows: list[dict] = []
    for x in raw:
        if not isinstance(x, dict):
            continue
        textnet = x.get("textnet") if isinstance(x.get("textnet"), dict) else {}
        seq = x.get("trending_seq") if isinstance(x.get("trending_seq"), list) else []
        rows.append({
            "query_id": str(x.get("query_id_str") or x.get("query_id") or ""),
            "query": str(x.get("query_text") or "").strip(),
            "popularity": _to_int(x.get("popularity")),
            "popularity_v2": _to_int(x.get("popularity_v2")),
            "video_num": _to_int(x.get("video_num")),
            "trend_seq": [_to_int(v) for v in seq],
            "category_l1": str(textnet.get("layer1") or ""),
            "category_l2": str(textnet.get("layer2") or ""),
            "business_types": [str(b) for b in x.get("business_types", []) if b],
        })
    return rows


# ── 达人资料 ─────────────────────────────────────────────────────────

def parse_user_profile(data: dict) -> dict:
    u = data.get("user") if isinstance(data, dict) else None
    if not isinstance(u, dict):
        return {}
    return {
        "user_id": str(u.get("uid") or u.get("id") or ""),
        "sec_user_id": str(u.get("sec_uid") or ""),
        "unique_id": str(u.get("unique_id") or ""),
        "nickname": str(u.get("nickname") or ""),
        "followers": _to_int(u.get("follower_count")),
        "following": _to_int(u.get("following_count")),
        "aweme_count": _to_int(u.get("aweme_count")),
        "favoriting_count": _to_int(u.get("favoriting_count")),
        "signature": str(u.get("signature") or ""),
        "custom_verify": str(u.get("custom_verify") or ""),
        "is_star": bool(u.get("is_star")),
        "avatar_url": _first_url(u.get("avatar_larger")) or _first_url(u.get("avatar_medium")),
    }


# ── Shop 选品 ────────────────────────────────────────────────────────

def parse_shop_products(data: dict) -> list[dict]:
    """search/seller products → 统一商品行。"""
    raw = data.get("products") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    rows: list[dict] = []
    for p in raw:
        if not isinstance(p, dict):
            continue
        price = p.get("product_price_info") if isinstance(p.get("product_price_info"), dict) else {}
        rate = p.get("rate_info") if isinstance(p.get("rate_info"), dict) else {}
        sold = p.get("sold_info") if isinstance(p.get("sold_info"), dict) else {}
        seller = p.get("seller_info") if isinstance(p.get("seller_info"), dict) else {}
        brand = p.get("brand_info") if isinstance(p.get("brand_info"), dict) else {}
        seo = p.get("seo_url") if isinstance(p.get("seo_url"), dict) else {}
        labels = []
        pmi = p.get("product_marketing_info")
        if isinstance(pmi, dict):
            placement = pmi.get("placement_labels")
            if isinstance(placement, dict):
                for lab_list in placement.values():
                    if isinstance(lab_list, list):
                        for lab in lab_list:
                            if isinstance(lab, dict) and lab.get("text"):
                                labels.append(str(lab["text"]))
        rows.append({
            "product_id": str(p.get("product_id") or ""),
            "title": str(p.get("title") or "").strip(),
            "image_url": _first_url(p.get("image")),
            "price": str(price.get("sale_price_decimal") or ""),
            "original_price": str(price.get("origin_price_decimal") or ""),
            "discount": str(price.get("discount_format") or ""),
            "currency": str(price.get("currency_symbol") or price.get("currency_name") or ""),
            "rating": _to_float(rate.get("score")),
            "review_count": _to_int(rate.get("review_count")),
            "sold_count": _to_int(sold.get("sold_count")),
            "seller_id": str(seller.get("seller_id") or ""),
            "seller_name": str(seller.get("shop_name") or ""),
            "brand": str(brand.get("brand_name") or ""),
            "url": str(seo.get("canonical_url") or ""),
            "labels": sorted(set(labels)),
        })
    return rows


def parse_shop_page(data: dict) -> dict:
    if not isinstance(data, dict):
        return {"has_more": False}
    lm = data.get("load_more_params") if isinstance(data.get("load_more_params"), dict) else {}
    return {
        "has_more": bool(data.get("has_more")),
        "offset": _to_int(lm.get("offset"), 0),
        "page_token": str(lm.get("page_token") or ""),
    }


def parse_product_detail(data: dict) -> dict:
    """product_detail_v3 → 商品深度信息。

    真机验证：匿名访客拿到的详情组件中 product_model 不含价格（价格由前端异步加载），
    因此价格以搜索列表为准，这里不编造；只解析真实可得的图集/规格/SKU/销量/店铺。
    """
    if not isinstance(data, dict):
        return {}
    pc = data.get("page_config") if isinstance(data.get("page_config"), dict) else {}
    comps = pc.get("components_map") if isinstance(pc.get("components_map"), list) else []
    by_name = {c.get("component_name"): c for c in comps if isinstance(c, dict)}
    pi = by_name.get("product_info") or {}
    cd = pi.get("component_data") if isinstance(pi.get("component_data"), dict) else {}
    pm = (cd.get("product_info") or {}).get("product_model") if isinstance(cd.get("product_info"), dict) else None
    if not isinstance(pm, dict):
        return {}
    images = [_first_url(im) for im in (pm.get("images") or []) if isinstance(im, dict)]
    # 描述是图文块数组，只取图片 URL
    desc_images: list[str] = []
    for blk in (pm.get("description") or []):
        if isinstance(blk, dict) and isinstance(blk.get("image"), dict):
            u = _first_url(blk["image"])
            if u:
                desc_images.append(u)
    specs = []
    for pp in (pm.get("product_properties") or []):
        if isinstance(pp, dict):
            vals = [str(v.get("property_value_name") or "") for v in (pp.get("property_values") or [])
                    if isinstance(v, dict)]
            specs.append({"name": str(pp.get("property_name") or ""), "values": [v for v in vals if v]})
    variants = []
    for sp in (pm.get("sale_properties") or []):
        if isinstance(sp, dict):
            vals = [str(v.get("property_value_name") or "") for v in (sp.get("property_values") or [])
                    if isinstance(v, dict)]
            variants.append({"name": str(sp.get("property_name") or ""), "values": [v for v in vals if v]})
    videos = pm.get("videos") if isinstance(pm.get("videos"), dict) else {}
    video_urls = [str(v.get("post_url") or "") for v in videos.values()
                  if isinstance(v, dict) and v.get("post_url")]
    shop = cd.get("shop_info") if isinstance(cd.get("shop_info"), dict) else {}
    return {
        "product_id": str(pm.get("product_id") or ""),
        "seller_id": str(pm.get("seller_id") or ""),
        "name": str(pm.get("name") or ""),
        "sold_count": _to_int(pm.get("sold_count")),
        "images": [u for u in images if u],
        "desc_images": desc_images,
        "specs": specs,
        "variants": variants,
        "sku_count": len(pm.get("skus") or []),
        "video_urls": video_urls,
        "shop": {
            "seller_id": str(shop.get("seller_id") or ""),
            "shop_name": str(shop.get("shop_name") or ""),
            "shop_rating": _to_float(shop.get("shop_rating")),
            "review_count": _to_int(shop.get("review_count")),
            "followers": _to_int(shop.get("followers_count")),
            "shop_sold": _to_int(shop.get("sold_count")),
            "on_sell_count": _to_int(shop.get("on_sell_product_count")),
        },
    }


def parse_shop_categories(data: list) -> list[dict]:
    """类目树 → 精简两级（self + children，只留必要字段）。"""
    def node(n: dict) -> dict:
        s = n.get("self", {}) if isinstance(n.get("self"), dict) else {}
        children = n.get("children") if isinstance(n.get("children"), list) else []
        return {
            "category_id": str(s.get("category_id") or ""),
            "name": str(s.get("category_name") or ""),
            "level": _to_int(s.get("category_level")),
            "is_leaf": bool(s.get("is_leaf")),
            "children": [node(c) for c in children if isinstance(c, dict)],
        }
    if not isinstance(data, list):
        return []
    return [node(n) for n in data if isinstance(n, dict)]


def parse_product_reviews(data: dict) -> dict:
    if not isinstance(data, dict):
        return {"reviews": [], "summary": {}}
    raw = data.get("product_reviews") if isinstance(data.get("product_reviews"), list) else []
    reviews = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        imgs = r.get("review_images") if isinstance(r.get("review_images"), list) else []
        img_urls = []
        for im in imgs:
            u = _first_url(im) if not isinstance(im, str) else im
            if u:
                img_urls.append(u)
        reviews.append({
            "review_id": str(r.get("review_id") or ""),
            "rating": _to_int(r.get("review_rating")),
            "time": str(r.get("review_time") or ""),
            "verified": bool(r.get("is_verified_purchase")),
            "incentivized": bool(r.get("is_incentivized_review")),
            "reviewer": str(r.get("reviewer_name") or ""),
            "text": str(r.get("review_text") or "").strip(),
            "images": img_urls,
            "sku_spec": str(r.get("sku_specification") or ""),
            "country": str(r.get("review_country") or ""),
        })
    rr = data.get("review_ratings") if isinstance(data.get("review_ratings"), dict) else {}
    summary = {
        "total": str(data.get("total_reviews") or rr.get("review_count") or ""),
        "avg": _to_float(rr.get("overall_score")),
        "distribution": rr.get("rating_result") if isinstance(rr.get("rating_result"), dict) else {},
        "has_more": bool(data.get("has_more")),
    }
    return {"reviews": reviews, "summary": summary}


# ── Instagram 话题帖子 ───────────────────────────────────────────────

def parse_ig_hashtag_posts(data: dict) -> dict:
    if not isinstance(data, dict):
        return {"posts": [], "pagination_token": ""}
    raw = data.get("items") if isinstance(data.get("items"), list) else []
    posts = []
    for p in raw:
        if not isinstance(p, dict):
            continue
        imgs = p.get("image_versions") if isinstance(p.get("image_versions"), list) else []
        thumb = ""
        if imgs and isinstance(imgs[0], dict):
            thumb = str(imgs[0].get("url") or "")
        user = p.get("user") if isinstance(p.get("user"), dict) else {}
        tags = p.get("caption_hashtags") if isinstance(p.get("caption_hashtags"), list) else []
        posts.append({
            "media_id": str(p.get("id") or p.get("code") or ""),
            "code": str(p.get("code") or ""),
            "caption": str(p.get("caption_text") or p.get("title") or "").strip(),
            "hashtags": [str(t) for t in tags if t],
            "likes": _to_int(p.get("like_count")),
            "comments": _to_int(p.get("comment_count")),
            "plays": _to_int(p.get("play_count") or p.get("view_count")),
            "is_video": bool(p.get("is_video")),
            "media_type": _to_int(p.get("media_type")),
            "thumbnail": thumb or str(p.get("thumbnail_url") or ""),
            "video_url": str(p.get("video_url") or ""),
            "taken_at": _to_int(p.get("taken_at_ts") or p.get("taken_at")),
            "username": str(user.get("username") or ""),
            "user_fullname": str(user.get("full_name") or ""),
            "verified": bool(user.get("is_verified")),
        })
    return {"posts": posts, "pagination_token": str(data.get("pagination_token") or "")}
