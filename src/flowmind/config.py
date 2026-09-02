"""配置层：技能内置通用默认，用户配置文件可覆盖。

个性化定制只发生在终端用户的对话式初始化——由消费此包的 Agent
按 README 剧本引导用户，调用 save_config() 写出 flowmind.config.toml。
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import tomli_w
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path("flowmind.config.toml")  # 相对于当前工作目录（cwd）


class InventoryConfig(BaseModel):
    """库销比/库存风险技能的可配置阈值（附通用默认值）。"""
    dsi_healthy_max: float = 60.0   # 周转天数 <=此值：健康
    dsi_watch_max: float = 90.0     # <=此值：关注
    dsi_warn_max: float = 120.0     # <=此值：预警；超过：危险
    dsi_low: float = 15.0           # 低于此值：断货风险
    capital_high: float = 100000.0  # 资金占用高阈值（货币单位）
    currency: str = "USD"


class FeishuKbConfig(BaseModel):
    """飞书知识库 FAQ 检索技能的可配置参数（附通用默认值）。"""
    data_path: str = ""                # FAQ 数据 JSON 文件路径；空 = 用默认种子
    retrieval_top_n: int = 20           # 每路召回候选上限（融合前）
    min_top1_score: float = 0.015       # hard-gate 阈值：Top-1 final_score 低于此值 → 转人工
                                         # 默认值由 ~113 条 FAQ 真实分布校准：正常命中 0.05-0.20，
                                         # 话题外噪声 0.00-0.01。0.015 为分隔点。


class MarketingImageConfig(BaseModel):
    """营销生图技能的可配置默认值（附通用默认）。"""

    default_platform: str = "xiaohongshu"
    default_style: str = "literary"
    default_backend: str = "auto"  # mock | allin_api | auto（auto=有 key 用真实,无 key 用 mock）
    default_negative_prompt: str = "no text, no watermark, no blurry, no distorted faces"
    credit_per_image: int = 1
    max_variants: int = 4
    platform_aspect_ratio: dict[str, str] = Field(
        default_factory=lambda: {
            "wechat_moment": "1:1",
            "xiaohongshu": "3:4",
            "douyin": "9:16",
            "taobao_main": "1:1",
            "taobao_detail": "3:4",
            "banner": "16:9",
            "weibo": "1:1",
            "video_cover": "16:9",
            "generic": "1:1",
        },
    )
    platform_pixel_hint: dict[str, tuple[int, int]] = Field(
        default_factory=lambda: {
            "wechat_moment": (1080, 1080),
            "xiaohongshu": (1080, 1440),
            "douyin": (1080, 1920),
            "taobao_main": (1200, 1200),
            "taobao_detail": (750, 1000),
            "banner": (1920, 822),
            "weibo": (1080, 1080),
            "video_cover": (1920, 1080),
            "generic": (1024, 1024),
        },
    )

    # --- allin-api 后端 (PR #5) ---
    allin_api_base: str = "https://api.ciyuansky.com"
    allin_api_image_model: str = "MPT-Image-2"
    # 关键安全:API key 仅从环境变量读取,绝不放进 toml/commit。
    # 用户在终端对话初始化时由 Agent 询问,然后由运行环境导出。
    allin_api_key_env: str = "AI_IMAGE_API_KEY"
    allin_api_timeout_s: float = 60.0

    # --- 画面描述提取器 ---
    # auto = 有 key 走 chat 提取,无 key 走 passthrough;passthrough = 总是原文;chat = 总是 chat
    extractor_mode: str = "auto"
    extractor_model: str = "gpt-4o-mini"
    extractor_timeout_s: float = 30.0


class LocalizerConfig(BaseModel):
    """视频本地化（localize_* 5 个技能）的可配置参数（附通用默认值）。

    阈值类（HTTP 超时 / 批量上限 / 成本分界 / TTS 默认 / 字幕策略 / 允许扩展名 / 服务地址）
    全走 config——不带默认值硬编码进函数体。
    """
    # ── 服务地址 / 网络 ──
    api_base: str = "http://localhost:8000"
    api_prefix: str = "/api/v1"
    http_timeout: float = 30.0          # 业务 HTTP 调用超时（秒）
    health_timeout: float = 2.0         # /health 探活超时（≤3s，见 test_localize_failfast）

    # ── 语言默认值 ──
    target_lang_default: str = "en"     # Agent 不传时落到的目标语言
    source_lang_default: str = "zh"     # 同上，源语言
    supported_target_langs: list[str] = Field(
        default_factory=lambda: ["en", "th", "ja", "ko", "es", "fr", "de", "ru"],
    )
    supported_source_langs: list[str] = Field(default_factory=lambda: ["zh"])

    # ── 字幕 / TTS 默认 ──
    tts_default: bool = True            # 默认开启配音
    remove_subtitles_default: bool = True
    remove_subtitles_strategy_default: str = "ocr_erase_redraw"  # v0.3 唯一支持

    # ── 文件预检 ──
    allowed_extensions: list[str] = Field(default_factory=lambda: [".mp4"])

    # ── 阈值（告警 / 档位） ──
    max_videos_per_batch: int = 100     # 超过则自动 chunk
    cost_low_max: int = 20              # 视频数 ≤ 此值 → 成本档「低」
    cost_high_min: int = 100            # 视频数 ≥ 此值 → 成本档「高」
    poll_max_concurrency: int = 8       # 状态查询并发上限
    stall_threshold_seconds: int = 600  # running 任务超过此秒数标 stalled

    # ── v0.3 可交互初始化字段（init_for_user 设置） ──
    tts_voice: str | None = None        # None = 让 VL 按目标语言自动选
    subtitle_font_size: int = 22        # 横屏；竖屏自动 ×0.7
    subtitle_position: str = "bottom_safe"  # 防遮画面
    output_filename_suffix: str = "sub"     # 输出文件名后缀

    # ── 全云流水线（localize_video / voice_clone_enroll） ──
    asr_sample_rate: int = 16000        # 提取音轨采样率（qwen-audio-3.0-asr-flash-streaming 支持 16k）
    ocr_frame_count: int = 5            # 字幕定位离线抽帧数（均匀取样，非逐帧）
    localize_llm_model: str = "LongCat-2.0"  # 翻译模型（Anthropic 兼容协议）
    localize_tts_model: str = "qwen-audio-3.0-tts-flash"  # 配音 TTS 模型
    localize_voice: str = "longanhuan_v3.6"   # 配音音色（预设；空=不配音）


class ContentConfig(BaseModel):
    """内容创作中心（content_* 5 技能）的可配置参数（附通用默认值）。

    安全约定：云密钥只从环境变量读取（*_key_env 只存 env var 名，绝不存明文、
    绝不进 toml / commit）。LLM 走 LongCat（Anthropic 兼容协议），生图走 ciyuansky
    （OpenAI 兼容 /v1/images/generations），热点走聚合 API（DailyHotApi 协议）。
    """

    # ── LLM（LongCat / Anthropic 兼容 /v1/messages）──
    llm_api_base: str = "https://api.longcat.chat/anthropic"
    llm_api_key_env: str = "AI_LLM_API_KEY"
    llm_model: str = "LongCat-2.0"
    llm_max_tokens: int = 4096
    llm_timeout_s: float = 60.0

    # ── 生图（ciyuansky / OpenAI 兼容 /v1/images/generations）──
    image_api_base: str = "https://api.ciyuansky.com"
    image_api_key_env: str = "AI_IMAGE_API_KEY"
    image_model: str = "MPT-Image-2"
    image_timeout_s: float = 60.0
    image_max_variants: int = 4

    # ── 热点（聚合 API，DailyHotApi 协议；小红书/公众号无公开热榜 → 代理平台）──
    hot_topic_api_base: str = "https://api-hot.imsyy.top"
    hot_topic_endpoints: dict[str, str] = Field(
        default_factory=lambda: {
            "xhs": "weibo",      # 小红书无公开热榜 → 微博全网热点代理
            "wechat": "toutiao", # 公众号无公开热榜 → 头条代理
            "douyin": "douyin",  # 抖音真榜
        },
    )
    hot_topic_limit: int = 20
    hot_topic_timeout_s: float = 10.0

    # ── 生成约束 ──
    max_ideas: int = 6          # 思路设计上限
    max_tags: int = 6           # 文案标签上限
    max_copy_length: int = 2000 # 文案正文上限
    audit_llm_enabled: bool = True  # 审计是否启用 LLM 复核（规则扫描始终执行）


class OrchestratorConfig(BaseModel):
    """A2A 编排器配置。"""
    llm_key_env: str = "AI_LLM_API_KEY"
    llm_base_url: str = "https://api.longcat.chat/anthropic"
    llm_model: str = "LongCat-2.0"
    max_plan_steps: int = 5
    max_retries_per_step: int = 1
    enable_streaming: bool = True


class KeywordTrendConfig(BaseModel):
    """B端关键词趋势榜单（b2b_keyword_trends / b2b_longtail_keywords）可配置参数。

    - tiktok 默认走 TikHub 第三方 API（tiktok_trend_source="tikhub"），
      服务端代抓 Creative Center 热门话题榜，无需 cookie/浏览器即给全量榜单；
      API Key 只从环境变量 AI_TRENDS_API_KEY 读取，绝不进 toml / commit。
      大陆环境 api_base 用加速域名 https://api.tikhub.dev，海外用 https://api.tikhub.io。
      旧自建三级降级路径保留为回退（tiktok_trend_source="cc_scraper"）。
    - instagram 同样默认走 TikHub（instagram_trend_source="tikhub"）：
      话题搜索端点（关键词 → 话题榜，heat=media_count），无需登录会话；
      旧 IG 网页会话直连保留为回退（instagram_trend_source="self_host"）。
    - alibaba → TOP 热销词统计。
    登录会话由调用方经工具参数注入，绝不进 toml / commit。
    """
    trend_timeout_s: float = 15.0
    default_country: str = "US"

    # ── 数据源选择：tikhub（默认）| 旧自建回退（tiktok: cc_scraper / instagram: self_host） ──
    tiktok_trend_source: str = "tikhub"
    instagram_trend_source: str = "tikhub"

    # ── TikHub API（docs.tikhub.io；key 只走环境变量） ──
    # 大陆默认加速域名 .dev（直连免代理）；海外部署改为 https://api.tikhub.io
    tikhub_api_base: str = "https://api.tikhub.dev"
    tikhub_key_env: str = "AI_TRENDS_API_KEY"
    tikhub_timeout_s: float = 30.0
    tikhub_max_pages: int = 5          # 分页聚合上限（每页 20，最多 100 条）

    # ── TikHub 响应磁盘缓存（_tikhub_cache）：默认保守，学习到免费窗才投机 ──
    # soft_ttl 内直回本地缓存（零成本）；过期后真实外呼（默认计费）；
    # 仅当端点从响应头学习到 TikHub 服务端免费缓存窗口时，soft_ttl 过期后的
    # 外呼升级 speculative（大概率免费命中对方缓存，白拿最新数据）。
    tikhub_cache_enabled: bool = True
    tikhub_cache_soft_ttl_s: float = 1800.0     # 30 分钟
    tikhub_cache_max_window_s: float = 21600.0  # 学习窗口上限 6h
    tikhub_cache_db_path: str = ""              # 空 = <cwd>/.cache/tikhub_cache.db

    # ── TikTok Creative Center 自建抓取（回退路径；旧 URL 已 301 到 TikTok One） ──
    cc_scrape_page_url: str = "https://ads.tiktok.com/creative/creativeCenter/trends/hashtag"
    cc_scrape_period_days: int = 7
    cc_scrape_timeout_s: float = 90.0
    cc_scrape_headless: bool = True
    # 海外出口代理（可选，直连被风控时使用）：http://host:port / socks5://host:port
    cc_scrape_proxy: str = ""

    seed_keywords: dict[str, list[dict]] = Field(default_factory=dict)

    # ── 长尾词生成（复用 LongCat LLM，走 _llm_client）──
    longtail_llm_model: str = "LongCat-2.0"
    max_longtail: int = 50


class AlibabaConfig(BaseModel):
    """阿里国际站开放 API（alibaba_* 技能）配置。

    走 TOP 协议 + OAuth 授权。AppKey/Secret/Session 只从环境变量读取，
    绝不进 toml / commit。未授权时 alibaba_* 技能返回结构化 degraded。
    """
    api_base: str = "https://eco.taobao.com/router/rest"
    app_key_env: str = "ALIBABA_APP_KEY"
    app_secret_env: str = "ALIBABA_APP_SECRET"
    session_env: str = "ALIBABA_SESSION"
    sign_method: str = "hmac"
    timeout_s: float = 20.0
    token_url: str = "https://oauth.alibaba.com/token"
    # 国际站 Listing 字段硬约束（运营提供字段规则前的通用默认，见 _alibaba_client.py）
    title_max_len: int = 128
    listing_rules: list[str] = Field(
        default_factory=lambda: [
            "标题 ≤ 128 字符，核心关键词前置",
            "禁用特殊符号：& | # * % 及全角符号（（））等",
            "卖点结构：核心词 + 属性 + 用途 + 场景，每条卖点 ≤ 80 字，最多 6 条卖点",
            "关键词 ≤ 3 个，单个关键词 ≤ 50 字符，关键词之间用英文逗号分隔",
            "详情页说明 ≥ 500 字符，突出采购场景、尺寸、材质、包装、售后",
            "FOB 价格必须为数字（保留两位小数），支持区间格式 X.XX - Y.YY",
            "MOQ（最小起订量）≥ 1，且为整数；支持阶梯价展示",
            "类目 ID（category_id）必填；如未提供请先在国际站后台匹配叶子类目再发布",
            "主图 ≥ 800×800 像素，白底或浅底，PNG/JPG，建议 5 张主图，首图无 Logo 水印",
            "组 ID（group_id）可空，填写后可在产品组批量管理",
        ],
    )


class ImageSkillConfig(BaseModel):
    """生图 skill 化（image_prompt_reverse）配置。

    提示词反推走视觉 LLM（复用 LongCat Anthropic 兼容 /v1/messages + image 块）。
    """
    reverse_prompt_api_base: str = "https://api.longcat.chat/anthropic"
    reverse_prompt_key_env: str = "AI_LLM_API_KEY"
    reverse_prompt_model: str = "LongCat-2.0"
    reverse_prompt_timeout_s: float = 45.0


class B2bPushConfig(BaseModel):
    """B端每日推送（b2b_push_feishu/wecom / b2b_daily_digest）可配置参数。

    webhook URL 只从环境变量读取（*_env 只存 env var 名，绝不存明文、
    绝不进 toml / commit）；入参显式传 webhook_url 优先于 env（供「测试推送」即时校验）。
    """
    feishu_webhook_url_env: str = "FEISHU_WEBHOOK_URL"
    wecom_webhook_url_env: str = "WECOM_WEBHOOK_URL"
    webhook_timeout_s: float = 10.0


class WechatPublishConfig(BaseModel):
    """微信公众号发布（content_wechat_publish / _wechat_client）可配置参数。

    凭证只从环境变量读取（*_env 只存 env var 名，绝不存明文、绝不进 toml / commit）。
    """
    app_id_env: str = "WECHAT_APP_ID"
    app_secret_env: str = "WECHAT_APP_SECRET"
    api_base: str = "https://api.weixin.qq.com/cgi-bin"
    timeout_s: float = 30.0


class CrawlerConfig(BaseModel):
    """智能爬虫套件（content_web_fetch / crawler_sentiment / crawler_deadlink）可配置参数。"""
    timeout_s: float = 15.0
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
    max_links_per_check: int = 30      # 死链检测单次上限（防滥用）
    max_concurrent: int = 10           # 死链检测并发数


class FlowmindConfig(BaseModel):
    """FlowMind 总配置：每技能一段。"""
    inventory: InventoryConfig = Field(default_factory=InventoryConfig)
    feishu_kb: FeishuKbConfig = Field(default_factory=FeishuKbConfig)
    marketing_image: MarketingImageConfig = Field(default_factory=MarketingImageConfig)
    localizer: LocalizerConfig = Field(default_factory=LocalizerConfig)
    content: ContentConfig = Field(default_factory=ContentConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    keyword_trend: KeywordTrendConfig = Field(default_factory=KeywordTrendConfig)
    alibaba: AlibabaConfig = Field(default_factory=AlibabaConfig)
    image_skill: ImageSkillConfig = Field(default_factory=ImageSkillConfig)
    b2b_push: B2bPushConfig = Field(default_factory=B2bPushConfig)
    wechat_publish: WechatPublishConfig = Field(default_factory=WechatPublishConfig)
    crawler: CrawlerConfig = Field(default_factory=CrawlerConfig)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> FlowmindConfig:
    """读取配置文件；不存在则全用通用默认。用户值覆盖默认，缺项回落默认。"""
    if not path.exists():
        return FlowmindConfig()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return FlowmindConfig.model_validate(data)


# 单例缓存：避免每次调用都重读磁盘 + 解析 TOML。
# 调用 init_for_user / save_config 后用 reload_config() 强制失效。
_cached_config: FlowmindConfig | None = None


def get_config() -> FlowmindConfig:
    """返回缓存的 FlowmindConfig；首次调用从磁盘加载。"""
    global _cached_config
    if _cached_config is None:
        _cached_config = load_config()
    return _cached_config


def reload_config() -> FlowmindConfig:
    """强制从磁盘重读，清空缓存。"""
    global _cached_config
    _cached_config = None
    return get_config()


def _tomlify(obj):
    """递归把 dict/list 里的 tuple 转 list（TOML 不支持 tuple）。

    MarketingImageConfig.platform_pixel_hint 用 tuple[int, int]；
    model_dump 后是 Python repr，tomli_w.dumps 写出来 TOML 反序列化会丢类型。
    """
    if isinstance(obj, dict):
        return {k: _tomlify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tomlify(v) for v in obj]
    return obj


def save_config(cfg: FlowmindConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    """把配置写回 TOML 文件（供初始化对话调用）。

    v0.3 修复：
    - model_dump(exclude_none=True) 剔除 None（TOML 不支持 null）
    - _tomlify 把 tuple 转 list（TOML 不支持 tuple 类型）
    """
    dumped = _tomlify(cfg.model_dump(exclude_none=True))
    path.write_text(tomli_w.dumps(dumped), encoding="utf-8")


def init_for_user(
    target_lang: str,
    source_lang: str = "zh",
    enable_tts: bool = True,
    remove_subtitles: bool = True,
    remove_subtitles_strategy: str = "ocr_erase_redraw",
    tts_voice: str | None = None,
    subtitle_font_size: int | None = None,
    subtitle_position: str | None = None,
    output_filename_suffix: str | None = None,
    save_path: Path | None = None,
) -> FlowmindConfig:
    """可交互式初始化：一键设全 localizer 偏好，写入 flowmind.config.toml。

    调用后所有后续 `invoke("localize_*", ...)` 自动应用这套偏好，不用每次传。
    None 参数视为「不覆盖」（保留现有值或 config 默认）。

    想要对话式分步引导（适合 Agent 引导用户）？用 `flowmind.interactive.run_interactive_init()`。
    """
    target = save_path or DEFAULT_CONFIG_PATH
    # 读现 config（从 target 路径，便于 save_path 一致性）
    if target.exists():
        cfg = FlowmindConfig.model_validate(tomllib.loads(target.read_text(encoding="utf-8")))
    else:
        cfg = FlowmindConfig()
    overrides = {
        "target_lang_default": target_lang,
        "source_lang": source_lang,
        "tts_default": enable_tts,
        "remove_subtitles_default": remove_subtitles,
        "remove_subtitles_strategy_default": remove_subtitles_strategy,
        "tts_voice": tts_voice,
        "subtitle_font_size": subtitle_font_size,
        "subtitle_position": subtitle_position,
        "output_filename_suffix": output_filename_suffix,
    }
    non_none = {k: v for k, v in overrides.items() if v is not None}
    if source_lang is not None:
        non_none["source_lang_default"] = source_lang
        non_none.pop("source_lang", None)
    cfg.localizer = cfg.localizer.model_copy(update=non_none)
    save_config(cfg, target)
    # 强制 reload 走相同 path（避免默认路径污染）
    global _cached_config
    _cached_config = None
    _cached_config = FlowmindConfig.model_validate(tomllib.loads(target.read_text(encoding="utf-8")))
    return _cached_config


def is_initialized(path: Path = DEFAULT_CONFIG_PATH) -> bool:
    """判断用户是否已完成个性化初始化。"""
    return path.exists()
