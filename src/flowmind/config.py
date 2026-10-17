"""配置层：技能内置通用默认，用户配置文件可覆盖。

个性化定制只发生在终端用户的对话式初始化——由消费此包的 Agent
按 README 剧本引导用户，调用 save_config() 写出 flowmind.config.toml。
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

import tomli_w
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationInfo, field_validator

DEFAULT_CONFIG_PATH = Path("flowmind.config.toml")  # 相对于当前工作目录（cwd）

# 密钥/环境文件落点（与 skills/_secrets.py 同口径）：__file__ = <repo>/src/flowmind/config.py
#   parents[0]=flowmind [1]=src [2]=<repo> [3]=工作区根
_PROJECT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
_WORKSPACE_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"

_env_loaded = False


def ensure_env_loaded() -> None:
    """把工作区根 ``.env`` 与项目 ``.env`` 载入进程环境（幂等）。

    先工作区根、后项目；``load_dotenv`` 不覆盖已存在变量，故根优先。
    权威实现放在 config.py（各 Config 要按「**env 优先，config 兜底**」解析字段，
    必须先有 env）；``skills/_secrets.py`` 复用本函数，避免两处漂移。
    """
    global _env_loaded
    if _env_loaded:
        return
    load_dotenv(_WORKSPACE_ENV_PATH)
    load_dotenv(_PROJECT_ENV_PATH)
    _env_loaded = True


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

    # ── 任务引擎（TaskManager；2026-09-08 从 9b45745 恢复——config 重构丢失，
    #    导致 TaskManager 初始化必 AttributeError、任务引擎整体不可用）──
    data_dir: str = "~/flowmind-data"   # 任务工作目录基准（~ 由 manager 展开为绝对路径）
    max_pending_tasks: int = 100        # 待处理任务上限（超出拒绝受理，背压）
    task_ttl_seconds: int = 3600        # 终态任务保留时长（秒），超时回收

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

    # ── 热榜引擎多榜（content_hot_boards）：榜型 → DailyHotApi 端点映射 ──
    # 4 榜型：general 综合 / vertical 垂类 / topic 话题 / inspiration 灵感
    # 小红书/公众号无公开对应榜，用真实聚合 API 的可代理端点；端点名随源可用性调整。
    hot_board_endpoints: dict[str, str] = Field(
        default_factory=lambda: {
            "general": "thepaper",      # 综合 → 澎湃热榜
            "vertical": "douyin",       # 垂类 → 抖音热点
            "topic": "toutiao",         # 话题 → 头条热榜
            "inspiration": "baidu",     # 灵感 → 百度热搜
        },
    )
    hot_board_labels: dict[str, str] = Field(
        default_factory=lambda: {
            "general": "综合热榜",
            "vertical": "垂类热榜",
            "topic": "话题热榜",
            "inspiration": "灵感热榜",
        },
    )

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


# dsp 非密钥配置项的 env 名（「env 优先，config 兜底」；密钥不在此，走 *_key_env + _secrets）
_DSP_ENV_FIELDS: dict[str, str] = {
    "storage_endpoint": "FLOWMIND_DSP_S3_ENDPOINT",
    "storage_bucket": "FLOWMIND_DSP_S3_BUCKET",
    "storage_region": "FLOWMIND_DSP_S3_REGION",
    # 公网预签名端点（项目 I2V 前置条件）：内部端点只给本仓读写；厂商拉素材必须用这条。
    # 未配置时**只有内部端点字面公网**才沿用它，否则如实跳过图生视频（不冒充公网可达）。
    "storage_public_endpoint": "FLOWMIND_DSP_S3_PUBLIC_ENDPOINT",
    "storage_public_addressing_style": "FLOWMIND_DSP_S3_PUBLIC_ADDRESSING_STYLE",
    "mediakit_api_base": "FLOWMIND_DSP_MEDIAKIT_API_BASE",
    "mediakit_queue_id": "FLOWMIND_DSP_MEDIAKIT_QUEUE_ID",
    "mediakit_output_destination": "FLOWMIND_DSP_MEDIAKIT_OUTPUT_DESTINATION",
    "mediakit_transition_region": "FLOWMIND_DSP_MEDIAKIT_TRANSITION_REGION",
    "tts_api_base": "FLOWMIND_DSP_TTS_API_BASE",
    "tts_default_speaker": "FLOWMIND_DSP_TTS_SPEAKER",
    "timeline_silent_shot_ms": "FLOWMIND_DSP_TIMELINE_SILENT_SHOT_MS",
    "timeline_shot_min_ms": "FLOWMIND_DSP_TIMELINE_SHOT_MIN_MS",
    "timeline_shot_max_ms": "FLOWMIND_DSP_TIMELINE_SHOT_MAX_MS",
    "timeline_shot_limit_min": "FLOWMIND_DSP_TIMELINE_SHOT_LIMIT_MIN",
    "timeline_shot_limit_max": "FLOWMIND_DSP_TIMELINE_SHOT_LIMIT_MAX",
    "videogen_api_base": "FLOWMIND_DSP_VIDEOGEN_API_BASE",
    "videogen_model": "FLOWMIND_DSP_VIDEOGEN_MODEL",
    "videogen_aspect_ratio": "FLOWMIND_DSP_VIDEOGEN_ASPECT_RATIO",
    "videogen_size": "FLOWMIND_DSP_VIDEOGEN_SIZE",
    "videogen_quality": "FLOWMIND_DSP_VIDEOGEN_QUALITY",
}


class DspConfig(BaseModel):
    """dsp · 抖音成片技能套件（8 技能）的可配置参数（附通用默认值）。

    契约真源（cross-dashboard 侧）：
    ``docs/architecture/2026-10-11-dsp-backend-skill-contract.md``
    （§2 共同契约 / §3 逐技能 / §7 MediaKit 参数级映射）。
    厂商接口字段已按 ``docs/vendor/`` 的**入仓真文档**逐条核实。

    安全约定（同 MarketingImageConfig / ContentConfig）：云厂商密钥只从环境变量
    读取——``*_key_env`` 只存 **env var 名**，绝不存明文、绝不进 toml / commit；
    未配置时相关技能**如实降级或抛结构化错误**，绝不返回假产物。

    配置源（全仓库统一「**env 优先，config 兜底**」）：端点/桶/队列等**非密钥**项
    经下方 ``_env_strings`` / ``_env_secure`` 解析 env（见 ``_DSP_ENV_FIELDS``），
    ``flowmind.config.toml`` 与内置默认兜底。

    部署：本族以独立 backend 注册（``prefix=dsp``，网关聚合出 ``dsp__{tool}``），
    见 ``skills/__init__.py`` 的族筛选与 ``FLOWMIND_SKILL_FAMILIES``。
    """

    # ── 对象存储（S3/MinIO 兼容；契约 §2.4「目录 = 云存储 prefix」）──
    # 选 S3/MinIO 而非火山 TOS：回传的 key 必须能被 cross-dashboard
    # ``/api/assets/<key>`` 按**当前租户段**取到（其 files.ts fail-closed 只认本租户段）。
    storage_endpoint: str = Field(default="", validate_default=True)   # env FLOWMIND_DSP_S3_ENDPOINT
    storage_region: str = Field(default="", validate_default=True)     # env FLOWMIND_DSP_S3_REGION
    storage_bucket: str = Field(default="flowmind", validate_default=True)  # env FLOWMIND_DSP_S3_BUCKET
    storage_access_key_env: str = "S3_ACCESS_KEY"
    storage_secret_key_env: str = "S3_SECRET_KEY"
    storage_secure: bool = Field(default=True, validate_default=True)  # env FLOWMIND_DSP_S3_SECURE
    storage_timeout_s: float = 30.0
    presign_ttl_s: int = 3600             # 预签名 URL 有效期（供 MediaKit 取素材）

    # ── **公网**预签名端点（图生视频参考素材，ADR-123）──
    # 内部端点（storage_endpoint）只给本仓读写；给第三方厂商的预签名 URL 必须由
    # **公网端点**签发——SigV4 把 Host 签进 URL，签完再改 hostname 必然 403。
    # 未配置时：内部端点**字面公网**才沿用它；本机/内网端点一律视为「未配置公网入口」
    # （技能逐镜跳过并回落静图，绝不冒充已具备公网访问能力）。
    storage_public_endpoint: str = Field(  # env FLOWMIND_DSP_S3_PUBLIC_ENDPOINT
        default="", validate_default=True)
    storage_public_secure: bool = Field(  # env FLOWMIND_DSP_S3_PUBLIC_SECURE
        default=True, validate_default=True)
    # 公网端点若是虚拟主机风格（bucket.s3.example.com）设 `virtual`；自建 MinIO 反代通常是 `path`。
    storage_public_addressing_style: str = Field(  # env FLOWMIND_DSP_S3_PUBLIC_ADDRESSING_STYLE
        default="path", validate_default=True)

    # ── 火山 AI MediaKit「多轨道剪辑」（render_compose 的实现落点 · §7）──
    mediakit_api_base: str = Field(  # env FLOWMIND_DSP_MEDIAKIT_API_BASE
        default="https://mediakit.cn-beijing.volces.com", validate_default=True)
    mediakit_key_env: str = "DSP_MEDIAKIT_API_KEY"   # Bearer；集群 Secret，密钥只在后端
    mediakit_multi_track_path: str = "/api/v1/tools/multi-track-edit"
    mediakit_camera_motion_path: str = "/api/v1/tools/apply-camera-motion"  # 运镜（§7.5 第 2 步）
    mediakit_task_path: str = "/api/v1/tasks/{task_id}"
    # **本地上传位**（契约 §7.3① 的另一支）：申请预签名 PUT 后把原始字节直传 MediaKit，
    # 任务里用 ``mediakit://{file_id}`` 引用 ⇒ 素材**不必公网可达**。
    mediakit_request_upload_path: str = "/api/v1/tools-sync/request-media-upload-url"
    mediakit_upload_timeout_s: float = 300.0   # 素材 PUT（原始字节）可能很大/很慢，单独给上限
    mediakit_timeout_s: float = 60.0       # 单次 HTTP（提交 / 查询）
    mediakit_poll_interval_s: float = 5.0
    mediakit_poll_max_s: float = 1800.0    # 轮询上限（超时如实报错，不假装完成）
    mediakit_canvas_width: int = 1080      # 9:16
    mediakit_canvas_height: int = 1920
    mediakit_output_format: str = "mp4"
    mediakit_output_fps: int = 30
    mediakit_queue_id: str = Field(default="", validate_default=True)  # env FLOWMIND_DSP_MEDIAKIT_QUEUE_ID
    mediakit_aigc_label: str = "1"         # §7.3④ AIGC 标识（"1" = 含 AI 生成内容）
    # 转场资源 ID 的**区域**（ADR-080）：厂商给的是**两张 ID 表**（华北 / 柔佛），同一效果两个号。
    # 端点默认是 ``mediakit.cn-beijing``（华北）⇒ 缺省 ``cn-north``；换区域只改这个值（表里两个号都在）。
    mediakit_transition_region: str = Field(default="cn-north", validate_default=True)
    mediakit_output_destination: str = Field(default="", validate_default=True)  # env FLOWMIND_DSP_MEDIAKIT_OUTPUT_DESTINATION

    # ── 火山豆包语音「音频生成」（tts_generate · §3.1）──
    tts_api_base: str = Field(  # env FLOWMIND_DSP_TTS_API_BASE
        default="https://openspeech.bytedance.com", validate_default=True)
    tts_create_path: str = "/api/v3/tts/create"
    tts_key_env: str = "DSP_TTS_API_KEY"   # 新版控制台 X-Api-Key 单头
    tts_model: str = "seed-audio-1.0"
    tts_default_speaker: str = Field(default="", validate_default=True)  # env FLOWMIND_DSP_TTS_SPEAKER
    tts_format: str = "mp3"                # 契约要 mp3（文档：mp3 默认采样率 44100）
    tts_sample_rate: int = 24000           # 文档允许集 8000/16000/24000/32000/40000/44100/48000
    tts_speech_rate: int = 0               # [-50, 100]；-50=0.5x，100=2.0x
    tts_loudness_rate: int = 0             # [-50, 100]（文档 audio_config.loudness_rate）
    tts_pitch_rate: int = 0                # [-12, 12]（文档 audio_config.pitch_rate）
    tts_timeout_s: float = 120.0           # 单次合成上限（文档：单次最长 120s）
    tts_max_chars: int = 3000              # text_prompt ≤ 3000 字符
    tts_enable_subtitle: bool = True       # 取 voSegments（句级 / 词级时间戳）
    tts_aigc_watermark: bool = False       # watermark.aigc_watermark（显式标识，按合规口径定）

    # ── 时间线对齐（timeline_build · 契约 §3.2，ADR-069）──
    # 无口播镜（配音句子数 < 镜头数，或该镜无文案）的展示时长（ms；env FLOWMIND_DSP_TIMELINE_SILENT_SHOT_MS）。
    # **显式常量、非实测**——`_dsp_align` 用它兜底并在 warnings 里如实说明，绝不编造实测值。
    timeline_silent_shot_ms: int = Field(default=1500, validate_default=True)

    # ── 切镜节奏收口（timeline_build · ADR-070 / ADR-072）──
    # 在**实测**轴上把分镜并/拆成终版镜头：过短的镜并入相邻镜、过长的镜在其内部一分为二
    # （**先句边界、句内没有句边界就下沉到词边界**，切点恒为实测 token）。
    # 与 cross-dashboard 的 lib/content/dsp-cut.ts **同值**（改口径两边一起改，同 silent_shot_ms 先例）。
    timeline_shot_min_ms: int = Field(default=1500, validate_default=True)
    timeline_shot_max_ms: int = Field(default=6000, validate_default=True)
    # 收口后镜头数的合法区间；越界**只告警不夹取**（数据诚实）。
    timeline_shot_limit_min: int = Field(default=2, validate_default=True)
    timeline_shot_limit_max: int = Field(default=10, validate_default=True)

    # ── 图生视频 · 视频生成（image_to_video 的实现落点 · 契约 §7.5 第 1 步）──
    # 厂商 = 晨星 AI（api.chenxingai.vip，第三方聚合，转售 Seedance / MiniMax 等）。
    # **2026-10-17 真调更正**（见 ADR-111）：
    #   ① 视频接口在 ``/v1`` 下且路径是**单数** ``video``：``POST /v1/video/generations``；
    #      旧记的 ``/videos/generations`` 现在返**前端 SPA 的 HTML**（200 text/html），
    #      ``/v1/videos/generations`` 也是 404 —— 前者曾被误判成「提交返回非 JSON」。
    #   ② 轮询 ``GET /v1/videos/{task_id}``（平铺 ``{task_id,status,progress,metadata.url}``；
    #      ``/v1/video/generations/{task_id}`` 亦可，但返回包一层 ``data``）。
    #   ③ 默认模型 ``seedance-2.5-global-sp`` 在该渠道视频路由上被上游判 ``invalid_request``；
    #      真跑可用的视频模型是 ``doubao-seedance-2-0-fast``（同 key、同端点，200 → queued）。
    #   ④ 请求体字段：``model`` / ``prompt`` / ``duration`` / ``size`` / ``image``；
    #      ``aspect_ratio`` / ``quality`` **不再注入**（非该接口字段，避免上游 invalid_request）。
    #   ⑤ **硬约束**：参考素材必须是**公网可访问的 http(s) URL** 或**已入库的 ``asset://``**；
    #      base64 / ``mediakit://`` / 本机与内网地址一律被拒（真跑逐条验证，见 ADR-111）。
    # ⚠️ 这是**独立第三方**，与火山 MediaKit / 豆包语音不共享端点与密钥。
    videogen_api_base: str = Field(  # env FLOWMIND_DSP_VIDEOGEN_API_BASE
        default="https://api.chenxingai.vip", validate_default=True)
    videogen_key_env: str = "DSP_VIDEOGEN_API_KEY"   # Authorization: Bearer sk-…（密钥只在后端）
    videogen_submit_path: str = "/v1/video/generations"
    videogen_task_path: str = "/v1/videos/{task_id}"
    videogen_content_path: str = "/v1/videos/{task_id}/content"   # 带鉴权代理下载（备用）
    videogen_timeout_s: float = 60.0        # 单次 HTTP
    videogen_poll_interval_s: float = 8.0   # 实测 5–10s 合适
    videogen_poll_max_s: float = 1800.0     # 轮询上限（超时如实报错，不假装完成）
    videogen_model: str = Field(  # env FLOWMIND_DSP_VIDEOGEN_MODEL
        default="doubao-seedance-2-0-fast", validate_default=True)
    videogen_aspect_ratio: str = Field(  # env FLOWMIND_DSP_VIDEOGEN_ASPECT_RATIO
        default="9:16", validate_default=True)   # **不进请求体**（比例由 size 表达）
    # 文档可选 size = 1280x720 / 720x1280 / 1024x1024 / 1792x1024 / 1024x1792
    # —— **无 1080x1920**；9:16 里只有 720x1280 是精确比例，成片由 MediaKit 缩到画布。
    videogen_size: str = Field(  # env FLOWMIND_DSP_VIDEOGEN_SIZE
        default="720x1280", validate_default=True)
    videogen_quality: str = Field(  # env FLOWMIND_DSP_VIDEOGEN_QUALITY
        default="standard", validate_default=True)  # **不进请求体**（该接口无此字段）
    videogen_max_ref_images: int = 9        # 文档：seedance 系列参考图 ≤ 9 张
    videogen_prompt_suffix: str = ""        # 逐镜 prompt 后缀（如统一加「运镜」提示）
    # ── 首尾帧桥（ADR-074；env FLOWMIND_DSP_VIDEOGEN_BRIDGE_NEXT_IMAGE=0 关）──
    # 逐镜把**下一镜的图**当**尾帧**下发（放 ``images`` 数组第 2 位），让相邻镜在画面里首尾咬合
    # ——这是「厂商转场资源 ID 未登记 ⇒ 一律硬切」的**更根本解法**（不依赖资源 ID 登记）。
    # ⚠️ 聚合商请求字段里**没有**指南首尾帧用的 ``content.role``——本实现是**该接口面上最接近的落法**，
    #    语义**待真跑验证**；若聚合商按「多参考图」处理，效果退化为**无桥**（不产生错误画面）。
    videogen_bridge_next_image: bool = Field(default=True, validate_default=True)
    # ── 下发前按输出分辨率降采样（ADR-074；env FLOWMIND_DSP_VIDEOGEN_DOWNSCALE_INPUT=0 关）──
    # 指南 FAQ：「输入图片**不超过输出视频分辨率**」（超标画面易出指纹状纹理；输出 1080p 概率低于 720p）。
    # 素材图（补生 ``1152x2048``）是**静图主轨**口径（长边 ≥2048，缩到画布才不虚）——与 I2V 输入是
    # **同一张图的两个用途**，故在**输入侧**等比缩到 ``videogen_size`` 以内，**不动补生尺寸常量**。
    videogen_downscale_input: bool = Field(default=True, validate_default=True)
    # ── 出片后**本地裁到镜头窗口**（ADR-080 阶段 5；env FLOWMIND_DSP_VIDEOGEN_TRIM_WINDOW=0 关）──
    # 图生视频模型按「建议时长附近」出片，与 timeline 锁定的镜头窗口**不是同一个数**：多出来的会顶到
    # 下一镜（画面多停一截）、不足的会让该镜提前卡住。故在转存前用 ffmpeg 把每镜裁到窗口。
    # 关掉 ⇒ 片段原样入库（``trim`` 实况按 0 上报，前端剪辑记录如实标「未裁切」）。
    # 裁切失败（ffmpeg 缺失 / 编码器缺失）**回落原片段 + 如实告警**，不丢镜。
    videogen_trim_window: bool = Field(default=True, validate_default=True)
    # ── 不足时**末帧定格**补齐（ADR-080；env FLOWMIND_DSP_VIDEOGEN_FREEZE_PAD=0 关）──
    # 源片段比窗口短 ⇒ ``tpad=stop_mode=clone`` 把末帧拖满窗口（画面在结尾静止一拍）。
    # 关掉 ⇒ **不补**（该镜比时间线短），但 ``shortfallMs`` **照实报出**——补不补是策略，短不短是事实。
    videogen_freeze_pad: bool = Field(default=True, validate_default=True)
    # ── 提交前**公网可达性预检**（ADR-123；env FLOWMIND_DSP_VIDEOGEN_PUBLIC_PROBE=0 关）──
    # 用公网预签名 URL 真读 1 字节（Range: bytes=0-0）：能提前发现「签名与公网 Host 不匹配 /
    # 反代没透传 Host / 公网入口没通」，**不浪费按镜计费的厂商提交**。只证明「后端此刻读得到」，
    # 不承诺厂商网络一定能读——所以仍保留逐镜回落与如实状态。
    videogen_public_probe: bool = Field(default=True, validate_default=True)

    @field_validator(*_DSP_ENV_FIELDS, mode="before")
    @classmethod
    def _env_strings(cls, v: str, info: ValidationInfo) -> str:
        """env 覆盖非密钥字段（空值不生效，回落 toml / 内置默认）。"""
        ensure_env_loaded()
        env = os.environ.get(_DSP_ENV_FIELDS[info.field_name], "").strip()
        return env or v

    @field_validator("storage_secure", mode="before")
    @classmethod
    def _env_secure(cls, v: bool) -> bool:
        """env FLOWMIND_DSP_S3_SECURE 覆盖（0/false/no 关；其余不生效）。"""
        ensure_env_loaded()
        raw = os.environ.get("FLOWMIND_DSP_S3_SECURE", "").strip().lower()
        if raw in ("0", "false", "no"):
            return False
        if raw in ("1", "true", "yes"):
            return True
        return v

    @field_validator("storage_public_secure", mode="before")
    @classmethod
    def _env_public_secure(cls, v: bool) -> bool:
        """env FLOWMIND_DSP_S3_PUBLIC_SECURE 覆盖（0/false/no 关；其余不生效）。

        公网端点通常带 scheme（``https://…``）：带 scheme 时本开关不参与拼接；
        只有 ``S3_PUBLIC_ENDPOINT=host:port`` 这种写法才用它决定 http/https。
        """
        ensure_env_loaded()
        raw = os.environ.get("FLOWMIND_DSP_S3_PUBLIC_SECURE", "").strip().lower()
        if raw in ("0", "false", "no"):
            return False
        if raw in ("1", "true", "yes"):
            return True
        return v

    @field_validator(
        "videogen_bridge_next_image", "videogen_downscale_input",
        "videogen_trim_window", "videogen_freeze_pad", "videogen_public_probe",
        mode="before",
    )
    @classmethod
    def _env_videogen_flags(cls, v: bool, info: ValidationInfo) -> bool:
        """I2V 行为开关的 env 覆盖（1/true/yes 开，0/false/no 关；其余不生效）。"""
        ensure_env_loaded()
        env_names = {
            "videogen_bridge_next_image": "FLOWMIND_DSP_VIDEOGEN_BRIDGE_NEXT_IMAGE",
            "videogen_downscale_input": "FLOWMIND_DSP_VIDEOGEN_DOWNSCALE_INPUT",
            "videogen_trim_window": "FLOWMIND_DSP_VIDEOGEN_TRIM_WINDOW",
            "videogen_freeze_pad": "FLOWMIND_DSP_VIDEOGEN_FREEZE_PAD",
            "videogen_public_probe": "FLOWMIND_DSP_VIDEOGEN_PUBLIC_PROBE",
        }
        raw = os.environ.get(env_names[info.field_name], "").strip().lower()
        if raw in ("1", "true", "yes"):
            return True
        if raw in ("0", "false", "no"):
            return False
        return v


class CrawlerConfig(BaseModel):
    """智能爬虫套件（content_web_fetch / crawler_sentiment / crawler_deadlink）可配置参数。"""
    timeout_s: float = 15.0
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
    max_links_per_check: int = 30      # 死链检测单次上限（防滥用）
    max_concurrent: int = 10           # 死链检测并发数


class AuthConfig(BaseModel):
    """调用方凭证校验（ECO-ADR-0011 A 阶段）。

    token 格式的契约真源在生态 ADR §4b（Go/Python/TS 三侧逐字节一致）。
    密钥与 go-kernel 的 ``KERNEL_TRUSTED_SECRET`` 同一值：上游 → 网关 →
    后端交换后的下游 token 均用同一 HMAC 格式（网关不直通客户端凭证）。

    **空 = 鉴权整体关闭**（dev / 独立部署语义：现有 examples/*_demo.py
    与直连用法不受影响，任务不记租户）。非空时所有未豁免路径必须带
    合法 Bearer，且任务访问变为 fail-closed。
    """

    secret: str = Field(default="", validate_default=True)  # env FLOWMIND_AUTH_SECRET

    # 豁免在凭证之外的路径前缀：健康探针（K8s probe）与发现面（manifest）
    # 必须匿名可达，否则集群探针永久失败。
    exempt_prefixes: list[str] = Field(
        default_factory=lambda: ["/api/v1/health", "/api/v1/manifest"])

    @field_validator("secret", mode="before")
    @classmethod
    def _env_secret(cls, v: str) -> str:
        """env FLOWMIND_AUTH_SECRET 覆盖（空值不生效，回落 config/默认）。"""
        return os.environ.get("FLOWMIND_AUTH_SECRET", "").strip() or v


class InfraConfig(BaseModel):
    """任务引擎基础设施配置（tasks 包：PG 存储 / MQTT 事件 / Milvus / BGE 嵌入）。

    配置源决策（2026-09-04，全仓库统一）：**env 优先，config 兜底**——
    解析顺序恒为 ``环境变量 → 本字段 → 模块内置默认``。
    理由：12-factor 风格，容器部署经 env 注入集群 svc 地址即可覆盖（
    集群内注入 http://milvus.agentic.svc:19530 之类），无需改文件；
    flowmind.config.toml 提供开发机/部署默认值。字段留空 = 未配置，
    由各模块回落 env / 内置默认，缺失即显式报错（不静默降级）。

    ⚠ 本段在 2026-09-07 前后的 config 重构中丢失（f62b49f3 可考），导致
    tasks/events・federation・vectors・store・_bge_embed・localize_video 六处
    ``get_config().infra.*`` 全部 AttributeError 靠调用方降级带病运行
    （MQTT 事件与联邦 MQTT 注册通道静默失效）。2026-09-08 逐字恢复。
    """

    # ── PostgreSQL（TaskStore；经 PgBouncer 事务模式）──
    pg_dsn: str = ""                     # 完整连接串；env FLOWMIND_PG_DSN 优先
    pg_db: str = "rak"                   # RAK_PG_* 兜底路径使用的库名（统一库，schema 见 tasks/store.py）

    # ── MQTT（EMQX，任务事件推送 mcp-base-gpu/tasks/{id}/events）──
    mqtt_host: str = ""                  # 空 = 发布器禁用（纯 PG 落库降级）
    mqtt_port: int = 1883                # EMQX 明文端口
    mqtt_use_tls: bool = False           # True 时走 TLS（默认系统 CA）
    mqtt_username: str = ""             # env FLOWMIND_MQTT_USERNAME 优先；空 = 匿名
    mqtt_password: str = ""             # env FLOWMIND_MQTT_PASSWORD 优先；空 = 匿名

    # ── Milvus（字幕分段向量库 localize_segments）──
    milvus_uri: str = ""                 # env FLOWMIND_MILVUS_URI 优先

    # ── BGE 嵌入服务（bge-base-zh-v1.5，768 维；TEI/OpenAI 双形状自适应）──
    embedding_base_url: str = ""         # env FLOWMIND_EMBEDDING_BASE_URL 优先

    # ── 字幕向量化开关（localize_video 流水线尾部增值步骤）──
    vectorize: bool = True               # env FLOWMIND_VECTORIZE（"0"/"false" 关）


class FederationBackendConfig(BaseModel):
    """联邦自注册配置（federation 包：功能包加入 MCP 网关联邦）。"""

    # ── 开关（默认关：不影响现有 demo 与独立部署） ──
    # validate_default=True 必须显式声明：pydantic v2 默认不验证默认值，
    # 不加的话下方 env 覆盖 validator 在用户未传值时永不触发。
    # （本类定义自 f62b49f 恢复：224a194 合并时曾把本段丢失，导致
    #  register.py 读 get_config().federation 抛 AttributeError 被吞成
    #  warning，联邦自注册静默失效。）
    enabled: bool = Field(default=False, validate_default=True)  # env FLOWMIND_FEDERATION_REGISTER（"1"/"true"）

    # ── 注册元数据 ──
    backend_id: str = Field(default="video_localizer", validate_default=True)  # env FLOWMIND_FEDERATION_BACKEND_ID
    prefix: str = Field(default="video_localizer", validate_default=True)      # env FLOWMIND_FEDERATION_PREFIX
    # 本机可达地址；空 = http://127.0.0.1:{port}/mcp 兜底
    # （集群部署注入外部可达地址，如 http://<pod-ip>:8002/mcp）
    url: str = Field(default="", validate_default=True)  # env FLOWMIND_FEDERATION_URL
    heartbeat_interval: float = Field(default=30.0, validate_default=True)  # env FLOWMIND_FEDERATION_HEARTBEAT_INTERVAL

    @field_validator("enabled", mode="before")
    @classmethod
    def _env_enabled(cls, v: bool) -> bool:
        """env FLOWMIND_FEDERATION_REGISTER 覆盖（1/true 开，0/false 关）。"""
        raw = os.environ.get("FLOWMIND_FEDERATION_REGISTER", "").strip().lower()
        if raw in ("1", "true", "yes"):
            return True
        if raw in ("0", "false", "no"):
            return False
        return v

    @field_validator("backend_id", "prefix", "url", mode="before")
    @classmethod
    def _env_strings(cls, v: str, info: ValidationInfo) -> str:
        """env 覆盖三个注册元数据字段（空值不生效，回落 config/默认）。"""
        env_names = {
            "backend_id": "FLOWMIND_FEDERATION_BACKEND_ID",
            "prefix": "FLOWMIND_FEDERATION_PREFIX",
            "url": "FLOWMIND_FEDERATION_URL",
        }
        env = os.environ.get(env_names[info.field_name], "").strip()
        return env or v

    @field_validator("heartbeat_interval", mode="before")
    @classmethod
    def _env_heartbeat_interval(cls, v: float) -> float:
        """env FLOWMIND_FEDERATION_HEARTBEAT_INTERVAL 覆盖（秒，>0）。"""
        raw = os.environ.get("FLOWMIND_FEDERATION_HEARTBEAT_INTERVAL", "").strip()
        if not raw:
            return v
        try:
            val = float(raw)
        except ValueError as exc:
            raise ValueError(
                "FLOWMIND_FEDERATION_HEARTBEAT_INTERVAL 必须是数字: "
                f"{raw!r}") from exc
        return val if val > 0 else v


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
    dsp: DspConfig = Field(default_factory=DspConfig)
    crawler: CrawlerConfig = Field(default_factory=CrawlerConfig)
    infra: InfraConfig = Field(default_factory=InfraConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    federation: FederationBackendConfig = Field(
        default_factory=FederationBackendConfig)


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
