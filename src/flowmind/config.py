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
    chunk_size: int = 400               # 切块字符数（占位用）
    chunk_overlap: int = 60             # 切块重叠（占位用）
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
    allin_api_base: str = "https://allin-api.com"
    allin_api_image_model: str = "gpt-image-2"
    # 关键安全:API key 仅从环境变量读取,绝不放进 toml/commit。
    # 用户在终端对话初始化时由 Agent 询问,然后由运行环境导出。
    allin_api_key_env: str = "ALLIN_API_KEY"
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


class FlowmindConfig(BaseModel):
    """FlowMind 总配置：每技能一段。"""
    inventory: InventoryConfig = Field(default_factory=InventoryConfig)
    feishu_kb: FeishuKbConfig = Field(default_factory=FeishuKbConfig)
    marketing_image: MarketingImageConfig = Field(default_factory=MarketingImageConfig)
    localizer: LocalizerConfig = Field(default_factory=LocalizerConfig)


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
