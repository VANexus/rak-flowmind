"""配置层：技能内置通用默认，用户配置文件可覆盖。

mcp-base-gpu 裁剪后仅保留视频本地化链路配置（LocalizerConfig）；
旧 toml 中的无关段落（如 [content]）由 Pydantic v2 默认 extra='ignore' 静默忽略，
无需手工迁移。
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

DEFAULT_CONFIG_PATH = Path("flowmind.config.toml")  # 相对于当前工作目录（cwd）


class LocalizerConfig(BaseModel):
    """视频本地化（localize_* 技能）的可配置参数（附通用默认值）。

    阈值类（HTTP 超时 / 批量上限 / 成本分界 / TTS 默认 / 字幕策略 / 允许扩展名 / 服务地址）
    全走 config——不带默认值硬编码进函数体。
    """

    # ── 服务地址 / 网络 ──
    api_base: str = "http://localhost:8000"
    api_prefix: str = "/api/v1"
    http_timeout: float = 30.0          # 业务 HTTP 调用超时（秒）
    health_timeout: float = 2.0         # /health 探活超时（≤3s，fast-fail）

    # ── 任务治理（SaaS 化：工作目录基准 + 并发上限 + 生命周期） ──
    data_dir: str = Field(
        default="~/flowmind-data",  # 任务工作目录基准（~ 自动展开为绝对路径）
        validate_default=True,      # 默认值也要过 validator 展开 ~
    )
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

    # ── 字段化偏好（Agent 可显式覆盖） ──
    tts_voice: str | None = None        # None = 让 VL 按目标语言自动选
    subtitle_font_size: int = 22        # 横屏；竖屏自动 ×0.7
    subtitle_position: str = "bottom_safe"  # 防遮画面
    output_filename_suffix: str = "sub"     # 输出文件名后缀

    # ── 全云流水线（localize_video） ──
    asr_sample_rate: int = 16000        # 提取音轨采样率（qwen-audio-3.0-asr-flash-streaming 支持 16k）
    ocr_frame_count: int = 8            # 字幕定位离线抽帧数（均匀取样，非逐帧）
    localize_llm_model: str = "LongCat-2.0"  # 翻译模型（Anthropic 兼容协议）
    localize_tts_model: str = "qwen-audio-3.0-tts-flash"  # 配音 TTS 模型
    localize_voice: str = "longanhuan_v3.6"   # 配音音色（预设或复刻音色 ID；空=不配音）
    voice_clone_prefix: str = "flwm"    # 复刻音色名前缀（仅字母数字，≤10 字符）

    # ── 本地/云后端开关（GPU 化升级：本地优先，云为回落）──
    # auto = 本地库可导入即用本地，否则回落云；两端都不可用显式报错（不静默降级）
    # local = 强制本地（库缺失显式报错）；cloud = 强制云
    asr_backend: str = "auto"           # ASR：local(faster-whisper) / cloud(dashscope) / auto
    local_asr_model: str = "small"      # faster-whisper 模型名（small/medium；8GB 显存够 medium）
    local_asr_device: str = "cuda"      # cuda / cpu
    ocr_backend: str = "auto"           # OCR：local(RapidOCR CPU) / cloud(qwen3.5-ocr) / auto
    erase_backend: str = "auto"         # 擦除：auto(LaMa 可用则用否则 delogo) / local(强制 LaMa) / delogo
    tts_backend: str = "auto"           # 配音：auto(本地栈可用则克隆原片人声否则云) / local(强制本地克隆) / cloud(强制云)
    bgm_vocal_sep: bool = True          # 背景音人声分离（demucs htdemucs）：True=纯伴奏做 BGM；False=整条原声

    @field_validator("data_dir")
    @classmethod
    def _expand_data_dir(cls, v: str) -> str:
        """把 data_dir 规范化为绝对路径：~ 展开；相对路径锚定 cwd（部署时显式配置绝对路径）。"""
        p = Path(v).expanduser()
        if not p.is_absolute():
            p = Path.cwd() / p
        return str(p)


class FlowmindConfig(BaseModel):
    """FlowMind 总配置：目前仅 localizer 一段。"""
    localizer: LocalizerConfig = Field(default_factory=LocalizerConfig)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> FlowmindConfig:
    """读取配置文件；不存在则全用通用默认。用户值覆盖默认，缺项回落默认。"""
    if not path.exists():
        return FlowmindConfig()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return FlowmindConfig.model_validate(data)


# 单例缓存：避免每次调用都重读磁盘 + 解析 TOML。
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
