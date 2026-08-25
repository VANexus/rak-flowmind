"""对话式可交互初始化 —— 让 Agent / 用户一步步设偏好，不用一次性塞 9 个参数。

两种用法：
1. CLI：用户跑 `uv run flowmind-init`，按问题顺序回答（适合真人）
2. 库：Agent 调 `run_interactive_init(ask_fn=...)`，每步问用户一次（适合 AI Agent）

设计原则：
- 每步只问一件事，给出「默认 / 解释 / 示例」，让用户能直接按 Enter 接受默认
- 选项型问题（bool / 语言）提供清晰的选择，避免无效输入
- 收集完所有答案 → 一次性调用 init_for_user()，写 TOML + reload_config
- 任何步骤 Ctrl-C 都安全退出，不留半成品配置
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from flowmind.config import DEFAULT_CONFIG_PATH, FlowmindConfig, init_for_user


@dataclass(frozen=True)
class Question:
    """一问一答的题目定义。"""
    id: str                                # 给 init_for_user 用的参数名
    prompt: str                            # 给用户看的问题（中文）
    hint: str = ""                         # 解释为什么问 / 默认是什么
    default: object = ...                  # 默认值；用户空回车 = 用这个
    choices: tuple[str, ...] | None = None # 可选枚举；显示给用户做选择
    required: bool = True                  # 是否必填（False 时默认会被使用）


QUESTIONS: tuple[Question, ...] = (
    Question(
        id="target_lang",
        prompt="目标语言（要把视频本地化到哪种语言？）",
        hint="适合出海通用场景：en / th / ja / ko / es / fr / de / ru",
        default="en",
        choices=("en", "th", "ja", "ko", "es", "fr", "de", "ru"),
    ),
    Question(
        id="source_lang",
        prompt="源语言（你的视频原本是什么语言？）",
        hint="出海营销场景常见是中文 → 多语种",
        default="zh",
        choices=("zh", "en"),
    ),
    Question(
        id="enable_tts",
        prompt="是否开启 TTS 配音？（为本地化视频生成新语音）",
        hint="y=配音，n=只换字幕。建议：营销视频必开",
        default="y",
        choices=("y", "n"),
    ),
    Question(
        id="tts_voice",
        prompt="TTS 音色（None=让 VL 按目标语言自动选）",
        hint="例：th-TH-NiwatNeural（泰语男声）/ ja-JP-NanamiNeural / en-US-AriaNeural",
        default="",
    ),
    Question(
        id="remove_subtitles",
        prompt="是否擦除硬烧中文字幕？（v0.3 推荐 True）",
        hint="目标观众不读中文，留着是视觉噪音",
        default="y",
        choices=("y", "n"),
    ),
    Question(
        id="remove_subtitles_strategy",
        prompt="字幕处理策略（v0.3 唯一支持 ocr_erase_redraw）",
        hint="OCR 定位 bbox → 擦除原字幕 → 用目标语言重绘",
        default="ocr_erase_redraw",
        choices=("ocr_erase_redraw",),
    ),
    Question(
        id="subtitle_font_size",
        prompt="字幕字号（默认 22；竖屏自动 ×0.7）",
        hint="数字大小，写整数",
        default="22",
    ),
    Question(
        id="subtitle_position",
        prompt="字幕位置（默认 bottom_safe 防遮画面）",
        hint="bottom_safe = 避开 YouTube Shorts / TikTok UI 控件区",
        default="bottom_safe",
        choices=("bottom", "bottom_safe"),
    ),
    Question(
        id="output_filename_suffix",
        prompt="输出文件名后缀（默认 'sub'，例 output_sub.mp4）",
        hint="用来在文件管理器里区分本地化前后版本",
        default="sub",
    ),
)


def _parse_bool(s: str) -> bool:
    """'y' / 'n' / 'yes' / 'no' / 'true' / 'false' → bool"""
    s = s.strip().lower()
    if s in ("y", "yes", "true", "1", ""):
        return True
    if s in ("n", "no", "false", "0"):
        return False
    raise ValueError(f"无法识别 y/n: {s!r}")


def _parse_choice(value: str, choices: tuple[str, ...], default: str) -> str:
    """验证用户输入在 choices 列表里；不在就用默认"""
    v = value.strip()
    if not v:
        return default
    if v.lower() in (c.lower() for c in choices):
        # 返回原始大小写版本
        for c in choices:
            if c.lower() == v.lower():
                return c
    raise ValueError(f"必须是 {'/'.join(choices)} 之一：{value!r}")


def _ask(ask_fn: Callable[[str], str], q: Question) -> str:
    """问用户一个问题。返回 raw string。"""
    print()
    print(f"── {q.prompt} ──")
    if q.hint:
        print(f"  ℹ {q.hint}")
    if q.choices:
        print(f"  选择：{' / '.join(q.choices)}")
    print(f"  默认（直接回车）：{q.default!r}")
    while True:
        try:
            raw = ask_fn("  > ").strip()
            if not raw:
                return str(q.default)
            if q.choices:
                return _parse_choice(raw, q.choices, str(q.default))
            return raw
        except ValueError as exc:
            print(f"  ✗ {exc}")


# 类型转换表：缺省（不在表里）= 原样透传给 init_for_user
_COERCERS: dict[str, Callable[[str], object]] = {
    "enable_tts": _parse_bool,
    "remove_subtitles": _parse_bool,
    "tts_voice": lambda raw: raw or None,
    "subtitle_font_size": int,
}


def _coerce(qid: str, raw: str) -> object:
    """把 raw 字符串转成 init_for_user 期望的类型。"""
    conv = _COERCERS.get(qid)
    return conv(raw) if conv else raw


def run_interactive_init(
    ask_fn: Callable[[str], str] | None = None,
    save_path: Path | None = None,
    quiet: bool = False,
) -> FlowmindConfig:
    """对话式初始化：逐个问 QUESTIONS 里的问题，最后一次性 init_for_user()。

    ask_fn: 单参数「提示词」→ 用户回答字符串的 callable。
            None 时用内置 `input()`（CLI 用）。
            Agent 可以传自己实现的 ask_fn（比如调 LLM 问用户）。

    save_path: 写到哪个 flowmind.config.toml。None 用 config.DEFAULT_CONFIG_PATH。

    quiet: True 时不打印 banner（库调用时用）。

    返回最终的 FlowmindConfig。
    """
    if ask_fn is None:
        ask_fn = input

    if not quiet:
        print()
        print("=" * 60)
        print("🎬 FlowMind 视频本地化偏好初始化")
        print("=" * 60)
        print()
        print("这个向导会引导你设置视频本地化的偏好。")
        print(f"共有 {len(QUESTIONS)} 个问题，每个都有合理默认值，")
        print("直接按 Enter 接受默认即可。Ctrl-C 安全退出。")

    answers: dict[str, object] = {}
    try:
        for q in QUESTIONS:
            raw = _ask(ask_fn, q)
            answers[q.id] = _coerce(q.id, raw)
    except (KeyboardInterrupt, EOFError):
        print("\n\n✗ 中断，未写入配置。")
        raise SystemExit(1)

    if not quiet:
        print()
        print("─" * 60)
        print("📋 配置摘要：")
        for q in QUESTIONS:
            print(f"  • {q.prompt:40s} → {answers[q.id]!r}")
        print("─" * 60)

    # 一次性写文件
    cfg = init_for_user(save_path=save_path or DEFAULT_CONFIG_PATH, **answers)  # type: ignore[arg-type]

    if not quiet:
        out_path = save_path or DEFAULT_CONFIG_PATH
        print(f"\n✓ 已写入 {out_path}")
        print("✓ 所有后续 invoke('localize_*', ...) 会自动应用这套偏好。")

    return cfg


def main() -> None:
    """CLI 入口：`uv run flowmind-init`"""
    run_interactive_init()