"""LLM 翻译客户端：Anthropic / OpenAI 兼容双协议（httpx 直调）。

云优先原则：翻译必须走云端 LLM；无 key 显式报错，不静默降级。
协议经 protocol 参数选择（anthropic → /v1/messages；openai → /chat/completions），
默认 anthropic（LongCat）。base/model/api_key 由调用方传入。
输入是 ASR 句段（index/begin/end/text），输出同结构译文——时间轴字段原样保留，
供后续字幕对齐与 TTS 逐句合成使用。
key 由调用方从环境变量读出后传入，本模块不直接读 env、不进 config 文件。

本土化扩展（2026-09 多人对话出海升级）：可选传入术语/禁译表、文化替换规则与
逐句长度预算，三者拼进系统提示的运行时扩展段（不新增 LLM 调用轮次）；译后按
长度预算标注 over_budget，供配音/字幕环节择优。均不传时行为与扩展前完全一致。
"""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import httpx
import requests  # noqa: F401  保留模块级引用：测试 fixture 经 <mod>.requests 打桩拦截

DEFAULT_BASE = "https://api.longcat.chat/anthropic"
DEFAULT_MODEL = "LongCat-2.0"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_SYSTEM_PROMPT = (
    "你是专业视频字幕本地化译者。把用户给出的字幕句段从源语言翻译成目标语言。"
    "严格要求：1) 只输出 JSON 数组，不要解释/前缀/代码块；"
    "2) 每个元素保留 index 字段不变，text 换成译文；"
    "3) 口语化、简洁、适合字幕阅读（每句不超过原文长度的 1.5 倍）；"
    "4) 不软化、不替换、不 euphemize 技术术语与品牌名，必要时附原文括注。"
)

# 各语言「每秒可念字符数」经验表（配音语速参考）。
# ⚠️ 经验值，非实测：按常见配音语速（中文 ~5 字/秒、拉丁语系 ~13-15 字符/秒、
# 日韩 ~6-7 字符/秒）粗估，**必须按真实成片校准**。可按语言经 config 覆盖：
# `localizer.dub_speed_factors`（如 {"en": 12.5}）由流水线透传，或直接调用
# `estimate_length_budget(..., speed_factor=<覆盖表>)`；键同本表。
LANG_SPEED_FACTOR: dict[str, float] = {
    "zh": 5.0,
    "en": 14.0,
    "es": 15.0,
    "pt": 14.0,
    "ja": 7.0,
    "ko": 6.0,
    "fr": 15.0,
    "de": 14.0,
    "ru": 14.0,
    "ar": 13.0,
    "th": 10.0,
    "vi": 13.0,
    "id": 14.0,
}
# 未收录语言的保守默认：取偏低值，宁可低估预算（约束更紧）也不放任超长。
_DEFAULT_SPEED_FACTOR = 8.0
# 句段时长下限（秒）：过短/非正时长按下限计，避免算出 0 字符预算。
_MIN_SEG_DURATION = 0.3


class LLMTtranslateError(Exception):
    """翻译调用失败。category/retriable 语义与 errors.py 一致。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def translate_segments(
    segments: list[dict],
    *,
    target_lang: str,
    source_lang: str = "zh",
    api_key: str,
    api_base: str = DEFAULT_BASE,
    model: str = DEFAULT_MODEL,
    protocol: str = "anthropic",
    batch_size: int = 3,
    timeout_s: float = 180.0,
    client: httpx.Client | None = None,
    glossary: dict[str, str] | None = None,
    culture_rules: list[str] | None = None,
    length_budget: dict[int, int] | None = None,
) -> list[dict]:
    """分批翻译句段，返回同结构列表（text 为译文，begin/end/index 原样）。

    可选本土化扩展（均不传时与扩展前行为完全一致）：
    - glossary：术语/禁译表（源词 → 目标语固定译法），作硬约束注入系统提示；
    - culture_rules：文化替换规则（货币/单位/节日/称谓/度量衡…）；
    - length_budget：{句段 index: 目标语最大字符数}，按批取子集注入提示，
      译后超限的句段加 over_budget=True（不因超限报错，仅标注）。
    """
    if not api_key:
        raise ValueError(
            "收到空 API key。请检查环境变量 AI_LLM_API_KEY 是否设置。"
            "云优先原则：翻译必须走云 LLM，不做本地降级。"
        )
    step = max(1, batch_size)
    out: list[dict] = []
    for start in range(0, len(segments), step):
        batch = segments[start:start + step]
        # 长度预算按全局 index 索引，但 LLM 只见批内局部序号（0-based），
        # 故按批折算成本地序号 → 预算的子集，与 user payload 的 index 对齐
        batch_budget: dict[int, int] | None = None
        if length_budget is not None:
            batch_budget = {}
            for local_i, seg in enumerate(batch):
                global_i = seg.get("index", start + local_i)
                if global_i in length_budget:
                    batch_budget[local_i] = length_budget[global_i]
        system_prompt = _build_system_prompt(
            glossary=glossary, culture_rules=culture_rules,
            length_budget_batch=batch_budget,
        )
        out.extend(_translate_batch(
            batch, target_lang=target_lang, source_lang=source_lang,
            api_key=api_key, api_base=api_base, model=model,
            protocol=protocol, timeout_s=timeout_s, client=client,
            system_prompt=system_prompt,
        ))
    if length_budget is not None:
        # 传入预算时统一加 over_budget 标记（无预算的句段恒 False）
        out = [
            {**seg, "over_budget": bool(
                seg.get("index") in length_budget
                and len(seg.get("text", "")) > length_budget[seg["index"]]
            )}
            for seg in out
        ]
    return out


def _build_system_prompt(
    *, glossary: dict[str, str] | None = None,
    culture_rules: list[str] | None = None,
    length_budget_batch: dict[int, int] | None = None,
) -> str:
    """基础系统提示 + 运行时本土化扩展段（三者皆空时原样返回常量）。

    length_budget_batch 用批内局部序号（与 user payload 的 index 一致）。
    """
    parts = [_SYSTEM_PROMPT]
    if glossary:
        lines = ["术语/禁译表（硬约束）：下列源词一旦出现，必须使用指定译法，"
                 "不得意译、改写、缩写或省略："]
        lines += [f'- "{src}" 必须译为 "{dst}"' for src, dst in glossary.items()]
        parts.append("\n".join(lines))
    if culture_rules:
        lines = ["文化替换规则（必须遵守）：按目标语言市场的习惯做等价替换，"
                 "保持原意与信息量不变："]
        lines += [f"- {rule}" for rule in culture_rules]
        parts.append("\n".join(lines))
    if length_budget_batch:
        lines = ["逐句配音时长约束（硬约束 + 尽力而为）：下列 index 的译文"
                 "需适配配音时长，字符数尽量不超过给定上限；"
                 "无法满足时优先保证语义完整，不要为凑长度删改信息："]
        lines += [f"- index {i}：不超过 {n} 个字符"
                  for i, n in sorted(length_budget_batch.items())]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _translate_batch(
    batch: list[dict], *, target_lang: str, source_lang: str,
    api_key: str, api_base: str, model: str, protocol: str,
    timeout_s: float, client: httpx.Client,
    system_prompt: str | None = None,
) -> list[dict]:
    # system_prompt 由调用方经 _build_system_prompt 拼好；缺省回落到基础常量
    system_prompt = system_prompt or _SYSTEM_PROMPT
    # LLM 只见批内局部序号（0-based），返回后映射回全局 index
    local = [{"index": i, "text": s["text"]} for i, s in enumerate(batch)]
    user_payload = {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "segments": local,
    }
    user_content = json.dumps(user_payload, ensure_ascii=False)

    if protocol == "openai":
        url = f"{api_base.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type": "application/json"}
        body = {
            "model": model,
            "max_tokens": 8192,
            # 关闭推理模式的思考（Qwen3 系等混合推理模型默认开启，
            # 思考会耗尽 max_tokens 导致 content 为空）；不认该参数的
            # 供应商返回 400 时在下方自动去参重发
            "enable_thinking": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
    else:  # anthropic（默认）
        url = f"{api_base.rstrip('/')}/v1/messages"
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type": "application/json"}
        body = {
            "model": model,
            "max_tokens": 8192,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_content}],
        }

    try:
        if client is not None:
            resp = client.post(url, headers=headers, json=body)
        else:
            with httpx.Client(timeout=timeout_s) as c:
                resp = c.post(url, headers=headers, json=body)
        if resp.status_code == 400 and "enable_thinking" in body:
            # 供应商不认 enable_thinking：去掉该参数重发一次
            body = {k: v for k, v in body.items() if k != "enable_thinking"}
            if client is not None:
                resp = client.post(url, headers=headers, json=body)
            else:
                with httpx.Client(timeout=timeout_s) as c:
                    resp = c.post(url, headers=headers, json=body)
    except requests.exceptions.Timeout as exc:
        raise LLMTtranslateError("LLM 超时", category="environment", retriable=False) from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        cat = "transient" if status >= 500 else "video"
        raise LLMTtranslateError(f"LLM HTTP {status}", category=cat,
                                 retriable=status >= 500) from exc

    if resp.status_code >= 500:
        raise LLMTtranslateError(f"LLM HTTP {resp.status_code}",
                                 category="transient", retriable=True)
    if resp.status_code >= 400:
        raise LLMTtranslateError(f"LLM HTTP {resp.status_code}", category="video")

    data = resp.json()
    if protocol == "openai":
        # OpenAI 兼容协议：choices[0].message.content
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
    else:
        # Anthropic 兼容协议：content 是块数组；推理模型的 thinking 块在前，
        # 取第一个 type=text 的块（无则结构异常）
        text = next(
            (blk.get("text") for blk in data.get("content", [])
             if isinstance(blk, dict) and blk.get("type") == "text"),
            None,
        )
    if not text:
        raise LLMTtranslateError("LLM 返回结构异常")

    return _parse_reply(text, batch)


def _parse_reply(content: str, batch: list[dict]) -> list[dict]:
    """解析 LLM 的 JSON 数组回复（批内局部 index），映射回原句段结构。"""
    # 推理模型可能把思考过程以 <think> 块混进正文，先剥掉
    text = _THINK_RE.sub("", content).strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        items = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMTtranslateError("LLM 译文不是合法 JSON") from exc
    if not isinstance(items, list) or len(items) != len(batch):
        raise LLMTtranslateError("LLM 译文章数与原文不一致")
    by_local_index = {it.get("index"): it.get("text", "") for it in items}
    out: list[dict] = []
    for local_i, seg in enumerate(batch):
        if local_i not in by_local_index or not str(by_local_index[local_i]).strip():
            raise LLMTtranslateError(f"LLM 缺少第 {local_i} 句的译文")
        out.append({**seg, "text": str(by_local_index[local_i]).strip()})
    return out


def load_glossary(path: str) -> tuple[dict[str, str], list[str]]:
    """读取术语/禁译表文件，返回 (terms, culture_rules)。

    支持两种格式（按扩展名分派，仅用标准库）：
    - `.json`：{"terms": {"老干妈": "Lao Gan Ma"}, "culture_rules": ["把人民币换算成美元"]}
    - `.toml`：[terms] 表 + 顶层 culture_rules = [...]

    路径不存在 / 解析失败 / 结构不合法 → LLMTtranslateError(category="environment")，
    消息含路径与原因；**绝不静默返回空表**（空表会让术语约束静默失效）。
    """
    p = Path(path)
    if not p.is_file():
        raise LLMTtranslateError(f"术语表文件不存在：{path}", category="environment")
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise LLMTtranslateError(
            f"术语表读取失败：{path}（{exc.strerror or exc}）",
            category="environment",
        ) from exc

    suffix = p.suffix.lower()
    if suffix == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMTtranslateError(
                f"术语表 JSON 解析失败：{path}（第 {exc.lineno} 行：{exc.msg}）",
                category="environment",
            ) from exc
    elif suffix == ".toml":
        try:
            data = tomllib.loads(raw)
        except tomllib.TOMLDecodeError as exc:
            raise LLMTtranslateError(
                f"术语表 TOML 解析失败：{path}（{exc}）", category="environment"
            ) from exc
    else:
        raise LLMTtranslateError(
            f"术语表格式不支持（仅支持 .json / .toml）：{path}", category="environment"
        )

    if not isinstance(data, dict) or not ({"terms", "culture_rules"} & set(data)):
        raise LLMTtranslateError(
            f"术语表结构不合法：{path}（顶层需含 terms 表与/或 culture_rules 数组）",
            category="environment",
        )
    terms_raw = data.get("terms", {})
    rules_raw = data.get("culture_rules", [])
    if not isinstance(terms_raw, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in terms_raw.items()
    ):
        raise LLMTtranslateError(
            f"术语表 terms 必须是「字符串 → 字符串」映射：{path}", category="environment"
        )
    if not isinstance(rules_raw, list) or any(not isinstance(r, str) for r in rules_raw):
        raise LLMTtranslateError(
            f"术语表 culture_rules 必须是字符串数组：{path}", category="environment"
        )
    return dict(terms_raw), list(rules_raw)


def estimate_length_budget(
    segments: list[dict],
    target_lang: str,
    *,
    speed_factor: dict[str, float] | float | None = None,
) -> dict[int, int]:
    """按句段时长估算目标语字符数上限，返回 {index: max_chars}。

    计算：max_chars = 原句时长(秒, end-begin) × 该语言每秒可念字符数。
    speed_factor 用于整体覆盖：传 dict 则优先查表（未命中的语言回落 LANG_SPEED_FACTOR，
    再回落保守默认）；传 float 则对所有语言统一覆盖（如按真实成片测得的全局语速）。
    时长过短（< _MIN_SEG_DURATION）或非正/缺失的句段按下限计，结果至少 1 字符，
    避免算出 0 预算让下游无解。
    """
    def rate_for(lang: str) -> float:
        if isinstance(speed_factor, dict):
            if lang in speed_factor:
                return float(speed_factor[lang])
        elif isinstance(speed_factor, (int, float)):
            return float(speed_factor)
        return LANG_SPEED_FACTOR.get(lang, _DEFAULT_SPEED_FACTOR)

    rate = rate_for(target_lang)
    out: dict[int, int] = {}
    for pos, seg in enumerate(segments):
        try:
            duration = float(seg["end"]) - float(seg["begin"])
        except (KeyError, TypeError, ValueError):
            duration = 0.0
        duration = max(duration, _MIN_SEG_DURATION)
        out[seg.get("index", pos)] = max(1, int(duration * rate))
    return out
