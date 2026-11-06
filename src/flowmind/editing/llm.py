"""创意层 LLM 客户端：Anthropic 兼容 /v1/messages（httpx 直调）。

与 _llm_translate 同协议（LongCat），但面向创意阶段（copy/script/storyboard）：
- 每阶段独立系统提示词，要求只输出 JSON；parse 后经 pydantic 模型校验；
- 两层韧性：瞬时失败（网络/408/429/5xx/空 content/max_tokens 截断）指数
  退避重试；解析/schema 失败再做多轮对话自纠一轮（报错喂回要求修正）；
  配额型 429 重试耗尽升级 environment（调用方快速失败）；
- 解析失败显式抛 EditingLLMError（错误永不静默），调用方映射 degraded；
- feedback（revise 反馈）与 extra_brief 注入 user 内容，驱动定向重生成；
- mock=True（config.editing.mock / env FLOWMIND_EDIT_MOCK）返回内置桩产物，
  供 demo 与无 key 环境冒烟——桩数据结构与真实输出同 schema。
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Callable

import httpx
from pydantic import BaseModel

from flowmind.editing.models import (
    TRANSITIONS,
    CopyDoc,
    ScriptDoc,
    StoryboardDoc,
    clamp_transition,
)

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class EditingLLMError(Exception):
    """创意层 LLM 调用/解析失败。category 语义与 errors.py 一致。

    quota=True 标记配额型 429：重试耗尽后调用方应升级为 environment
    （配额墙需人工介入，区别于瞬时限流）。retry_after 携带服务端
    Retry-After 头（秒），供退避策略优先采用。
    """

    def __init__(self, message: str, category: str = "unknown",
                 retriable: bool = False, quota: bool = False,
                 retry_after: float | None = None) -> None:
        super().__init__(message)
        self.category = category
        self.retriable = retriable
        self.quota = quota
        self.retry_after = retry_after


# ── 各阶段系统提示词（中文；结构化 JSON 输出契约写死在提示词里）──

_COPY_SYSTEM = (
    "你是资深短视频编导。根据素材概览、内容点清单（content_points：素材里"
    "真实出现的画面与文字信息）与创作要求，产出一条成片的文案。"
    "严格要求：1) 只输出 JSON，不要解释/前缀/代码块；"
    "2) JSON 结构：{\"title\":标题,\"hook\":开头钩子一句话,"
    "\"body_sections\":[{\"purpose\":\"段落职能\",\"text\":\"文案\"}],"
    "\"cta\":\"结尾行动号召\",\"full_narration\":\"完整口播稿（连贯成文）\","
    "\"style_notes\":\"风格说明\"}；"
    "3) 事实锚定（最重要）：正文与口播必须使用 content_points 里的具体信息"
    "——认证/设备/数字/工艺/产能，全稿至少落地 3 个具体名词或数字"
    "（如 GMPC 认证、20 台压粉机、7 天出样）；"
    "4) 先判断素材类型（工厂实力展示/产品种草/知识科普/剧情），用匹配的"
    "叙事框架，不要硬套「N个干货点」模板——素材是实力展示就讲实力；"
    "5) 禁止空洞排比与套话：「第一点…第二点…」「全是干货」「句句精华」"
    "「直击要害」这类没有实际信息的句子一律禁止——每句话都要答得上"
    "「具体说了什么」；"
    "6) 口语化、有节奏感，钩子 3 秒内抓人；7) 时长预算内可承载，别灌水。"
)

_SCRIPT_SYSTEM = (
    "你是分镜脚本编剧。把文案拆成结构化脚本段落，每个段落必须绑定素材里的"
    "真实内容点（content_points）。"
    "严格要求：1) 只输出 JSON；"
    "2) 结构：{\"sections\":[{\"id\":\"s1\",\"purpose\":\"hook\","
    "\"narration\":\"该段口播\",\"visual_brief\":\"画面需求（写明引用哪个"
    "内容点的时刻/场景）\",\"duration_s\":5.0,\"subtitle_text\":\"该段字幕\"}]}；"
    "3) 段落数 3~8，duration_s 之和不超过时长预算；"
    "4) 字幕与口播讲事实：每段 subtitle_text/narration 必须含该段绑定的"
    "具体信息（名词/数字），subtitle_text ≤16 字独立成句；"
    "禁止只有序号没有内容的段落（如「第一点：直击要害」）；"
    "5) visual_brief 具体可执行（景别/主体/运动，引用内容点），不要空话。"
)

_STORYBOARD_SYSTEM = (
    "你是短视频剪辑设计师。根据脚本段落与素材内容目录（content_notes：每个"
    "抽帧时刻的真实画面描述与画面字幕文本）设计一支有设计感的成片，不是简单"
    "拼接。严格要求："
    "1) 只输出 JSON；"
    "2) 结构：{\"title_card\":{\"main\":\"开场大字\",\"sub\":\"副标题\","
    "\"duration_s\":2.0},\"shots\":[{\"id\":\"sh1\",\"section_id\":\"s1\","
    "\"source_asset\":\"素材id\",\"source_in\":0.0,\"duration_s\":4.0,"
    "\"mode\":\"video_clip\",\"pan\":\"none\",\"transition\":\"zoomin\","
    "\"speed\":1.0,\"freeze_s\":0.0,\"subtitle_text\":\"字幕\","
    "\"subtitle_emphasis\":[\"关键词\"],\"badge\":\"01/08\","
    "\"zoom_start\":1.0,\"zoom_end\":1.08,\"hook\":false,"
    "\"note\":\"选材+设计理由\"}],\"end_card\":{\"main\":\"CTA大字\","
    "\"sub\":\"副标题\",\"duration_s\":2.0},\"bgm_note\":\"配乐意向\"}；"
    "3) 效果白名单：mode 只能 video_clip / image_kenburns；transition 只能取 "
    "fade / none / zoomin / slideleft / slideright / circleopen / dissolve / "
    "fadeblack / radial；speed 0.25~4；freeze_s 0~3；zoom_start/zoom_end "
    "0.9~1.2（缓推 zoom_end>zoom_start，缓拉反之，静止素材必须有运动）；"
    "4) 设计规则（铁律）："
    "a. title_card 必有——主标题即钩子（≤14字，口语化，制造好奇/利益点），"
    "第一镜 hook=true（字幕居中放大）；"
    "b. 相邻镜头 transition 不许重复，按情绪选：zoomin 推进强调 / slideleft "
    "滑切换场 / circleopen 揭示 / dissolve 柔和叙事 / fadeblack 章节收束；"
    "c. 每镜 zoom_start/zoom_end 至少一组 ≠1.0（画面必须有运动）；"
    "d. subtitle_emphasis 每镜标 1~3 个 subtitle_text 里真实出现的关键词"
    "（渲染金色高亮），优先数字与名词；badge 用 01/08 式进度角标（可选）；"
    "e. end_card 必有——CTA 大字 + 引导关注/收藏的副标题；"
    "5) 叙事硬约束：镜头按 content_notes 的时间顺序推进（source_in 单调不减，"
    "允许结尾一次回到开头做呼应）；把语义最贴合的内容点选进对应段落；"
    "6) 只能引用素材目录里真实存在的素材 id，source_in 优先落在内容点附近；"
    "单镜 duration_s 不小于 0.5，总时长贴近预算；"
    "7) note 写一句中文设计理由（画面+为什么这么转场/推拉），供人工审批参考；"
    "8) subtitle_text 直接沿用脚本段落字幕（可微调断句），必须是有具体信息"
    "的独立成句——禁止「第一点/核心点」式空标题；subtitle_emphasis 选具体"
    "数字与名词（认证名/设备数/天数），不选虚词。"
)

_VISION_SYSTEM = (
    "你是视频内容分析员。用户给出一段视频在若干时刻的抽帧（按时间顺序）。"
    "输出 JSON 数组，每帧一个元素：{\"t_s\":<原样返回该帧时刻秒数>,"
    "\"scene\":\"画面内容一句话（主体/场景/动作）\","
    "\"subtitle\":\"画面中烧录的字幕或文字（逐字读出；没有则空字符串）\","
    "\"importance\":<1~5 整数，该时刻的信息与情绪重要度>}。"
    "只输出 JSON 数组，不要解释。"
)

# LLM 常见转场/模式同义词 → schema 枚举（模型爱用自然词汇，做输入归一化
# 而不是苛求提示词；未识别值按最安全档处理并记日志，不静默改语义）
_TRANSITION_ALIAS = {
    "cut": "none", "hard_cut": "none", "hardcut": "none", "jump": "none",
    "jumpcut": "none", "straight": "none", "straight_cut": "none",
    "flash": "fade", "flash_cut": "fade", "crossfade": "fade",
    "cross_fade": "fade", "dissolve": "dissolve",
    "push": "zoomin", "push_in": "zoomin", "punch": "zoomin",
    "zoom": "zoomin", "slide": "slideleft", "wipe": "slideleft",
    "wipeleft": "slideleft", "circle": "circleopen", "reveal": "circleopen",
    "black": "fadeblack", "fade_to_black": "fadeblack",
    "radialblur": "radial", "blur": "radial",
}
_MODE_ALIAS = {
    "clip": "video_clip", "video": "video_clip", "shot": "video_clip",
    "kenburns": "image_kenburns", "ken_burns": "image_kenburns",
    "image": "image_kenburns", "photo": "image_kenburns",
}


def _normalize_storyboard(data: object) -> dict:
    """LLM 输出归一化：转场/模式同义词 + 镜头时长钳制。

    模型爱用自然词汇（cut/hard_cut）与超短闪切（0.2s）——做输入归一化
    而不是苛求提示词；未识别值按最安全档处理并记日志，不静默改语义。
    顶层 JSON 数组（模型偶尔漏掉外层对象）包成 {"shots": [...]};
    非 dict/list 结构显式报错（不再以 AttributeError 逃逸成 unknown）。
    """
    if isinstance(data, list):
        data = {"shots": data}
    if not isinstance(data, dict):
        raise EditingLLMError(
            f"storyboard 输出不是 JSON 对象：{type(data).__name__}")
    # v1.3 设计卡归一化：时长钳制 + 文本截断（缺 key 容错）
    for card_key in ("title_card", "end_card"):
        card = data.get(card_key)
        if isinstance(card, dict):
            card["main"] = str(card.get("main", ""))[:40]
            card["sub"] = str(card.get("sub", ""))[:60]
            try:
                card["duration_s"] = min(
                    4.0, max(1.0, float(card.get("duration_s", 2.0))))
            except (TypeError, ValueError):
                card["duration_s"] = 2.0
        else:
            data[card_key] = None
    for shot in data.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        t = str(shot.get("transition", "fade")).strip().lower()
        if t in TRANSITIONS:
            pass
        else:
            mapped = _TRANSITION_ALIAS.get(t)
            if mapped is None:
                logger.info("未知转场 %r，归一化为 none（硬切）", t)
                mapped = "none"
            shot["transition"] = clamp_transition(mapped)
        m = str(shot.get("mode", "video_clip")).strip().lower()
        if m not in ("video_clip", "image_kenburns"):
            shot["mode"] = _MODE_ALIAS.get(m, "video_clip")
        try:
            dur = float(shot.get("duration_s"))
        except (TypeError, ValueError):
            dur = 3.0
        clamped = min(60.0, max(0.3, dur))
        if clamped != dur:
            logger.info("镜头 %s 时长 %.2fs 钳制为 %.2fs",
                        shot.get("id", "?"), dur, clamped)
        shot["duration_s"] = clamped
        try:
            speed = float(shot.get("speed", 1.0) or 1.0)
        except (TypeError, ValueError):
            speed = 1.0
        # 上限与提示词/storyboard 模型一致（0.25~4）
        shot["speed"] = min(4.0, max(0.25, speed))
        try:
            freeze = float(shot.get("freeze_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            freeze = 0.0
        shot["freeze_s"] = min(3.0, max(0.0, freeze))
        # v1.3 设计参数归一化：zoom 钳制 + 关键词去重截断 + 角标长度
        for key in ("zoom_start", "zoom_end"):
            try:
                z = float(shot.get(key, 1.0) or 1.0)
            except (TypeError, ValueError):
                z = 1.0
            shot[key] = min(1.2, max(0.9, z))
        if isinstance(shot.get("subtitle_emphasis"), list):
            shot["subtitle_emphasis"] = [
                str(w)[:20] for w in shot["subtitle_emphasis"][:3]
                if str(w).strip()]
        else:
            shot["subtitle_emphasis"] = []
        shot["badge"] = str(shot.get("badge", "") or "")[:12]
        shot["hook"] = bool(shot.get("hook", False))
    return data


def _retry_sleep(attempt: int, retry_after: float | None) -> None:
    """指数退避 + 抖动；服务端 Retry-After 优先（上限 60s）。attempt 从 0 计。"""
    delay = (retry_after if retry_after is not None
             else min(30.0, 1.5 * (2 ** attempt)))
    time.sleep(min(60.0, max(0.0, delay)) + random.uniform(0.0, 1.0))


def _call_with_retries(label: str, fn: Callable[[], str], *,
                       max_attempts: int) -> str:
    """瞬时失败退避重试（网络/超时/408/429/5xx/空 content/max_tokens 截断）。

    配额型 429 在重试耗尽后升级为 environment（配额墙需人工介入，
    调用方据此快速失败而不是盲剪降级）。
    """
    attempts = max(1, max_attempts)
    last: EditingLLMError | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except EditingLLMError as exc:
            if not exc.retriable or attempt == attempts - 1:
                if exc.quota:
                    raise EditingLLMError(
                        f"LLM 配额受限（429 退避重试 {attempts} 次后仍失败）"
                        f"：{exc}", category="environment") from exc
                raise
            last = exc
            logger.warning("%s LLM 调用失败（可重试，第 %d/%d 次）：%s",
                           label, attempt + 1, attempts, exc)
            _retry_sleep(attempt, exc.retry_after)
    raise last or EditingLLMError("LLM 调用失败")  # pragma: no cover


def _validate_raw(stage: str, raw: str, schema: type[BaseModel]) -> BaseModel:
    """解析 + 归一化 + schema 校验；失败带原文片段显式报错。"""
    data = _parse_json(raw)
    if stage == "storyboard":
        data = _normalize_storyboard(data)
    try:
        return schema.model_validate(data)
    except Exception as exc:  # noqa: BLE001 LLM 输出不合 schema 显式报错
        raise EditingLLMError(
            f"LLM 输出不合 schema：{exc}；原始输出片段：{raw[:500]}",
            category="unknown") from exc


def generate_json(
    stage: str, payload: dict, *, schema: type[BaseModel],
    api_base: str, api_key: str, model: str,
    timeout_s: float = 90.0, max_tokens: int = 16384,
    mock: bool = False, client: httpx.Client | None = None,
    max_attempts: int = 5, protocol: str = "anthropic",
) -> BaseModel:
    """按阶段生成并校验 JSON 产物。stage ∈ copy/script/storyboard。

    两层韧性：瞬时失败退避重试（_call_with_retries）；解析/校验失败再做
    一轮多轮对话自纠——把报错和原输出喂回去，要求只输出修正后的 JSON
    （实测 MiniMax 会输出同义词枚举/超限值，自纠一轮可救回大半）。
    """
    if not api_key and not mock:
        raise EditingLLMError(
            "创意层缺少 API key（环境变量 AI_LLM_API_KEY）。"
            "云优先原则：不做本地降级，显式报错。", category="environment")
    system, user_payload = _stage_prompt(stage, payload)
    user_msg = {"role": "user", "content": [
        {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)}]}
    if mock:
        raw = json.dumps(_mock_payload(stage, payload), ensure_ascii=False)
        return _validate_raw(stage, raw, schema)

    def _call() -> str:
        return _post_llm(
            protocol=protocol, system=system, messages=[user_msg],
            api_base=api_base, api_key=api_key, model=model,
            timeout_s=timeout_s, max_tokens=max_tokens, client=client)

    raw = _call_with_retries(f"阶段 {stage}", _call, max_attempts=max_attempts)
    try:
        return _validate_raw(stage, raw, schema)
    except EditingLLMError as first_err:
        correction = (
            "你上一次的输出无法通过校验：" + str(first_err)[:600]
            + "。请重新输出完整、修正后的 JSON（结构与系统提示词要求严格一致），"
              "不要解释，不要输出 JSON 以外的内容。")
        followup = [
            user_msg,
            {"role": "assistant", "content": raw},
            {"role": "user", "content": [{"type": "text",
                                          "text": correction}]},
        ]

        def _correct() -> str:
            return _post_llm(
                protocol=protocol, system=system, messages=followup,
                api_base=api_base, api_key=api_key, model=model,
                timeout_s=timeout_s, max_tokens=max_tokens, client=client)

        logger.warning("阶段 %s LLM 输出校验失败，自纠一轮：%s",
                       stage, str(first_err)[:200])
        raw2 = _call_with_retries(f"阶段 {stage} 自纠", _correct,
                                  max_attempts=max_attempts)
        try:
            return _validate_raw(stage, raw2, schema)
        except EditingLLMError as exc:
            raise EditingLLMError(
                f"阶段 {stage} 的 LLM 输出自纠后仍不合格：{exc}",
                category="unknown") from exc


def _stage_prompt(stage: str, payload: dict) -> tuple[str, dict]:
    if stage == "copy":
        return _COPY_SYSTEM, payload
    if stage == "script":
        return _SCRIPT_SYSTEM, payload
    if stage == "storyboard":
        return _STORYBOARD_SYSTEM, payload
    raise EditingLLMError(f"未知创意阶段：{stage}", category="unknown")


def _parse_retry_after(value: str | None) -> float | None:
    """Retry-After 头解析（秒）；HTTP 日期形式不支持，返回 None 走默认退避。"""
    if not value:
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


def _coerce_float(v: object, default: float | None = None) -> float | None:
    """宽松数值提取：模型爱输出 "5.14s"/"≈3.2" 之类，取首个数字。"""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        if m:
            return float(m.group())
    return default


def _post_messages(*, system: str, messages: list[dict], api_base: str,
                   api_key: str, model: str, timeout_s: float,
                   max_tokens: int, client: httpx.Client | None) -> str:
    """Anthropic /v1/messages 底层调用。

    messages 为完整轮次列表（content 为块列表，兼容 vision 与自纠多轮）。
    408/429/5xx/网络错误 → transient 可重试；429 额外带 quota 标记与
    Retry-After；stop_reason=max_tokens 显式报错（截断输出必不合格）。
    """
    url = f"{api_base.rstrip('/')}/v1/messages"
    headers = {
        # 双头兼容：标准 Anthropic 用 x-api-key；Claude Code 风格 token
        # （ANTHROPIC_AUTH_TOKEN，如 MiniMax 兼容端点）用 Authorization
        "x-api-key": api_key,
        "Authorization": f"Bearer {api_key}",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    owns = client is None
    cli = client or httpx.Client(timeout=timeout_s)
    try:
        resp = cli.post(url, headers=headers, json=body)
        if resp.status_code >= 400:
            retriable = resp.status_code in (408, 429) or resp.status_code >= 500
            # 错误消息不回显完整响应体（可能含上游内部信息），状态码+截断片段足够定位
            raise EditingLLMError(
                f"LLM HTTP {resp.status_code}: {resp.text[:200]}",
                category="transient" if retriable else "unknown",
                retriable=retriable, quota=resp.status_code == 429,
                retry_after=(_parse_retry_after(resp.headers.get("retry-after"))
                             if resp.status_code == 429 else None))
        data = resp.json()
        if data.get("stop_reason") == "max_tokens":
            # 思考型模型 thinking 块先吃 token，截断输出必然不完整——
            # 重采样大概率同样截断，但报错要显式指向调大 llm_max_tokens
            raise EditingLLMError(
                "LLM 输出被 max_tokens 截断（stop_reason=max_tokens）——"
                "请调大 config.editing.llm_max_tokens",
                category="transient", retriable=True)
        parts = data.get("content") or []
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        if not text.strip():
            # 思考型模型（MiniMax M 系）偶发把 token 预算全花在 thinking 块上，
            # content 为空——这是可重试的瞬时失败，交由调用方重试
            raise EditingLLMError(
                "LLM 返回空 content（可能被 max_tokens 截断或内容被过滤）",
                category="transient", retriable=True)
        return text
    except httpx.HTTPError as exc:
        raise EditingLLMError(f"LLM 网络错误：{exc}",
                              category="transient", retriable=True) from exc
    finally:
        if owns:
            cli.close()


def _post_openai(*, system: str, messages: list[dict], api_base: str,
                 api_key: str, model: str, timeout_s: float,
                 max_tokens: int, client: httpx.Client | None) -> str:
    """OpenAI 兼容 /chat/completions 底层调用（千问 MaaS 等仅此协议的平台）。

    错误/重试语义与 _post_messages 完全对齐：408/429/5xx/网络错误 retriable，
    429 带 quota 与 Retry-After，finish_reason=length 显式报错。
    """
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}",
               "content-type": "application/json"}
    body = {"model": model, "max_tokens": max_tokens,
            "messages": _to_openai_messages(system, messages)}
    owns = client is None
    cli = client or httpx.Client(timeout=timeout_s)
    try:
        resp = cli.post(url, headers=headers, json=body)
        if resp.status_code >= 400:
            retriable = resp.status_code in (408, 429) or resp.status_code >= 500
            raise EditingLLMError(
                f"LLM HTTP {resp.status_code}: {resp.text[:200]}",
                category="transient" if retriable else "unknown",
                retriable=retriable, quota=resp.status_code == 429,
                retry_after=(_parse_retry_after(resp.headers.get("retry-after"))
                             if resp.status_code == 429 else None))
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            raise EditingLLMError(
                "LLM 输出被 max_tokens 截断（finish_reason=length）——"
                "请调大 config.editing.llm_max_tokens",
                category="transient", retriable=True)
        text = choice.get("message", {}).get("content") or ""
        if isinstance(text, list):   # 防平台把 content 拆成块数组
            text = "".join(p.get("text", "") for p in text
                           if isinstance(p, dict) and p.get("type") == "text")
        if not text.strip():
            raise EditingLLMError(
                "LLM 返回空 content（可能被 max_tokens 截断或内容被过滤）",
                category="transient", retriable=True)
        return text
    except httpx.HTTPError as exc:
        raise EditingLLMError(f"LLM 网络错误：{exc}",
                              category="transient", retriable=True) from exc
    finally:
        if owns:
            cli.close()


def _to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    """Anthropic 块风格 messages → OpenAI chat 格式（system 前置 + 块转换）。"""
    out: list[dict] = [{"role": "system", "content": system}]
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
            continue
        blocks: list[dict] = []
        for p in content or []:
            if p.get("type") == "text":
                blocks.append({"type": "text", "text": p.get("text", "")})
            elif p.get("type") == "image":
                src = p.get("source") or {}
                media = src.get("media_type", "image/jpeg")
                blocks.append({"type": "image_url", "image_url": {
                    "url": f"data:{media};base64,{src.get('data', '')}"}})
        out.append({"role": msg["role"], "content": blocks or ""})
    return out


def _post_llm(*, protocol: str, system: str, messages: list[dict],
              api_base: str, api_key: str, model: str, timeout_s: float,
              max_tokens: int, client: httpx.Client | None) -> str:
    """协议分发：anthropic → /v1/messages；openai → /chat/completions。"""
    post = _post_openai if protocol == "openai" else _post_messages
    return post(system=system, messages=messages, api_base=api_base,
                api_key=api_key, model=model, timeout_s=timeout_s,
                max_tokens=max_tokens, client=client)


def _parse_json(raw: str) -> dict:
    """剥代码栅栏后 json.loads；失败带原文片段显式报错。"""
    text = raw.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        # 兜底：截取首个 { 到末个 }（部分模型会带前后缀废话）
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise EditingLLMError(
            f"LLM 输出不是合法 JSON：{exc}；片段：{text[:300]}") from exc


def describe_frames(
    frames: list[tuple[float, str]], *,
    api_base: str, api_key: str, model: str,
    timeout_s: float = 180.0, max_tokens: int = 4096,
    mock: bool = False, client: httpx.Client | None = None,
    max_attempts: int = 5, protocol: str = "anthropic",
) -> list[dict]:
    """v1.1 视觉理解：抽帧 → MiniMax vision → 每帧 {t_s,scene,subtitle,
    importance}。

    frames 为 [(时刻秒, jpeg路径)]；返回 dict 列表（调用方过 ContentNote
    校验）。无 key 显式报错；mock 返回桩笔记。失败抛 EditingLLMError，
    由 understand 阶段捕获——transient 类显式记 vision_degraded（盲剪
    降级不静默），environment/配额类向上快速失败。
    """
    import base64

    if mock:
        return [{"t_s": round(t, 2),
                 "scene": f"mock 画面（{t:.1f}s）",
                 "subtitle": "", "importance": 3} for t, _ in frames]
    if not api_key:
        raise EditingLLMError(
            "视觉理解缺少 API key（环境变量 AI_LLM_API_KEY）",
            category="environment")

    content: list[dict] = [{
        "type": "text",
        "text": f"以下是同一段视频按时间顺序抽取的 {len(frames)} 帧。"
                "请按要求输出每帧的 JSON 分析。",
    }]
    for t, path in frames:
        try:
            data = base64.b64encode(Path(path).read_bytes()).decode()
        except OSError as exc:
            raise EditingLLMError(f"抽帧文件不可读：{path}：{exc}",
                                  category="environment") from exc
        content.append({"type": "text", "text": f"t={t:.2f}s 的帧："})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg",
                       "data": data},
        })
    # vision 同样可能吃到空 content（thinking 预算耗尽），重试语义同 generate_json
    raw = _call_with_retries(
        "vision",
        lambda: _post_llm(
            protocol=protocol, system=_VISION_SYSTEM,
            messages=[{"role": "user", "content": content}],
            api_base=api_base, api_key=api_key, model=model,
            timeout_s=timeout_s, max_tokens=max_tokens, client=client),
        max_attempts=max_attempts)
    parsed = _parse_json(raw)
    if isinstance(parsed, dict):   # 模型偶尔包一层 {"frames": [...]}
        parsed = parsed.get("frames") or parsed.get("notes") or []
    if not isinstance(parsed, list):
        raise EditingLLMError(
            f"视觉理解输出不是 JSON 数组：{type(parsed).__name__}")
    out: list[dict] = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            continue
        # v1.3 数值清洗：模型会把 t_s 返回成 "5.14s" 之类带单位字符串——
        # 提取首个数字；取不到回落到该帧的请求时刻（调用方知道真值）
        t = _coerce_float(item.get("t_s"))
        if t is None:
            t = frames[i][0] if i < len(frames) else 0.0
        item["t_s"] = t
        try:
            item["importance"] = min(
                5, max(1, int(_coerce_float(item.get("importance"), 3))))
        except (TypeError, ValueError):
            item["importance"] = 3
        out.append(item)
    if not out:
        raise EditingLLMError("视觉理解返回空数组")
    logger.info("vision 看图完成：%d 帧笔记", len(out))
    return out


# ── mock 桩（schema 与真实一致；演示审批/revise/渲染全流程）──

def _mock_payload(stage: str, payload: dict) -> dict:
    brief = str(payload.get("extra_brief", "") or "")
    feedback = payload.get("_feedback") or []   # revise 反馈注入展示
    suffix = f"【修订{len(feedback)}】" if feedback else ""
    if stage == "copy":
        return {
            "title": f"《{payload.get('theme', '未命名')}》{suffix}",
            "hook": f"开头钩子：{brief or '三秒抓人'}",
            "body_sections": [
                {"purpose": "body", "text": "正文段落一：核心卖点展开"},
                {"purpose": "body", "text": "正文段落二：场景与细节"},
            ],
            "cta": "行动号召：点赞关注不迷路",
            "full_narration": "完整口播稿：钩子 → 正文 → CTA 连贯成文。",
            "style_notes": payload.get("style") or "明快节奏",
        }
    if stage == "script":
        return {"sections": [
            {"id": "s1", "purpose": "hook", "narration": "你知道吗？",
             "visual_brief": "特写开场，主体居中", "duration_s": 3.0,
             "subtitle_text": "你知道吗？"},
            {"id": "s2", "purpose": "body", "narration": "这就是本期主角。",
             "visual_brief": "中景推进，展示细节", "duration_s": 5.0,
             "subtitle_text": "这就是本期主角。"},
            {"id": "s3", "purpose": "cta", "narration": "点个赞再走。",
             "visual_brief": "定格收尾", "duration_s": 2.0,
             "subtitle_text": "点个赞再走。"},
        ], }
    if stage == "storyboard":
        # catalog 在 _stage_storyboard 已序列化为 dict（LLM 载荷形态）
        assets: list[dict] = payload.get("assets") or []
        video = next((a for a in assets if a.get("kind") == "video"), None)
        image = next((a for a in assets if a.get("kind") == "image"), None)
        # v1.3 设计层：转场轮换 + 每镜推拉 + 角标/关键词，演练渲染新路径
        trans_cycle = ["none", "zoomin", "slideleft", "circleopen", "dissolve"]
        shots: list[dict] = []
        sid = 1
        for sec in payload.get("sections", []):
            src = video
            mode = "video_clip"
            if src is None and image is not None:
                src, mode = image, "image_kenburns"
            if src is None:
                continue
            sub = sec.get("subtitle_text") or sec.get("narration", "")
            shots.append({
                "id": f"sh{sid}", "section_id": sec["id"],
                "source_asset": src["id"], "source_in": 0.0,
                "duration_s": sec.get("duration_s", 3.0),
                "mode": mode, "pan": "in" if mode == "image_kenburns" else "none",
                "transition": trans_cycle[(sid - 1) % len(trans_cycle)],
                "subtitle_text": sub,
                "subtitle_emphasis": [sub[:4]] if sub else [],
                "badge": f"{sid:02d}/{len(payload.get('sections', [])):02d}",
                "zoom_start": 1.0, "zoom_end": 1.08 if sid % 2 else 0.94,
                "hook": sid == 1,
                "note": f"mock 设计：{sec.get('visual_brief', '')}{suffix}",
            })
            sid += 1
        return {
            "shots": shots,
            "title_card": {"main": "标题卡·钩子", "sub": "mock 副标题",
                           "duration_s": 1.5},
            "end_card": {"main": "点赞关注", "sub": "mock CTA 副标题",
                         "duration_s": 1.5},
            "bgm_note": "轻快电子乐",
        }
    raise EditingLLMError(f"未知 mock 阶段：{stage}")


# 类型别名方便调用方标注（copy/script/storyboard 三产物）
StageDoc = CopyDoc | ScriptDoc | StoryboardDoc
StageGenerator = Callable[..., StageDoc]
