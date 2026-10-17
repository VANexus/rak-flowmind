"""火山 · 豆包语音「音频生成」客户端（契约 §3.1）。

调用面（契约 §3.1，接口文档 2026-10-12 对齐）：
- ``POST {tts_api_base}/api/v3/tts/create``；模型 ``seed-audio-1.0``；**非流式 HTTP**
- 鉴权 = 新版控制台 **``X-Api-Key`` 单头**；另传 ``X-Api-Request-Id``（内部 trace id）
- ``audio_config.format`` = mp3 / 24k；``speech_rate`` ∈ [-50, 100]
- ``enable_subtitle=true`` → 响应带句级 / 词级时间戳 ``subtitle.sentences[].words[]``
  （见契约 §3.1「✅ voSegments 的来源已确定」——**不用估算、不用改双向流式**）。
  **词级 token 解析为 ``VoWord`` 并随 ``voSegments[].words`` 透出**——供 ``_dsp_align``
  在超长镜的句内**下沉切到词边界**（``_dsp_align._split_point``）。
- ``text_prompt`` 上限 3000 字符，单次最长 120 秒。
  **2026-10-13 改判（ADR-069）**：旧版在这里用自然语言做「时间轴控制」（总时长 + 人声时段）
  来让配音适配**已锁定**的画面；但那是**求模型配合**、厂商不保证精度，而现在的时长真源是
  **配音的逐句实测**（画面去适配配音），故 ``build_text_prompt`` **只发纯文案**——见该函数。
- ``duration`` / ``original_duration``（秒）与 ``subtitle.sentences[].{start_time,end_time}``
  **全部读入并透出**（``TtsAudio.duration_ms`` / ``sentences``）——它们是 ``_dsp_align``
  的输入，即时间线的真源。

错误分类（对齐 ``errors.py``）：429 / 5xx → ``transient``（可重试）；
其余 4xx → ``video``；连接 / 超时 → ``environment``。空 key / 空音色**显式报错**，
绝不静默降级、绝不返回假音频。

字段已按**入仓真文档**逐条核实：``docs/vendor/豆包语音_音频生成_1790593987.pdf``
（8 页，含请求头/请求体 ``audio_config``/``watermark`` 全字段、响应顶层 ``audio``（Base64）
/``url``/``duration``(秒)/``original_duration``(秒，计费基准，上限 120s)/``subtitle`` 结构）。

错误码按**入仓真文档** ``docs/vendor/豆包语音_错误码查询_1789027400.pdf`` 分类
（已按中文释义逐条核对；该 PDF 用 ``pymupdf`` 可正常抽文，``pdftotext`` 会丢中文
——见 ``docs/vendor/README.md``）。要点：``45001115`` 参考音色不存在 → environment；
``45001116``/``45001117``/``45001001`` 输入超限、``45001104``/``45001127``/``45001130``/
``45001125`` 审核不过 → video；``45001131``/``45001132`` 参考音频/图片下载失败 →
video（改素材地址，**不是** transient）；``55001309``–``55001311`` 合成失败（内部错 /
未过安审 / 未过声纹）→ transient（文档均「重试」）；``5500130x`` DB·TOS 内部错、
``45002000`` 超并发、``55000000`` 服务端通用错 → transient。**文档总则**：
8 位码 5 开头 = 服务端报错（transient）/ 4 开头 = 客户操作错误（video）。
错误经**顶层 ``code``(int)** 返回——故先解析响应体判码，解析不到才回落 HTTP 状态。
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field

import requests

from flowmind.config import DspConfig
from flowmind.skills._secrets import get_api_key

# 单次合成上限（文档：单次最长 120 秒）；超限须分片，本契约短视频场景在限内。
MAX_SINGLE_MS = 120_000

# 错误码分类（真源 docs/vendor/豆包语音_错误码查询_1789027400.pdf，已按中文释义逐条核对）。
# 文档总则：8 位码「以 5 开头 = 服务端报错」（可重试）、「以 4 开头 = 客户操作错误」（改输入）。
_TTS_ENVIRONMENT_CODES = frozenset({
    45001115,  # 参考音色不存在 → 检查/更正音色 id（属配置，别重试）
})
_TTS_TRANSIENT_CODES = frozenset({
    45002000,  # 超并发（请求并发数超限）
    55000000,  # 服务端通用错误 → 文档「请稍后重试」
    55001301, 55001302, 55001303, 55001304,  # DB 查询/插入/更新/删除失败（服务异常，可重试）
    55001305, 55001306,  # TOS 上传/下载失败（对象存储不可用/超时）
    55001307,  # 音色克隆失败（下游失败/超时）
    55001309,  # 合成失败：内部错误 → 重试
    55001310,  # 合成失败：合成音频未通过安审 → 重试
    55001311,  # 合成失败：合成音频未通过声纹检查 → 重试
})
# **需查环境** = 凭据 / 音色 / 模型配置问题（文档里这些多为 message 而非码值）。
_TTS_ENV_MESSAGE_KEYWORDS = (
    "invalidmodel", "invaliddialect", "speaker not found", "tts invalid speaker",
    "resource id is mismatched",
)


class DspTtsError(Exception):
    """豆包语音调用失败。category/retriable 语义与 errors.py 对齐。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


@dataclass
class VoWord:
    """**词级**（token）时间戳（ms，距音频开始；厂商 ``sentences[].words[]``，与句级同形）。

    它是比句**更细**的**实测**边界：某个镜实测超长、而其内部没有句边界可切时，
    ``_dsp_align._split_point`` 靠它把切点**下沉到词**（词边界同样是实测 token，
    故「画面与配音逐帧对齐」不受影响——只是切得更细）。
    """
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass
class VoSentence:
    """句级时间戳（ms，距音频开始）+ 其下的**词级** token（可空）。"""
    start_ms: int
    end_ms: int
    text: str
    words: list[VoWord] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass
class TtsAudio:
    """合成结果：音频字节 + 逐句时间戳 + 总时长。"""
    audio_bytes: bytes
    content_type: str
    sentences: list[VoSentence]
    duration_ms: int
    voice: str


def tts_configured(cfg: DspConfig) -> bool:
    """密钥是否齐备（音色是**可选**项，见 ``_build_payload``，不参与「可配置」判定）。"""
    return bool(get_api_key(cfg.tts_key_env))


def build_text_prompt(shots: list[dict], *, max_chars: int) -> str:
    """把逐镜文案拼成 ``text_prompt``——**纯文案，无时间轴控制**。

    ``shots`` = ``[{id, copy}]``（按镜头序展开）。

    **2026-10-13 改判（ADR-069）**：旧版在这里追加一句自然语言「时间轴控制」
    （「请让整段人声总时长不超过 N ms，并按下列镜头时段排布」）——那是在**求模型配合**，
    厂商不保证精度；而现在的时长真源是**配音的逐句实测**（画面去适配配音），
    故这句既不可靠、也不再需要，**删掉**。
    """
    lines = [str(s.get("copy", "")).strip() for s in shots if str(s.get("copy", "")).strip()]
    script = "\n".join(lines)
    if len(script) > max_chars:
        raise DspTtsError(
            f"配音文本超长：{len(script)} > {max_chars} 字符（契约 §3.1 上限）。请拆分镜头或精简文案。",
            category="video",
        )
    return script


def _build_payload(cfg: DspConfig, *, text_prompt: str, speaker: str, speech_rate: int) -> dict:
    """按《豆包语音 · 音频生成》文档逐字段构造请求体。

    ``speaker`` 与 ``audio_data`` / ``audio_url`` **三者互斥**；留空（未指定音色）
    即**不带该字段**——文档允许「无参考 → 纯文本生成」，故缺音色不报错（由调用方
    如实提示「未指定音色，将用模型默认音色」）。
    """
    payload: dict = {
        "model": cfg.tts_model,
        "text_prompt": text_prompt,
        "audio_config": {
            "format": cfg.tts_format,
            "sample_rate": cfg.tts_sample_rate,
            "speech_rate": speech_rate,
            "loudness_rate": cfg.tts_loudness_rate,
            "pitch_rate": cfg.tts_pitch_rate,
            "enable_subtitle": cfg.tts_enable_subtitle,
        },
        "watermark": {
            "aigc_watermark": cfg.tts_aigc_watermark,
        },
    }
    if speaker:
        payload["speaker"] = speaker
    return payload


def _parse_words(raw: object) -> list[VoWord]:
    """解析 ``sentences[].words[]``（词级 token）——形状不对的条目**丢弃**（不臆造时间戳）。"""
    out: list[VoWord] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        start, end = item.get("start_time"), item.get("end_time")
        if start is None or end is None:
            continue
        try:
            out.append(VoWord(start_ms=int(start), end_ms=int(end),
                              text=str(item.get("text", ""))))
        except (TypeError, ValueError):
            continue
    return out


def _parse_sentences(raw: dict) -> tuple[list[VoSentence], int]:
    """从响应里取 ``subtitle.sentences``（含词级 ``words``）与总时长（缺失即如实报错，不臆造）。"""
    subtitle = None
    if isinstance(raw, dict):
        subtitle = raw.get("subtitle")
        if not isinstance(subtitle, dict) and isinstance(raw.get("data"), dict):
            subtitle = raw["data"].get("subtitle")
    sentences: list[VoSentence] = []
    duration_ms = 0
    if isinstance(subtitle, dict):
        for sent in subtitle.get("sentences") or []:
            start = int(sent.get("start_time", 0))
            end = int(sent.get("end_time", 0))
            sentences.append(VoSentence(
                start_ms=start, end_ms=end, text=str(sent.get("text", "")),
                words=_parse_words(sent.get("words")),
            ))
            duration_ms = max(duration_ms, end)
    return sentences, duration_ms


def _extract_audio(raw: dict, default_type: str) -> tuple[bytes, str]:
    """取音频字节。文档：成功响应里音频是**顶层** ``audio``（Base64）；
    另有顶层 ``url``（2 小时有效）。``default_type`` 为无 Content-Type 时的回落。"""
    data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
    b64 = data.get("audio") or data.get("audio_base64") or data.get("binary_data_base64")
    if isinstance(b64, list):
        b64 = b64[0] if b64 else None
    if isinstance(b64, str) and b64:
        try:
            return base64.b64decode(b64), default_type
        except Exception as exc:  # noqa: BLE001
            raise DspTtsError(f"音频 base64 解码失败：{exc}", category="video") from exc
    url = data.get("url") or data.get("audio_url")
    if isinstance(url, str) and url:
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise DspTtsError(f"音频下载失败：{url}（{exc}）", category="transient",
                              retriable=True) from exc
        return resp.content, resp.headers.get("Content-Type", default_type)
    raise DspTtsError(
        f"豆包语音响应里找不到音频字段（audio / url）。"
        f"实际顶层字段：{sorted(data)[:12]}",
        category="video",
    )


# 输出格式 → Content-Type（文档 audio_config.format 四值）
_FORMAT_TYPES = {"mp3": "audio/mpeg", "wav": "audio/wav",
                 "pcm": "audio/pcm", "ogg_opus": "audio/ogg"}


def _classify_tts_error(code: object, message: str) -> tuple[str, bool]:
    """按豆包错误码/消息分类为 ``(category, retriable)``（见 ``_TTS_*_CODES``）。

    优先级：显式登记 → 文档总则的**码段规则** → message 关键词 → 兜底 ``video``。
    码段规则（文档总则）：8 位码 **5 开头 = 服务端报错**（transient）、
    **4 开头 = 客户操作错误**（video，改输入）。与 ``errors.py`` 语义一致。
    """
    if isinstance(code, int):
        if code in _TTS_ENVIRONMENT_CODES:
            return "environment", False
        if code in _TTS_TRANSIENT_CODES:
            return "transient", True
        if 50_000_000 <= code < 60_000_000:
            return "transient", True
        if 40_000_000 <= code < 50_000_000:
            return "video", False
    if any(kw in (message or "").lower() for kw in _TTS_ENV_MESSAGE_KEYWORDS):
        return "environment", False
    return "video", False


def synthesize(
    cfg: DspConfig, *, text_prompt: str, speaker: str, speech_rate: int = 0,
    request_id: str = "",
) -> TtsAudio:
    """合成整段口播（非流式）；返回音频字节 + 逐句时间戳。

    这是隔离出的薄适配层（便于 demo 替换）：所有网络动作都在此函数内。
    """
    api_key = get_api_key(cfg.tts_key_env)
    if not api_key:
        raise DspTtsError(
            f"未配置豆包语音密钥 {cfg.tts_key_env}（dsp 配音必需）。",
            category="environment",
        )
    if speech_rate < -50 or speech_rate > 100:
        raise DspTtsError(f"speech_rate 越界（-50..100）：{speech_rate}", category="video")

    payload = _build_payload(cfg, text_prompt=text_prompt, speaker=speaker, speech_rate=speech_rate)
    headers = {"X-Api-Key": api_key, "Content-Type": "application/json"}
    if request_id:
        headers["X-Api-Request-Id"] = request_id
    url = cfg.tts_api_base.rstrip("/") + cfg.tts_create_path

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=cfg.tts_timeout_s)
    except requests.exceptions.Timeout as exc:
        raise DspTtsError(f"豆包语音超时（>{cfg.tts_timeout_s}s）", category="environment") from exc
    except requests.exceptions.RequestException as exc:
        raise DspTtsError(f"豆包语音连接失败：{exc}", category="environment") from exc

    if resp.status_code == 429:
        raise DspTtsError("豆包语音限流（429）", category="transient", retriable=True)

    # 文档：错误经**顶层 code(int)** 返回，HTTP 可能仍 200/4xx/5xx——**先解析体判码**，
    # 拿不到才回落 HTTP 状态（否则 45001131 这类可重试错会被状态码误判成 video）。
    try:
        raw = resp.json()
    except ValueError:
        raw = None

    if isinstance(raw, dict) and raw.get("code") not in (0, None):
        code = raw.get("code")
        message = str(raw.get("message", ""))
        category, retriable = _classify_tts_error(code, message)
        raise DspTtsError(
            f"豆包语音返回错误码 {code}：{message or '(无 message)'}",
            category=category, retriable=retriable,
        )

    if resp.status_code >= 500:
        raise DspTtsError(f"豆包语音服务端错误（{resp.status_code}）", category="transient",
                          retriable=True)
    if resp.status_code >= 400:
        raise DspTtsError(f"豆包语音拒绝请求（{resp.status_code}）：{resp.text[:300]}",
                          category="video")
    if not isinstance(raw, dict):
        raise DspTtsError("豆包语音返回非 JSON", category="video")

    audio_bytes, content_type = _extract_audio(raw, _FORMAT_TYPES.get(cfg.tts_format, "audio/mpeg"))
    sentences, sentences_end_ms = _parse_sentences(raw)
    if not sentences:
        raise DspTtsError(
            "豆包语音未返回逐句时间戳（subtitle.sentences 为空）——字幕依赖它，"
            "请确认 audio_config.enable_subtitle=true。",
            category="video",
        )
    # 时长优先取文档的 original_duration / duration（**秒**，浮点），缺失才回落到句尾
    seconds = raw.get("original_duration") or raw.get("duration") or 0
    duration_ms = int(float(seconds) * 1000) or sentences_end_ms
    return TtsAudio(
        audio_bytes=audio_bytes, content_type=content_type,
        sentences=sentences, duration_ms=duration_ms, voice=speaker or "",
    )
