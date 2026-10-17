"""dsp 时间线对齐（**纯函数**，零 I/O、零第三方依赖）：把配音的**逐句实测**时间轴归属到镜头，
再按实测时长做**节奏收口**（并镜 / 拆镜）。

契约 §3.2（ADR-069 / ADR-070 改判后）：镜头时长 = 该镜**分到的配音句子的实测时长**——
真源是配音，不是模板常量（**六个内置模板已整块删除**，ADR-070：镜头的「有几个、切在哪」
由文案说了算）。比例兜底的**权重**由**文案字数**自算（``speakable_len``）——不再是前端下发的 ``weight``。

## 两级归属（先精确、后比例）

- **一级 · 精确（文配）**：发给 TTS 的 script 就是各镜文案按 ``\\n`` 拼接（见 ``_dsp_tts.build_text_prompt``），
  故「每句来自哪个镜头」**本已知**——按顺序把返回句子与各镜文案做**归一文配**（只留字母/数字、折叠大小写）
  即得精确归属（``matched=True``）。**某一镜没对上不牵连后面的镜**：跳过该镜、把句子留给后面的镜继续试。
- **二级 · 比例兜底**：只有那些**没对上**的镜才走比例——把它们**所在空档**里的剩余句子按权重
  连续切分（顺序不变、句子不拆、每镜至少一句）。空档里没有句子 ⇒ 该镜是**无口播镜**。
  该镜若本有文案却拿不到句子，**如实告警**「归属为推断 / 无对应句子」。

## 轨严格连续（画面与配音逐帧对齐）

时间边界落在**实测 token（句 / 词）轴**上：``durationMs_i = end_ms[q_i] − end_ms[q_{i−1}]``（第一镜自 0 起）、
``atMs_i = atMs_{i−1} + durationMs_{i−1}``——token 间停顿归**前**一镜（保持前镜画面）。
``total_ms`` 即音频总长（末尾静默并入末镜），故**不存在「配音与时间线偏差」这个类别**
（旧甲案靠自然语言求配音配合，才有 drift）。

## 节奏收口（ADR-070 的 L2）：`consolidate_shots`

「有几个镜」由文案定，但**文案没写好的地方由规则兜底**（都在实测轴上做，故只能运行期做）：

- **并镜**：实测 < ``min_ms``（默认 1500）的镜并入相邻镜——防「一句话一闪」。
- **拆镜**：实测 > ``max_ms``（默认 6000）的镜在**其内部最近的合法边界**一分为二——防「一张图钉太久」。
  **两级下沉**（``_split_point``）：一级切**句边界**（句间停顿，观感最干净）；一级切不出（典型 =
  该镜就是一整句且超长）则**下沉到词边界**（厂商词级 token）；连词级 token 都没有时才真的「不拆」。
  切点永远是**实测** token 边界 ⇒ 画面与配音仍逐帧对齐。
- 收口后镜数越界 ``[2, 10]`` 时**只告警、不夹取**（数据诚实）；终版镜的 id 重排为 ``s1..sK``，
  故下游**必须**读 EDL 的 ``shots[].copy``（``timeline`` 已把文案落进 EDL），**不得回查 ``inp.shots``**。

## 无口播镜：吃「空隙」，不够才用显式常量

配音是**一个音频文件**，所以往中间**插时间**会把后面的画面整体推离配音。故：

- **中间**的无口播镜：只吃它所在位置的**停顿**（按镜数均分）。停顿不够就取**显式常量**
  ``silent_shot_ms``，并**如实告警「画面比音频长 N ms（这一段会失同步）」**——绝不假装对齐。
- **尾部**的无口播镜：吃**尾部静默**（音频尾部没有语音的部分），不足则取 ``silent_shot_ms``
  ——尾部拉长**不破坏同步**（后面没有画面了），但**照样如实标注**「时长取静默常量、非实测」。

常量就是常量：凡是用到它，一律进 ``warnings``，**绝不伪装成实测值**。
"""
from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

# 无口播镜的展示时长（ms）——**显式常量**（非实测）。仅在「空隙不够」时使用。见模块 docstring。
DEFAULT_SILENT_SHOT_MS = 1500

# 段时长下限（ms）：厂商若给出 start == end 的退化句，段长为 0 会让渲染侧拿到零长段。
# 夹到 1ms 并**如实告警**（本仓禁止的是「静默夹取」，不是「标注后夹取」）。
MIN_SEGMENT_MS = 1


@dataclass(frozen=True)
class AlignShot:
    """对齐所需的最小镜头视图。"""
    id: str
    copy: str
    weight: float = 1.0


@dataclass(frozen=True)
class AlignWord:
    """**词级**（token）实测边界（来自厂商 ``subtitle.sentences[].words[]``，与句级同形）。

    用途单一：某镜实测超过上限、而其**内部没有句边界**可切时，切点**下沉到词**
    ——词边界同为**实测** token，故「画面与配音逐帧对齐」不受影响（只是切得更细）。
    """
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class AlignSentence:
    """配音的**逐句实测**时间轴（来自厂商 ``subtitle.sentences``，非估算）。

    ``words`` = 该句的**词级** token（可空）——供 ``consolidate_shots`` 的**词级切分**下沉。
    """
    start_ms: int
    end_ms: int
    text: str
    words: tuple[AlignWord, ...] = ()


@dataclass(frozen=True)
class ShotSpan:
    """单镜在时间线上的落点。"""
    shotId: str
    atMs: int
    durationMs: int
    sentenceCount: int
    """归属是否由**文配精确**得到（False = 比例推断 / 无口播）。"""
    matched: bool
    """是否无口播镜（未分到任何句子）。"""
    silent: bool


@dataclass(frozen=True)
class AlignResult:
    """对齐结果：逐镜落点 + 总量 + 如实说明 + 计数（供推理链与告警用）。"""
    spans: list[ShotSpan]
    total_ms: int
    warnings: list[str]
    anchored_shots: int
    inferred_shots: int
    silent_shots: int
    mismatched_shots: int
    unused_sentences: int
    trailing_ms: int
    """画面因空隙不足而被拉长、超出音频的毫秒数（> 0 即该段与配音失同步）。"""
    desync_ms: int


def _norm(text: str) -> str:
    """归一：只保留**字母与数字**（Unicode 类别 ``L*`` / ``N*``）、折叠大小写。

    这样「你好，世界！」与「你好世界」等价（标点不发音），空白的差异也被抹平。
    **不做**中文数字 ↔ 阿拉伯数字的互转——那是另一回事，对不上就如实走比例兜底。
    """
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in ("L", "N")).lower()


def _dur(sentence: AlignSentence) -> int:
    return max(0, sentence.end_ms - sentence.start_ms)


def _consistent(shot_norm: str, consumed_norm: str) -> bool:
    """前缀一致（任一方是对方的前缀）即认为「还在对的这一镜上」。"""
    return (
        consumed_norm == shot_norm
        or shot_norm.startswith(consumed_norm)
        or consumed_norm.startswith(shot_norm)
    )


def _anchor(shots: Sequence[AlignShot], sentences: Sequence[AlignSentence]) -> list[tuple[int, int] | None]:
    """一级：按文案原文**精确**锚定「句子 → 镜头」。

    某一镜对不上即**跳过该镜**（不消耗句子），把句子留给后面的镜继续试——
    这样「一镜的文案被厂商改写」不会牵连其后所有镜。
    """
    m = len(sentences)
    ranges: list[tuple[int, int] | None] = [None] * len(shots)
    qi = 0
    for i, shot in enumerate(shots):
        want = _norm(shot.copy)
        if not want or qi >= m:
            continue  # 无文案 / 句子已用尽 → 无口播镜
        start = qi
        consumed = ""
        while qi < m:
            consumed += _norm(sentences[qi].text)
            qi += 1
            if consumed == want:
                break
            if not _consistent(want, consumed):
                qi = start  # 对不上 → 句子原样留给后面的镜
                consumed = ""
                break
        if consumed and _consistent(want, consumed):
            ranges[i] = (start, qi - 1)
    return ranges


def _proportional_split(
    shot_idx: list[int],
    weights: Sequence[float],
    sentence_idx: list[int],
    sentences: Sequence[AlignSentence],
) -> dict[int, tuple[int, int] | None]:
    """把 ``sentence_idx`` 按 ``weights`` 比例**连续**切给 ``shot_idx`` 各镜。

    不变量：顺序不变、句子不拆、非末镜**给后面的镜各留至少一句**（句子够时）。
    ``None`` = 该镜没分到句子（句子已用尽）→ 作无口播镜处理。
    """
    out: dict[int, tuple[int, int] | None] = {}
    if not shot_idx:
        return out
    total_w = sum(max(0.0, weights[i]) for i in shot_idx) or 1.0
    total_ms = float(sum(_dur(sentences[j]) for j in sentence_idx))
    k = 0
    for n, i in enumerate(shot_idx):
        remaining = len(shot_idx) - n - 1
        limit = len(sentence_idx) - remaining  # 每镜至少一句 ⇒ 后面各留 1 句
        if k >= limit:
            out[i] = None
            continue
        target = float("inf") if remaining == 0 else total_ms * (max(0.0, weights[i]) / total_w)
        start_k = k
        acc = 0.0
        while k < limit and acc < target:
            acc += _dur(sentences[sentence_idx[k]])
            k += 1
        if k == start_k:  # 目标极小（权重占比很小）时也至少吃一句（limit 已保证后面有货）
            k += 1
        out[i] = (sentence_idx[start_k], sentence_idx[k - 1])
    return out


def _fill_gaps(
    shots: Sequence[AlignShot],
    ranges: list[tuple[int, int] | None],
    sentences: Sequence[AlignSentence],
) -> None:
    """二级：把「没被锚定镜占用的句子」按权重分给**相邻空档**里的未锚定镜（原地改 ``ranges``）。

    空档 = 相邻两个锚定范围之间（含首尾）的句子。空档里没有句子 ⇒ 那几镜保持 ``None``（无口播）。
    """
    n, m = len(shots), len(sentences)
    i = 0
    free_from = 0  # 此下标之前的句子已被锚定镜占用
    while i < n:
        if ranges[i] is not None:
            free_from = ranges[i][1] + 1  # type: ignore[index]
            i += 1
            continue
        k = i
        while k < n and ranges[k] is None:
            k += 1
        next_start = ranges[k][0] if k < n and ranges[k] is not None else m  # type: ignore[index]
        targets = [j for j in range(i, k) if _norm(shots[j].copy)]
        available = list(range(free_from, max(free_from, next_start)))
        if targets and available:
            for j, rng in _proportional_split(
                targets, [s.weight for s in shots], available, sentences
            ).items():
                ranges[j] = rng
        i = k


def _silent_durations(run: int, available: int, silent_ms: int) -> tuple[list[int], int, bool]:
    """静默段时长切分 → ``(逐镜时长, 超出音频的毫秒数, 是否用了显式常量)``。

    ``available`` = 该段可用的空隙（中间 = 停顿；尾部 = 尾部静默）。
    **均分**（静默段没有语音，均分是诚实的neutral口径，不做权重脑补）。
    空隙够 → 吃空隙、零超出；不够 → 取显式常量 ``silent_ms`` 并如实报出超出量。
    """
    available = max(0, available)
    if available >= run * MIN_SEGMENT_MS:
        base, rem = divmod(available, run)
        durs = [base] * run
        durs[-1] += rem  # 余数归末镜 ⇒ 合计严格 == available
        return durs, 0, False
    durs = [silent_ms] * run
    return durs, run * silent_ms - available, True


def align_sentences_to_shots(
    shots: Sequence[AlignShot],
    sentences: Sequence[AlignSentence],
    *,
    silent_shot_ms: int = DEFAULT_SILENT_SHOT_MS,
    audio_total_ms: int | None = None,
) -> AlignResult:
    """把**逐句实测**归属到各镜 → 逐镜 ``(atMs, durationMs)`` + 如实告警。

    ``audio_total_ms``：音频文件总长（ms）。给了即以此为 ``total_ms``，末镜**吸收末尾静默**，
    确保「画面总长 ≡ 音频总长」（除空隙不足被拉长的那部分，会显式计入 ``desync_ms``）。
    """
    warnings: list[str] = []
    m = len(sentences)
    n = len(shots)

    if n == 0:
        return AlignResult([], 0, ["无镜头：无可对齐内容"], 0, 0, 0, 0, m, 0, 0)

    silenced = max(1, int(silent_shot_ms))
    if m == 0:
        warnings.append(
            f"配音未返回逐句时间戳：{n} 镜全部无口播，时长一律取静默常量 {silenced}ms（**非实测**）"
        )
        spans = [ShotSpan(s.id, i * silenced, silenced, 0, False, True) for i, s in enumerate(shots)]
        total = _int_or(audio_total_ms, n * silenced)
        return AlignResult(spans, total, warnings, 0, 0, n, 0, 0, 0, 0)

    # ── 归属：一级锚定 → 二级补空档 ──
    ranges = _anchor(shots, sentences)
    # 「未对上」必须在**填档之前**捕获：填档后这些镜也会拿到句子（比例推断），无法再区分
    mismatched = sum(1 for i in range(n) if ranges[i] is None and _norm(shots[i].copy))
    _fill_gaps(shots, ranges, sentences)

    inferred = sum(1 for i in range(n) if ranges[i] is not None and not _matches(shots[i], sentences, ranges[i]))

    # ── 时间解析：边界全部落在句级轴上 ⇒ 轨严格连续 ──
    spans: list[ShotSpan] = []
    warnings_local: list[str] = []
    cursor = 0
    last_end = 0  # 已消费到的音频时刻（上一句的结束）
    trailing = 0
    desync = 0

    i = 0
    while i < n:
        rng = ranges[i]
        if rng is not None:
            p, q = rng
            end = sentences[q].end_ms
            dur = end - cursor
            if dur < MIN_SEGMENT_MS:
                warnings_local.append(
                    f"镜头 {shots[i].id} 的实测时长为 {dur}ms（厂商时间戳退化）——"
                    f"已夹到 {MIN_SEGMENT_MS}ms 以保证渲染可用（如实报出，非静默夹取）"
                )
                dur = MIN_SEGMENT_MS
            spans.append(ShotSpan(shots[i].id, cursor, dur, q - p + 1, True, False))
            cursor += dur
            last_end = end
            i += 1
            continue

        # 连续无口播段 [i, k)
        k = i
        while k < n and ranges[k] is None:
            k += 1
        run = k - i
        if k < n:
            # 中间段：只能吃「停顿」（下一句起点 − 已消费时刻）——插时间会推开后面的画面
            next_range = ranges[k]
            assert next_range is not None  # k 是第一个有范围的镜
            available = max(0, sentences[next_range[0]].start_ms - last_end)
            is_trailing = False
        else:
            # 尾部段：吃「尾部静默」（音频总长 − 已消费时刻）——拉长不破坏同步
            available = max(0, _int_or(audio_total_ms, 0) - last_end)
            is_trailing = True
        durs, over, used_const = _silent_durations(run, available, silenced)
        for off, dur in enumerate(durs):
            spans.append(ShotSpan(shots[i + off].id, cursor, dur, 0, False, True))
            cursor += dur
        if used_const:
            where = "尾部静默不足" if is_trailing else "该处停顿不足"
            warnings_local.append(
                f"{run} 个镜头无配音句（{where}）：时长取静默常量 {silenced}ms（**非实测**）"
            )
        if over > 0 and not is_trailing:
            desync += over
            warnings_local.append(
                f"上段无配音镜使画面比音频长 {over}ms（这些镜之后**画面与配音失同步**）——"
                "请补文案或删镜后重跑。"
            )
        i = k

    # 末尾静默（音频尾巴）并入末镜 ⇒ 画面总长 ≡ 音频总长
    if audio_total_ms and audio_total_ms > cursor:
        trailing = audio_total_ms - cursor
        last = spans[-1]
        spans[-1] = replace(last, durationMs=last.durationMs + trailing)
        cursor += trailing
    elif audio_total_ms and audio_total_ms < last_end:
        warnings_local.append(
            f"配音句级时间轴终点 {last_end}ms 超过音频总长 {audio_total_ms}ms（厂商时间戳异常）——"
            "时间线以句级轴为准，请试听核对。"
        )

    silent_shots = sum(1 for s in spans if s.silent)
    if mismatched:
        got = max(0, mismatched - silent_shots)
        parts: list[str] = []
        if got:
            parts.append(f"其中 {got} 镜按权重比例**推断**时长")
        if silent_shots:
            parts.append(f"{silent_shots} 镜没有对应句子（按无口播处理）")
        warnings_local.append(
            f"{mismatched} 个镜头的文案与配音逐句**未对上**："
            + ("；".join(parts) if parts else "无法归属")
            + "——请试听核对。"
        )
    warnings.extend(warnings_local)

    return AlignResult(
        spans=spans,
        total_ms=cursor,
        warnings=warnings,
        anchored_shots=sum(1 for s in spans if s.matched),
        inferred_shots=inferred,
        silent_shots=silent_shots,
        mismatched_shots=mismatched,
        unused_sentences=max(0, m - sum(s.sentenceCount for s in spans)),
        trailing_ms=trailing,
        desync_ms=desync,
    )


def _matches(shot: AlignShot, sentences: Sequence[AlignSentence],
             rng: tuple[int, int]) -> bool:
    """该镜的范围是否由**文配精确**得到（用于区分「锚定」与「比例推断」）。"""
    p, q = rng
    consumed = "".join(_norm(sentences[j].text) for j in range(p, q + 1))
    return _consistent(_norm(shot.copy), consumed)


def _int_or(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def speakable_len(text: str) -> int:
    """**可念字数**（只数字母与数字，忽略标点空白）——比例兜底的**权重**。

    ADR-070 起不再由前端下发 ``weight``（模板已删）：权重改由**文案字数**自算——
    字数即说话时长，比作者手填的相对权重更有依据，且少一个可漂移的字段。
    """
    return len(_norm(text))


# ── 节奏收口（ADR-070）：按实测时长并镜 / 拆镜 ──────────────────────

# 收口后镜头数的合法区间（与 cross-dashboard 的 `DSP_SHOT_LIMIT` 同值；越界只告警不夹取）
SHOT_LIMIT_MIN = 2
SHOT_LIMIT_MAX = 10


@dataclass(frozen=True)
class CutShot:
    """收口所需的镜头视图（比 ``AlignShot`` 多带重建终版镜头所需的字段）。"""
    id: str
    copy: str
    clip_keys: tuple[str, ...] = ()
    """该镜的**运镜方案**（ADR-076「运镜导演」看图产出；``None`` = 摄影机固定）。
    **并镜取主导镜的方案、拆镜两半共用**（同一画面意图），与 ``visual_prompt`` 同款语义。
    ⚠️ 用 ``Any`` 而非 ``common.DspShotMotion``：本模块是**纯算法模块**（只依赖标准库 + 本文件），
    引入 pydantic 模型会把它绑死在协议层——这里只需**原样搬运**该对象。"""
    motion: Any = None
    """该镜的**画面提示词**（ADR-073；与 ``copy`` 口播文案分开）——喂图生视频模型的画面描述。
    **并镜 = 各镜画面描述按行拼接；拆镜 = 两半共用**（同一画面意图）。下游 ``image_to_video``
    优先用它出 prompt，缺则回落 ``copy`` 并如实告警。"""
    visual_prompt: str = ""
    """该镜**末尾**的转场覆盖（符号名；None = 硬切）。**只作用于该镜末尾**——若该镜被拆成两半，
    内部那刀按硬切，用户设的转场保留在**整镜末尾**（ADR-070）。"""
    transition_after: str | None = None


@dataclass(frozen=True)
class CutResult:
    """收口结果：终版镜头（id 重排 ``s1..sK``）+ 与之一一对应的 ``spans`` + 如实说明。"""
    shots: list[CutShot]
    spans: list[ShotSpan]
    warnings: list[str]
    merged: int
    """被并掉的镜头数（原镜数 − 终版镜数里因并镜减少的部分）。"""
    split: int
    """被拆开的镜头数。"""
    boundary_split: list[bool]
    """相邻终版镜之间的边界成因（长度 = 终版镜数 − 1）：``True`` = 同一镜被**时长拆开**的边界
    （按规则应硬切），``False`` = **信息点之间**的边界（用户设的转场落在这里）。"""


def _span_end(span: ShotSpan) -> int:
    return span.atMs + span.durationMs


def _merge_groups(spans: Sequence[ShotSpan], min_ms: int) -> list[list[int]]:
    """把**过短**（实测 < ``min_ms``）的镜并入相邻镜（按声明序，贪心向前吸收）。

    只在「当前组仍太短」时继续吸收下一镜——故短镜会连着并，直到够长为止。
    """
    groups: list[list[int]] = [[0]]
    for i in range(1, len(spans)):
        cur = groups[-1]
        cur_dur = sum(spans[j].durationMs for j in cur)
        if cur_dur < min_ms:
            cur.append(i)
        else:
            groups.append([i])
    # 收尾：末组太短且不止一组 → 并进前一组（否则末尾会留一个一闪而过的镜）
    if len(groups) > 1 and sum(spans[j].durationMs for j in groups[-1]) < min_ms:
        groups[-2].extend(groups.pop())
    return groups


def consolidate_shots(
    shots: Sequence[CutShot],
    spans: Sequence[ShotSpan],
    sentences: Sequence[AlignSentence],
    *,
    min_ms: int,
    max_ms: int,
    limit_min: int = SHOT_LIMIT_MIN,
    limit_max: int = SHOT_LIMIT_MAX,
) -> CutResult:
    """**节奏收口**（ADR-070 的 L2）：按**实测**时长把分镜并/拆成终版镜头。

    - **并镜**：实测 < ``min_ms`` 的镜并入相邻镜（一次没够就继续并）——防「一句话一闪」。
    - **拆镜**：实测 > ``max_ms`` 的镜在**其内部最近的合法边界**一分为二——防「一张图钉死太久」。
      **两级下沉**（``_split_point``）：一级切**句边界**（句间停顿，观感最干净）；一级切不出则
      **下沉到词边界**（厂商词级 token）。拆点取后半段首个 token 起点（停顿归前一半，与
      ``align_sentences_to_shots`` 的连续口径一致），且两半都尽量 ≥ ``min_ms``；
      句、词边界都没有合适的就**不拆**（如实告警）。

    切点**永远落在实测 token（句 / 词）边界上** ⇒ 画面与配音仍逐帧对齐（ADR-069 的承重结构不动）。
    收口后镜数越界 ``[limit_min, limit_max]`` 时**只告警、不夹取**（数据诚实）。
    ``shots`` 与 ``spans`` 必须**索引对齐**（``spans`` 由 ``align_sentences_to_shots`` 产出）。
    """
    warnings: list[str] = []
    if not shots or not spans or len(shots) != len(spans):
        return CutResult(list(shots), list(spans), warnings, 0, 0, [max(0, len(shots) - 1) * False])

    lo = max(1, int(min_ms))
    hi = max(lo + 1, int(max_ms))
    groups = _merge_groups(spans, lo)
    merged = len(spans) - len(groups)

    final_shots: list[CutShot] = []
    final_spans: list[ShotSpan] = []
    group_of: list[int] = []  # 每个终版镜头来自哪个输入镜组（用于推导边界成因）
    split_count = 0
    sentence_splits = 0  # 其中切在**句边界**上的
    word_splits = 0      # 其中**下沉到词边界**的（该处内部无句边界）

    for gi, group in enumerate(groups):
        at = spans[group[0]].atMs
        dur = sum(spans[j].durationMs for j in group)
        # 主导成员 = 实测最长的那一镜（决定画面意图与运镜；并镜后仍要有个明确的表达）
        dominant = max(group, key=lambda j: spans[j].durationMs)
        copy = "".join(shots[j].copy for j in group)
        # 画面提示词（ADR-073）：并镜 = 各镜画面描述**按行拼接**（跳过空串）；拆镜时两半共用 base，自然继承
        visual_prompt = "\n".join(
            v for j in group if (v := shots[j].visual_prompt.strip())
        )
        clip_keys: list[str] = []
        for j in group:
            for key in shots[j].clip_keys:
                if key and key not in clip_keys:
                    clip_keys.append(key)
        base = CutShot(
            id="",  # 收尾统一重排
            copy=copy,
            visual_prompt=visual_prompt,
            clip_keys=tuple(clip_keys),
            motion=shots[dominant].motion,  # ADR-076：并镜取主导镜的运镜方案（拆镜两半共用）
            transition_after=shots[group[-1]].transition_after,  # 转场只属于**整镜末尾**
        )

        # ── 拆镜：整组超长时，在内部**最近的合法边界**一分为二（先句边界，句内下沉到词边界） ──
        piece: int | None = None
        boundary_kind: str | None = None
        if dur > hi:
            piece, boundary_kind = _split_point(sentences, at, dur, lo)
        if dur > hi and piece is None:
            warnings.append(
                f"镜头 {shots[dominant].id} 实测 {dur}ms 超过上限 {hi}ms，"
                f"但其内部没有可用的句/词边界（或拆开后两半都不足 {lo}ms）——**不拆**，如实保留。"
            )
        if piece is not None:
            first_dur = piece - at
            second_dur = dur - first_dur
            # 内部那刀按硬切：用户设的转场是给**整镜末尾**的，不该落到拆出来的中间边界上
            final_shots.append(replace(base, transition_after=None))
            final_spans.append(ShotSpan("", at, first_dur, 0, spans[group[0]].matched, spans[group[0]].silent))
            final_shots.append(base)
            final_spans.append(ShotSpan("", at + first_dur, second_dur, 0, spans[group[-1]].matched, spans[group[-1]].silent))
            group_of.extend([gi, gi])
            split_count += 1
            if boundary_kind == "word":
                word_splits += 1
            else:
                sentence_splits += 1
            continue

        final_shots.append(base)
        final_spans.append(
            ShotSpan(
                shotId="",
                atMs=at,
                durationMs=dur,
                sentenceCount=sum(spans[j].sentenceCount for j in group),
                matched=all(spans[j].matched for j in group),
                silent=all(spans[j].silent for j in group),
            )
        )
        group_of.append(gi)

    # ── 重排 id 并与 spans 对齐（``s1..sK``；EDL 的 shots 与 image 轨段都按这个序） ──
    k = len(final_shots)
    final_shots = [replace(s, id=f"s{i + 1}") for i, s in enumerate(final_shots)]
    final_spans = [replace(sp, shotId=f"s{i + 1}") for i, sp in enumerate(final_spans)]

    # 边界成因：相邻两终版镜来自**同一输入镜组** ⇒ 这刀是「同镜被时长拆开」（按规则硬切），
    # 否则是**信息点之间**的边界（用户设的转场落在这里）。
    kinds = [group_of[i] == group_of[i + 1] for i in range(max(0, k - 1))]

    if merged:
        warnings.append(f"并镜 {merged} 个：实测不足 {lo}ms 的镜已并入相邻镜（切点仍在句边界上）。")
    if split_count:
        parts: list[str] = []
        if sentence_splits:
            parts.append(f"{sentence_splits} 处切在句边界")
        if word_splits:
            parts.append(f"{word_splits} 处**下沉到词边界**（该处内部无句边界）")
        warnings.append(
            f"拆镜 {split_count} 个：实测超过 {hi}ms 的镜已在其内部一分为二——{'、'.join(parts)}。"
        )
    if k < limit_min or k > limit_max:
        warnings.append(
            f"收口后镜头数 {k} 越界 [{limit_min}, {limit_max}]——**如实报出、不静默夹取**，"
            "请调整文案长度或目标时长后重跑。"
        )

    return CutResult(final_shots, final_spans, warnings, merged, split_count, kinds)


def _split_point(
    sentences: Sequence[AlignSentence], at_ms: int, dur_ms: int, min_ms: int
) -> tuple[int | None, str | None]:
    """在 ``[at_ms, at_ms+dur_ms)`` 内找**最近的合法内部边界** → ``(切点ms, 边界类型)``。

    两级（**先句后词** = 「下沉」）：

    1. **句边界**（一级）：内部**句起点**里取离 ``mid`` 最近者。切在句间停顿上，画面观感最干净。
    2. **词边界**（二级 · 下沉）：内部**句内词起点**里取离 ``mid`` 最近者。**仅当一级无合法切点时启用**
       ——典型场景 = 该镜就是一整句且超长（句级切不出任何内部边界，只能切词）；厂商未返回词级
       token 时自然退化为「仍不拆」。

    合法性：两半都 ≥ ``min_ms``。都找不到返回 ``(None, None)``（该镜不拆）。
    切点取「后半段第一个 token 的起点」⇒ token 间停顿归**前**一半
    （与 ``align_sentences_to_shots`` 的连续口径一致），故画面与配音仍逐帧对齐。
    """
    mid = at_ms + dur_ms / 2
    end = at_ms + dur_ms

    def nearest(cuts: Sequence[int]) -> int | None:
        best: int | None = None
        best_gap: float | None = None
        for cut in cuts:
            if cut <= at_ms or cut >= end:
                continue  # 不是**内部**边界
            if cut - at_ms < min_ms or end - cut < min_ms:
                continue  # 拆开后有半段太短
            gap = abs(cut - mid)
            if best_gap is None or gap < best_gap:
                best, best_gap = cut, gap
        return best

    cut = nearest([s.start_ms for s in sentences])
    if cut is not None:
        return cut, "sentence"
    cut = nearest([w.start_ms for s in sentences for w in s.words])
    if cut is not None:
        return cut, "word"
    return None, None

