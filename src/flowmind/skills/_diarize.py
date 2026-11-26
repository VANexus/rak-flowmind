"""说话人分离：VAD 切段 + CAM++ 声纹嵌入 + 谱聚类（funasr 原语，无额外 ASR）。

多人对话出海本地化的前置环节：把一条音轨切成「谁在什么时候说话」的轮次
（turns），供 ASR 句段打说话人标签、再逐说话人裁参考音频做零样本音色克隆。

技术路线（只用声纹，不加载 ASR 模型——转写已由 faster-whisper 负责）：
    1. ``fsmn-vad`` 检出语音区间（毫秒级起止）；
    2. 对每个（切分后的）语音窗抽 ``cam++`` 192 维声纹嵌入；
    3. ``ClusterBackend`` 聚类得到说话人标签（谱聚类自动估类数 + 簇心余弦合并，
       可固定类数）；
    4. 小簇按支持度折叠进时间上最近的大簇（去掉噪声离群窗被误判的「说话人」）；
    5. 合并相邻同说话人窗为轮次。

关于「VAD 段切窗」：fsmn-vad 对连续朗读/对话（停顿 < 800ms）会输出**整段**语音
（实测 40s 素材返回单段 [0, 40220]）。整段抽一个嵌入会把多个说话人混在一起，
故对超过 ``_WINDOW_SECONDS`` 的 VAD 段再等分切窗，逐窗抽嵌入——这是分离可行的
前提，不是可选优化。

关于模型版本：``cam++`` 经 funasr 名字表映射到 modelscope 仓库
``iic/speech_campplus_sv_zh-cn_16k-common``；该仓库**只有 master 一个版本**
（实测 ``model_revision="v2.0.4"`` 返回 404），故这里不钉 revision，用默认 master。

模型缓存：VAD 与 CAM++ 实例进程内缓存（``_models`` + 双检锁），首次调用经代理
下载（``MODELSCOPE_CACHE`` / ``HF_HOME`` 指向 /srv/data/models）。

lazy import：``funasr`` 未安装 → ``DiarizeError(category="environment")``；
错误永不静默，所有失败路径带 category 抛出。
"""
from __future__ import annotations

import logging
from pathlib import Path

from flowmind.skills._media import run_ffmpeg
from flowmind.tasks.gpu import model_cache_guard

logger = logging.getLogger(__name__)

DEFAULT_SPK_MODEL = "cam++"     # CAM++ 中文通用声纹（192 维）
DEFAULT_VAD_MODEL = "fsmn-vad"  # FSMN 语音端点检测（毫秒级）
DEFAULT_DEVICE = "cpu"          # 声纹/VAD 轻量，CPU 足够，规避 Pascal 显存争用

# 嵌入窗口长度（秒）：过长会把多个说话人混进一个嵌入，过短嵌入不稳。
# 1.5s 是 CAM++ 常用短窗，兼顾稳定性与说话人切换分辨率。
_WINDOW_SECONDS = 1.5
# 单窗过短阈值（秒）：< 0.4s 的嵌入方差大、易误聚，丢弃或并入相邻窗。
_MIN_SEGMENT_SECONDS = 0.4
# 相邻同说话人窗合并的最大时间间隙（秒）：跨 VAD 静音区（>0.5s）不合并，保留轮次边界。
_MERGE_GAP_SECONDS = 0.5

# 簇心余弦合并阈值（funasr ClusterBackend 默认值）：两簇簇心余弦 > 该值即合并。
# 取值偏保守——真正的说话人区分靠它；噪声离群窗由支持度折叠兜底。
DEFAULT_MERGE_THR = 0.78
# 小簇支持度下限（秒）：累计语音时长 < 该值的簇视为噪声离群，按时间折叠进最近大簇。
# 与 speaker_ref_min_seconds 同量级——低于此的「说话人」本就产不出可信参考音频。
DEFAULT_MIN_CLUSTER_SECONDS = 3.0


class DiarizeError(Exception):
    """说话人分离失败。category/retriable 语义与 errors.py 对齐。"""

    def __init__(self, message: str, *, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def available() -> bool:
    """funasr 是否可导入（离线检查，**绝不触发网络下载**）。

    funasr 已随 environment.yml 安装；本函数只做 import 探测，不加载、不下载模型。
    真正的模型下载/加载发生在**首次 ``diarize()`` 调用**时（经代理
    http://127.0.0.1:7890，缓存到 MODELSCOPE_CACHE=/srv/data/models）；若下载
    或加载失败，``diarize()`` 会抛 ``DiarizeError(category="environment")``。
    """
    try:
        import funasr  # noqa: F401  # 导入探测本身即目的，勿删
        return True
    except ImportError:
        return False


# 进程内模型缓存：key=(role, model, device)，role ∈ {"vad", "spk"}
_models: dict[tuple, object] = {}


def _get_model(role: str, model: str, device: str):
    """进程内缓存 + 双检锁懒加载（并发首调不重复加载，懒加载语义不变）。"""
    key = (role, model, device)
    cached = _models.get(key)
    if cached is not None:
        return cached
    with model_cache_guard():
        if key not in _models:
            try:
                from funasr import AutoModel
            except ImportError as exc:
                raise DiarizeError(
                    "未安装 funasr（environment.yml 已含，"
                    "conda env update -n flowmind -f environment.yml 后可用）",
                    category="environment",
                ) from exc
            try:
                _models[key] = AutoModel(
                    model=model, device=device, disable_update=True,
                )
            except Exception as exc:
                # 模型下载（网络/代理）与权重加载失败同属环境问题，显式抛出。
                logger.warning(
                    "说话人分离模型加载失败 role=%s model=%s", role, model, exc_info=True,
                )
                raise DiarizeError(
                    f"说话人分离模型加载/下载失败（{model}@{device}）: {type(exc).__name__}",
                    category="environment",
                ) from exc
    return _models[key]


def _read_wav(wav_path: str):
    """读取 wav 为 (float32 单声道 ndarray, sample_rate)。

    soundfile 是 funasr 的传递依赖（environment.yml 的 funasr 已带入），非新增依赖。
    """
    try:
        import numpy as np
        import soundfile as sf

        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    except Exception as exc:
        logger.warning("音频读取失败 path=%s", wav_path, exc_info=True)
        raise DiarizeError(
            f"音频读取失败: {type(exc).__name__}", category="video",
        ) from exc
    if audio.ndim > 1:  # 立体声 → 取均值降为单声道
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio, dtype="float32")
    if audio.size == 0:
        raise DiarizeError(f"音频为空: {wav_path}", category="video")
    return audio, int(sr)


def _run_vad(vad_model, wav_path: str) -> list[tuple[float, float]]:
    """VAD 检出语音区间，返回 [(begin_s, end_s), ...]（按时间升序）。"""
    try:
        res = vad_model.generate(input=wav_path)
    except Exception as exc:
        raise _classify_infer_error(exc, "VAD") from exc
    regions: list[tuple[float, float]] = []
    for item in res or []:
        for span in (item or {}).get("value") or []:
            # funasr VAD 输出 [begin_ms, end_ms]；末段可能以 -1 结尾表示语音未结束。
            if not span or len(span) < 2:
                continue
            begin_ms, end_ms = float(span[0]), float(span[1])
            if begin_ms < 0 or end_ms <= begin_ms:
                continue
            regions.append((begin_ms / 1000.0, end_ms / 1000.0))
    regions.sort(key=lambda r: r[0])
    return regions


def _split_windows(
    regions: list[tuple[float, float]], total_seconds: float,
) -> list[tuple[float, float]]:
    """把 VAD 区间切成嵌入窗，并处理过短窗。

    - 区间超过 _WINDOW_SECONDS 时等分切窗（连续语音整段返回时必需）；
    - 窗口按总时长裁剪到 [0, total_seconds]；
    - 时长 < _MIN_SEGMENT_SECONDS 的窗并入相邻窗（优先前一个，其次后一个），
      避免用极短音频抽嵌入导致误聚。
    """
    windows: list[tuple[float, float]] = []
    for begin, end in regions:
        begin = max(0.0, begin)
        end = min(total_seconds, end)
        if end <= begin:
            continue
        dur = end - begin
        if dur <= _WINDOW_SECONDS:
            windows.append((begin, end))
            continue
        n = int(dur / _WINDOW_SECONDS) + 1  # ceil
        step = dur / n
        for i in range(n):
            windows.append((begin + i * step, begin + (i + 1) * step))

    # 过短窗合并：按时间升序线性扫描，能并入前窗就并入，否则并入后窗。
    merged: list[list[float]] = []
    for begin, end in windows:
        if end - begin < _MIN_SEGMENT_SECONDS and merged:
            merged[-1][1] = max(merged[-1][1], end)
            continue
        merged.append([begin, end])
    # 首个窗仍过短 → 并入后一个窗（若有）。
    if len(merged) >= 2 and merged[0][1] - merged[0][0] < _MIN_SEGMENT_SECONDS:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    return [(b, e) for b, e in merged if e > b]


def _embed_windows(spk_model, audio, sr: int, windows: list[tuple[float, float]]):
    """逐窗抽 CAM++ 声纹嵌入，返回 (embeddings[N,192] ndarray, kept_windows)。

    非有限（NaN/Inf）嵌入所在窗直接丢弃——聚类前清洗，避免污染相似度矩阵。
    """
    import numpy as np

    embs: list = []
    kept: list[tuple[float, float]] = []
    for begin, end in windows:
        clip = audio[int(begin * sr): int(end * sr)]
        if clip.size < int(_MIN_SEGMENT_SECONDS * sr):
            continue
        try:
            res = spk_model.generate(input=clip, fs=sr)
        except Exception as exc:
            raise _classify_infer_error(exc, "声纹嵌入") from exc
        if not res:
            continue
        emb = res[0].get("spk_embedding")
        if emb is None:
            continue
        arr = np.asarray(emb.detach().cpu().numpy() if hasattr(emb, "detach") else emb)
        arr = arr.reshape(-1).astype("float32")
        if arr.size == 0 or not np.isfinite(arr).all():
            logger.warning("丢弃非有限声纹嵌入 window=(%.2f, %.2f)", begin, end)
            continue
        embs.append(arr)
        kept.append((begin, end))
    if not embs:
        return np.zeros((0, 0), dtype="float32"), []
    return np.vstack(embs), kept


def _cluster(
    embeddings, num_speakers: int | None, *, merge_thr: float = DEFAULT_MERGE_THR,
) -> list[int]:
    """聚类得到每窗的说话人标签（int 列表，长度 = 窗数）。

    用 funasr 官方 ``ClusterBackend``（而非裸 ``SpectralCluster``）——它比裸谱聚类
    多两道关键保护：
      * 窗数 < 20 直接判单说话人（短素材自动估类数极不稳）；
      * 自动估类数后再按簇心余弦 ``merge_thr`` 合并相似簇。
    实测 40s 单说话人素材：裸 ``SpectralCluster`` 分出 14 类（噪声离群窗被当成
    独立说话人），``ClusterBackend`` 降到 10 类，再经 ``_fold_small_clusters``
    正确判为 1 人。``num_speakers`` 给定则固定类数（此时不做余弦合并）。
    """
    n = embeddings.shape[0]
    if n == 1:
        return [0]
    k = num_speakers if num_speakers else None
    if k is not None and k > n:
        logger.warning("请求说话人数 %s 超过可用窗数 %s，按窗数聚类", k, n)
        k = n
    try:
        from funasr.models.campplus.cluster_backend import ClusterBackend

        labels = ClusterBackend(merge_thr=merge_thr)(embeddings, oracle_num=k)
    except Exception as exc:
        logger.warning("说话人聚类失败", exc_info=True)
        raise DiarizeError(
            f"说话人聚类失败: {type(exc).__name__}", category="unknown",
        ) from exc
    return [int(x) for x in labels]


def _fold_small_clusters(
    windows: list[tuple[float, float]], labels: list[int], *, min_seconds: float,
) -> list[int]:
    """把累计语音时长 < ``min_seconds`` 的小簇按时间折叠进最近的大簇。

    ``merge_by_cos`` 只按簇心余弦合并：噪声离群窗自成一簇、簇心远离主体，
    相似度低于阈值因而存活，被误当成独立说话人。这里按**支持度**（累计时长）
    兜底——小簇的窗改判为时间上最近的大簇标签。
    若不存在大簇（所有说话人都很短），原样返回，避免误并真实短对话。
    """
    dur: dict[int, float] = {}
    for (begin, end), lab in zip(windows, labels):
        dur[lab] = dur.get(lab, 0.0) + (end - begin)
    big = {lab for lab, d in dur.items() if d >= min_seconds}
    if not big:
        return labels
    big_idx = [i for i, lab in enumerate(labels) if lab in big]
    out: list[int] = []
    for i, ((begin, end), lab) in enumerate(zip(windows, labels)):
        if lab in big:
            out.append(lab)
            continue
        nearest = min(
            big_idx,
            key=lambda j: min(abs(windows[j][1] - begin), abs(end - windows[j][0])),
        )
        out.append(labels[nearest])
    return out


def _merge_turns(windows: list[tuple[float, float]], labels: list[int]) -> list[dict]:
    """合并相邻同说话人窗为轮次，并给说话人稳定编号 spk0/spk1...（按首次出现排序）。"""
    merged: list[dict] = []
    for (begin, end), label in zip(windows, labels):
        if (
            merged
            and merged[-1]["_label"] == label
            and begin - merged[-1]["end"] <= _MERGE_GAP_SECONDS
        ):
            merged[-1]["end"] = max(merged[-1]["end"], end)
        else:
            merged.append({"_label": label, "begin": begin, "end": end})

    # 稳定编号：按轮次首次出现的时间顺序给 spk0/spk1...（同一音频多次跑标签一致）。
    order: dict[int, str] = {}
    for turn in merged:
        if turn["_label"] not in order:
            order[turn["_label"]] = f"spk{len(order)}"
    return [
        {
            "begin": round(t["begin"], 3),
            "end": round(t["end"], 3),
            "speaker": order[t["_label"]],
        }
        for t in merged
    ]


def _classify_infer_error(exc: Exception, stage: str) -> DiarizeError:
    """把推理异常归类：OOM / CUDA → transient(retriable)，其余 → unknown。"""
    msg = str(exc).lower()
    if "out of memory" in msg or "cuda" in msg or "memory" in msg:
        return DiarizeError(
            f"{stage}推理显存/CUDA 失败: {type(exc).__name__}",
            category="transient", retriable=True,
        )
    return DiarizeError(
        f"{stage}推理异常: {type(exc).__name__}", category="unknown",
    )


def diarize(
    wav_path: str, *,
    num_speakers: int | None = None,
    device: str = DEFAULT_DEVICE,
    model: str = DEFAULT_SPK_MODEL,
    merge_thr: float = DEFAULT_MERGE_THR,
    min_cluster_seconds: float = DEFAULT_MIN_CLUSTER_SECONDS,
) -> list[dict]:
    """wav → 说话人轮次列表 [{begin, end, speaker}]（秒，按 begin 升序，同说话人相邻已合并）。

    - ``num_speakers`` 给定 → 固定类数聚类（不做小簇折叠，尊重显式类数）；
      None → 自动估类数（谱聚类特征间隙 + 簇心余弦合并 + 小簇支持度折叠）。
    - ``merge_thr``：簇心余弦合并阈值；``min_cluster_seconds``：小簇折叠下限。
    - 说话人标签为稳定字符串 spk0/spk1...（按首次出现时间编号）。
    - 失败一律抛 ``DiarizeError``（带 category），**绝不返回空列表掩盖失败**；
      仅当音频确实没有任何有效语音窗时返回空列表（合法结果，非错误）。
    """
    if not available():
        raise DiarizeError(
            "未安装 funasr（说话人分离不可用）。请执行："
            "conda env update -n flowmind -f environment.yml",
            category="environment",
        )
    if not Path(wav_path).exists():
        raise DiarizeError(f"音频文件不存在: {wav_path}", category="video")

    audio, sr = _read_wav(wav_path)
    total_seconds = len(audio) / float(sr)

    vad_model = _get_model("vad", DEFAULT_VAD_MODEL, device)
    regions = _run_vad(vad_model, wav_path)
    windows = _split_windows(regions, total_seconds)
    if not windows:
        logger.warning("VAD 未检出语音区间，返回空轮次 path=%s", wav_path)
        return []

    spk_model = _get_model("spk", model, device)
    embeddings, kept = _embed_windows(spk_model, audio, sr, windows)
    if not kept:
        logger.warning("所有语音窗声纹嵌入无效，返回空轮次 path=%s", wav_path)
        return []

    labels = _cluster(embeddings, num_speakers, merge_thr=merge_thr)
    if num_speakers is None:
        # 仅自动估类数时折叠小簇：显式给定类数应被尊重
        labels = _fold_small_clusters(kept, labels, min_seconds=min_cluster_seconds)
    return _merge_turns(kept, labels)


def _overlap(begin: float, end: float, turn: dict) -> float:
    """句段与轮次的时间重叠长度（秒，无重叠为 0）。"""
    return max(0.0, min(end, turn["end"]) - max(begin, turn["begin"]))


def _turn_distance(begin: float, end: float, turn: dict) -> float:
    """句段与轮次的时间距离（秒，重叠为 0；否则为两区间间隙）。"""
    return max(0.0, max(turn["begin"] - end, begin - turn["end"]))


def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """给 ASR 句段打说话人标签（**纯函数**，不碰磁盘/模型）。

    - 归属规则：取与句段**重叠最大**的轮次；完全无重叠时取时间距离最近的轮次。
    - 原键 index/begin/end/text **原样保留**，只**新增** ``speaker`` 键。
    - ``turns`` 为空 → 原样返回（不加 speaker 键），即单说话人回落路径。
    - 返回**新列表**（元素为浅拷贝），不原地修改入参。
    """
    if not turns:
        return [dict(seg) for seg in segments]

    out: list[dict] = []
    for seg in segments:
        new_seg = dict(seg)
        begin, end = seg.get("begin"), seg.get("end")
        if begin is None or end is None:
            logger.warning("句段缺少 begin/end，跳过说话人归属: %s", seg.get("index"))
            out.append(new_seg)
            continue
        begin, end = float(begin), float(end)
        best = turns[0]
        best_overlap = _overlap(begin, end, best)
        best_dist = _turn_distance(begin, end, best)
        for turn in turns[1:]:
            ov = _overlap(begin, end, turn)
            dist = _turn_distance(begin, end, turn)
            if ov > best_overlap or (ov == best_overlap and dist < best_dist):
                best, best_overlap, best_dist = turn, ov, dist
        if best_overlap <= 0.0:
            # 无重叠 → 取时间距离最近的轮次（距离相同取更早者，稳定）
            best = min(
                turns,
                key=lambda t: (_turn_distance(begin, end, t), t["begin"]),
            )
        new_seg["speaker"] = best["speaker"]
        out.append(new_seg)
    return out


def _select_clips(segs: list[dict], total_seconds: float, *,
                  target_seconds: float, max_seconds: float) -> list[dict]:
    """按时间顺序贪心选取句段，使累计时长尽量接近 target 且不超过 max。

    返回被选中的句段（含裁切后的 begin/end），供音频裁切与 ref_text 对应。
    """
    clips: list[dict] = []
    total = 0.0
    for seg in segs:
        begin, end = seg.get("begin"), seg.get("end")
        if begin is None or end is None:
            continue
        begin = max(0.0, float(begin))
        end = min(total_seconds, float(end))
        dur = end - begin
        if dur <= 0:
            continue
        if total + dur > max_seconds:
            if not clips:  # 单个句段就超上限 → 截断到 max_seconds（总比一个都不产出好）
                end = begin + max_seconds
                clips.append({"begin": begin, "end": end, "text": str(seg.get("text", ""))})
            break
        clips.append({"begin": begin, "end": end, "text": str(seg.get("text", ""))})
        total += dur
        if total >= target_seconds:
            break
    return clips


def _concat_clips(wav_path: str, clips: list[dict], out_path: str) -> None:
    """用 ffmpeg atrim + concat 把多个时间段裁切拼接为单个 wav。"""
    parts = []
    labels = []
    for i, clip in enumerate(clips):
        parts.append(
            f"[0:a]atrim=start={clip['begin']:.3f}:end={clip['end']:.3f},"
            f"asetpts=N/SR/TB[a{i}]"
        )
        labels.append(f"[a{i}]")
    filter_str = ";".join(parts) + ";" + "".join(labels) + \
        f"concat=n={len(clips)}:v=0:a=1[out]"
    rc, _, err = run_ffmpeg([
        "ffmpeg", "-y", "-i", wav_path,
        "-filter_complex", filter_str, "-map", "[out]", out_path,
    ])
    if rc != 0:
        raise DiarizeError(f"参考音频拼接失败: {err[-200:]}", category="video")


def build_speaker_refs(
    segments: list[dict], wav_path: str, out_dir: str, *,
    min_seconds: float = 3.0, target_seconds: float = 10.0, max_seconds: float = 15.0,
) -> dict[str, dict]:
    """按说话人裁切拼接参考音频 + 对应转写，供零样本音色克隆。

    对每个 ``seg["speaker"]`` 分组，按时间顺序贪心裁切句段音频拼成
    ``out_dir/ref_<speaker>.wav``，累计时长尽量接近 ``target_seconds``、
    不超过 ``max_seconds``；``ref_text`` 严格对应被裁进去的那些句段的 text。

    返回 ``{speaker: {"ref_audio": 绝对路径, "ref_text": str, "seconds": float,
    "low_confidence": bool}}``；累计可用音频 < ``min_seconds`` 时仍产出（用全部
    可用音频）但 ``low_confidence=True``（**不报错**）。

    ``segments`` 中无 ``speaker`` 键的句段跳过；ffmpeg 失败抛
    ``DiarizeError(category="video")``。
    """
    if not Path(wav_path).exists():
        raise DiarizeError(f"音频文件不存在: {wav_path}", category="video")

    _, sr = _read_wav(wav_path)  # 顺带校验可读 + 拿时长
    import soundfile as sf

    total_seconds = sf.info(wav_path).frames / float(sr)

    groups: dict[str, list[dict]] = {}
    for seg in segments:
        speaker = seg.get("speaker")
        if not speaker:
            continue
        groups.setdefault(str(speaker), []).append(seg)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    refs: dict[str, dict] = {}
    for speaker in sorted(groups):
        segs = sorted(groups[speaker], key=lambda s: float(s.get("begin", 0.0)))
        clips = _select_clips(
            segs, total_seconds,
            target_seconds=target_seconds, max_seconds=max_seconds,
        )
        if not clips:
            logger.warning("说话人 %s 无有效音频，跳过参考音频生成", speaker)
            continue
        out_path = out_root / f"ref_{speaker}.wav"
        _concat_clips(wav_path, clips, str(out_path))
        seconds = round(sum(c["end"] - c["begin"] for c in clips), 3)
        refs[speaker] = {
            "ref_audio": str(out_path.resolve()),
            "ref_text": "".join(c["text"] for c in clips),
            "seconds": seconds,
            "low_confidence": seconds < min_seconds,
        }
    return refs
