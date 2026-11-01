"""流水线步骤注册表 —— flowmind 所有技能步骤的唯一真源（SaaS 前端契约）。

设计目标（2026-10-31）：
  1. **声明式**：步骤 id / 中文名 / 进度区间 / 依赖 / 是否必需 / 产物键，
     一处定义，流水线执行、跳过校验、前端渲染、manifest 暴露全部读它；
  2. **可跳过**：前端能把 `steps`（白名单）或 `skip_steps` 传进来，
     依赖不满足时**显式 422**（绝不静默跑半截）；
  3. **可观测**：每步产出结构化状态（pending/running/done/skipped/failed
     + 起止时间 + 实际后端 + 跳过原因），落 PG 并随任务状态返回。

与 dsp 族的关系：dsp 的 DAG 由 cross-dashboard 的 DSP_STEPS 编排（外部 runner），
本注册表只覆盖 **flowmind 内部自跑的流水线**（当前 = localize_video 九阶段）。
"""
from __future__ import annotations

import threading
import time as _time
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class StepSpec:
    """单个流水线步骤的声明。"""

    id: str                       # 稳定 id（前端/API 用，永不改名）
    label: str                    # 中文名（前端展示）
    pct_start: float              # 进度区间下限（0-100）
    pct_end: float                # 进度区间上限
    required: bool                # True=不可跳过（跳过会破坏产物契约）
    depends_on: tuple[str, ...] = ()   # 硬依赖：运行时步骤存在才可执行
    soft_deps: tuple[str, ...] = ()    # 软依赖：缺失时降级而非报错
    produces: tuple[str, ...] = ()      # 产物键（前端可据此判断后续步骤可用）
    note: str = ""                # 跳过/依赖语义说明（前端 tooltip）
    skip_hint: str = ""           # 被跳过后对产物影响的一句话


LOCALIZE_STEPS: tuple[StepSpec, ...] = (
    StepSpec(
        id="extract_audio", label="提取音轨", pct_start=0, pct_end=8,
        required=True, produces=("audio",),
        note="所有后续步骤都建立在音轨上，不可跳过。",
    ),
    StepSpec(
        id="asr", label="ASR 转写", pct_start=8, pct_end=20,
        required=True, depends_on=("extract_audio",),
        produces=("segments", "transcript"),
        note="翻译/配音/字幕的时间轴来源，不可跳过。",
    ),
    StepSpec(
        id="diarize", label="说话人分离", pct_start=20, pct_end=32,
        required=False, depends_on=("asr",),
        produces=("speakers",),
        note="多人对话配音时按说话人分配音色；跳过则整片单音色。",
        skip_hint="跳过：整片使用单一音色配音，多人场景表现下降。",
    ),
    StepSpec(
        id="ocr", label="OCR 定位字幕区", pct_start=32, pct_end=44,
        required=False, depends_on=("asr",),
        produces=("regions",),
        note="定位原字幕区域供擦除；跳过则原字幕/水印保留在画面上。",
        skip_hint="跳过：不擦除原字幕，成片保留原语言字幕。",
    ),
    StepSpec(
        id="translate", label="LLM 翻译", pct_start=44, pct_end=57,
        required=False, depends_on=("asr",),
        produces=("translated",),
        note="生成目标语言台词；跳过则无译文（配音/字幕都用原文）。",
        skip_hint="跳过：无译文，配音与字幕使用原文（同语言场景适用）。",
    ),
    StepSpec(
        id="erase", label="擦除字幕/水印", pct_start=57, pct_end=75,
        required=False, depends_on=("ocr",),
        produces=("erased_video", "ass"),
        note="依赖 OCR 区域；跳过则原字幕保留。同时负责烧录新字幕轨道。",
        skip_hint="跳过：不擦除、不烧录新字幕（画面保持原样）。",
    ),
    StepSpec(
        id="tts", label="逐句配音", pct_start=75, pct_end=90,
        required=False, depends_on=("asr",), soft_deps=("translate", "diarize"),
        produces=("dub_audio",),
        note="依赖 ASR 时间轴 + 译文（缺失时回落原文）；跳过则无配音。",
        skip_hint="跳过：成片无配音，仅保留画面（与字幕）。",
    ),
    StepSpec(
        id="mix", label="混音合成", pct_start=90, pct_end=97,
        required=False, depends_on=("extract_audio",), soft_deps=("tts", "erase"),
        produces=("output",),
        note="无配音时负责音轨处理（保留/替换背景音）；跳过则输出未混音。",
        skip_hint="跳过：不做音轨混音，输出沿用上一步画面与原音轨。",
    ),
    StepSpec(
        id="vectorize", label="字幕向量化", pct_start=97, pct_end=100,
        required=False, depends_on=("translate",),
        produces=("milvus_rows",),
        note="写入 Milvus 供语义检索；跳过则不影响成片，只影响搜索。",
        skip_hint="跳过：不做向量化，该任务不出现在字幕语义检索里。",
    ),
)

# 技能 id → 步骤注册表（未来 localize/edit/dsp 各自的流水线在此登记）
STEP_REGISTRIES: dict[str, tuple[StepSpec, ...]] = {
    "localize_video": LOCALIZE_STEPS,
}


def steps_for(skill_id: str) -> tuple[StepSpec, ...]:
    """返回技能的步骤表（未登记技能的流水线 = 单步整体）。"""
    return STEP_REGISTRIES.get(skill_id, ())


def step_by_id(skill_id: str) -> dict[str, StepSpec]:
    return {s.id: s for s in steps_for(skill_id)}


def resolve_selection(
    skill_id: str,
    *,
    steps: list[str] | None = None,
    skip_steps: list[str] | None = None,
) -> tuple[list[StepSpec], dict[str, str]]:
    """把前端的「白名单 / 跳过列表」解析成实际执行步骤 + 跳过原因。

    规则：
      - 未传任何参数 → 全跑（向后兼容）；
      - `steps` 白名单：只跑列出的；除此之外的步骤记 skipped（含原因）；
      - `skip_steps`：从全集中扣除；
      - 两者同时传 → 取交集（白名单优先，skip 再剔除）；
      - **必需步骤被剔除 → ValueError**（调用方映射 422）；
      - **硬依赖被剔除 → ValueError**（显式失败，绝不静默半跑）；
      - 软依赖缺失 → 保留步骤但记 degraded 说明。

    返回 (按注册顺序的实际执行步骤, {被跳过步骤 id: 原因})。
    """
    specs = steps_for(skill_id)
    if not specs:
        return [], {}
    by_id = {s.id: s for s in specs}
    unknown = [s for s in (steps or []) + (skip_steps or []) if s not in by_id]
    if unknown:
        raise ValueError(
            f"未知步骤: {sorted(set(unknown))}；可选: {sorted(by_id)}")

    wanted: set[str] | None = None
    if steps:
        wanted = set(steps)
    enabled = {s.id for s in specs} if wanted is None else set(wanted)
    if skip_steps:
        enabled -= set(skip_steps)

    skipped: dict[str, str] = {}
    for s in specs:
        if s.id in enabled:
            continue
        if s.required:
            raise ValueError(f"步骤 {s.id}（{s.label}）是必需步骤，不能跳过")
        skipped[s.id] = "调用方未选择该步骤"

    # 硬依赖校验：依赖步骤必须在 enabled 里
    for s in specs:
        if s.id not in enabled:
            continue
        for dep in s.depends_on:
            if dep not in enabled:
                dep_spec = by_id[dep]
                raise ValueError(
                    f"步骤 {s.id}（{s.label}）依赖 {dep}（{dep_spec.label}），"
                    f"但后者未启用；请一并启用或跳过 {s.id}")

    ordered = [s for s in specs if s.id in enabled]
    if not ordered:
        raise ValueError("未选择任何可执行步骤")
    return ordered, skipped


def stage_schema(skill_id: str) -> list[dict]:
    """前端/网关用的步骤 schema（随 manifest 下发，前端无需硬编码步骤名）。"""
    return [
        {
            "id": s.id,
            "label": s.label,
            "required": s.required,
            "depends_on": list(s.depends_on),
            "soft_deps": list(s.soft_deps),
            "produces": list(s.produces),
            "progress_range": [s.pct_start, s.pct_end],
            "note": s.note,
            "skip_hint": s.skip_hint,
        }
        for s in steps_for(skill_id)
    ]

# ── 运行期步骤跟踪器 ──


class StepTracker:
    """流水线步骤运行期状态跟踪（线程安全；每步一次回调）。

    流水线用 ``with tracker.step("asr") as st:`` 包裹阶段主体：
      - 进入时记 running + started_at，并推一次状态；
      - 正常退出记 done；
      - 抛异常记 failed（error 一句话）；
      - 通过 ``st.skip(reason)`` 标记为 skipped（阶段被裁剪时用）。

    每次状态变化调用 ``on_change(build())``，把**全量步骤快照**交给上层
    （TaskManager → PG/MQTT）。全量快照而非增量：前端一次拿到完整面板。
    """

    def __init__(self, skill_id: str, *, enabled: tuple[str, ...] = (),
                 skipped: dict[str, str] | None = None,
                 on_change=None):
        self._skill_id = skill_id
        self._on_change = on_change
        self._lock = threading.Lock()
        specs = steps_for(skill_id)
        # enabled 为空 = 全跑（调用方未传选择）；非空 = 严格按它裁剪。
        # 注意：不能用 `set(enabled) or 全集`——空集本身是合法输入（全跳过），
        # 但业务上空集只会来自「未传参」，故此处显式区分 None/空 与有值。
        enabled_set = set(enabled) if enabled else {sp.id for sp in specs}
        self._states: dict[str, dict] = {}
        for s in specs:
            if s.id in enabled_set:
                self._states[s.id] = {
                    "id": s.id, "label": s.label, "status": "pending",
                    "started_at": None, "ended_at": None, "duration_ms": None,
                    "backend": None, "skipped": False, "skip_reason": None,
                    "note": None,
                }
            else:
                reason = (skipped or {}).get(s.id, "调用方未选择该步骤")
                self._states[s.id] = {
                    "id": s.id, "label": s.label, "status": "skipped",
                    "started_at": None, "ended_at": None, "duration_ms": None,
                    "backend": None, "skipped": True, "skip_reason": reason,
                    "note": s.skip_hint or None,
                }
        self._order = [s.id for s in specs]
        self._emit()

    def _emit(self) -> None:
        if self._on_change is not None:
            try:
                self._on_change(self.build())
            except Exception:  # noqa: BLE001  上报失败绝不阻断流水线
                pass

    def build(self) -> dict:
        """全量快照（按注册顺序）。"""
        with self._lock:
            steps = [dict(self._states[i]) for i in self._order if i in self._states]
        done = sum(1 for s in steps if s["status"] in ("done", "skipped"))
        return {
            "skill_id": self._skill_id,
            "total": len(steps),
            "completed": done,
            "steps": steps,
        }

    def step(self, step_id: str) -> "_StepHandle":
        return _StepHandle(self, step_id)

    # 内部状态迁移（供 _StepHandle 调用）
    def _mark(self, step_id: str, **fields) -> None:
        with self._lock:
            st = self._states.get(step_id)
            if st is None:
                return
            st.update(fields)
        self._emit()


class _StepHandle:
    """``with tracker.step("asr") as st:`` 的上下文管理器。"""

    def __init__(self, tracker: StepTracker, step_id: str):
        self._t = tracker
        self._id = step_id
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = _time.monotonic()
        self._t._mark(self._id, status="running",
                      started_at=_now_iso(), ended_at=None,
                      duration_ms=None, skipped=False, skip_reason=None)
        return self

    def skip(self, reason: str) -> None:
        """标记本步被跳过（阶段内部条件判断后才决定不跑时用）。"""
        self._t._mark(self._id, status="skipped", skipped=True,
                      skip_reason=reason, ended_at=_now_iso(),
                      duration_ms=_elapsed_ms(self._t0), note=reason)

    def backend(self, name: str) -> None:
        """记录本步实际使用的后端（local/cloud/...）。"""
        self._t._mark(self._id, backend=name)

    def note(self, text: str) -> None:
        self._t._mark(self._id, note=text)

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self._t._mark(self._id, status="failed", ended_at=_now_iso(),
                          duration_ms=_elapsed_ms(self._t0),
                          note=f"{exc_type.__name__}: {exc}"[:300])
            return False  # 不吞异常
        with self._t._lock:
            st = self._t._states.get(self._id) or {}
            if st.get("status") == "skipped":
                return False
        self._t._mark(self._id, status="done", ended_at=_now_iso(),
                      duration_ms=_elapsed_ms(self._t0))
        return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(t0: float) -> int:
    return int((_time.monotonic() - t0) * 1000)
