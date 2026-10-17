"""dsp_timeline_build 技能演示 —— 素材对齐 · 切镜收口 · 时间线锁定（唯一时间线写者）。

运行：PYTHONPATH=<repo>/src conda run -n flowmind python examples/dsp_timeline_build_demo.py

**ADR-069（定时长真源）与 ADR-070（删模板改自动切镜）之后**，本技能有三段职责，演示逐条覆盖：
1. discover() 自动字段发现
2. **实测路径**（有 `deps.tts`）：逐句实测归属到镜头；轨严格连续、末尾静默并入末镜
3. **比例兜底**：文案与配音逐句对不上 → 按**文案字数**为权重连续切分 + **如实告警「归属为推断」**
4. **中间无口播镜**：吃「停顿」⇒ **画面与配音仍逐帧对齐**（不插时间）
5. **尾部无口播镜**：吃「尾部静默」；不足才取显式常量并如实标注（**非实测**）
6. **估算路径**（无 `deps.tts`，配音开关关掉）：按 shots[].durationMs 累加 + 如实标注
7. **收口 · 并镜**（ADR-070）：实测 < 1.5s 的镜并入相邻镜 → id 重排、文案随镜落进 EDL
8. **收口 · 拆镜 + 转场**（ADR-070）：实测 > 6s 在内部最近的合法边界一分为二——**先句边界，
   句内下沉到词边界**（8b 专测后者）；被拆开的**内部**那刀按硬切，用户设的转场保留在**整镜末尾**
9. **转场**：非拆分边界上的符号值原样落 EDL（厂商资源 ID 未登记时由 render 侧如实回落硬切）
10. error：镜头时长非法（ge=1）→ VALIDATION
11. ADR-077 四块：旧扁平字段提升（老 run 的 spec 原样重跑仍能跑）
12. **类目退役**（2026-10-15）：旧报文的 `needKind` / `materials[].category` 被忽略，
    选图**只看指名图**（`clipKeys` / `image.assetKey`），没指名即留空位

mock 方式：替换 ``_dsp_storage.put_text`` 为内存实现（本文件内），不需要对象存储。
"""

from __future__ import annotations

import flowmind.skills  # noqa: F401  触发 @skill 注册
from flowmind.discover import field_names
from flowmind.skill import invoke
from flowmind.skills import _dsp_storage as storage
from flowmind.skills._dsp_common import DspShot

_WRITTEN: dict[str, str] = {}


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _install_storage() -> None:
    def put_text(key: str, text: str, content_type: str | None = None) -> str:
        _WRITTEN[key] = text
        return key

    def put_bytes(key: str, data: bytes, content_type: str | None = None) -> str:
        _WRITTEN[key] = f"<{len(data)}B>"
        return key

    def presign_get(key: str, ttl_s: int | None = None) -> str:
        return f"https://assets.example/{key}"

    storage.put_text = put_text
    storage.put_bytes = put_bytes
    storage.presign_get = presign_get


def _shots() -> list[dict]:
    # 注：**不再有 weight 字段**（ADR-070：比例兜底的权重改由 `timeline` 从文案字数自算），
    # **也不再需要 needKind**（2026-10-15 类目退役）：选图只看镜头指名的图（`clipKeys` /
    # `image.assetKey`），没指名就留空位。旧 spec 快照里若还带 weight / needKind，pydantic
    # 会忽略（extra 默认 ignore），故旧快照仍可跑——见第 12 节。
    return [
        {"id": "s1", "label": "门头开场", "durationMs": 3000,
         "copy": "开门的瞬间", "clipKeys": ["dsp/scratch/t1/门店/门头1.jpg"],
         "motion": {"type": "push_in", "intensity": "subtle", "direction": "toward_subject",
                    "target": "店门", "reason": "门头留白充足，轻微推近强化第一眼",
                    "positive": "", "negative": ""}},
        {"id": "s2", "label": "产品特写", "durationMs": 4000,
         "copy": "保温12小时"},
        {"id": "s3", "label": "人物", "durationMs": 2000,
         "copy": "上班族的早晨"},
    ]


def _seg(sentences: list[tuple], audio_ms: int | None = None) -> dict:
    """配音产物（`deps.tts`）——逐句实测轴（含**词级 token**）+ 实测总时长。

    每项 = ``(start, end, text)`` 或 ``(start, end, text, words)``；``words`` =
    ``[(start, end, text), …]`` 词级 token（供 ``_dsp_align`` 的**句内下沉切分**）。
    ``audio_ms`` 显式给音频文件总长（可为末句之后留一段**尾部静默**）；缺省 = 末句结束时刻。
    """
    segs = []
    for i, item in enumerate(sentences, start=1):
        a, b, t = item[0], item[1], item[2]
        words = [{"atMs": wa, "durationMs": wb - wa, "text": wt}
                 for (wa, wb, wt) in (item[3] if len(item) > 3 else [])]
        segs.append({"id": f"s{i}", "atMs": a, "durationMs": b - a, "text": t, "words": words})
    total = audio_ms if audio_ms is not None else (sentences[-1][1] if sentences else 0)
    return {"audioKey": "dsp/scratch/t1/audio/voice.mp3", "durationMs": total,
            "voSegments": segs, "voice": "zh_male_test", "warnings": []}


def _ctx(**over) -> dict:
    base = {
        "mode": "content", "platform": "douyin",
        "cachePrefix": "dsp/scratch/t1", "outputPrefix": "dsp/output/t1",
        "runId": "run-demo-1", "stepId": "timeline", "idempotencyKey": "run-demo-1:timeline",
        "shots": _shots(),
        "materials": [
            {"kind": "image", "key": "dsp/scratch/t1/门店/门头1.jpg",
             "width": 1080, "height": 1920},
            {"kind": "image", "key": "dsp/scratch/t1/产品/杯1.jpg"},
        ],
    }
    base.update(over)
    return base


def _dump(d) -> None:
    print(f"  edlKey    : {d.edlKey}")
    print(f"  durationSource: {d.durationSource}")
    print(f"  totalMs   : {d.edl.totalMs}")
    for seg, shot in zip(d.edl.tracks[0].segments, d.edl.shots, strict=True):
        print(f"    • {shot.id}: {seg.atMs}–{seg.atMs + seg.durationMs}ms "
              f"src={seg.sourceKey or '(空位)'} "
              f"motion={(shot.motion.type if shot.motion else '固定机位')} transition={seg.transition}")
    for note in d.notes:
        print(f"  note      : {note}")


def main() -> None:
    section("0) discover('timeline_build') —— Agent 自查字段")
    for path, names in field_names("timeline_build").items():
        print(f"  {path}: {names}")

    section("1) 实测路径：3 镜逐句对上（指名选到图 / 未指名留空位）")
    _install_storage()
    _WRITTEN.clear()
    r = invoke("timeline_build", _ctx(deps={"tts": _seg([
        (0, 2000, "开门的瞬间"),
        (2400, 5200, "保温12小时"),      # 前有 400ms 停顿 → 归前一镜
        (5600, 8300, "上班族的早晨"),
    ], audio_ms=9000)}))
    check(r.ok, "happy path 应 ok")
    d = r.data
    _dump(d)
    check(d.durationSource == "measured", "有 deps.tts → 时长来源 = 实测")
    check(d.edl.totalMs == 9000, "总时长 = 音频实测总长 9000ms（末尾 700ms 静默并入末镜）")
    segs = d.edl.tracks[0].segments
    check((segs[0].atMs, segs[0].durationMs) == (0, 2000), "s1 = 0–2000ms")
    check((segs[1].atMs, segs[1].durationMs) == (2000, 3200), "s2 = 2000–5200ms（含 400ms 停顿）")
    check((segs[2].atMs, segs[2].durationMs) == (5200, 3800), "s3 = 5200–9000ms（含 700ms 尾部静默）")
    check(not hasattr(segs[0], "effect"), "EDL segment 不再有 effect 字段（ADR-076 已退役）")
    check(d.edl.shots[0].motion is not None and d.edl.shots[0].motion.type == "push_in",
          "第一镜的运镜方案原样落进 edl.shots[].motion")
    check(d.edl.shots[1].motion is None and d.edl.shots[2].motion is None,
          "未给方案的镜头 motion 为 None（= 摄影机固定）")
    check(segs[0].sourceKey == "dsp/scratch/t1/门店/门头1.jpg", "第一镜指名了图 → 用它")
    check(segs[1].sourceKey == "" and segs[2].sourceKey == "",
          "未指名的镜**不会**再从素材池按类目挑（类目已退役）→ 留空位")
    check(any("留空位" in n for n in d.notes), "留空位须在 notes 里如实说明")
    check(not any("推断" in n or "静默常量" in n for n in d.notes), "逐句都对上 → 不应有推断/静默告警")
    check(d.edlKey in _WRITTEN, "EDL 快照须落对象存储")
    check(d.edl.tracks[1].kind == "vo" and d.edl.tracks[1].segments[0].durationMs == 9000,
          "音轨占位 [0,totalMs)（sourceKey 由 render 挂）")

    section("2) 比例兜底：文案与配音逐句**对不上** → 按权重切分 + 如实告警")
    r = invoke("timeline_build", _ctx(deps={"tts": _seg([
        (0, 1500, "开场白"),
        (1500, 4500, "保温时长"),
        (4500, 7000, "早高峰"),
    ])}))
    check(r.ok, "比例兜底路径应 ok")
    d = r.data
    _dump(d)
    check(d.durationSource == "measured", "归属为推断，但时长仍来自实测轴")
    check(d.edl.totalMs == 7000, "总时长 = 音频实测总长")
    check(any("未对上" in n and "推断" in n for n in d.notes),
          "文配失败须如实告警「归属为推断」（绝不假装精确）")
    check([(s.atMs, s.durationMs) for s in d.edl.tracks[0].segments]
          == [(0, 1500), (1500, 3000), (4500, 2500)], "三镜按句连续切分（1 句 1 镜）")

    section("3) 中间无口播镜：吃「停顿」⇒ **画面与配音仍逐帧对齐**")
    r = invoke("timeline_build", _ctx(deps={"tts": _seg([
        (0, 1500, "开门的瞬间"),
        (4000, 6000, "上班族的早晨"),   # s2 的文案没有对应句子
    ])}))
    check(r.ok, "中间无口播镜路径应 ok")
    d = r.data
    _dump(d)
    check(d.edl.totalMs == 6000, "总时长仍 = 音频总长（未插时间，no desync）")
    check([(s.atMs, s.durationMs) for s in d.edl.tracks[0].segments]
          == [(0, 1500), (1500, 2500), (4000, 2000)],
          "s2 吃掉 1500–4000ms 的停顿 2500ms；s3 自 4000ms 起（画面未偏移）")
    check(any("无对口" in n or "没有对应句子" in n for n in d.notes), "无口播镜须如实说明")
    check(not any("失同步" in n for n in d.notes), "中间吃停顿 → 不应报失同步")

    section("4) 尾部无口播镜：吃「尾部静默」，不足则取显式常量（**非实测**）")
    r = invoke("timeline_build", _ctx(deps={"tts": _seg([(0, 1500, "开门的瞬间")])}))
    check(r.ok, "尾部无口播镜路径应 ok")
    d = r.data
    _dump(d)
    segs = d.edl.tracks[0].segments
    check((segs[0].atMs, segs[0].durationMs) == (0, 1500), "s1 实测 1500ms")
    check((segs[1].durationMs, segs[2].durationMs) == (1500, 1500),
          "s2/s3 无句可吃（音频尾部无静默）→ 取显式常量 1500ms")
    check(any("静默常量" in n and "非实测" in n for n in d.notes),
          "用常量必须如实标注「非实测」")
    check(not any("失同步" in n for n in d.notes), "尾部拉长不破坏同步 → 不应报失同步")

    section("5) 估算路径：无 deps.tts（配音开关关掉）→ 按估算累加 + 如实标注")
    r = invoke("timeline_build", _ctx())
    check(r.ok, "估算路径应 ok")
    d = r.data
    _dump(d)
    check(d.durationSource == "estimated", "无 deps.tts → 时长来源 = 估算")
    check(d.edl.totalMs == 9000, "总时长 = 3+4+2 = 9s（shots[].durationMs 累加）")
    check(any("非实测" in n for n in d.notes), "估算路径须如实标注「非实测」")
    check(0.0 < r.reasoning[0].confidence < 0.6, "估算路径置信度应低于实测路径（已如实下调）")
    print(f"  confidence: {r.reasoning[0].confidence}（实测路径为 0.6，估算再乘 0.7）")

    section("7) 收口 · 并镜（ADR-070）：实测 < 1.5s 的镜并入相邻镜 → id 重排、文案随镜入 EDL")
    r = invoke("timeline_build", _ctx(deps={"tts": _seg([
        (0, 2000, "开门的瞬间"),
        (2000, 2600, "保温12小时"),     # 实测仅 600ms < 1500ms → 会被并掉
        (2600, 5000, "上班族的早晨"),
    ])}))
    check(r.ok, "并镜路径应 ok")
    d = r.data
    _dump(d)
    segs = d.edl.tracks[0].segments
    check(len(segs) == 2, f"3 镜收口成 2 镜（600ms 的那镜并入相邻镜），实际 {len(segs)}")
    check([(s.atMs, s.durationMs) for s in segs] == [(0, 2000), (2000, 3000)],
          "并镜后两段仍**严格连续**（切点仍在句边界上）")
    check([s.id for s in d.edl.shots] == ["s1", "s2"], "终版镜头 id 重排为 s1..sK")
    check(d.edl.totalMs == 5000, "总时长不变（并镜只改镜界，不改总长）")
    check(d.edl.shots[1].copy == "保温12小时上班族的早晨",
          "并镜后**文案随镜落进 EDL**（下游据 edl.shots[].copy 出提示词，不回查 inp.shots）")
    check(any("并镜 1 个" in n for n in d.notes), "并镜须如实说明")

    section("8) 收口 · 拆镜 + 转场（ADR-070）：> 6s 在内部句边界一分为二；内部那刀按硬切")
    shots_split = [
        {"id": "s1", "label": "门头+产品", "durationMs": 7000,
         "copy": "开门的瞬间保温12小时", "transitionAfter": "dissolve"},
        {"id": "s2", "label": "人物", "durationMs": 2000, "copy": "上班族的早晨"},
    ]
    r = invoke("timeline_build", _ctx(shots=shots_split, deps={"tts": _seg([
        (0, 3000, "开门的瞬间"),
        (3000, 7000, "保温12小时"),
        (7000, 9000, "上班族的早晨"),
    ])}))
    check(r.ok, "拆镜路径应 ok")
    d = r.data
    _dump(d)
    segs = d.edl.tracks[0].segments
    check(len(segs) == 3, f"7s 的那镜在内部句边界一分为二 → 3 镜，实际 {len(segs)}")
    check([(s.atMs, s.durationMs) for s in segs] == [(0, 3000), (3000, 4000), (7000, 2000)],
          "拆点落在**句起点**上（句间停顿归前一半，与连续口径一致）")
    check(segs[0].transition is None, "被拆开的**内部**那刀按硬切（用户设的转场不该落到中间边界）")
    check(segs[1].transition == "dissolve", "用户设的转场保留在**整镜末尾**（即后半段的末尾）")
    check(segs[2].transition is None, "末镜无下一镜 → 不写转场")
    check(any("拆镜 1 个" in n for n in d.notes), "拆镜须如实说明")
    check(any("硬切" in n and "未登记" in n for n in d.notes),
          "转场须如实说明「厂商资源 ID 未登记即回落硬切」")
    check(d.edl.shots[0].copy == "开门的瞬间保温12小时" and d.edl.shots[1].copy == "开门的瞬间保温12小时",
          "拆开的两半**共用该镜文案**（它们讲的是同一段话）")
    check(all(s.sourceKey == "" for s in segs),
          "没指名图的镜（含拆出的两半）一律留空位——类目退役后**不再**有「按类目挑图」这一档")

    section("8b) 收口 · 拆镜**下沉到词边界**：整镜就是一整句且超长（句内无句边界）→ 切词")
    shots_word = [
        {"id": "s1", "label": "长句一镜", "durationMs": 8000,
         "copy": "开门的瞬间保温十二小时", "visualPrompt": "长句一镜的画面描述"},
        {"id": "s2", "label": "人物", "durationMs": 2000, "copy": "上班族的早晨"},
    ]
    r = invoke("timeline_build", _ctx(shots=shots_word, deps={"tts": _seg([
        (0, 8000, "开门的瞬间保温十二小时", [
            (0, 2000, "开门的"), (2000, 4000, "瞬间"), (4000, 6000, "保温"), (6000, 8000, "十二小时"),
        ]),
        (8000, 10000, "上班族的早晨"),
    ])}))
    check(r.ok, "词级下沉拆分路径应 ok")
    d = r.data
    _dump(d)
    segs = d.edl.tracks[0].segments
    check(len(segs) == 3, f"8s 的整句一镜内部**无句边界** ⇒ 下沉到词边界一分为二 → 3 镜，实际 {len(segs)}")
    check([(s.atMs, s.durationMs) for s in segs] == [(0, 4000), (4000, 4000), (8000, 2000)],
          "切点落在**词边界** 4000ms 上（离中点最近、且两半各 ≥ 1.5s）")
    check(segs[0].transition is None and segs[1].transition is None,
          "词级拆出的两道边界都按硬切（未设转场）")
    check(any("下沉到词边界" in n for n in d.notes),
          "词级下沉须如实说明（该处切在句内、非句边界）")
    check(d.edl.shots[0].visualPrompt == d.edl.shots[1].visualPrompt == "长句一镜的画面描述",
          "拆镜：两半**共用**同一画面提示词（ADR-073）")

    section("9b) 画面提示词（ADR-073）：并镜按行拼接、随 EDL 透出给 image_to_video")
    shots_vp = [
        {"id": "s1", "label": "门头", "durationMs": 1000,
         "copy": "开门", "visualPrompt": "A 画面：门头晨光"},
        {"id": "s2", "label": "产品", "durationMs": 1000,
         "copy": "保温", "visualPrompt": "B 画面：杯身特写"},
        {"id": "s3", "label": "人物", "durationMs": 2000,
         "copy": "上班族", "visualPrompt": "C 画面：通勤人流"},
    ]
    r = invoke("timeline_build", _ctx(shots=shots_vp, deps={"tts": _seg([
        (0, 1000, "开门"), (1000, 2000, "保温"), (2000, 4000, "上班族"),
    ])}))
    check(r.ok, "画面提示词路径应 ok")
    d = r.data
    _dump(d)
    check(len(d.edl.shots) == 2, "两个 1s 镜实测 < 1.5s 并入后共 2 镜")
    check(d.edl.shots[0].visualPrompt == "A 画面：门头晨光\nB 画面：杯身特写",
          "并镜：画面提示词**按行拼接**（不丢、不臆造）")
    check(d.edl.shots[1].visualPrompt == "C 画面：通勤人流", "未并镜的镜原样透出")

    section("9) 转场：非拆分边界上的符号值原样落 EDL（厂商 ID 未登记 ⇒ render 侧回落硬切）")
    shots_tr = [
        {"id": "s1", "label": "门头", "durationMs": 3000,
         "copy": "开门的瞬间", "transitionAfter": "fade-black"},
        {"id": "s2", "label": "人物", "durationMs": 2000, "copy": "上班族的早晨"},
    ]
    r = invoke("timeline_build", _ctx(shots=shots_tr, deps={"tts": _seg([
        (0, 3000, "开门的瞬间"),
        (3000, 5000, "上班族的早晨"),
    ])}))
    check(r.ok, "转场路径应 ok")
    d = r.data
    _dump(d)
    check([s.transition for s in d.edl.tracks[0].segments] == ["fade-black", None],
          "信息点之间的边界带上所选转场；末镜不写")
    check(any("1 处转场" in n for n in d.notes), "记录了几处转场须如实说明")

    section("10) error：镜头时长非法（durationMs=0 < ge=1）→ VALIDATION")
    bad = _ctx(shots=[{"id": "s1", "durationMs": 0}])
    r = invoke("timeline_build", bad)
    print(f"  ok         : {r.ok}")
    print(f"  error.code : {r.error.code if r.error else None}")
    check(r.ok is False and r.error and r.error.code == "VALIDATION",
          "非法入参应被 schema 拦为 VALIDATION")

    section("11) ADR-077 四块：旧扁平字段提升（老 run 的 spec 原样重跑仍能跑）")
    lifted = DspShot(
        id="s1", durationMs=3000, copy="开门",
        clipKeys=["k1", "k2"], visualPrompt="门头晨光",
        motion={"type": "push_in", "intensity": "subtle", "reason": "留白充足"},
        transitionAfter="dissolve")
    check(lifted.image is not None and lifted.asset_key == "k1", "旧 clipKeys → image.assetKey（取首位）")
    check(lifted.image.auto is False and "旧快照" in lifted.image.note,
          "来源不可考 ⇒ auto=false 且 note 如实记明")
    check(lifted.image.candidates == ["k1", "k2"], "候选全留（质检换图要取次优）")
    check(lifted.author_prompt == "门头晨光", "旧 visualPrompt → i2v.authorPrompt")
    check(lifted.motion_plan is not None and lifted.motion_plan.type == "push_in",
          "旧 motion → i2v.motion（并据在场置 motionEnabled）")
    check(lifted.i2v is not None and lifted.i2v.attempts == 0,
          "提升后 attempts=0 —— **不会被误判成已回落静图**")
    check(lifted.edit is None, "旧快照**不**提升出 edit（转场请求 ≠ 剪辑实况）")
    check(lifted.transitionAfter == "dissolve", "转场请求留在镜头上，与 edit.transition（实况）成对")

    modern = DspShot(id="s2", durationMs=3000, copy="保温",
                     image={"assetKey": "k9"}, i2v={"videoKey": "v1", "attempts": 1})
    check(modern.asset_key == "k9" and modern.i2v.videoKey == "v1" and modern.i2v.attempts == 1,
          "四块在场时 lifter 幂等（不改写）")

    r2 = invoke("timeline_build", _ctx(shots=[
        {"id": "s1", "durationMs": 3000, "copy": "开门", "image": {"assetKey": "k1"}},
    ]))
    check(r2.ok, "显式四块的入参应 ok")
    check(r2.data.edl.shots[0].imageSourceKey == "k1",
          "显式 image.assetKey 照常被 timeline 选图（与旧 clipKeys 口径等价）")

    section("12) 类目退役（2026-10-15）：旧报文的 needKind / category 被忽略，不再选图")
    legacy = _ctx(
        shots=[
            {"id": "s1", "label": "产品特写", "needKind": "产品", "durationMs": 3000,
             "copy": "保温12小时"},
            {"id": "s2", "label": "门头", "durationMs": 2000, "copy": "开门的瞬间",
             "clipKeys": ["dsp/scratch/t1/门店/门头1.jpg"]},
        ],
        materials=[
            {"kind": "image", "key": "dsp/scratch/t1/产品/杯1.jpg", "category": "产品"},
            {"kind": "image", "key": "dsp/scratch/t1/门店/门头1.jpg", "category": "门头"},
        ],
    )
    r = invoke("timeline_build", legacy)
    check(r.ok, "旧报文（带类目字段）**必须照常跑通**——pydantic 默认忽略多余键")
    d = r.data
    _dump(d)
    check(d.edl.tracks[0].segments[0].sourceKey == "",
          "旧 needKind 不再从素材池挑图（池里明明有「产品」那张）→ 留空位")
    check(d.edl.tracks[0].segments[1].sourceKey == "dsp/scratch/t1/门店/门头1.jpg",
          "指名的图照常选上（选图只看指名）")
    check(not hasattr(d.edl.shots[0], "needKind"), "EDL 镜头不再记类目（字段已从模型删除）")
    check(not any(k.startswith("shots") and "needKind" in v for k, v in field_names("timeline_build").items()),
          "inp 的 shots[] 里已没有 needKind（契约字段删除，非仅置空）")

    print("\n✅ dsp_timeline_build_demo 全部通过")


if __name__ == "__main__":
    main()
