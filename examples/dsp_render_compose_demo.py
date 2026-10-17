"""dsp_render_compose 技能演示 —— 合成渲染（MediaKit；唯一渲染 / DAG 汇点）。

运行：PYTHONPATH=<repo>/src conda run -n flowmind python examples/dsp_render_compose_demo.py

展示：
1. discover() 自动字段发现
2. happy path：EDL → track[][] → 提交渲染 → **成片转存**为不可变 key（§7.3②）
3. error：缺渲染密钥 → 明确报错（abort 步，不伪造成片）
4. error：EDL 各镜均无图 → 轨道为空，如实报错
5. **运镜不在这里施加**（ADR-076）：运镜已随 I2V 生成落地，本步**不得**再施加（否则同一运动两次）
   ——含一条**结构性哨兵**：``_dsp_mediakit`` 不再暴露 ``apply_camera_motion``
6. **逐镜剪辑记录 ``editSegments[]``**（ADR-080 阶段 5）：时间轴位置 / 剪辑时长 / 音轨 / **实际施加**的
   转场（未登记 ⇒ 实况硬切，**请求 ≠ 实况**）/ 裁切实况（i2v 带出的 ``trim``）/ 降级模式（静图回落 / 空段）/
   BGM 恒降级如实标出
7. **素材本地上传**（契约 §7.3①）：静图 / 音轨 / 字幕先把**字节**传 MediaKit，任务里只用
   ``mediakit://{file_id}`` 引用——不再依赖对象存储公网可达（本机 ``127.0.0.1:9000`` 也能真渲染）

mock 方式：替换 ``_dsp_mediakit.render`` / ``_dsp_mediakit.upload_media``（模块级薄适配层）
与 ``_dsp_storage`` 的 get_bytes/fetch/put_file，不需要网络 / MediaKit 密钥 / 对象存储。
密钥用假值经 env 注入。
"""

from __future__ import annotations

import importlib
import os

import flowmind.skills  # noqa: F401  触发 @skill 注册
from flowmind.discover import field_names
from flowmind.skill import invoke
from flowmind.skills import _dsp_mediakit as mediakit
from flowmind.skills import _dsp_storage as storage

_KEY_ENV = "DSP_MEDIAKIT_API_KEY"
_BODY: dict = {}
_PUTS: list[str] = []    # 所有 put_file 的 key 序列（运镜退役后只有成片）
_UPLOADS: list[tuple[str, str]] = []  # 本地上传 MediaKit 的 (素材 key, content-type) 序列


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _install_fakes() -> None:
    def render(cfg, body):
        _BODY.clear()
        _BODY.update(body)
        return {"task_id": "mediakit-task-1", "video_url": "https://mediakit.example/tmp/final.mp4?sig=x",
                "duration": 9.0, "resolution": "1080p"}

    def fetch_to_file(url: str, dest_path: str, timeout_s: float | None = None) -> str:
        _BODY["_fetched"] = url
        return dest_path

    def put_file(path: str, key: str, content_type: str | None = None) -> str:
        _PUTS.append(key)
        _BODY["_stored"] = key
        return key

    def get_bytes(key: str) -> bytes:
        return f"bytes:{key}".encode()

    def upload_media(cfg, data: bytes, content_type: str) -> str:  # noqa: ANN001
        key = data.decode().removeprefix("bytes:")
        _UPLOADS.append((key, content_type))
        return f"mediakit://file-{len(_UPLOADS)}"

    mediakit.render = render
    mediakit.upload_media = upload_media
    storage.fetch_to_file = fetch_to_file
    storage.put_file = put_file
    storage.get_bytes = get_bytes


def _edl() -> dict:
    return {"timeline": {"edl": {
        "version": 1, "totalMs": 9000,
        "tracks": [
            {"kind": "image", "segments": [
                {"atMs": 0, "durationMs": 3000, "sourceKey": "dsp/scratch/t1/门店/门头1.jpg"},
                {"atMs": 3000, "durationMs": 6000, "sourceKey": "dsp/scratch/t1/产品/杯1.jpg"},
            ]},
            {"kind": "vo", "segments": [{"atMs": 0, "durationMs": 9000, "sourceKey": ""}]},
        ],
        "shots": [{"id": "s1", "atMs": 0, "durationMs": 3000,
                   "imageSourceKey": "dsp/scratch/t1/门店/门头1.jpg"}],
    }}}


def _edl_motion() -> dict:
    """带**运镜方案**的 EDL：2 镜，s1 带 push_in 方案（render **不应**因此做任何事），s2 无方案。"""
    return {"timeline": {"edl": {
        "version": 1, "totalMs": 7000,
        "tracks": [
            {"kind": "image", "segments": [
                {"atMs": 0, "durationMs": 3000, "sourceKey": "dsp/scratch/t1/门店/门头1.jpg"},
                {"atMs": 3000, "durationMs": 4000, "sourceKey": "dsp/scratch/t1/产品/杯1.jpg"},
            ]},
            {"kind": "vo", "segments": [{"atMs": 0, "durationMs": 7000, "sourceKey": ""}]},
        ],
        "shots": [
            {"id": "s1", "atMs": 0, "durationMs": 3000,
             "imageSourceKey": "dsp/scratch/t1/门店/门头1.jpg",
             "motion": {"type": "push_in", "intensity": "subtle", "reason": "留白充足",
                        "positive": "", "negative": ""}},
            {"id": "s2", "atMs": 3000, "durationMs": 4000,
             "imageSourceKey": "dsp/scratch/t1/产品/杯1.jpg"},
        ],
    }}}


def _ctx(**over) -> dict:
    base = {
        "mode": "content", "platform": "douyin",
        "cachePrefix": "dsp/scratch/t1", "outputPrefix": "dsp/output/t1",
        "runId": "run-demo-7", "stepId": "render", "idempotencyKey": "run-demo-7:render",
        "shots": [], "materials": [],
        "deps": _edl() | {
            "tts": {"audioKey": "dsp/scratch/t1/run-demo-7/audio/voice.mp3"},
            "subtitle": {"srtKey": "dsp/scratch/t1/run-demo-7/subtitle/captions.srt"},
            "cover": {"coverKey": "dsp/scratch/t1/门店/门头1.jpg", "atMs": 0},
            "cart": {"stickers": [{"atMs": 0, "durationMs": 9000,
                                   "productRef": {"kind": "manual", "value": "sku-123"}}]},
        },
    }
    base.update(over)
    return base


def main() -> None:
    section("0) discover('render_compose') —— Agent 自查字段")
    for path, names in field_names("render_compose").items():
        print(f"  {path}: {names}")

    section("1) Happy path：EDL → track[][] → 渲染 → 成片转存")
    os.environ[_KEY_ENV] = "demo-mediakit-key"
    _install_fakes()
    r = invoke("render_compose", _ctx())
    check(r.ok, "happy path 应 ok")
    d = r.data
    print(f"  storedKey : {d.storedKey}")
    print(f"  duration  : {d.durationMs}ms  分辨率: {d.width}×{d.height}")
    print(f"  轨道数    : {len(_BODY['track'])}（图片主轨 / 音轨 / 字幕轨）")
    print("  轨内元素  :")
    for idx, track in enumerate(_BODY["track"]):
        print(f"    [{idx}] " + ", ".join(e["type"] for e in track))
    print(f"  output    : {_BODY['output']}（封面走 output.cover 抽帧）")
    print(f"  AIGC 元数据: {_BODY.get('extra_param', {}).get('upload', {}).get('aigc_meta')}")
    print(f"  warnings  : {d.warnings}")
    check(d.storedKey == "dsp/output/t1/run-demo-7/final.mp4", "成片须落成片目录")
    check(_BODY["_stored"] == d.storedKey, "24h 签名 URL 必须转存为不可变 key（§7.3②）")
    check(_BODY["output"]["cover"]["cover_time"] == [0], "封面走 output.cover 抽帧")
    check(len(_BODY["track"][0]) == 2, "图片主轨两段")
    check(any("平台组件" in w for w in d.warnings), "商品链接不烧进画面须如实说明")
    # 契约 §7.3① 的本地上传支：静图 / 音轨 / 字幕都先传字节，任务里只用 mediakit:// 引用
    check(len(_UPLOADS) == 4, f"应上传 4 个素材（2 静图 + 1 音轨 + 1 字幕），实际 {len(_UPLOADS)}")
    check(all(u[0].startswith("dsp/scratch/") for u in _UPLOADS), "上传的是对象 key（字节），不是 URL")
    check(_BODY["track"][0][0]["source"].startswith("mediakit://"), "主轨素材须用 mediakit:// 引用")
    check(any("mediakit://" in w for w in d.warnings), "上传了几个素材须如实进 warnings")

    section("2) error：缺渲染密钥（abort 步）→ 明确报错，不伪造成片")
    os.environ.pop(_KEY_ENV, None)
    r = invoke("render_compose", _ctx())
    print(f"  ok        : {r.ok}")
    print(f"  error.msg : {(r.error.message if r.error else '')[:80]}")
    check(r.ok is False, "无密钥必须失败（成片字段保持空）")
    os.environ[_KEY_ENV] = "demo-mediakit-key"

    section("3) error：EDL 各镜均无图 → 轨道为空")
    empty = {"timeline": {"edl": {"version": 1, "totalMs": 9000, "tracks": [
        {"kind": "image", "segments": [{"atMs": 0, "durationMs": 9000, "sourceKey": ""}]}]}}}
    r = invoke("render_compose", _ctx(deps=empty))
    print(f"  ok        : {r.ok}")
    print(f"  error.msg : {(r.error.message if r.error else '')[:60]}")
    check(r.ok is False and "轨道为空" in (r.error.message if r.error else ""), "空轨道须如实报错")

    section("4a) 有片段的镜头走 video 元素，无片段的回落 image（运镜**不在此施加**）")
    _BODY.clear()
    _PUTS.clear()
    r = invoke("render_compose", _ctx(deps=_edl_motion() | {
        "video": {"clips": [{"shotId": "s1", "key": "dsp/scratch/t1/run-demo-7/clips/s1.mp4",
                             "atMs": 0, "durationMs": 3000}]},
        "tts": {"audioKey": "dsp/scratch/t1/run-demo-7/audio/voice.mp3"},
    }))
    check(r.ok, "有片段应渲染成功")
    kinds = [e["type"] for e in _BODY["track"][0]]
    print(f"  主轨元素  : {kinds}（s1 有片段=video / s2 无片段=image 回落）")
    print(f"  put_file  : {_PUTS}（运镜退役 ⇒ 只有成片，没有 motion/*.mp4）")
    print(f"  warnings  : {r.data.warnings}")
    check(kinds == ["video", "image"], "有片段的镜头走 video 元素，无片段的回落 image")
    check(_PUTS == ["dsp/output/t1/run-demo-7/final.mp4"], "只转存成片——不再有运镜产物")
    check(not any("运镜" in w for w in r.data.warnings), "不再因运镜产生告警（本步已不参与运镜）")

    section("4b) 结构性哨兵：运镜施加路径**整块退役**（防重复施加）")
    # 用 importlib 取模块：main() 后面还有一处**局部** import 同名（转场表那节），
    # 直接用模块级 `mediakit` 会被 Python 当成局部变量而在此处未绑定。
    _mk = importlib.import_module("flowmind.skills._dsp_mediakit")
    check(not hasattr(_mk, "apply_camera_motion"),
          "MediaKit 客户端不再暴露 apply_camera_motion（ADR-076）")
    check(not hasattr(_mk, "motion_style_for") and not hasattr(_mk, "MOTION_STYLES"),
          "旧运镜词表/映射一并退役（新适配表在 _dsp_video_gen.MOTION_ENGINE）")
    _BODY.clear()
    r = invoke("render_compose", _ctx(deps=_edl_motion()))
    check(r.ok, "带运镜方案的 EDL 仍可静图成片")
    check([e["type"] for e in _BODY["track"][0]] == ["image", "image"], "无片段全部回落静图")

    section("4c) 配音实测总长 ≠ 锁定时间线 → 自检告警（ADR-069 新不变量）")
    _BODY.clear()
    deps = _edl_motion()
    deps["tts"] = {"audioKey": "dsp/scratch/t1/run-demo-7/audio/voice.mp3", "durationMs": 9999}
    r = invoke("render_compose", _ctx(deps=deps))
    check(r.ok, "不一致仍可渲染（如实告警，不静默重算）")
    print(f"  warnings  : {r.data.warnings}")
    check(any("不一致" in w for w in r.data.warnings),
          "EDL 与配音不同源须如实告警（时间线应即音频总长）")

    section("4d) 转场（ADR-070）：**厂商目录里没有可用项** → 不加 extra、如实回落硬切")
    _BODY.clear()
    deps = _edl_motion()
    # `wipe`（擦除）在本仓符号词表内，但厂商目录里的擦除类（对角擦除/圆形擦开/时钟扫开…）
    # **全部标「不可用于非交叠」**，而本仓只允许非交叠 ⇒ **查不到即不可用**（如实硬切）
    deps["timeline"]["edl"]["tracks"][0]["segments"][0]["transition"] = "wipe"
    r = invoke("render_compose", _ctx(deps=deps))
    check(r.ok, "不可用的转场不应让渲染失败（回落硬切即可）")
    print(f"  warnings  : {[w for w in r.data.warnings if '转场' in w]}")
    first = _BODY["track"][0][0]
    check(not any(f.get("type") == "transition" for f in (first.get("extra") or [])),
          "不可用的转场**不得**编 ID 编 extra（如实回落硬切）")
    check(any("wipe" in w and "未登记" in w and "硬切" in w for w in r.data.warnings),
          "不可用转场须如实告警（绝不编造厂商 ID）")
    check(any("不可用于非交叠" in w for w in r.data.warnings),
          "告警须**说清是哪一种不可用**（厂商目录里没有可用于非交叠的项 ≠ 我们漏登记）")

    section("4e) 转场：已登记项**开箱即用** → 编成厂商 extra，加在**前一个**元素上（两个区域各一套号）")
    _BODY.clear()
    deps = _edl_motion()
    deps["timeline"]["edl"]["tracks"][0]["segments"][0]["transition"] = "dissolve"
    r = invoke("render_compose", _ctx(deps=deps))
    check(r.ok, "已登记转场应正常渲染")
    first = _BODY["track"][0][0]
    extra = first.get("extra") or []
    print(f"  华北 extra: {extra}")
    check(extra == [{"type": "transition", "source": "1182358", "duration": 400}],
          f"dissolve 应编成华北资源 ID 的 extra 加在前一个元素上，实际 {extra}")
    check(not any("未登记" in w for w in r.data.warnings), "已登记不应再报「未登记」")
    # 换区域（柔佛）⇒ 同一符号换另一套号，代码零改动（厂商按区域给两套 ID）
    os.environ["FLOWMIND_DSP_MEDIAKIT_TRANSITION_REGION"] = "johor"
    try:
        _BODY.clear()
        r = invoke("render_compose", _ctx(deps=deps))
        extra_jb = (_BODY["track"][0][0].get("extra") or [])
        print(f"  柔佛 extra: {extra_jb}")
        check(extra_jb == [{"type": "transition", "source": "1109560", "duration": 400}],
              "换区域 ⇒ 同一符号换成柔佛那套 ID（表里两个号都在，改 env 即可）")
    finally:
        os.environ.pop("FLOWMIND_DSP_MEDIAKIT_TRANSITION_REGION", None)
    # 「登记后自动生效」这条路径仍要给新符号留证：模拟把 wipe 登记到厂商可用的「交替出场」上
    from flowmind.skills import _dsp_mediakit as mediakit
    _BODY.clear()
    mediakit.TRANSITION_VENDOR["wipe"] = {"cnNorth": "1182359", "johor": "1109556",
                                          "mode": "non-overlap", "durationMs": 300}
    try:
        deps = _edl_motion()
        deps["timeline"]["edl"]["tracks"][0]["segments"][0]["transition"] = "wipe"
        r = invoke("render_compose", _ctx(deps=deps))
        extra = _BODY["track"][0][0].get("extra") or []
        print(f"  登记 wipe 后: {extra}")
        check(extra == [{"type": "transition", "source": "1182359", "duration": 300}],
              "登记后自动生效（只填表，代码零改动）")
        check(not any("未登记" in w for w in r.data.warnings), "登记后不应再报「未登记」")
    finally:
        mediakit.TRANSITION_VENDOR.pop("wipe", None)  # 表是进程级的，用完恢复

    section("5) 逐镜剪辑记录 editSegments[]（ADR-080 阶段 5）")
    # 5a) 有片段 / 无片段混排 + **裁切实况透传**（trim 由 i2v 产出，render 只透出不改写）
    _BODY.clear()
    r = invoke("render_compose", _ctx(deps=_edl_motion() | {
        "video": {"clips": [
            {"shotId": "s1", "key": "dsp/scratch/t1/run-demo-7/clips/s1.mp4",
             "atMs": 0, "durationMs": 3000,
             "trim": {"sourceMs": 5200, "outMs": 3000, "shortfallMs": 0}},
        ]},
        "tts": {"audioKey": "dsp/scratch/t1/run-demo-7/audio/voice.mp3"},
        "bgm": {"bgmKey": "", "durationMs": 7000},
    }))
    check(r.ok, "带剪辑记录应渲染成功")
    segs = r.data.editSegments
    for s in segs:
        print(f"  {s.shotId} @{s.atMs}ms +{s.durationMs}ms  video={s.videoKey.split('/')[-1] or '-'} "
              f"image={s.imageKey.split('/')[-1]} degrade={s.degradeMode or '-'} note={s.note}")
    check(len(segs) == 2 and [s.shotId for s in segs] == ["s1", "s2"], "逐镜一条剪辑记录（EDL 索引对齐）")
    s1, s2 = segs[0], segs[1]
    check(s1.atMs == 0 and s1.durationMs == 3000 and s2.atMs == 3000 and s2.durationMs == 4000,
          "时间轴位置与剪辑时长取自锁定 EDL（该段在成片里的绝对区间）")
    check(s1.videoKey.endswith("clips/s1.mp4") and s1.degradeMode == "",
          "有片段的镜头：videoKey 在场、无降级")
    check(s2.videoKey == "" and s2.imageKey.endswith("杯1.jpg") and s2.degradeMode == "fallback_still",
          "无片段的镜头：如实标 fallback_still 并带上回落的静图 key")
    check(s1.voKey.endswith("audio/voice.mp3") and s1.bgmKey == "",
          "音轨如实：voKey = 配音 key；bgmKey 空（degrade 节点如实为空轨）")
    check(any("无 BGM" in s.note and "恒降级" in s.note for s in segs),
          "BGM 未接通（恒降级）在**剪辑记录里如实标出**（bgm 节点在场但没选出曲子）")
    check(s1.trim.sourceMs == 5200 and s1.trim.outMs == 3000,
          "裁切实况**透过 render 原样带出**（i2v 测的，render 不改写）")
    check(any("裁掉 2200ms" in s.note for s in segs[:1]), "裁掉多少写进该段 note（可事后核）")
    check(s1.transition is None and s2.transition is None, "未请求转场 ⇒ 实况硬切（None）")
    check(any("剪辑记录 2 段" in w for w in r.data.warnings), "run 级如实汇总剪辑记录段数")

    # 5b) **请求 ≠ 实况**：EDL 上标了转场但厂商目录里没有可用于非交叠的项 ⇒ 该段 transition 为 None 并写明原因
    deps = _edl_motion() | {
        "video": {"clips": [{"shotId": "s1", "key": "dsp/scratch/t1/run-demo-7/clips/s1.mp4",
                             "atMs": 0, "durationMs": 3000}]},
        "tts": {"audioKey": "dsp/scratch/t1/run-demo-7/audio/voice.mp3"},
    }
    deps["timeline"]["edl"]["tracks"][0]["segments"][0]["transition"] = "wipe"
    r = invoke("render_compose", _ctx(deps=deps))
    print(f"  s1 note   : {r.data.editSegments[0].note}")
    check(r.data.editSegments[0].transition is None,
          "不可用的转场 ⇒ 该段 transition 如实为 None（**不把请求当实况**）")
    check("硬切" in r.data.editSegments[0].note and "wipe" in r.data.editSegments[0].note,
          "回落原因写进该段 note")

    # 5c) 已登记项（开箱即用）⇒ 该段 transition 记下**真加进轨的**符号名，并**如实告警适用范围**
    #     厂商口径（用户 2026-10-14）：TransitionFilter 加在**前一个**视频的 Extra 上，
    #     适用范围是「同轨道**相邻两 Video**」——本处 s2 是**静图**（无片段回落 image），
    #     故必须告警「静图邻接是否生效未经实测」（文档支持范围 ≠ 实际用到的形态）。
    _BODY.clear()
    deps = _edl_motion() | {
        "video": {"clips": [{"shotId": "s1", "key": "dsp/scratch/t1/run-demo-7/clips/s1.mp4",
                             "atMs": 0, "durationMs": 3000}]},
        "tts": {"audioKey": "dsp/scratch/t1/run-demo-7/audio/voice.mp3"},
    }
    deps["timeline"]["edl"]["tracks"][0]["segments"][0]["transition"] = "dissolve"
    r = invoke("render_compose", _ctx(deps=deps))
    print(f"  s1 transition: {r.data.editSegments[0].transition}")
    print(f"  s1 extra  : {_BODY['track'][0][0].get('extra')}")
    print(f"  warnings  : {[w for w in r.data.warnings if 'Video' in w]}")
    check(r.data.editSegments[0].transition == "dissolve",
          "已登记（开箱即用）的转场 ⇒ 该段 transition = 实况转场")
    check("硬切" not in r.data.editSegments[0].note, "已施加的转场不在该段 note 里写「硬切」")
    check(_BODY["track"][0][0].get("extra") == [{"type": "transition", "source": "1182358",
                                                 "duration": 400}],
          "转场编成 extra={type,source,duration} 加在**前一个**元素上（与厂商 TransitionFilter 逐字对齐）")
    check(any("相邻两 Video" in w and "静图" in w and "未经实测" in w for w in r.data.warnings),
          "相邻含静图须如实告警（厂商文档只写了 Video↔Video）")

    # 5d) 既无片段也无图 ⇒ missing_clip（该段没进片子，如实落一条而非从记录里消失）
    _BODY.clear()
    empty_one = {"timeline": {"edl": {"version": 1, "totalMs": 3000, "tracks": [
        {"kind": "image", "segments": [{"atMs": 0, "durationMs": 3000, "sourceKey": ""}]}],
        "shots": [{"id": "s1", "atMs": 0, "durationMs": 3000, "imageSourceKey": ""}]}}}
    r = invoke("render_compose", _ctx(deps=empty_one | {
        "tts": {"audioKey": "dsp/scratch/t1/run-demo-7/audio/voice.mp3"}}))
    check(r.ok, "全空段仍可渲染（渲染侧本就跳过空元素）")
    check(len(r.data.editSegments) == 1 and r.data.editSegments[0].degradeMode == "missing_clip",
          "既无片段也无图 ⇒ 该段如实标 missing_clip（不静默消失）")

    print("\n✅ dsp_render_compose_demo 全部通过")


if __name__ == "__main__":
    main()
