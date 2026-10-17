"""dsp_image_to_video 技能演示 —— 图生视频（逐镜产视频底 · 契约 §7.5 第 1 步）。

运行：PYTHONPATH=<repo>/src conda run -n flowmind python examples/dsp_image_to_video_demo.py

展示：
1. discover() 自动字段发现
2. happy path：2 镜 → **输入降采样**（1152×2048 → ≤720×1280，真跑 PIL）+ **首尾帧桥**
   （s1 尾帧 = s2 的图）→ 提交 → 完成 → **直链转存**为不可变 key（24h 直链不当 key）
3. build_payload：有/无尾帧的两种提交体形状（``images`` 数组第 2 位 = 尾帧）
4. degrade：未配密钥 → 如实降级（成片回落静图，不伪造片段）
5. degrade：缺锁定时间线 → 明确 degradation_reason
6. degrade：某镜出片失败 → 该镜回落静图 + warnings 如实说明（其余镜照常出片）
7. 降采样失败 → **回落原图**并如实告警（画质风险，**不丢镜**）
8. 成本闸：maxShots 限制生成镜数
9. 开关：``FLOWMIND_DSP_VIDEOGEN_BRIDGE_NEXT_IMAGE=0`` / ``..._DOWNSCALE_INPUT=0``
10. **运镜（ADR-076）**：经引擎适配器并入 prompt——「运镜」开关关 ⇒ 逐镜下**固定机位**条款；
   开 ⇒ 用该镜方案（Agent 的 positive 优先、否则确定性短语）；有方案但开关关 ⇒ 忽略并如实告警；
   词表外符号 ⇒ 空短语不硬凑；**引擎参数位恒空**（本聚合端点无运动入参）
12. **质检的测量侧（ADR-079）**：有界重试 / 按镜复用 / qc 开=抽帧+客观量、关=不抽不测 / 抽帧失败不丢镜
13. **裁到镜头窗口（ADR-080 阶段 5）**：超长裁掉 / 不足**末帧定格**补齐并如实报 `shortfallMs` /
    两个 env 开关（`TRIM_WINDOW` / `FREEZE_PAD`）/ 裁切失败**回落原片段**不丢镜
    —— 本 demo 要**离线可跑**（CI 无 ffmpeg），故裁切靠**注入假 ffmpeg/ffprobe** 断言命令行与 trim
    统计；**真 ffmpeg 的行为复核另见 `examples/dsp_trim_realcheck.py`**（合成片段验 `-t` 真裁 / `tpad`
    真补 / `-an` 真丢音轨，并用 ffprobe 复核输出时长）

mock 方式：替换 ``_dsp_video_gen`` 的 submit/wait（模块级薄适配层）与 ``_dsp_storage``
的 ``presign_get_public`` / ``probe_public_read`` / fetch/put_file，不需要网络 / 厂商密钥 /
对象存储。密钥用假值经 env 注入。``fetch_to_file`` **真写一张 PIL 图**（按文件名给尺寸）——
故降采样是**真跑**而非假过。

**ADR-123**：6b（7b）段专验「输入素材公网可达性」——未配公网端点 / 端点字面内网 / HTTP 预检失败
三种情况都必须**逐镜 skipped（source_url_unreachable）且零厂商提交**；逐镜状态 `shots[]` 与
`summary` 同时断言（厂商失败记 `failed` / `vendor_generation_failed`，与前置失败可区分）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import flowmind.skills  # noqa: F401  触发 @skill 注册
from flowmind.config import load_config
from flowmind.discover import field_names
from flowmind.skill import invoke
from flowmind.skills import _dsp_storage as storage
from flowmind.skills import _dsp_video_gen as videogen
from flowmind.skills import _media as media

_KEY_ENV = "DSP_VIDEOGEN_API_KEY"
_PUTS: list[str] = []
_SUBMITTED: list[dict] = []
_FETCHED: list[str] = []
_FAIL_SHOTS: set[str] = set()
# ADR-079 用：提交时**前 N 次**抛「瞬时（retriable）」错（验有界重试）；以及「已存在的对象」（验按镜复用）。
_TRANSIENT_FAILS = {"n": 0}
_EXISTS: set[str] = set()
# ADR-080 用：假 ffmpeg/ffprobe（本开发机**没有** ffmpeg）——按文件名给「源片段时长」，
# 于是「超长裁掉」与「不足末帧定格」两条路径都能在本机**离线真跑**（断言命令行与 trim 统计）。
_FFMPEG_CMDS: list[list[str]] = []
_SOURCE_MS = {"s1.mp4": 5200, "s2.mp4": 1500}   # s1 超窗口(3000) / s2 不足窗口(4000)

# 素材池里这些图的**实际像素**（按文件名匹配）：补生图 = 9:16 @2K（超 I2V 输出 720×1280），
# 另一张 = 小图（未超 ⇒ 原图直用，不落派生件）。
_IMAGE_PIXELS = {"门头1.jpg": (1152, 2048), "杯1.jpg": (600, 900)}
_REAL_PROBE = media.probe_image_size
_REAL_FFMPEG = media.run_ffmpeg
_REAL_FFPROBE = media.run_ffprobe


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _write_image(dest_path: str, url: str) -> None:
    """按素材文件名写一张**真图**（降采样路径因此真跑 PIL，不是假过）。"""
    from PIL import Image

    w, h = next((v for k, v in _IMAGE_PIXELS.items() if k in url), (600, 900))
    Image.new("RGB", (w, h), (32, 64, 96)).save(dest_path)


def _install_fakes() -> None:
    def submit(cfg, *, prompt: str, image_url: str, seconds: str,
               tail_image_url: str = "", motion_params: dict | None = None) -> str:
        _SUBMITTED.append({"prompt": prompt, "image_url": image_url, "seconds": seconds,
                           "tail_image_url": tail_image_url,
                           "motion_params": dict(motion_params or {})})
        # **有界重试**（ADR-079）：前 N 次抛瞬时错（厂商把错误分成 retriable/其它，本步只重试前者）
        if _TRANSIENT_FAILS["n"] > 0:
            _TRANSIENT_FAILS["n"] -= 1
            raise videogen.DspVideoGenError("网关抖动（瞬时）", category="transient", retriable=True)
        return f"task_{len(_SUBMITTED)}"

    def wait(cfg, task_id: str, *, api_key: str = "") -> dict:
        if task_id in _FAIL_SHOTS:
            raise videogen.DspVideoGenError(f"任务 {task_id} 生成失败", category="video")
        return {"id": task_id, "status": "completed",
                "metadata": {"url": f"https://ivd.example/video_cache/{task_id}.mp4"}}

    def fetch_to_file(url: str, dest_path: str, timeout_s: float | None = None) -> str:
        _FETCHED.append(url)
        if url.startswith("https://assets.example/"):
            _write_image(dest_path, url)          # 素材图：真写（供真跑降采样）
        else:
            Path(dest_path).write_bytes(b"\x00\x00\x00\x18ftypmp42")   # 片段：占位字节
        return dest_path

    def put_file(path: str, key: str, content_type: str | None = None) -> str:
        _PUTS.append(key)
        return key

    def presign_get_public(key: str, ttl_s: int | None = None) -> str:
        return f"https://assets.example/{key}"

    def probe_public_read(url: str, timeout_s: float = 8.0) -> tuple[bool, str]:
        return True, ""

    def exists(key: str) -> bool:
        return key in _EXISTS

    # ── 假 ffmpeg / ffprobe（ADR-080）：本 demo 要离线可跑（CI 无 ffmpeg），故裁切路径靠**注入假件** ──
    # ffprobe 按**文件名**报源片段时长（`_SOURCE_MS`），ffmpeg 只记录命令行并落一个占位文件。
    # 于是「超长裁掉 / 不足末帧定格」两条路径**在无 ffmpeg 的机器上也跑得通**，命令行与 trim 统计都被断言；
    # **真 ffmpeg 下的实际行为**由 `examples/dsp_trim_realcheck.py` 复核（本机有 ffmpeg 时按需真跑）。
    def run_ffprobe(cmd: list[str], **kw) -> tuple[int, str, str]:
        ms = _SOURCE_MS.get(Path(cmd[-1]).name, 3000)
        out = json.dumps({"format": {"duration": f"{ms / 1000:.3f}"},
                          "streams": [{"width": 720, "height": 1280}]})
        return 0, out, ""

    def run_ffmpeg(cmd: list[str], **kw) -> tuple[int, str, str]:
        _FFMPEG_CMDS.append(list(cmd))
        Path(cmd[-1]).write_bytes(b"\x00\x00\x00\x18ftypmp42")
        return 0, "", ""

    videogen.submit = submit
    videogen.wait = wait
    storage.fetch_to_file = fetch_to_file
    storage.put_file = put_file
    storage.presign_get_public = presign_get_public
    storage.probe_public_read = probe_public_read
    storage.exists = exists
    media.run_ffprobe = run_ffprobe
    media.run_ffmpeg = run_ffmpeg


def _edl() -> dict:
    return {"timeline": {"edl": {
        "version": 1, "totalMs": 7000,
        "tracks": [
            {"kind": "image", "segments": [
                {"atMs": 0, "durationMs": 3000, "sourceKey": "dsp/scratch/t1/门店/门头1.jpg"},
                {"atMs": 3000, "durationMs": 4000, "sourceKey": "dsp/scratch/t1/产品/杯1.jpg"},
            ]},
        ],
        "shots": [
            {"id": "s1", "atMs": 0, "durationMs": 3000,
             "imageSourceKey": "dsp/scratch/t1/门店/门头1.jpg",
             # ADR-073：s1 带**画面提示词**（主路径），s2 刻意不给（走 copy 兜底 + 如实告警）
             "visualPrompt": "中景，平视，固定机位；店员弯腰拉起卷帘门，晨光斜射进门头；写实拍摄质感"},
            {"id": "s2", "atMs": 3000, "durationMs": 4000,
             "imageSourceKey": "dsp/scratch/t1/产品/杯1.jpg"},
        ],
    }}}


def _ctx(**over) -> dict:
    base = {
        "mode": "content", "platform": "douyin",
        "cachePrefix": "dsp/scratch/t1", "outputPrefix": "dsp/output/t1",
        "runId": "run-demo-8", "stepId": "video", "idempotencyKey": "run-demo-8:video",
        "shots": [
            {"id": "s1", "durationMs": 3000, "copy": "早上七点，卷帘门刚拉起"},
            {"id": "s2", "durationMs": 4000, "copy": "这杯保温 12 小时"},
        ],
        "materials": [],
        "deps": _edl(),
    }
    base.update(over)
    return base


def _reset() -> None:
    _PUTS.clear()
    _SUBMITTED.clear()
    _FETCHED.clear()
    _FAIL_SHOTS.clear()
    _EXISTS.clear()
    _TRANSIENT_FAILS["n"] = 0
    _FFMPEG_CMDS.clear()


def main() -> None:
    section("0) discover('image_to_video') —— Agent 自查字段")
    for path, names in field_names("image_to_video").items():
        print(f"  {path}: {names}")

    section("1) Happy path：2 镜 → 输入降采样 + 首尾帧桥 → 提交 → 完成 → 直链转存")
    os.environ[_KEY_ENV] = "sk-demo-key"
    _install_fakes()
    _reset()
    r = invoke("image_to_video", _ctx())
    check(r.ok, "happy path 应 ok")
    print(f"  提交参数  : seconds={[s['seconds'] for s in _SUBMITTED]}（镜头建议时长原样下发）")
    print(f"  首镜 prompt: {_SUBMITTED[0]['prompt']}")
    print(f"  首帧      : {_SUBMITTED[0]['image_url']}")
    print(f"  尾帧      : {_SUBMITTED[0]['tail_image_url']}")
    print(f"  片段      : {[c.key for c in r.data.clips]}")
    print(f"  转存      : {_PUTS}")
    print(f"  结论      : {r.reasoning[0].conclusion}")
    print(f"  失败标记  : degraded={r.metrics.degraded}")
    check(len(r.data.clips) == 2, "两镜都应出片")
    check(_SUBMITTED[0]["seconds"] == "3" and _SUBMITTED[1]["seconds"] == "4",
          "秒数按镜头时长下发（3000ms→'3' / 4000ms→'4'）")
    # ADR-073：prompt 取**画面提示词**（不是台词）；末尾统一追加负向约束（禁文字/字幕/水印）
    check(_SUBMITTED[0]["prompt"].startswith("中景，平视，固定机位"),
          "prompt 取画面提示词（visualPrompt），不是口播文案")
    check("不要出现任何文字" in _SUBMITTED[0]["prompt"], "prompt 末尾须带负向约束（禁文字/字幕/水印）")
    check(_SUBMITTED[1]["prompt"].startswith("这杯保温 12 小时"),
          "无画面提示词的镜**回落用文案**出 prompt（如实兜底）")
    check(any("没有画面提示词" in w and "回落" in w for w in r.data.warnings),
          "回落用文案出 prompt 须**如实告警**（ADR-073）")
    # ADR-074 ① 首尾帧桥：s1 的尾帧 = s2 的图；末镜没有下一镜 ⇒ 无尾帧
    check("杯1.jpg" in _SUBMITTED[0]["tail_image_url"],
          "s1 尾帧 = **下一镜**的图（首尾帧桥，ADR-074）")
    check(_SUBMITTED[1]["tail_image_url"] == "", "末镜无下一镜 ⇒ 不下发尾帧")
    # ADR-074 ② 输入降采样：1152×2048 超输出 720×1280 ⇒ 先降采样再下发（派生件转存）
    check("/i2v_input/" in _SUBMITTED[0]["image_url"],
          "超输出分辨率的素材须**降采样后**下发（不是原图直发）")
    check(_SUBMITTED[0]["image_url"].startswith(
        "https://assets.example/dsp/scratch/t1/run-demo-8/i2v_input/"),
        "派生件落 {cachePrefix}/{runId}/i2v_input/…（工作缓存目录内）")
    check(sum(1 for k in _PUTS if "/i2v_input/" in k) == 1,
          "降采样派生件恰 1 个（只给超标的 s1 那张）")
    check(_SUBMITTED[1]["image_url"].endswith("产品/杯1.jpg"),
          "未超输出分辨率的素材**原图直用**（不重编码、不落派生件）")
    check(sum(1 for u in _FETCHED if u.endswith("杯1.jpg")) == 1,
          "同一素材只准备一次（s1 的尾帧与 s2 的图共用，不重复下载）")
    check("降采样" in (r.reasoning[0].causal_analysis or ""),
          "结论须如实说明做过输入降采样")
    check("首尾帧桥" in r.reasoning[0].conclusion, "结论须如实说明用了首尾帧桥")
    check("dsp/scratch/t1/run-demo-8/clips/s1.mp4" in _PUTS, "直链须转存为不可变 key")
    check(r.metrics.degraded is False, "全出片不应标记降级")
    # ADR-123：逐镜真实结局**结构化**带出（成功时含片段 key + runId），运行器据此判阶段进度
    by_status = {s.shotId: s for s in r.data.shots}
    check(by_status["s1"].status == "generated" and by_status["s2"].status == "generated",
          "happy path：两镜状态都是 generated")
    check(by_status["s1"].key.endswith("/clips/s1.mp4") and by_status["s1"].runId == "run-demo-8",
          "success 带片段 key 与 runId（可回查产物）")
    check(by_status["s1"].imageKey.endswith("门头1.jpg"),
          "逐镜状态保留素材绑定关系（回落静图时也保留）")
    check(r.data.summary.total == 2 and r.data.summary.generated == 2
          and r.data.summary.skipped == 0 and r.data.summary.failed == 0,
          "summary 按真实结局汇总（2/2 generated）")

    section("2) build_payload：有/无尾帧的两种提交体形状")
    cfg = load_config().dsp
    with_tail = videogen.build_payload(cfg, prompt="p", image_url="A", seconds="3",
                                       tail_image_url="B")
    without = videogen.build_payload(cfg, prompt="p", image_url="A", seconds="3")
    print(f"  有尾帧    : {with_tail}")
    print(f"  无尾帧    : {without}")
    check(with_tail["images"] == ["A", "B"] and with_tail["image"] == "A",
          "有尾帧 ⇒ images = [首帧, 尾帧]（第 2 位 = 尾帧）")
    check("images" not in without and without["image"] == "A",
          "无尾帧 ⇒ 只发 image（不再塞单元素 images）")
    check(with_tail["duration"] == 3 and isinstance(with_tail["duration"], int),
          "时长字段必须是 duration（整数秒）")
    check(not ({"aspect_ratio", "quality", "seconds"} & set(with_tail)),
          "不得再发 aspect_ratio / quality / seconds（非该接口字段，ADR-111）")
    check(videogen.parse_size("720x1280") == (720, 1280)
          and videogen.parse_size("720X1280") == (720, 1280)
          and videogen.parse_size("") == (0, 0) and videogen.parse_size("abc") == (0, 0),
          "parse_size 形状守卫（大小写不敏感；坏串 → (0,0)）")

    section("3) degrade：未配密钥 → 如实降级（不伪造片段）")
    os.environ.pop(_KEY_ENV, None)
    r = invoke("image_to_video", _ctx())
    print(f"  ok        : {r.ok}  degraded={r.metrics.degraded}")
    print(f"  reason    : {r.metrics.degradation_reason}")
    print(f"  warnings  : {r.data.warnings}")
    check(r.ok and r.metrics.degraded and r.metrics.degradation_reason == "videogen-not-configured",
          "无密钥须降级且原因明确")
    check(r.data.clips == [], "降级不得给出片段（不伪造）")
    os.environ[_KEY_ENV] = "sk-demo-key"

    section("4) degrade：缺锁定时间线（deps.timeline.edl）")
    r = invoke("image_to_video", _ctx(deps={}))
    print(f"  reason    : {r.metrics.degradation_reason}")
    check(r.metrics.degradation_reason == "missing-timeline", "缺时间线须降级")

    section("5) degrade：某镜出片失败 → 该镜回落静图（其余照常）")
    _reset()
    _FAIL_SHOTS.add("task_2")   # 第二镜（s2）失败
    r = invoke("image_to_video", _ctx())
    print(f"  clips     : {[c.shotId for c in r.data.clips]}")
    print(f"  degraded  : {r.metrics.degraded} / {r.metrics.degradation_reason}")
    print(f"  warnings  : {r.data.warnings}")
    check([c.shotId for c in r.data.clips] == ["s1"], "失败镜不出片段，成功的照常出片")
    check(r.metrics.degraded and r.metrics.degradation_reason == "partial-shots-fell-back",
          "部分失败须如实标记降级")
    check(any("回落静图" in w for w in r.data.warnings), "失败镜须如实说明回落")
    # ADR-123：**厂商调用失败**与前置条件失败的原因码不同（前者 vendor_*，后者 source_url_unreachable）
    by_status = {s.shotId: s for s in r.data.shots}
    check(by_status["s2"].status == "failed" and by_status["s2"].reason == "vendor_generation_failed",
          "厂商侧失败记 failed / vendor_generation_failed（不是 skipped）")
    check(r.data.summary.generated == 1 and r.data.summary.failed == 1,
          "summary：1 生成 / 1 失败（不从 clips 反推）")
    _FAIL_SHOTS.clear()

    section("6) 成本闸：maxShots 限制生成镜数")
    _reset()
    r = invoke("image_to_video", _ctx(maxShots=1))
    print(f"  clips     : {[c.shotId for c in r.data.clips]}")
    print(f"  warnings  : {r.data.warnings}")
    check(len(r.data.clips) == 1 and r.data.clips[0].shotId == "s1", "maxShots=1 只生成首镜")
    check(any("maxShots" in w for w in r.data.warnings), "截断须如实说明")
    max_status = {s.shotId: s for s in r.data.shots}
    check(max_status["s2"].status == "skipped" and max_status["s2"].reason == "max_shots",
          "被成本闸截断的镜记 skipped / max_shots（不是失败，也没提交厂商）")

    section("7) 降采样失败 → 回落原图并如实告警（画质风险，不丢镜）")
    _reset()

    def _boom(path: str) -> tuple[int, int]:
        raise media.MediaError("坏图（演示用）")

    media.probe_image_size = _boom
    try:
        r = invoke("image_to_video", _ctx())
    finally:
        media.probe_image_size = _REAL_PROBE
    print(f"  首镜下发的图: {_SUBMITTED[0]['image_url']}")
    print(f"  warnings  : {[w for w in r.data.warnings if '降采样' in w]}")
    check(_SUBMITTED[0]["image_url"] == "https://assets.example/dsp/scratch/t1/门店/门头1.jpg",
          "降采样失败 ⇒ **回落原图**下发")
    check(not any("/i2v_input/" in k for k in _PUTS), "降采样失败 ⇒ 不落派生件")
    check(any("回落原图" in w for w in r.data.warnings), "降采样失败须**如实告警**（不静默）")
    check(len(r.data.clips) == 2, "降采样失败**不丢镜**（画质风险 ≠ 成片失败）")

    section("7b) 公网输入前置检查（ADR-123）：未配公网端点 / 私网地址 / 预检失败 -> 逐镜 skipped，不提交厂商")

    def _fake_presign_raises(key: str, ttl_s: int | None = None) -> str:
        raise storage.DspStorageError(
            "未配置公网对象存储端点 FLOWMIND_DSP_S3_PUBLIC_ENDPOINT",
            category="environment", code="public_endpoint_unconfigured")

    _reset()
    storage.presign_get_public = _fake_presign_raises
    r = invoke("image_to_video", _ctx())
    print(f"  未配公网端点 → shots: {[(s.shotId, s.status, s.reason) for s in r.data.shots]}")
    check(len(_SUBMITTED) == 0, "公网端点未配置 ⇒ **一次厂商提交都不发**（不浪费按镜计费）")
    check(all(s.status == "skipped" and s.reason == "source_url_unreachable" for s in r.data.shots),
          "逐镜记 skipped / source_url_unreachable（不是 failed）")
    check(r.data.summary.skipped == 2 and r.metrics.degradation_reason == "all-shots-fell-back",
          "全部跳过 ⇒ summary 如实计数，阶段不会标成完成")
    check(r.data.clips == [], "不伪造片段")

    def _fake_presign_private(key: str, ttl_s: int | None = None) -> str:
        return f"http://127.0.0.1:9000/flowmind/{key}"

    _reset()
    storage.presign_get_public = _fake_presign_private
    r = invoke("image_to_video", _ctx())
    print(f"  私网端点 → shots: {[(s.shotId, s.status, s.reason) for s in r.data.shots]}")
    check(len(_SUBMITTED) == 0, "公网端点字面内网 ⇒ 不提交厂商")
    check(all(s.reason == "source_url_unreachable" for s in r.data.shots),
          "内网预签名地址被识别为不适合给公网厂商（source_url_unreachable）")

    _install_fakes()          # 恢复「公网签发」假件，只让**预检**失败
    _reset()

    def _probe_fails(url: str, timeout_s: float = 8.0) -> tuple[bool, str]:
        return False, "HTTP 403（签名与公网 Host 不匹配，或反向代理未透传 Host）"

    storage.probe_public_read = _probe_fails
    r = invoke("image_to_video", _ctx())
    print(f"  预检失败 → shots: {[(s.shotId, s.status, s.reason) for s in r.data.shots]}")
    check(len(_SUBMITTED) == 0, "公网预检失败 ⇒ 不提交厂商（签名/反代问题提前暴露）")
    check(all(s.reason == "source_url_unreachable" for s in r.data.shots),
          "预检失败与厂商失败区分：前者 skipped / source_url_unreachable")
    check(any("公网预检失败" in w for w in r.data.warnings),
          "预检失败须如实告警并给出排查方向（Host / 签名）")
    _install_fakes()          # 恢复

    section("8) 开关：关首尾帧桥 / 关输入降采样（env 覆盖）")
    os.environ["FLOWMIND_DSP_VIDEOGEN_BRIDGE_NEXT_IMAGE"] = "0"
    _reset()
    invoke("image_to_video", _ctx())
    print(f"  关桥后尾帧: {[s['tail_image_url'] for s in _SUBMITTED]}")
    check(all(s["tail_image_url"] == "" for s in _SUBMITTED), "关桥 ⇒ 全部单图下发")
    check(any("/i2v_input/" in k for k in _PUTS), "关桥不影响降采样（两者独立）")
    os.environ.pop("FLOWMIND_DSP_VIDEOGEN_BRIDGE_NEXT_IMAGE")

    os.environ["FLOWMIND_DSP_VIDEOGEN_DOWNSCALE_INPUT"] = "0"
    _reset()
    invoke("image_to_video", _ctx())
    print(f"  关降采样后首帧: {_SUBMITTED[0]['image_url']}")
    check(not any("/i2v_input/" in k for k in _PUTS), "关降采样 ⇒ 不落派生件")
    check(_SUBMITTED[0]["image_url"].endswith("门店/门头1.jpg"), "关降采样 ⇒ 原图直发")
    os.environ.pop("FLOWMIND_DSP_VIDEOGEN_DOWNSCALE_INPUT")

    # ── 10) 运镜（ADR-076）：经引擎适配器并入 prompt ──
    # 口径：**关**（缺省）= 图照做、要求摄影机固定；**开** = 下发逐镜方案（缺方案 = 固定机位）。
    section("10) 运镜：适配器并入 prompt（关=固定机位 / 开=逐镜方案 / 词表外不硬凑）")

    def _edl_with_motion(extra: dict, shot_id: str = "s1") -> dict:
        """把运镜方案塞进 EDL 对应镜头——I2V **从 EDL 读**（并/拆会重排 id，回查 inp.shots 必错配）。"""
        edl = _edl()
        for shot in edl["timeline"]["edl"]["shots"]:
            if shot["id"] == shot_id:
                shot["motion"] = extra
        return edl

    plan = {"type": "push_in", "intensity": "subtle", "direction": "toward_subject",
            "target": "门头", "reason": "留白充足", "positive": "", "negative": ""}

    # 10a) 「运镜」开关**关** ⇒ 逐镜下固定机位条款（**不是**「不做 I2V」）
    _reset()
    r = invoke("image_to_video", _ctx(deps=_edl_with_motion(plan)))
    check(r.ok and len(r.data.clips) == 2, "运镜关也不影响出片（图照做）")
    print(f"  关运镜 prompt 尾部: ...{_SUBMITTED[0]['prompt'][-56:]}")
    check(videogen.MOTION_STATIC_CLAUSE in _SUBMITTED[0]["prompt"],
          "运镜关 ⇒ prompt 带**固定机位条款**")
    check("保持固定" in _SUBMITTED[0]["prompt"] and "自然" in _SUBMITTED[0]["prompt"],
          "条款含「保持固定」且**允许**自然动态（不越界到「画面动态强度」）")
    check(any("开关已关" in w for w in r.data.warnings),
          "有方案但运镜开关关 ⇒ 如实告警被忽略（不静默）")

    # 10b) 「运镜」开关**开** ⇒ 用该镜方案；无 positive 时出确定性短语（运动 + 幅度 + 方向 + 焦点）
    _reset()
    r = invoke("image_to_video", _ctx(switches={"motion": True}, deps=_edl_with_motion(plan)))
    p1 = _SUBMITTED[0]["prompt"]
    print(f"  开运镜 s1 prompt 尾部: ...{p1[-64:]}")
    check("推近" in p1 and "幅度轻微" in p1 and "朝主体" in p1 and "门头" in p1,
          "无 positive ⇒ 适配器出确定性短语（运动+幅度+方向+焦点）")
    check(videogen.MOTION_STATIC_CLAUSE in _SUBMITTED[1]["prompt"],
          "同批里**没给方案**的镜仍按固定机位（逐镜粒度）")
    # 计数是**结论里的摘要**（不是告警——告警只留「方案被开关忽略」那种需要动作的信号）
    _concl = r.reasoning[0].conclusion
    print(f"  结论摘要  : {_concl}")
    check("运镜方案 1 镜" in _concl and "固定机位 1 镜" in _concl,
          "结论如实报出「运镜方案 N 镜 / 固定机位 M 镜」")

    # 10c) Agent 给了 positive ⇒ 优先用它（它看过图、能写到具体主体）
    _reset()
    invoke("image_to_video", _ctx(switches={"motion": True}, deps=_edl_with_motion(
        plan | {"positive": "Slow, smooth camera push-in toward the shopfront."})))
    check("Slow, smooth camera push-in toward the shopfront." in _SUBMITTED[0]["prompt"],
          "Agent 的 positive 优先于确定性短语")

    # 10d) 词表外符号 ⇒ 空短语（不硬凑）+ 如实告警；该镜不冒充方案
    _reset()
    r = invoke("image_to_video", _ctx(switches={"motion": True},
                                      deps=_edl_with_motion(plan | {"type": "warp_speed"})))
    check(any("未登记到引擎适配表" in w for w in r.data.warnings),
          "词表外运镜符号 ⇒ 如实告警（不硬凑运动）")
    check(videogen.MOTION_STATIC_CLAUSE not in _SUBMITTED[0]["prompt"]
          and "warp_speed" not in _SUBMITTED[0]["prompt"],
          "词表外符号既不下发也不冒充方案")

    # 10e) **引擎参数位恒空**：本聚合端点没有运动入参 ⇒ 不往请求体塞自造字段
    _reset()
    invoke("image_to_video", _ctx(switches={"motion": True}, deps=_edl_with_motion(plan)))
    check(all(item["motion_params"] == {} for item in _SUBMITTED),
          "本引擎无运动入参 ⇒ motion_params 恒空（参数位留给将来引擎）")
    check(videogen.motion_expression(None)[0] == videogen.MOTION_STATIC_CLAUSE,
          "未给方案（None）也走固定机位条款——「不运镜」要明确告诉模型别动镜头")



    section("12) ADR-079：有界重试 / 按镜复用 / 质检的测量侧（抽帧 + 客观量）")
    from flowmind.skills import _dsp_common as _common12
    from flowmind.skills import _dsp_qc as qc

    # 12a) **有界重试**：第一次提交抛瞬时错（retriable）⇒ 重试后成功，该镜**不**回落静图
    _reset()
    _TRANSIENT_FAILS["n"] = 1
    r = invoke("image_to_video", _ctx())
    check(r.ok and len(r.data.clips) >= 1, "瞬时错重试后仍出片（不因一次抖动丢镜）")
    check(any("重试" in w for w in r.data.warnings), "重试须如实告警（不静默吞掉）")

    # 12b) **按镜复用**：regenerateShotIds 只列 s1 ⇒ s2 复用已有产物（不重新出片、不重复计费）
    _reset()
    s2_key = _common12.step_key("dsp/scratch/t1", "run-demo-8", "clips", "s2.mp4")
    _EXISTS.add(s2_key)
    r = invoke("image_to_video", _ctx(regenerateShotIds=["s1"]))
    keys = [c.key for c in r.data.clips]
    check(s2_key in keys, "未列入 regenerateShotIds 且已有产物 ⇒ 直接复用该 key")
    check(len(_SUBMITTED) == 1, "只重生成被点名的那一镜（实际提交 %d 次）" % len(_SUBMITTED))
    check(any("复用" in w for w in r.data.warnings), "复用须如实说明（未重复出片 / 未重复计费）")
    reuse_status = {s.shotId: s for s in r.data.shots}
    check(reuse_status["s2"].status == "reused" and reuse_status["s2"].key == s2_key,
          "复用的镜记 reused 并带出复用的 key（不是 generated）")

    # 12c) **质检测量侧**：qc 开关开 ⇒ 抽帧上传 + 客观量随片段带出；关 ⇒ 完全不抽不测
    _reset()
    calls = {"n": 0}
    def _fake_sample(video_path, out_dir, duration_s, ratios=None):
        calls["n"] += 1
        paths = []
        for i in range(3):
            fp = out_dir + "/f" + str(i) + ".png"
            Path(fp).write_bytes(b"png")
            paths.append(fp)
        return qc.FrameSample(paths=paths, ratios=list(ratios or qc.FRAME_RATIOS))
    def _fake_stats(paths):
        return {"width": 720, "height": 1280, "frames": len(paths),
                "meanLuma": [40.0, 42.0, 41.0], "meanDiff": [12.0, 11.0],
                "blackFrames": 0, "frozenPairs": 0}
    real_sample, real_stats = qc.sample_frames, qc.frame_stats
    qc.sample_frames, qc.frame_stats = _fake_sample, _fake_stats
    try:
        r = invoke("image_to_video", _ctx(switches={"qc": True}))
        c0 = r.data.clips[0]
        check(len(c0.frames) == 3 and all("/qa/" in k for k in c0.frames),
              "qc 开 ⇒ 逐镜抽 3 帧并上传到 qa/（帧 key 随片段带出）")
        check(all(k.endswith(".png") for k in c0.frames), "帧以 PNG 上传（与抽帧格式一致，不谎报类型）")
        check(c0.frameStats.get("frozenPairs") == 0 and c0.frameStats.get("width") == 720,
              "客观量随片段带出（分辨率 / 亮度 / 差异 / 黑帧与冻结帧计数）")
        check(calls["n"] == len(r.data.clips), "每镜各抽一次（不重复抽）")

        _reset()
        calls["n"] = 0
        r = invoke("image_to_video", _ctx(switches={"qc": False}))
        check(calls["n"] == 0 and all(not c.frames for c in r.data.clips),
              "qc 关 ⇒ 不抽帧不测（不产生额外开销）")
        check(all(c.frameStats == {} for c in r.data.clips), "qc 关 ⇒ 不带客观量（不留空壳）")

        # 12d) **抽帧失败不丢镜**：测量挂了仍保住片段并如实告警（画质风险 != 丢镜）
        _reset()
        def _boom(video_path, out_dir, duration_s, ratios=None):
            raise media.MediaError("本机没有 ffmpeg（演示注入）")
        qc.sample_frames = _boom
        r = invoke("image_to_video", _ctx(switches={"qc": True}))
        check(len(r.data.clips) >= 1 and all(not c.frames for c in r.data.clips),
              "抽帧失败 ⇒ 片段仍保住（该镜不丢）")
        check(any("抽帧/测量失败" in w for w in r.data.warnings),
              "抽帧失败须如实告警（不伪造帧 / 不伪造统计）")
    finally:
        qc.sample_frames, qc.frame_stats = real_sample, real_stats


    # ── 13) 裁到镜头窗口（ADR-080 阶段 5「剪辑引擎」）──
    section("13) 裁到镜头窗口（ADR-080）：超长裁掉 / 不足末帧定格补齐 / 两个 env 开关")

    # 13a) 超长与不足：s1 源 5200ms > 窗口 3000ms ⇒ 裁掉 2200ms；
    #      s2 源 1500ms < 窗口 4000ms ⇒ tpad 克隆末帧补齐 2500ms 并如实报 shortfallMs
    _reset()
    r = invoke("image_to_video", _ctx())
    by_shot = {c.shotId: c for c in r.data.clips}
    print(f"  s1 trim   : {by_shot['s1'].trim}")
    print(f"  s2 trim   : {by_shot['s2'].trim}")
    for cmd in _FFMPEG_CMDS:
        print(f"  ffmpeg    : {' '.join(cmd)}")
    check(by_shot["s1"].trim.sourceMs == 5200 and by_shot["s1"].trim.outMs == 3000
          and by_shot["s1"].trim.shortfallMs == 0, "超长镜：源 5200ms 裁到窗口 3000ms（shortfall 0）")
    check(by_shot["s2"].trim.sourceMs == 1500 and by_shot["s2"].trim.outMs == 4000
          and by_shot["s2"].trim.shortfallMs == 2500,
          "不足镜：源 1500ms 补到窗口 4000ms，并**如实报出** shortfallMs")
    s1_cmd = next(c for c in _FFMPEG_CMDS if "s1.mp4" in " ".join(c))
    s2_cmd = next(c for c in _FFMPEG_CMDS if "s2.mp4" in " ".join(c))
    check("-t" in s1_cmd and "3.000" in s1_cmd and not any("tpad" in a for a in s1_cmd),
          "超长镜的命令行 = 纯 `-t` 裁到窗口（不补帧）")
    check(any(a == "tpad=stop_mode=clone:stop_duration=2.500" for a in s2_cmd) and "4.000" in s2_cmd,
          "不足镜的命令行 = tpad 克隆末帧 2.5s + `-t` 4.000（定格补齐到窗口）")
    check(all(c[-1].endswith(".fit.mp4") for c in _FFMPEG_CMDS), "裁切产物落派生件（原片段不就地改写）")
    check(any("裁到镜头窗口 1 镜" in w for w in r.data.warnings)
          and any("末帧定格补齐 1 镜" in w for w in r.data.warnings),
          "裁切与补齐**如实计入告警**（不静默）")
    check("dsp/scratch/t1/run-demo-8/clips/s1.mp4" in _PUTS
          and "dsp/scratch/t1/run-demo-8/clips/s2.mp4" in _PUTS,
          "裁切后片段仍落**确定性 key** clips/{shotId}.mp4（render / 按镜复用的 key 口径不变）")

    # 13b) 关「裁到镜头窗口」⇒ 一次 ffmpeg 都不跑、trim 全 0（未实测，不猜）、如实告警
    os.environ["FLOWMIND_DSP_VIDEOGEN_TRIM_WINDOW"] = "0"
    _reset()
    r = invoke("image_to_video", _ctx())
    print(f"  关裁切 trim: {[c.trim for c in r.data.clips]}")
    check(not _FFMPEG_CMDS, "关裁切 ⇒ 一次 ffmpeg 都不跑")
    check(all(c.trim.sourceMs == 0 and c.trim.outMs == 0 and c.trim.shortfallMs == 0
              for c in r.data.clips), "关裁切 ⇒ trim 全 0（未实测，不猜一个数出来）")
    check(any("开关关" in w for w in r.data.warnings), "关裁切须如实告警")
    os.environ.pop("FLOWMIND_DSP_VIDEOGEN_TRIM_WINDOW")

    # 13c) 关「末帧定格」⇒ 不补（outMs = 源长），但 shortfallMs **照实报出**
    os.environ["FLOWMIND_DSP_VIDEOGEN_FREEZE_PAD"] = "0"
    _reset()
    r = invoke("image_to_video", _ctx())
    by_shot = {c.shotId: c for c in r.data.clips}
    print(f"  关定格 s2 : {by_shot['s2'].trim}")
    check(by_shot["s2"].trim.sourceMs == 1500 and by_shot["s2"].trim.outMs == 1500
          and by_shot["s2"].trim.shortfallMs == 2500,
          "关定格 ⇒ **不补**（outMs = 源长），但**短多少照实报**（补不补是策略，短不短是事实）")
    check(by_shot["s1"].trim.outMs == 3000 and by_shot["s1"].trim.shortfallMs == 0,
          "关定格不影响**超长**镜的裁切（两个开关彼此独立）")
    check(any("未做定格补齐" in w for w in r.data.warnings), "未补须**如实列出**（不静默）")
    os.environ.pop("FLOWMIND_DSP_VIDEOGEN_FREEZE_PAD")

    # 13d) 裁切失败（ffmpeg 缺失）⇒ **回落原片段** + 如实告警（画质/节奏风险 ≠ 丢镜）
    _reset()

    def _no_ffmpeg(cmd, **kw):
        raise FileNotFoundError("ffmpeg")

    media.run_ffmpeg = _no_ffmpeg
    try:
        r = invoke("image_to_video", _ctx())
    finally:
        media.run_ffmpeg = _REAL_FFMPEG
    print(f"  裁切失败 warnings: {[w for w in r.data.warnings if '裁切' in w]}")
    check(len(r.data.clips) == 2, "裁切失败**不丢镜**（原片段照常入库）")
    check(all(c.trim.sourceMs == 0 and c.trim.outMs == 0 for c in r.data.clips),
          "裁切失败 ⇒ trim 按 0 如实上报（不假装裁过）")
    check(any("裁到镜头窗口失败" in w for w in r.data.warnings), "裁切失败须如实告警")
    check("裁切失败 2 镜" in r.reasoning[0].conclusion, "结论里如实带出裁切失败镜数")


    print("\n✅ dsp_image_to_video_demo 全部通过")


if __name__ == "__main__":
    main()
