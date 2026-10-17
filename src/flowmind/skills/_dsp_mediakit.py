"""火山 AI MediaKit「多轨道剪辑」客户端（契约 §7）。

调用面（契约 §7.1，API 文档 2026-10-12 对齐）：
- ``POST {base}/api/v1/tools/multi-track-edit``（提交，**同步返** ``{success, task_id, …}``）
- ``POST {base}/api/v1/tools/apply-camera-motion``（运镜，§7.5 第 2 步；**只吃视频**）
- ``GET  {base}/api/v1/tasks/{task_id}``（轮询至 ``completed``）
- 鉴权 ``Authorization: Bearer <API Key>``（集群 Secret，**密钥只在后端**）
- 形态 **异步任务**：``result{video_url, duration(秒), resolution}``；``video_url`` 是
  **24h 签名 URL**（§7.3②）——**不得**直接当 ``storedKey``，必须转存（见 render 技能）

参数模型（§7.1 / §7.2）：``canvas`` → ``track[][]``（**索引越大越上层**）→
``element{type, target_time, source|text, extra[]}`` → ``output{format, fps, cover}``。
主轨元素按镜头取：有 ``clip_urls``（图生视频片段）即 ``video``、否则回落 ``image``。

字段已按**入仓真文档**逐条核实（``docs/vendor/`` 四份：多轨道剪辑 37 页 /
提交多轨道剪辑任务 API 27 页 / 视频添加运镜 9 页 / 公共错误码 5 页）。关键核实点：
``extra_param.upload.aigc_meta`` 是**JSON 字符串**（非对象）；``output.cover.cover_time``
是**整数数组**；text/subtitle 元素**必须**给 ``extra.transform``；subtitle 轨独占且 ≤10 条；
``output.codec.*`` 是嵌套对象（本实现用默认值，不显式下发）；运镜 ``motion_style`` 四值、
``start_time``/``end_time`` 单位**秒**、输入**只接受视频**且 ≤1080p / ≤5 分钟。
错误码按**入仓真文档** ``docs/vendor/AIMediaKit_公共错误码_1775248816.pdf`` 分类
（火山公共错误码表，code 为**字符串名**，已按中文释义逐条核对；
**对照表见 ``docs/vendor/README.md``**）。

本模块只做「EDL → MediaKit 请求体」的映射与 HTTP 往返；**入参 / 出参形状与契约 §3
一字不改**（MediaKit 是 ``render_compose`` 的后端实现细节）。
"""
from __future__ import annotations

import json
import time

import requests

from flowmind.config import DspConfig
from flowmind.skills import _dsp_common as common
from flowmind.skills._secrets import get_api_key

_TERMINAL_OK = frozenset({"completed", "success", "done", "finished", "succeeded"})
_TERMINAL_FAIL = frozenset({"failed", "error", "canceled", "cancelled", "timeout"})

# 错误码分类（真源 docs/vendor/AIMediaKit_公共错误码_1775248816.pdf，已按中文释义逐条核对）。
# 该文档是火山「公共错误码」表，**code 是字符串名**（非数字）、HTTP 状态随错误语义走；
# 分类按 name 判定（比按 HTTP 状态稳——同一个 name 可能跨多状态返回）。
# **可重试** = 服务端临时故障 / 限流，稍后重试可恢复。
_MK_TRANSIENT_CODES = frozenset({
    "UndefinedError",  # 未知错误（500）—— 未定性，按可重试处理
    "InternalError", "InternalServiceError", "InternalServiceTimeout", "FailToConnect",
    "ServiceUnavailable", "ServiceUnavailableTemp", "Unavailable", "ServiceTimeout",
    "ServerIsBusy", "Busy", "Timeout",
    "FlowLimitExceeded", "RequestLimitExceeded", "LimitExceeded",
    "ResourceUnavailable",  # 资源不可用（含欠费等临时状态）
    "ResourceInUse",  # 资源被占用（409，临时冲突）
})
# **需查环境** = 鉴权 / 签名 / 策略 / 配额问题，别重试（改配置或提工单）。
_MK_ENVIRONMENT_CODES = frozenset({
    "MissingAuthenticationToken", "MissingSignature", "MissingRequestInfo",
    "InvalidClientTokenId", "SignatureDoesNotMatch", "InvalidTimestamp",
    "LackPolicy", "LackProjectPolicy", "AccessDenied", "Forbidden", "RequestForbidden",
    "OperationDenied", "Unauthorized", "UnauthorizedOperation",
    "QuotaInsufficient", "Insufficient", "ResourceInsufficient",
})


class DspMediakitError(Exception):
    """MediaKit 调用失败。category/retriable 语义与 errors.py 对齐。"""

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False):
        super().__init__(message)
        self.category = category
        self.retriable = retriable


def _mk_error(raw: object) -> tuple[str, str]:
    """从响应体里取 ``(code, message)``；兼容 ``error{}`` 与顶层 ``code``/``message``。"""
    if not isinstance(raw, dict):
        return "", ""
    err = raw.get("error")
    if isinstance(err, dict) and err:
        return str(err.get("code") or ""), str(err.get("message") or "")
    code = raw.get("code")
    if isinstance(code, str) and code and code.lower() != "success":
        return code, str(raw.get("message") or "")
    return "", ""


def _classify_mk_error(code: str) -> tuple[str, bool]:
    """按 MediaKit 错误码名分类为 ``(category, retriable)``。

    未登记的码一律 ``video``（参数 / 资源问题——改输入，别重试）。
    """
    if code in _MK_TRANSIENT_CODES:
        return "transient", True
    if code in _MK_ENVIRONMENT_CODES:
        return "environment", False
    return "video", False



def mediakit_configured(cfg: DspConfig) -> bool:
    """是否已配置 MediaKit 密钥。"""
    return bool(get_api_key(cfg.mediakit_key_env))


# ── 本地上传（mediakit://）——契约 §7.3① 的另一支 ────────────────────
# 为什么要有它：``source`` 接受四协议（http(s) / mediakit:// / vod:// / tos://），而**预签名 URL**
# 那条只有在对象存储**公网可达**时才成立——本机 dev 的 ``127.0.0.1:9000`` 厂商根本拉不到
# （2026-10-17 真跑：MediaKit 在 render 里如实失败于 ``DownloadFileError``）。
# 本地上传支：① 申请一个上传位（预签名 PUT）② 把**原始字节**直传 ③ 任务里用 ``mediakit://{file_id}``
# 引用 ⇒ 素材**不经过任何对外地址**，与对象存储的公网可达性彻底解耦。
# 厂商文档：``POST {base}/api/v1/tools-sync/request-media-upload-url``（Bearer、body ``{}``）
# → ``result{upload_url, method, upload_headers[{key,value}], file_id}``；``upload_url`` 24h 有效。

# 上传位响应里**必须有**的字段（缺一个就无法完成上传）——缺失即如实报，不猜。
_UPLOAD_SLOT_FIELDS = ("upload_url", "file_id")


def request_upload_slot(cfg: DspConfig) -> dict[str, object]:
    """申请一个媒体上传位，返回 ``{upload_url, method, headers, file_id}``。"""
    api_key = get_api_key(cfg.mediakit_key_env)
    if not api_key:
        raise DspMediakitError(
            f"未配置 MediaKit 密钥 {cfg.mediakit_key_env}（上传渲染素材必需）。",
            category="environment",
        )
    raw = _post(cfg, api_key, cfg.mediakit_request_upload_path, {})
    if not raw.get("success", True):
        code, message = _mk_error(raw)
        category, retriable = _classify_mk_error(code)
        raise DspMediakitError(
            f"MediaKit 申请上传位失败：{code} {message}".strip(),
            category=category, retriable=retriable,
        )
    result = raw.get("result")
    if not isinstance(result, dict):
        result = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    missing = [f for f in _UPLOAD_SLOT_FIELDS if not str(result.get(f) or "")]
    if missing:
        raise DspMediakitError(
            f"MediaKit 上传位响应缺字段（{'、'.join(missing)}）：{sorted(result)[:12]}",
            category="video",
        )
    headers: dict[str, str] = {}
    raw_headers = result.get("upload_headers")
    if isinstance(raw_headers, list):
        for item in raw_headers:
            if isinstance(item, dict) and item.get("key"):
                headers[str(item["key"])] = str(item.get("value") or "")
    return {
        "upload_url": str(result["upload_url"]),
        # 文档给 ``PUT``；不硬编码成唯一可能（若厂商改成 POST，照它说的做）
        "method": str(result.get("method") or "PUT").upper(),
        "headers": headers,
        "file_id": str(result["file_id"]),
    }


def upload_media(cfg: DspConfig, data: bytes, content_type: str) -> str:
    """把**原始字节**直传 MediaKit，返回任务里可用的 ``mediakit://{file_id}``。

    约束（厂商文档）：``PUT`` 原始二进制（**不是** multipart）、把所有 ``upload_headers`` 带上、
    ``Content-Type`` 必须与真实文件类型一致（缺省按扩展名推的 ``content_type`` 补）。
    """
    if not data:
        raise DspMediakitError("素材为空，拒绝上传（不拿空文件换一个 file_id）。", category="video")
    slot = request_upload_slot(cfg)
    headers = dict(slot["headers"])  # type: ignore[arg-type]
    if not any(k.lower() == "content-type" for k in headers):
        headers["Content-Type"] = content_type
    method = str(slot["method"])
    if method not in ("PUT", "POST"):
        raise DspMediakitError(
            f"MediaKit 上传位给了不认识的方法「{method}」（文档口径是 PUT）——拒绝盲发。",
            category="video",
        )
    try:
        resp = requests.request(
            method, str(slot["upload_url"]), data=data, headers=headers,
            timeout=cfg.mediakit_upload_timeout_s,
        )
    except requests.exceptions.Timeout as exc:
        raise DspMediakitError("MediaKit 素材上传超时", category="environment") from exc
    except requests.exceptions.RequestException as exc:
        raise DspMediakitError(f"MediaKit 素材上传连接失败：{exc}", category="environment") from exc
    if not resp.ok:
        raise DspMediakitError(
            f"MediaKit 素材上传失败（HTTP {resp.status_code}）：{resp.text[:200]}",
            category="transient" if resp.status_code >= 500 else "video",
            retriable=resp.status_code >= 500,
        )
    # 文档响应示例里 ``result.file_id`` 本身**已经**是 ``mediakit://…``（不是裸 id）——
    # 故两种形态都要接住：已带前缀就用它，裸 id 才补前缀。盲目再拼一次会得到 ``mediakit://mediakit://…``。
    file_id = str(slot["file_id"])
    return file_id if file_id.startswith("mediakit://") else f"mediakit://{file_id}"


# ── 运镜：整块退役（ADR-076）────────────────────────────────────
# 旧路径 = 对**已生成片段**调 MediaKit `apply-camera-motion` 做事后 pan-zoom（契约 §7.5 第 2 步）。
# 2026-10-14 改判：运镜改由「运镜导演」产出结构化方案 → `image_to_video` 经引擎适配器**并入 prompt、
# 随生成一起落地**（新单点适配表 = `_dsp_video_gen.MOTION_ENGINE`）。故本文件的 `MOTION_STYLES` /
# `EFFECT_MOTION_STYLE` / `motion_style_for` / `apply_camera_motion` **全部删除**——留着就会
# 「同一运动施加两次」（ADR-073 当初禁止把运镜写进 prompt，正是为了防这个）。
# 将来若要用引擎**参数级**运镜，落点同样是 `_dsp_video_gen.MOTION_ENGINE`。


# ── 转场（extra 滤镜 · ADR-070 / ADR-080）─────────────────────────
# **厂商口径（用户 2026-10-14 提供的 TransitionFilter 参数说明）**：``TransitionFilter`` 加在**资源的
# ``Extra`` 字段**里用来加转场；**同轨道相邻 A 视频与 B 视频**之间要转场，**只需加在 A（前一个）**上。
# 参数三个都是必选：``Type = "transition"`` / ``Source = 转场 ID`` / ``Duration = 转场时长(ms)``。
#   → 本仓映射（``transition_filters``）与之逐字对齐：``extra: [{type, source, duration}]`` 挂在
#     **该段自己的元素**上（= 段 i 末尾、桥接 i→i+1，正是「加在 A 上」）。
#   ⚠️ **文档写的是「两 Video」**，而本仓主轨在**未出片**时回落 ``image``（静图）元素——静图邻接是否
#     同样生效**未经实测**（登记 B1b ③）。故 ``transition_warnings`` 在转场**真被施加**时，若相邻
#     元素含静图会如实告警（**不假装**它和视频一样有效）。
# **只允许「非交叠」**：音轨是一整条实测音频、时间线严格连续，交叠会移动画面内容或改总时长
# ⇒ 音画失同步（硬约束，见 `_dsp_common.DspEdlSegment.transition`）。
#
# ⚠️ **本表当前只登记了一项**（`dissolve`）——厂商转场目录里**可用于非交叠**的那批里，
# 与我们符号词表能**对得上**的只有「泛开」这一项（见 `TRANSITION_CATALOG`）。我们的另两个符号
# （`wipe` / `fade-black`）在厂商目录里**没有可用于非交叠的对应项** ⇒ **查不到即不可用**，如实回落硬切。
# **绝不编造 ID、也绝不把「否」的项当「是」用**（数据诚实）。
# 拿到 ID 表后只填这一张表（与 cross-dashboard 的 `DSP_TRANSITION_VENDOR` 同源、同值）。
TRANSITION_VENDOR: dict[str, dict[str, object]] = {
    # 本仓符号 `dissolve`（柔切）→ 厂商「泛开」（非交叠可用）。`durationMs` 是**本仓取值**
    # （厂商只要求给 Duration，没给默认值）——400ms 是「看得出是个转场、又不吃掉太多镜头时长」的取值，
    # 可按真跑观感在这一行改。
    "dissolve": {"cnNorth": "1182358", "johor": "1109560",
                 "mode": "non-overlap", "durationMs": 400},
}

# 厂商转场**目录**（用户 2026-10-14 提供的《转场 ID》表；华北 / 柔佛两套号 + 是否可用于**非交叠**）。
#
# 登记它只为一件事：**让「未登记」与「不适用」不再混为一谈**——转场回落硬切时，能如实说清到底是
# 「我们没登记」还是「厂商目录里根本没有可用的那一项」。
# ⚠️ **本表按用户给的文档片段照实录入**：ID 序列在 `1182371 → 1182373` 处有跳号（`1182372` 不在片段里）
# ⇒ **可能不是全集**（片段可能被截断）。故它只作**参照**，不作「厂商一共就这些」的断言。
# 我们只可能用到 `nonOverlap=True` 的那批（硬约束：音轨一整条、时间线严格连续，交叠会破坏音画对齐）。
TRANSITION_CATALOG: dict[str, dict[str, object]] = {
    "叶片翻转": {"cnNorth": "1182355", "johor": "1109548", "nonOverlap": False},
    "百叶窗": {"cnNorth": "1182356", "johor": "1109550", "nonOverlap": False},
    "风吹": {"cnNorth": "1182357", "johor": "1109552", "nonOverlap": False},
    "交替出场": {"cnNorth": "1182359", "johor": "1109556", "nonOverlap": True},
    "旋转放大": {"cnNorth": "1182360", "johor": "1109558", "nonOverlap": True},
    "泛开": {"cnNorth": "1182358", "johor": "1109560", "nonOverlap": True},
    "风车": {"cnNorth": "1182362", "johor": "1109562", "nonOverlap": False},
    "多色混合": {"cnNorth": "1182363", "johor": "1109564", "nonOverlap": False},
    "遮罩转场": {"cnNorth": "1182364", "johor": "1109566", "nonOverlap": False},
    "六角形": {"cnNorth": "1182365", "johor": "1109568", "nonOverlap": True},
    "心型打开": {"cnNorth": "1182366", "johor": "1109570", "nonOverlap": False},
    "故障转换": {"cnNorth": "1182367", "johor": "1109572", "nonOverlap": True},
    "飞眼": {"cnNorth": "1182368", "johor": "1109574", "nonOverlap": True},
    "梦幻放大": {"cnNorth": "1182369", "johor": "1109576", "nonOverlap": True},
    "开门展现": {"cnNorth": "1182370", "johor": "1109578", "nonOverlap": False},
    "对角擦除": {"cnNorth": "1182371", "johor": "1109580", "nonOverlap": False},
    "立方转换": {"cnNorth": "1182373", "johor": "1109584", "nonOverlap": True},
    "透镜变换": {"cnNorth": "1182374", "johor": "1109586", "nonOverlap": True},
    "晚霞转场": {"cnNorth": "1182375", "johor": "1109588", "nonOverlap": True},
    "圆形打开": {"cnNorth": "1182376", "johor": "1109590", "nonOverlap": False},
    "圆形擦开": {"cnNorth": "1182377", "johor": "1109592", "nonOverlap": False},
    "圆形交替": {"cnNorth": "1182378", "johor": "1109594", "nonOverlap": True},
    "时钟扫开": {"cnNorth": "1182379", "johor": "1109596", "nonOverlap": False},
}


def resolve_transition(name: str | None, region: str = "cn-north") -> dict[str, object] | None:
    """EDL ``transition``（符号名）→ 该区域的 MediaKit 资源 ID；**未登记即返回 None**。

    ``region`` = ``cn-north``（华北，端点默认所在）| ``johor``（柔佛）——厂商给的是两套号，见 ``TRANSITION_CATALOG``。
    """
    key = (name or "").strip()
    if not key:
        return None
    entry = TRANSITION_VENDOR.get(key)
    if not entry:
        return None
    field = "johor" if region.strip().lower() in ("johor", "ap-southeast") else "cnNorth"
    resource_id = str(entry.get(field) or "").strip()
    if not resource_id:
        return None
    return {**entry, "id": resource_id}


def transition_filters(seg: common.DspEdlSegment, region: str = "cn-north") -> list[dict]:
    """该段末尾的转场 → ``extra`` 滤镜列表（**未登记 = 空列表 = 硬切**）。"""
    entry = resolve_transition(seg.transition, region)
    if not entry:
        return []
    return [{
        "type": "transition",
        "source": entry["id"],
        "duration": int(entry.get("durationMs") or 400),  # type: ignore[arg-type]
    }]


def transition_warnings(
    edl: common.DspEdl, clip_shot_ids: set[str] | None = None, region: str = "cn-north",
) -> list[str]:
    """未登记 / 非「非交叠」/ **相邻含静图** 的转场 → 如实的降级说明（``render_compose`` 并入 ``warnings``）。

    三种回落各有各的说法（**不混为一谈**）：
      ① **我们没登记**（符号不在 ``TRANSITION_VENDOR`` 里）；
      ② **登记了但该区域没号** / **只支持交叠**；
      ③ **已施加**但相邻元素含**静图**——厂商文档的适用范围是「同轨道**相邻两 Video**」，
         静图邻接是否生效**未经实测**（登记 B1b ③）。
    ``clip_shot_ids`` = 真出了视频片段的镜 id（主轨会用 ``video`` 元素的那批），给定它才做 ③ 的判定。
    """
    warnings: list[str] = []
    for track in edl.tracks:
        if track.kind != "image":
            continue
        for idx, seg in enumerate(track.segments):
            name = (seg.transition or "").strip()
            if not name:
                continue
            if name not in TRANSITION_VENDOR:
                # 区分「我们没登记」与「厂商目录里就没有可用于非交叠的那一项」——后者不是我们漏登记
                overlap_only = sum(1 for row in TRANSITION_CATALOG.values() if not row.get("nonOverlap"))
                warnings.append(
                    f"转场 {name} 未登记厂商资源 ID，已回落**硬切**（不编造 ID；登记后自动生效）。"
                    f"（本仓只允许**非交叠**转场，而厂商目录 {len(TRANSITION_CATALOG)} 项里有 "
                    f"{overlap_only} 项标「不可用于非交叠」——擦除/黑场一类过渡正在其中；"
                    f"当前已登记可用项：{', '.join(sorted(TRANSITION_VENDOR)) or '无'}）"
                )
                continue
            entry = resolve_transition(name, region)
            if entry is None:
                warnings.append(
                    f"转场 {name} 未登记区域 {region!r} 的资源 ID，已回落**硬切**"
                    "（厂商按区域给两套号，见 `TRANSITION_CATALOG`）。"
                )
                continue
            if str(entry.get("mode") or "") != "non-overlap":
                warnings.append(
                    f"转场 {name} 只支持交叠模式，与「一整条配音音频严格对齐」冲突，已回落**硬切**。"
                )
                continue
            if clip_shot_ids is not None and _has_still_neighbor(edl, idx, clip_shot_ids):
                warnings.append(
                    f"转场 {name} 已施加，但厂商文档的适用范围是「**同轨道相邻两 Video**」——"
                    "本处相邻元素含**静图**（未出片的镜回落 image），静图邻接是否生效**未经实测**"
                    "（登记 B1b ③）；若不生效则该处等同硬切。"
                )
    return warnings



def _has_still_neighbor(edl: common.DspEdl, idx: int, clip_shot_ids: set[str]) -> bool:
    """该转场涉及的**两个相邻元素**（本段与下一段）里是否有静图（即非 video 元素）。"""
    shots = edl.shots
    for pos in (idx, idx + 1):
        shot = shots[pos] if pos < len(shots) else None
        if shot is None or shot.id not in clip_shot_ids:
            return True
    return False



# ── EDL → track[][]（契约 §7.2 逐字段映射）────────────────────────

def build_tracks(
    edl: common.DspEdl,
    *,
    image_urls: dict[str, str],
    clip_urls: dict[str, str] | None = None,
    audio_url: str = "",
    bgm_url: str = "",
    srt_url: str = "",
    subtitle_font_size: int = 48,
    canvas_width: int = 1080,
    canvas_height: int = 1920,
    transition_region: str = "cn-north",
) -> list[list[dict]]:
    """把本仓 EDL 翻成 MediaKit ``track[][]``（索引越大越上层）。

    - 主轨 = ``image`` / ``video``：``clip_urls``（**按镜头 id**）命中即用 ``video`` 元素
      （图生视频产出的片段，已含运镜），否则回落 ``image``（静图）；两者**无 trim**、
      ``target_time`` 为绝对区间 ms（§7.2）
    - ``vo`` / ``bgm`` → ``audio`` 元素（走混音，非覆盖）
    - ``subtitle`` → ``subtitle`` 元素，**独占一轨**（文档：subtitle 轨 ≤ 10 条），
      且 text/subtitle 元素**必须**在 ``extra`` 里给 ``transform``（文档要求四字段齐全）

    封面走 ``output.cover`` 抽帧（见 ``build_body``），**不**做成叠层图片；
    小黄车贴片是**平台组件**（商品链接不烧进画面，§7.2），同样不入轨。

    **转场**（ADR-070）：读该段末尾的 ``transition``（符号名），经 ``TRANSITION_VENDOR`` 解析成
    ``extra`` 滤镜**加在「前一个」元素上**（厂商口径）；**未登记即不加**（= 硬切，由
    ``transition_warnings`` 如实说明）。

    ``edl.shots`` 与 image 轨 ``segments`` **索引对齐**（timeline_build 保证），
    故用 shots[i].id 取该段的片段；无 shots 时全部回落静图。
    """
    tracks: list[list[dict]] = []
    clips = clip_urls or {}
    shots = edl.shots

    media_elements: list[dict] = []
    for track in edl.tracks:
        if track.kind != "image":
            continue
        for idx, seg in enumerate(track.segments):
            span = [seg.atMs, seg.atMs + seg.durationMs]
            shot_id = shots[idx].id if idx < len(shots) else ""
            clip = clips.get(shot_id, "")
            if clip:
                elem: dict = {"type": "video", "source": clip, "target_time": span}
            else:
                src = image_urls.get(seg.sourceKey, "")
                if not src:
                    continue  # 留空位的镜头：无图可取，如实跳过（timeline 已在 notes 说明）
                elem = {"type": "image", "source": src, "target_time": span}
            extra = transition_filters(seg, transition_region)
            if extra:
                elem["extra"] = extra
            media_elements.append(elem)
    if media_elements:
        tracks.append(media_elements)

    if audio_url:
        tracks.append([{
            "type": "audio",
            "source": audio_url,
            "target_time": [0, edl.totalMs],
        }])
    if bgm_url:
        tracks.append([{
            "type": "audio",
            "source": bgm_url,
            "target_time": [0, edl.totalMs],
            # 配音起时压低 BGM（§7.2：音量 / 闪避写在 extra）
            "extra": [{"type": "a_volume", "volume": 0.4}],
        }])
    if srt_url:
        tracks.append([{
            "type": "subtitle",
            "text": srt_url,
            "target_time": [0, edl.totalMs],
            "font_size": subtitle_font_size,
            "align_type": 1,  # 横排居中（文档 align_type：横排 0 左 / 1 中 / 2 右）
            "extra": [{
                "type": "transform",
                "pos_x": 0, "pos_y": 0,
                "width": canvas_width, "height": canvas_height,
            }],
        }])

    return tracks


def build_body(
    cfg: DspConfig,
    *,
    tracks: list[list[dict]],
    client_token: str,
    cover_at_ms: int | None = None,
) -> dict:
    """组装提交体（canvas / track / output / 幂等 token / AIGC 元数据）。

    ``cover_at_ms`` 给定即用 MediaKit ``output.cover`` **抽帧**导出封面（§7.1）。
    """
    token = (client_token or "dsp")[:64]  # 契约：client_token ≤ 64 ASCII
    output: dict = {
        "format": cfg.mediakit_output_format,
        "fps": cfg.mediakit_output_fps,
    }
    if cover_at_ms is not None:
        output["cover"] = {
            "disable_cover": False,
            "cover_time": [max(0, int(cover_at_ms))],  # 文档：cover_time 为**整数数组**（ms）
            "format": "jpeg",
        }
    body: dict = {
        "canvas": {
            "width": cfg.mediakit_canvas_width,
            "height": cfg.mediakit_canvas_height,
            "background_color": "#000000FF",
        },
        "track": tracks,
        "output": output,
        "client_token": token,
    }
    if cfg.mediakit_queue_id:
        body["queue_id"] = cfg.mediakit_queue_id
    if cfg.mediakit_output_destination:
        # §7.3② 更优解：结果直写自己的 TOS，从源头免去 24h 转存
        body["media_output_destination"] = cfg.mediakit_output_destination
    if cfg.mediakit_aigc_label:
        # §7.3④ AIGC 标识合规（成片含 AI 生成内容：AI 补生图 / TTS 口播）
        # 文档核实：extra_param.upload.aigc_meta 是**JSON 字符串**（不是对象）——
        # 内层键 Label / ContentProducer / ProduceID / …
        body["extra_param"] = {"upload": {"aigc_meta": json.dumps(
            {"Label": cfg.mediakit_aigc_label, "ProduceID": token},
            ensure_ascii=False,
        )}}
    return body


def _post(cfg: DspConfig, api_key: str, path: str, body: dict) -> dict:
    url = cfg.mediakit_api_base.rstrip("/") + path
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = requests.post(url, json=body, headers=headers, timeout=cfg.mediakit_timeout_s)
    except requests.exceptions.Timeout as exc:
        raise DspMediakitError(f"MediaKit 提交超时（{url}）", category="environment") from exc
    except requests.exceptions.RequestException as exc:
        raise DspMediakitError(f"MediaKit 连接失败：{exc}", category="environment") from exc

    # 先解析体里的错误码名（文档：code 是字符串名，跨多 HTTP 状态——按名分类比按状态稳）；
    # 拿不到码名才回落 HTTP 状态。
    try:
        raw = resp.json()
    except ValueError:
        raw = None

    code, message = _mk_error(raw)
    if not resp.ok and code:
        category, retriable = _classify_mk_error(code)
        raise DspMediakitError(
            f"MediaKit 错误 {code}（HTTP {resp.status_code}）：{message or '(无 message)'}",
            category=category, retriable=retriable,
        )
    if resp.status_code == 429:
        raise DspMediakitError("MediaKit 限流（429）", category="transient", retriable=True)
    if resp.status_code >= 500:
        raise DspMediakitError(f"MediaKit 服务端错误（{resp.status_code}）", category="transient",
                               retriable=True)
    if resp.status_code >= 400:
        raise DspMediakitError(f"MediaKit 拒绝请求（{resp.status_code}）：{resp.text[:300]}",
                               category="video")
    if not isinstance(raw, dict):
        raise DspMediakitError("MediaKit 返回非 JSON", category="video")
    return raw


def submit(cfg: DspConfig, body: dict) -> str:
    """提交多轨道剪辑任务，返回 ``task_id``。"""
    api_key = get_api_key(cfg.mediakit_key_env)
    if not api_key:
        raise DspMediakitError(
            f"未配置 MediaKit 密钥 {cfg.mediakit_key_env}（dsp 渲染必需）。", category="environment")
    raw = _post(cfg, api_key, cfg.mediakit_multi_track_path, body)
    if not raw.get("success", True):
        code, message = _mk_error(raw)
        category, retriable = _classify_mk_error(code)
        raise DspMediakitError(
            f"MediaKit 提交失败：{code} {message}".strip(),
            category=category, retriable=retriable,
        )
    task_id = raw.get("task_id") or raw.get("data", {}).get("task_id")
    if not task_id:
        raise DspMediakitError(f"MediaKit 响应缺少 task_id：{sorted(raw)[:12]}", category="video")
    return str(task_id)


def _query(cfg: DspConfig, api_key: str, task_id: str) -> dict:
    url = cfg.mediakit_api_base.rstrip("/") + cfg.mediakit_task_path.format(task_id=task_id)
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(url, headers=headers, timeout=cfg.mediakit_timeout_s)
    except requests.exceptions.RequestException as exc:
        raise DspMediakitError(f"MediaKit 查询失败：{exc}", category="transient",
                               retriable=True) from exc
    try:
        raw = resp.json()
    except ValueError:
        raw = None

    code, message = _mk_error(raw)
    if not resp.ok and code:
        category, retriable = _classify_mk_error(code)
        raise DspMediakitError(
            f"MediaKit 查询错误 {code}（HTTP {resp.status_code}）：{message or '(无 message)'}",
            category=category, retriable=retriable,
        )
    if resp.status_code >= 500:
        raise DspMediakitError(f"MediaKit 查询服务端错误（{resp.status_code}）",
                               category="transient", retriable=True)
    if resp.status_code >= 400:
        raise DspMediakitError(f"MediaKit 查询被拒（{resp.status_code}）：{resp.text[:200]}",
                               category="video")
    if not isinstance(raw, dict):
        raise DspMediakitError("MediaKit 查询返回非 JSON", category="video")
    return raw


def render(cfg: DspConfig, body: dict) -> dict:
    """提交多轨道剪辑并轮询至终态，返回 ``result``（``{video_url, duration, resolution}``）。

    轮询上限 ``mediakit_poll_max_s``——超时**如实报错**，不假装完成。
    """
    api_key = get_api_key(cfg.mediakit_key_env)
    if not api_key:
        raise DspMediakitError(
            f"未配置 MediaKit 密钥 {cfg.mediakit_key_env}（dsp 渲染必需）。", category="environment")
    task_id = submit(cfg, body)
    return _wait_result(cfg, api_key, task_id, label="渲染")


def _wait_result(cfg: DspConfig, api_key: str, task_id: str, *, label: str) -> dict:
    """轮询至终态并返回 ``result``（``{video_url, duration, resolution}``）。

    ``label`` 只用于错误消息（渲染 / 运镜），行为一致；超时**如实报错**，不假装完成。
    """
    deadline = time.monotonic() + cfg.mediakit_poll_max_s
    last_status = "unknown"
    while True:
        raw = _query(cfg, api_key, task_id)
        status = str(raw.get("status") or raw.get("data", {}).get("status") or "").lower()
        if status:
            last_status = status
        if status in _TERMINAL_OK:
            result = raw.get("result") or raw.get("data", {}).get("result") or {}
            if not isinstance(result, dict) or not result.get("video_url"):
                raise DspMediakitError(
                    f"MediaKit {label}任务完成但缺少 result.video_url：{sorted(raw)[:12]}",
                    category="video")
            return {"task_id": task_id, **result}
        if status in _TERMINAL_FAIL:
            code, message = _mk_error(raw)
            category, retriable = _classify_mk_error(code)
            raise DspMediakitError(
                f"MediaKit {label}失败（status={status}）：{code} {message}".strip(),
                category=category, retriable=retriable,
            )
        if time.monotonic() >= deadline:
            raise DspMediakitError(
                f"MediaKit {label}超时（>{cfg.mediakit_poll_max_s}s，最后状态 {last_status}）",
                category="transient", retriable=True,
            )
        time.sleep(cfg.mediakit_poll_interval_s)


# 运镜（`apply_camera_motion`）已整块退役 —— 见本文件上方「运镜：整块退役」说明（ADR-076）。
