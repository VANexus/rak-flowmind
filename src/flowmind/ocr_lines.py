"""逐行 OCR 读取：RapidOCR（本地 ONNX / PP-OCRv6）→ ``[{text, box, score}, ...]``。

**用途**：cross-dashboard 的 dsp「生图质检 · 文字核查」（决策记录在那边 `ADR.md#ADR-097`）——
读**参考图**与**生成图**的文字，比对型号 / 规格 / 包装字样有没有被改。这条检查存在的全部理由是
**独立性**：它必须是**与生图模型相独立的一台引擎**（用户口径：不依赖生图模型是否正确理解了商品
文字），而这里是**纯本地 ONNX 推理**，既不经过生图模型，也不经过任何多模态大模型。

**为什么放在顶层模块而不是 ``skills/_*.py``（与那 18 个 helper 不同）**：``flowmind.skills``
包的 ``__init__`` 会在导入时执行**族注册**（把选中族的全部技能模块 import 进来）。本模块的消费者是
**本机 CPU 侧车**（``flowmind.ocr_sidecar``）——它不该为一个 OCR 函数去拉起 15 个技能模块的依赖，
更不该被任一技能的缺失依赖连坐（同样的理由，本模块**不复用** ``skills._cloud_ocr.OCRError``，
自带的 ``OcrLinesError`` 只沿用同一套类别词表）。将来要把 OCR 接进 MCP 面，写
``skills/dsp_*.py`` 时从这个顶层模块 import 即可。

**与 ``skills/_local_ocr.py`` 的分工（刻意不合并）**：那一条是**字幕条区域框**（视频本地化，多帧
聚合 + 尺寸/位置先验），**新旧两个包名都认**（``_engine_flavor()`` 双探、**新包 v3 优先**；
``environment.yml`` 自 2026-10-16 起钉的也是新包）；本模块用**新包 ``rapidocr`` v3**
（``from rapidocr import RapidOCR``），要的是
**逐行文字 + 四点框 + 置信度**，且**不做任何位置/置信度过滤**。两者的依赖与输出契约都不同，
合并只会让两边互相牵制。

**实测（2026-10-16，本机，PP-OCRv6 det/rec ``_small``）**：

* 有文字的图：12 行、置信度 0.92-1.0、单张 1.5-6.8 秒（首调含模型加载）；
* **图上没字时如实返回空列表**（不编、不退化——这正是换掉云端「聊天通道」那台引擎的原因：
  它在无字图上会重复枚举出上万字）；
* ``engine(bytes)`` **直接吃字节**（其 ``load_img`` 收 ``bytes``）⇒ 不用落临时文件。

**不做置信度判决**：``TEXT_SCORE`` 是**引擎内置的检测过滤**（显式钉住 0.5，防「库改了默认值」
造成口径漂移），**不是**「超过多少就算对」的判据——置信度不是错误概率，判定侧不使用它，
只作为校准素材原样留档（判定规则在 cross-dashboard 的纯函数 `dsp-ocr.ts`，那边可离线对账）。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from flowmind.errors import FailureCategory

#: 引擎内置的检测过滤阈值（**不是判定阈值**；显式传值 = 防库改默认造成口径漂移）。
TEXT_SCORE = 0.5

#: 引擎标识（进证据留档；实际档位从加载后的配置里读，见 ``engine_info``）。
ENGINE = "rapidocr/onnxruntime"
#: 缺哪一边都装不起来时给用户的那条命令。
INSTALL_HINT = "pip install rapidocr"

_engine: Any | None = None
_engine_lock = threading.Lock()
#: 推理串行化：引擎实例不保证线程安全，而 HTTP 侧车是多线程的（一次至多两张图，串行代价可忽略）。
_infer_lock = threading.Lock()


class OcrLinesError(Exception):
    """OCR 读取失败。``category`` 词表与 ``flowmind.errors.FailureCategory`` 一致
    （``environment`` 装不起来 / ``video`` 输入不是一张能读的图 / ``unknown`` 推理异常）。"""

    def __init__(self, message: str, category: str = FailureCategory.UNKNOWN.value) -> None:
        super().__init__(message)
        self.category = category


def available() -> bool:
    """``rapidocr``（新包）是否可导入——供自检判断，**不做任何初始化**（不加载模型）。"""
    try:
        import rapidocr  # noqa: F401  # 导入探测本身即目的，勿删
        return True
    except Exception:
        return False


def _get_engine() -> Any:
    """进程内单例 + 双检锁懒加载（并发首调不重复初始化，与 ``skills/_local_ocr.py`` 同口径）。"""
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:
                raise OcrLinesError(
                    f"未安装 rapidocr（新包；{INSTALL_HINT}；旧包 rapidocr_onnxruntime 不适用）",
                    category=FailureCategory.ENVIRONMENT.value,
                ) from exc
            try:
                _engine = RapidOCR()
            except Exception as exc:
                raise OcrLinesError(
                    f"本地 OCR 引擎初始化失败: {type(exc).__name__}: {exc}",
                    category=FailureCategory.ENVIRONMENT.value,
                ) from exc
    return _engine


def engine_info() -> dict[str, str]:
    """加载后的**实际**引擎信息（档位从配置里读，不写死我们「以为」的档位）。"""
    info = {"engine": ENGINE, "ocr_version": "", "det_model": "", "rec_model": ""}
    cfg = getattr(_get_engine(), "cfg", None)
    try:
        info["ocr_version"] = _enum_value(cfg.Rec.ocr_version)
        info["det_model"] = _enum_value(cfg.Det.model_type)
        info["rec_model"] = _enum_value(cfg.Rec.model_type)
    except Exception:
        pass  # 读不到就留空（= 未判），**不填一个好看的默认值**
    return info


def _enum_value(value: Any) -> str:
    """配置里的档位取**枚举的值**（``OCRVersion.PPOCRV6`` → ``PP-OCRv6``）。

    直接用 ``str()`` 会把枚举类名一起带进证据（``OCRVersion.PPOCRV6``）——那是实现细节，
    不是我们要留档的型号名。
    """
    return str(getattr(value, "value", value))


def model_label(info: dict[str, str] | None = None) -> str:
    """引擎 + 档位的一行标识（进证据留档），如 ``PP-OCRv6/small``；读不到档位就只说版本/引擎。"""
    data = info or engine_info()
    version, rec = data.get("ocr_version", ""), data.get("rec_model", "")
    if version and rec:
        return f"{version}/{rec}"
    return version or data.get("engine", ENGINE)


@dataclass(frozen=True)
class OcrLine:
    """一行识别结果：文字 + **引擎原生的四点框**（像素，未裁剪、未排序）+ 引擎自评置信度。

    ⚠️ 这里**只搬事实**：不做「四点 → 轴对齐矩形」这类派生（旋转文本外接框会略偏大，
    取不取整、要不要外接框是**判定侧的口径**，那边有离线门禁可钉住）。引擎没给框 ⇒ ``quad=None``
    （**不编一个框**，调用侧如实标位置不可用）。
    """

    text: str
    quad: tuple[tuple[float, float], ...] | None
    score: float


def read_lines(image: bytes | str, *, text_score: float = TEXT_SCORE) -> list[OcrLine]:
    """读一张图里的所有文字行（``bytes`` 或图片路径）。

    **图上没有文字 ⇒ 空列表**（合法答案，不是错误）。识别不了这张图 / 推理异常 ⇒ ``OcrLinesError``
    （调用方据此如实报失败，**绝不**退化成「没读到字 = 没问题」）。
    """
    if isinstance(image, bytearray):
        image = bytes(image)
    if isinstance(image, bytes) and len(image) == 0:
        raise OcrLinesError("图像字节为空（未识别，不当作「没有文字」）")
    engine = _get_engine()
    try:
        with _infer_lock:
            result = engine(image, text_score=text_score)
    except OcrLinesError:
        raise
    except Exception as exc:
        # 解不开的图与推理异常在这里都会冒出来：归一类「输入/资源」问题（词表见 errors.py）
        category = FailureCategory.VIDEO.value if "image" in str(exc).lower() else FailureCategory.UNKNOWN.value
        raise OcrLinesError(f"本地 OCR 推理异常: {type(exc).__name__}: {exc}", category=category) from exc

    texts = _seq(getattr(result, "txts", None))
    scores = _seq(getattr(result, "scores", None))
    quads = _seq(getattr(result, "boxes", None))
    lines: list[OcrLine] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            continue
        score = float(scores[index]) if index < len(scores) and scores[index] is not None else 0.0
        # 没有框的那一行**照样留下**（文字是有的）——框留 None，由调用侧如实标位置不可用
        lines.append(OcrLine(
            text=text.strip(),
            quad=_quad(quads[index]) if index < len(quads) else None,
            score=score,
        ))
    return lines


def _seq(value: Any) -> list[Any]:
    """引擎的字段取成 list。

    ⚠️ **不能写 `value or []`**：`txts`/`scores`/`boxes` 可能是 **numpy 数组**，而 numpy 数组的
    真值判断会抛 ``The truth value of an array with more than one element is ambiguous``
    （本 demo 第一版就是这么炸的——合成图一跑就现形）。
    """
    return [] if value is None else list(value)


def _quad(value: Any) -> tuple[tuple[float, float], ...] | None:
    """引擎给的点集 → 四点元组（**原样、不排序、不取整**）；不成形 ⇒ ``None``（不猜一个框出来）。"""
    try:
        points = [(float(point[0]), float(point[1])) for point in value]
    except (TypeError, IndexError, ValueError):
        return None
    return tuple(points) if len(points) == 4 else None
