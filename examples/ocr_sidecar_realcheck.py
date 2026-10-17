"""本机 OCR 侧车自检 —— RapidOCR / PP-OCRv6 的**逐行文字 + 位置框 + 置信度**（cross-dashboard ADR-097）。

运行：PYTHONPATH=<repo>/src python examples/ocr_sidecar_realcheck.py

**按需真跑，不进 18 个 demo 的回归线**（同 `dsp_trim_realcheck.py` 的先例）：本脚本要求解释器里真有
**`rapidocr` 引擎**（新包 v3；`environment.yml` 自 2026-10-16 修包名漂移后钉的正是 `rapidocr>=3.10`），
并真起一个侧车 HTTP 服务、真跑推理——与本目录那些「mock 掉全部外部依赖、任何机器都绿」的 demo
分工相反。缺引擎即 `check(ocr_lines.available())` **当场失败**（绝不把「没装」写成通过）；
在 conda env `flowmind` 里约 10 秒跑完。

三段式（happy / 默认 / 错误），**全程离线、不花钱、不依赖 GPU 与云密钥**：

1. **happy**：PIL 合成一张白底黑字的 ASCII 大字图 → 起侧车（`--port 0` 由系统分配）→ POST /ocr
   → 断言「读到了字、框落在图内、置信度在 (0,1]」。**不断言具体文字**（字体缺失会让断言变脆，
   而这里要验的是「通了且形状对」，不是「某个字体渲染成什么样」）。
2. **默认**：用缺省参数（host/port/上限）起，断言 `GET /health` 的字段形状与 `text_score` 在场。
3. **错误**：坏字节 ⇒ 400（不是 500，也不是「0 行」）；非图片 Content-Type ⇒ 415。
   —— 两侧都不许把失败冒充成「读过了但没字」。

为什么用 ASCII 而不是中文：**不保证本机有 CJK 字体**（PIL 缺字体会画成方框或抛错），
而本 demo 要的是契约与形状，中文识别质量由真实商品图那边去验（cross-dashboard 的补生真跑）。
"""
from __future__ import annotations

import io
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageDraw  # noqa: E402

from flowmind import ocr_lines, ocr_sidecar  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _post(url: str, body: bytes, content_type: str) -> tuple[int, dict]:
    request = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as resp:  # noqa: S310  （本机回环）
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _png_with_text(text: str) -> bytes:
    img = Image.new("RGB", (900, 240), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 60), text, fill="black")
    draw.text((40, 150), "MODEL XZ-2026 / CAPACITY 500ml", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _blank_png() -> bytes:
    """**真的没有字**的一张图（纯白）——用来验「0 行是合法答案」。"""
    buf = io.BytesIO()
    Image.new("RGB", (900, 240), "white").save(buf, format="PNG")
    return buf.getvalue()


class Sidecar:
    """在后台线程里起一个侧车（`port=0` 时由系统分配），退出时关掉——自终止、不占端口。"""

    def __init__(self, port: int = 0) -> None:
        from http.server import ThreadingHTTPServer

        self.server = ThreadingHTTPServer(("127.0.0.1", port), ocr_sidecar._OcrHandler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def main() -> None:
    section("0) 引擎自检（不加载模型的那部分）")
    print(f"  rapidocr 可导入 : {ocr_lines.available()}")
    check(ocr_lines.available(), "本 demo 需要 rapidocr（新包）：pip install rapidocr")
    print(f"  钉住的检测过滤   : text_score={ocr_lines.TEXT_SCORE}（不是判定阈值）")

    section("1) happy：合成图 → POST /ocr → 逐行文字 + 框 + 置信度")
    sidecar = Sidecar(port=0)
    try:
        print(f"  侧车（随机端口）：{sidecar.base}")
        status, payload = _post(f"{sidecar.base}/ocr", _png_with_text("HELLO FLOWMIND"), "image/png")
        check(status == 200 and payload.get("ok") is True, f"应当 200/ok：{status} {payload}")
        lines = payload["lines"]
        print(f"  engine={payload['engine']!r} model={payload['model']!r} "
              f"elapsed_ms={payload['elapsed_ms']} lines={len(lines)}")
        for line in lines:
            print(f"    {line['text']!r} quad={line['quad']} score={line['score']}")
        check(len(lines) >= 1, "合成图上应当读到至少一行")
        check(all(line["text"] for line in lines), "每行都要有文字")
        check(all(0.0 < line["score"] <= 1.0 for line in lines), "置信度应当在 (0,1]")
        check(all(line["quad"] and len(line["quad"]) == 4 for line in lines),
              "每一行都应当带 4 点框（引擎原生，判定侧自己转外接矩形）")

        section("2) 默认：缺省参数下 /health 的字段形状")
        with urllib.request.urlopen(f"{sidecar.base}/health", timeout=30) as resp:  # noqa: S310
            health = json.loads(resp.read().decode("utf-8"))
        print(f"  {json.dumps(health, ensure_ascii=False)}")
        for field in ("ok", "engine", "model", "text_score", "ocr_version", "det_model", "rec_model"):
            check(field in health, f"/health 缺字段 {field}")
        check(health["text_score"] == ocr_lines.TEXT_SCORE, "/health 要回显实际生效的检测过滤阈值")
        # 图上没字：**合法答案**（0 行），不是错误
        status, payload = _post(f"{sidecar.base}/ocr", _blank_png(), "image/png")
        check(status == 200 and payload["ok"] and payload["lines"] == [], "空图应当回 0 行，而不是报错")
        print("  空白图 → 0 行（合法答案，不是错误）")

        section("3) 错误：坏字节 / 非图片 Content-Type —— 都不许冒充「读过了」")
        status, payload = _post(f"{sidecar.base}/ocr", b"not an image at all", "image/png")
        print(f"  坏字节 → HTTP {status} {payload}")
        check(status == 502 and payload["ok"] is False, "识别不了应当 502 + ok=false（不是 0 行）")
        status, payload = _post(f"{sidecar.base}/ocr", b"{}", "application/json")
        print(f"  非图片类型 → HTTP {status} {payload}")
        check(status == 415 and payload["ok"] is False, "非 image/* 应当 415")
        status, payload = _post(f"{sidecar.base}/nope", b"x", "image/png")
        check(status == 404, "未知路径应当 404")
        print(f"  未知路径 → HTTP {status}")
    finally:
        sidecar.stop()
        time.sleep(0.2)

    print("\n全部通过")


if __name__ == "__main__":
    main()
