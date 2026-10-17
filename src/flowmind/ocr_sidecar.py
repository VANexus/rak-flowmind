"""本机 OCR 侧车（RapidOCR / PP-OCRv6）：给 cross-dashboard 的 dsp「生图质检 · 文字核查」提供
**逐行文字 + 位置框 + 置信度**。

**为什么是一个本机小服务**（2026-10-16 用户拍板，决策记录在 cross-dashboard `ADR.md#ADR-097`）：

* 这条通道要传**图片字节**，而 MCP 的参数是 JSON（``{inp: args}``）——cross-dashboard 全仓没有
  传字节的先例，dsp 一律只传对象存储 key 由后端自己读；字节走 MCP 会另开一条不合口径的路。
* cross-dashboard 侧「直连 HTTP 传字节」是**既定范式**（``ai.cvseg`` 的字节直传、``ai.imageEdit``
  的 multipart），本机侧车只是把端点换成本机地址 ⇒ 形状与既有能力一致，且**字节直传**才守得住
  ADR-094 的纪律（质检比对的必须是**送进模型的那一份字节**）。
* **未起 ⇒ 那一项如实记「没跑成」**（cross-dashboard 侧 ``not_run``），不冒充查过。
  将来要进集群，把 ``flowmind.ocr_lines`` 包成 ``@skill`` 走 MCP 即可（登记为下一步）。

**它不是技能**：不进 ``skills/__init__.py`` 的 FAMILIES、不碰 ``server*.py`` / ``contracts.py`` /
``manifest.py`` / ``skill.py``（照 `AGENTS.md` 的技能铁律）——它只把 ``flowmind.ocr_lines`` 那一个
函数搬上本机 HTTP。也**因此不 import ``flowmind.skills``**（那个包导入即执行族注册，会把 15 个技能
模块的依赖全拉起来）：一个 CPU 侧车不该为此付启动代价，也不该被任一技能的缺失依赖连坐。

**启动**（本机 dev，手动起）::

    # ① 容器（**默认这条路**，cross-dashboard ADR-100）：引擎/模型/依赖钉在镜像里，与开发机的 Python 无关
    docker compose -f deploy/ocr-sidecar/docker-compose.yml up -d      # 只发布到宿主 127.0.0.1:8765
    docker compose -f deploy/ocr-sidecar/docker-compose.yml down       # 停
    #   （本仓包成了 python scripts/ocr_sidecar_dev.py（--status / --down，先探活、不重复起）；cross-dashboard 侧是 `bun run ocr:serve`）
    # ② 原生（本机装好依赖时；也是与容器对账的基准）
    PYTHONPATH=src python -m flowmind.ocr_sidecar --port 8123
    # 装好包装后与 mcp-base-gpu 同形：rapidocr-sidecar --port 8123
    #   ⚠️ 这条路吃**本机 env** 的 rapidocr + onnxruntime（后者必须显式装：rapidocr 不拉它，
    #      缺了是**引擎初始化才炸**——`available()` 只探 import，探不到这一层）。真源见 environment.yml。

**两条路产出同一份结果**（ADR-100 实测对账：同一批图片字节分别打两边，文字 / 四点框 / 置信度三样全同）
——容器只是把「装没装对 Python 依赖」从每台开发机各答一遍的题，变成一份可复现的字节。⚠️ 镜像里钉死版本
**不是为了稳**：`engine_info()` 把档位当**证据**上报（`PP-OCRv6/small`），换版本 = 换口径 ⇒
历史留档与新批次不可比（钉的是 `rapidocr==3.10.0` + `onnxruntime==1.31.0`；模型随 wheel 进镜像）。

**接口**（默认只监听 127.0.0.1，无凭据；请求体**是原始图片字节**，不是 base64/multipart）::

  ⚠️ **容器里要监听 0.0.0.0**（`--host 0.0.0.0`）：监听回环的话宿主根本连不进来。**边界由端口映射给**
  （compose 只发布到宿主的 `127.0.0.1:8765`），**不是**由容器里的 bind 地址给——两件事别混。

    GET  /health → {"ok":true,"engine":...,"model":...,"text_score":0.5,...}
    POST /ocr    ← body = 图片字节（Content-Type: image/*）
                 → {"ok":true,"engine":...,"model":...,"elapsed_ms":1470,
                    "lines":[{"text":"容量 500ml","quad":[[x,y]×4],"score":0.951}, ...]}
                 → 4xx/5xx + {"ok":false,"error":"..."}（**绝不吞异常**，也绝不用空结果冒充读过了）

**为什么原始字节而不是 base64 JSON**：生成图 2.5-6MB，base64 会膨胀到 ~8MB，再各经一次 JSON
序列化/反序列化（TS 侧一份、Python 侧一份）⇒ 峰值内存无谓 ×3；而 stdlib 里解 multipart 自找麻烦
（Python 3.13 已删 ``cgi``）。原始字节一条 ``rfile.read(n)`` 搞定，且与 TS 侧那个
``readImageText({bytes, contentType})`` 的签名**一一对应**。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from flowmind import ocr_lines as ocr

#: 单次请求体的上限（本机服务；生成的 PNG 到 6MB 级，留足余量）。
MAX_BODY_BYTES = 24 * 1024 * 1024
#: 请求体的 MIME 白名单（只收图片；侧车不做格式转换）。
IMAGE_MIME_PREFIX = "image/"


class _OcrHandler(BaseHTTPRequestHandler):
    server_version = "flowmind-ocr/1.0"
    protocol_version = "HTTP/1.1"  # 响应必须显式给 Content-Length（下面都给了）

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802  （BaseHTTPRequestHandler 的接口名）
        if self.path.split("?")[0] != "/health":
            self._send(404, {"ok": False, "error": f"未知路径 {self.path}（只有 GET /health 与 POST /ocr）"})
            return
        info = ocr.engine_info()
        self._send(200, {
            "ok": True,
            "engine": info["engine"],
            "model": ocr.model_label(info),
            "text_score": ocr.TEXT_SCORE,
            "ocr_version": info["ocr_version"],
            "det_model": info["det_model"],
            "rec_model": info["rec_model"],
        })

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] != "/ocr":
            self._send(404, {"ok": False, "error": f"未知路径 {self.path}（只有 GET /health 与 POST /ocr）"})
            return
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if not ctype.startswith(IMAGE_MIME_PREFIX):
            self._send(415, {"ok": False, "error": f"Content-Type 必须是图片（收到 {ctype or '（无）'}）——"
                                                   "请求体是原始图片字节，不是 JSON"})
            return
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            # 不收 chunked（本仓两端都是自己人，让它显式给长度：省掉一整条解码路径）
            self._send(411, {"ok": False, "error": "缺少 Content-Length（本服务不收分块传输）"})
            return
        try:
            length = int(raw_length)
        except ValueError:
            self._send(400, {"ok": False, "error": f"Content-Length 不是整数：{raw_length}"})
            return
        if length <= 0:
            self._send(400, {"ok": False, "error": "请求体为空（0 字节）"})
            return
        if length > MAX_BODY_BYTES:
            self._send(413, {"ok": False, "error": f"请求体 {length} 字节超过上限 {MAX_BODY_BYTES}"})
            return
        image = self.rfile.read(length)
        if len(image) != length:
            self._send(400, {"ok": False, "error": f"请求体不完整（声明 {length} 字节，实收 {len(image)}）"})
            return

        started = time.time()
        try:
            lines = ocr.read_lines(image)
        except Exception as exc:
            # 不吞异常：识别失败就是失败（调用方据此如实记「没跑成」），绝不用空结果冒充读过了
            self._send(502, {"ok": False, "error": f"OCR 识别失败：{exc}"})
            print(f"[ocr] 失败 {length // 1024}KB：{exc}", file=sys.stderr, flush=True)
            return
        info = ocr.engine_info()
        elapsed_ms = round((time.time() - started) * 1000)
        self._send(200, {
            "ok": True,
            "engine": info["engine"],
            "model": ocr.model_label(info),
            "elapsed_ms": elapsed_ms,
            "lines": [
                {
                    "text": line.text,
                    # 引擎原生的四点框（不在这里做「外接矩形」这类派生：判定侧有离线门禁可钉住）
                    "quad": [list(point) for point in line.quad] if line.quad is not None else None,
                    "score": round(line.score, 4),
                }
                for line in lines
            ],
        })
        print(f"[ocr] {len(lines)} 行 · {elapsed_ms}ms · {length // 1024}KB", flush=True)

    def log_message(self, fmt: str, *args) -> None:
        """默认实现会把每个请求打一行 access log；这里静音（结论由 do_POST 自己打一行）。"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本机 OCR 侧车（RapidOCR / PP-OCRv6）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认只监听本机）")
    parser.add_argument("--port", type=int, default=8123, help="监听端口（默认 8123）")
    args = parser.parse_args(argv)

    if not ocr.available():
        print("✗ 未安装 rapidocr（新包）：pip install rapidocr", file=sys.stderr)
        return 1
    try:
        info = ocr.engine_info()  # 启动即加载模型：装不起来当场说清，不留给第一个请求去撞
    except Exception as exc:
        print(f"✗ OCR 引擎初始化失败：{exc}", file=sys.stderr)
        return 1
    print(f"✓ 引擎就绪：{json.dumps(info, ensure_ascii=False)}"
          f"（model={ocr.model_label(info)}，text_score={ocr.TEXT_SCORE}）", flush=True)
    print(f"  监听 http://{args.host}:{args.port}"
          "（cross-dashboard 侧把 AI_OCR_API_URL 指过来）", flush=True)
    if args.host not in ("127.0.0.1", "localhost"):
        print(f"  ⚠️ 监听 {args.host} 会暴露到本机之外（本服务无凭据）", file=sys.stderr, flush=True)
    ThreadingHTTPServer((args.host, args.port), _OcrHandler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
