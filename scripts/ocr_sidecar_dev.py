"""本仓 OCR 侧车的起 / 停 / 看（**容器那条路**；cross-dashboard `ADR.md#ADR-100` 的第三层）。

**它是谁**：``src/flowmind/ocr_sidecar.py``（RapidOCR 侧车，服务 cross-dashboard 的 dsp「生图质检 ·
文字核查」）跑在容器里的那条路。**它不是本服务的一部分**：``flowmind-mcp`` 不 import 它、不依赖它，
容器没起时主应用照常启动，消费方如实记 ``not_run``（不假报通过，也不挡启动）。

**为什么要这个脚本**：容器那层落在 ``deploy/ocr-sidecar/``，起它本来要手敲
``docker compose -f deploy/ocr-sidecar/docker-compose.yml up -d``，而**「在不在跑」没有任何入口可问**
（对面仓有 ``bun run ocr:serve`` / ``ocr:status`` 包着同一套 compose，本仓此前只能靠 ``docker ps`` 眼查）。
本脚本把起 / 停 / 看收进一条命令，并**先探活**：端点有人在答就如实说「已在跑」，
**绝不重复起第二个**（重复起会因宿主端口被占而失败，而那个错误里看不出「其实已经好了」）。

```bash
python scripts/ocr_sidecar_dev.py            # 起（默认；镜像不在则先构建，首次要几分钟）
python scripts/ocr_sidecar_dev.py --status   # 只看状态：在不在跑、引擎档位是什么、容器什么状态
python scripts/ocr_sidecar_dev.py --down     # 停（并删除容器；镜像保留）
```

⚠️ **不带 ``PYTHONPATH=src``**（与本仓其它 ``scripts/`` 的约定不同，故单独说明）：本脚本只做**进程编排**、
**不 import ``flowmind``**（纯 stdlib：``docker`` 子进程 + ``urllib`` 探活）——它得能在「本机 Python 环境
根本没装依赖」的机器上照样跑，那也是它要守的那条边界（第三层不连坐主应用的运行时）。

**端口只有一处真源** ``OCR_HOST_PORT``（默认 8765，与 compose 里 ``${OCR_HOST_PORT:-8765}`` 同一个变量）。
⚠️ ``docker compose`` 只读**项目目录**（= compose 文件所在的 ``deploy/ocr-sidecar/``）的 ``.env``，
**不读仓库根的 ``.env``**（根那个是 ``server_http.py`` 加载的、也是人自然会改的那个）⇒ 本脚本按下述顺序
解析后**显式传进 compose 子进程**，免得「两处各写一个数」拼出一个永远连不上的错配：

    进程 env  >  deploy/ocr-sidecar/.env  >  8765

⚠️ **真正连它的地址不在本仓**：cross-dashboard ``.env`` 的 ``AI_OCR_API_URL``（默认 ``http://127.0.0.1:8765``）。
两个数对不上 ⇒ 那边连不上 ⇒ 那一项如实记「没跑成」——**不会有人替你猜**，故本脚本每次都把解析出的端点
打出来，方便两边对账。

**退出码**：达成请求（``up`` 起成了 / ``--status`` 端点真在答 / ``--down`` 之后确实不在跑）= 0，否则 1。
起不来一律**如实失败**（缺 docker / 缺 compose 文件 / 端口被占各有各的话），**绝不假装成功**。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

#: 仓根（本文件在 scripts/ 下）。
_REPO = Path(__file__).resolve().parents[1]
#: compose 与 Dockerfile 的落点（本仓唯一的 Docker 落点；ADR-100）。
COMPOSE_REL = "deploy/ocr-sidecar/docker-compose.yml"
#: 容器名（与 compose 的 container_name 同值）。
CONTAINER = "flowmind-ocr-sidecar"
#: 默认宿主端口（与 compose 的缺省同值）。
DEFAULT_PORT = "8765"
#: 等引擎就绪的上限（首次起要加载 ONNX 模型；实测数秒，给足余量）。
READY_TIMEOUT_S = 90
#: 单次探活的等待上限（本机回环：连不上是毫秒级 ECONNREFUSED，不是超时）。
PROBE_TIMEOUT_S = 3


def _port_from_dotenv(path: Path) -> str | None:
    """从 compose 项目目录的 .env 取 OCR_HOST_PORT（**就是 compose 自己会读的那一份**）。"""
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "OCR_HOST_PORT" and value.strip():
                return value.strip()
    except OSError:
        pass
    return None


def resolve_port() -> tuple[str, str]:
    """解析宿主端口与它的来处（**与 compose 同一处真源**，故把来处一并打印出来备对账）。"""
    env_port = os.environ.get("OCR_HOST_PORT", "").strip()
    if env_port:
        return env_port, "环境变量 OCR_HOST_PORT"
    dotenv_port = _port_from_dotenv(_REPO / "deploy" / "ocr-sidecar" / ".env")
    if dotenv_port:
        return dotenv_port, COMPOSE_REL + " 同目录的 .env"
    # 非 f-string：里面那个 ${OCR_HOST_PORT:-8765} 是给 compose 看的字面量，不是本脚本的格式占位。
    return DEFAULT_PORT, "默认值（compose 里 ${OCR_HOST_PORT:-8765} 的缺省）"


def run(cmd: list[str], env_extra: dict[str, str] | None = None) -> tuple[bool, str]:
    """跑一条命令取输出（探测用；**失败不抛**——「没这容器」与「docker 报错」都不是异常）。"""
    try:
        proc = subprocess.run(  # noqa: S603 （命令字面量，无 shell）
            cmd,
            cwd=_REPO,
            env={**os.environ, **(env_extra or {})},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return False, f"找不到命令 {cmd[0]}（没装，或不在 PATH 上）"
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def compose(*args: str, port: str) -> tuple[bool, str]:
    """跑一条 docker compose 子命令；**宿主端口显式传进去**（见模块头：项目目录的 .env 才是它读的）。"""
    return run(["docker", "compose", "-f", COMPOSE_REL, *args], env_extra={"OCR_HOST_PORT": port})


def docker_available() -> tuple[bool, str]:
    ok, out = run(["docker", "compose", "version"])
    return ok, (out.splitlines()[0] if ok and out else out)


def container_state() -> str:
    """容器现在什么状态（running / exited / created…）；**不存在**回空串（≠「已停」）。"""
    ok, out = run(["docker", "inspect", "-f", "{{.State.Status}}", CONTAINER])
    return out if ok else ""


def probe(port: str) -> tuple[dict | None, str]:
    """打 /health：是侧车在答 ⇒ (info, "")；否则 (None, 原因)。

    原因**原样带出**（不吞成一句「连不上」）——「没人答」「有东西答但不是侧车」「HTTP 4xx/5xx」
    要做的事完全不同。判据是回包真带 ok/engine（端口上可能蹲着别的服务）。
    """
    try:
        with urllib.request.urlopen(  # noqa: S310 （本机回环）
            f"http://127.0.0.1:{port}/health", timeout=PROBE_TIMEOUT_S
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}"
    except Exception as exc:  # 连不上 / 超时 / 回的不是 JSON
        return None, str(exc)
    if isinstance(body, dict) and body.get("ok") is True and "engine" in body:
        return body, ""
    return None, "端口上有东西答了，但回包不是 OCR 侧车的 /health（可能是别的服务占了这个端口）"


def describe(info: dict) -> str:
    return (f"引擎 {info.get('engine', '（未给）')} · 档位 {info.get('model', '（未给）')}"
            f" · 检测过滤 text_score={info.get('text_score', '（未给）')}")


def tail(text: str, lines: int = 25) -> str:
    return "\n".join(text.splitlines()[-lines:])


def wait_ready(port: str) -> tuple[dict | None, str]:
    """轮询 /health 直到答上来或超时——**`up -d` 返回 ≠ 就绪**（HEALTHCHECK 有 start-period，模型还要加载）。"""
    deadline = time.monotonic() + READY_TIMEOUT_S
    why = ""
    while time.monotonic() < deadline:
        info, why = probe(port)
        if info is not None:
            return info, ""
        time.sleep(0.7)
    return None, why


def cmd_up(port: str, source: str) -> int:
    print(f"· 宿主端口：{port}（来自 {source}）")
    print(f"· 端点：http://127.0.0.1:{port}（对面仓要拿它填 AI_OCR_API_URL）")
    info, why = probe(port)
    if info is not None:
        print(f"✓ 侧车已在跑（没重复起第二个）：{describe(info)}")
        return 0
    print(f"· 端点上没有侧车（{why}）")
    if not (_REPO / COMPOSE_REL).exists():
        print(f"✗ 找不到 {COMPOSE_REL}（compose 与 Dockerfile 是本仓唯一的 Docker 落点）")
        return 1
    ok, ver = docker_available()
    if not ok:
        print(f"✗ 用不了 Docker（`docker compose version` 没跑通）：{ver}")
        print("  ① 装并启动 Docker Desktop（或别的 compose 实现）后重试；")
        print("  ② 或改用原生那条路（需要本机装了 rapidocr）：")
        print("     PYTHONPATH=src python -m flowmind.ocr_sidecar --port 8123")
        return 1

    print(f"· 起容器（{COMPOSE_REL}；镜像不在则先构建，首次要装 onnxruntime/opencv，"
          "几分钟、需网络）…")
    ok, out = compose("up", "-d", port=port)
    if not ok:
        print(f"✗ 起容器失败：\n{tail(out)}")
        print("  常见原因：① Docker 没启动；② 拉基础镜像 / pip 源不通"
              "（可传 PYTHON_IMAGE=<镜像地址> 或 OCR_PIP_INDEX_URL=<镜像源>）；"
              "③ 宿主端口被占（改 OCR_HOST_PORT，并同步对面仓的 AI_OCR_API_URL）——"
              "本机原生侧车占着 8765 时也会撞。")
        return 1

    print(f"· 容器已起，等引擎就绪（上限 {READY_TIMEOUT_S}s；加载 ONNX 模型要几秒）…")
    info, why = wait_ready(port)
    if info is None:
        state = container_state()
        print(f"✗ 容器起了但 http://127.0.0.1:{port}/health 一直没答上来"
              f"（容器状态：{state or '查不到'}；最后原因：{why}）")
        print(f"  看日志：docker logs {CONTAINER}")
        return 1
    print(f"✓ 侧车就绪（{CONTAINER}）：{describe(info)}")
    print(f"  停：python scripts/ocr_sidecar_dev.py --down      日志：docker logs -f {CONTAINER}")
    print("  容器在跑 ≠ 主应用依赖它：dsp 文字核查从此真跑；容器一停，那一项如实记没跑（不挡应用）。")
    return 0


def cmd_status(port: str, source: str) -> int:
    print(f"· 宿主端口：{port}（来自 {source}）")
    print(f"· 端点：http://127.0.0.1:{port}")
    compose_state = "" if (_REPO / COMPOSE_REL).exists() else "（**不存在**）"
    print(f"· compose：{COMPOSE_REL}{compose_state}")
    ok, ver = docker_available()
    print(f"· Docker：{ver if ok else f'**不可用**（{ver}）'}")
    # 「不存在」≠「没起过」：`down` 会把容器删掉，两种历史在这里长得一样 ⇒ 只如实说**现在**是什么。
    state = container_state() if ok else ""
    print(f"  容器 {CONTAINER}：{state or '**不存在**（没起，或已被 down 删掉）'}")

    info, why = probe(port)
    if info is not None:
        print(f"✓ 侧车在答：{describe(info)}")
        print("  （compose 里是 restart: unless-stopped ⇒ Docker 重启后它会自己回来，"
              "「在跑」不一定是这一会话起的）")
        return 0
    print(f"· 端点没有侧车在答（{why}）")
    if ok and state == "running":
        print(f"  容器是 running 但 /health 不应答 ⇒ 看日志：docker logs {CONTAINER}")
    elif ok and state:
        print(f"  容器状态是 {state}（不是 running）⇒ 起它：python scripts/ocr_sidecar_dev.py")
    else:
        print("  起它：python scripts/ocr_sidecar_dev.py")
    print("  （侧车没起不影响主应用：dsp 文字核查如实记「没跑成」，不假报通过、也不挡启动）")
    return 1


def cmd_down(port: str) -> int:
    ok, ver = docker_available()
    if not ok:
        print(f"✗ 用不了 Docker（`docker compose version` 没跑通）：{ver} ⇒ 没法停容器。")
        return 1
    if not container_state():
        # 同样不许猜历史：「不存在」就只说现在不存在。
        print(f"· 容器 {CONTAINER} 本就不存在（没起过，或已被 down 删掉）⇒ 无需停，不报「已停」。")
        return 0
    ok, out = compose("down", port=port)
    if not ok:
        print(f"✗ 停容器失败：\n{tail(out)}")
        return 1
    print(f"✓ 容器已停并删除（{CONTAINER}）——镜像还在，下次起不必重建。")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认码页未必编得出 ✓ / · （本脚本就是要能直接 `python scripts/...` 跑起来）
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="OCR 侧车（容器）的起 / 停 / 看 —— 本仓 dev 的第三层，不挡主应用启动",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true",
                       help="只看状态：在不在跑 / 引擎档位 / 容器现在什么状态")
    group.add_argument("--down", action="store_true",
                       help="停容器（并删除容器；镜像保留）")
    args = parser.parse_args(argv)

    port, source = resolve_port()
    if args.status:
        return cmd_status(port, source)
    if args.down:
        return cmd_down(port)
    return cmd_up(port, source)


if __name__ == "__main__":
    raise SystemExit(main())
