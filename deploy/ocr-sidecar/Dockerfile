# =============================================================================
# flowmind OCR 侧车（RapidOCR / PP-OCRv6）—— cross-dashboard dsp「生图质检 · 文字核查」的引擎
#
# 决策记录：cross-dashboard `ADR.md#ADR-100`（本镜像 / 三层开发环境）、`#ADR-097`（这条通道本身）。
#
# **为什么要有这个镜像**：侧车原本要求开发机上装 Python + `rapidocr` + `onnxruntime`，而「装没装对」
# 只在第一次真跑质检时才暴露——且表现为「这一项没跑成」（很容易被当成没事）。打进镜像后，引擎、
# 模型、依赖是**一份可复现的字节**：起容器 = 起引擎，与开发机上有没有 Python 无关。
#
# ⚠️ **版本必须钉死**（不是洁癖）：`engine_info()` 把档位当**证据**上报留档（`PP-OCRv6/small`），
#   换版本 = 换口径 ⇒ 历史上的 `referenceText` / `score` 与新批次不可比。故 `rapidocr` 精确钉
#   3.10.0（与 2026-10-16 那次实测同版本），`onnxruntime` 同步钉住。
# ⚠️ **模型文件随 wheel 进镜像**：rapidocr 3.10.0 的 wheel 里带 PP-OCRv6 det/rec `_small` 与
#   cls mobile（见其 `dist-info/RECORD`）⇒ **构建之后不再需要任何网络**。下面那次构建期自检就是
#   拿这一点当断言：模型缺失/装错平台会在**构建期**当场炸，而不是留给第一个请求去撞。
#
# 构建与启动（在 rak-flowmind 根目录）：
#   docker compose -f deploy/ocr-sidecar/docker-compose.yml up -d
# 在 cross-dashboard 侧不必记这条（那边 `bun run ocr:serve` 就是它，见 ADR-100）。
# =============================================================================

# 基础镜像可覆盖（国内拉 docker.io 慢时传镜像地址，同 cross-dashboard 主 Dockerfile 的 BUN_IMAGE 做法）：
#   docker compose build --build-arg PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.13-slim-bookworm
ARG PYTHON_IMAGE=python:3.13-slim-bookworm
FROM ${PYTHON_IMAGE}

# pip 源同样可覆盖（直连 PyPI 慢/断时：OCR_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple）。
# 用 pip 原生认的 PIP_INDEX_URL 环境变量而不是 --index-url 开关：一个开关只管一条命令，环境变量管全部。
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# 依赖：`rapidocr` **自己不依赖** onnxruntime（它在 `inference_engine/` 里懒加载），所以 onnxruntime
# 必须显式装——少了它引擎初始化会失败（侧车以退出码 1 明确报错，不静默降级，这是刻意的）。
# `pydantic` 是 `import flowmind`（包 `__init__` → `skill.py`）拉起来的：侧车用不到它，但缺了就起不来。
RUN pip install --no-cache-dir \
        rapidocr==3.10.0 \
        onnxruntime==1.31.0 \
        "pydantic>=2,<3"

# 只拷侧车真正要用的那个包（`docs/`、`examples/`、`skills/` 那 15 个技能模块的依赖一概不进镜像）
COPY src/flowmind /app/src/flowmind

# ── 构建期自检（**是可用性断言，不是装饰**）──────────────────────────────────
# ① 引擎真能起（`rapidocr` 可导入 + ONNX 模型在镜像里 + onnxruntime 与本平台对得上）并打出档位；
# ② 侧车模块真能导入。两条任一不过 ⇒ 构建失败 = 这份镜像根本没产出，而不是产出一份起不来的镜像。
RUN python -c "import json; from flowmind import ocr_lines as o; assert o.available(), 'rapidocr 不可导入'; print('引擎档位：' + json.dumps(o.engine_info(), ensure_ascii=False))" \
    && python -c "import flowmind.ocr_sidecar as s; print('侧车模块 ok，单次请求上限 %d MB' % (s.MAX_BODY_BYTES // 1024 // 1024))"

# 非 root 跑：侧车**只读不写**（模型在 site-packages、请求体全在内存里）⇒ 根文件系统可只读挂载
#（compose 里就是 `read_only: true`）。uid 固定 10001，与主镜像的 node 用户同理，避免特权。
RUN useradd --create-home --uid 10001 ocr
USER ocr

# 容器内监听 0.0.0.0（不是 127.0.0.1）：监听回环的话宿主根本连不进来。
# 边界由**端口映射**给（compose 里只发布到宿主的 127.0.0.1）——不是由容器内的 bind 地址给。
EXPOSE 8000

# 探活：`/health` 由侧车自己实现（回引擎档位），此处只是每次都真去打它。
# 用 python 而不是 curl：slim 镜像没带 curl，为一个探针装一整个包不值。
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"]

CMD ["python", "-m", "flowmind.ocr_sidecar", "--host", "0.0.0.0", "--port", "8000"]
