# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

`mcp-base-gpu` 是**视频本地化 MCP over HTTP SaaS 服务端**：把「中文视频 → 多语言
配音/字幕版本」的 GPU 流水线（ASR → OCR → 翻译 → 字幕擦除 → TTS 配音 → 混音）
包装成 **7 个 `localize_*` MCP 工具 + 一条异步任务 REST 通道**，单端口 8001 对外。

**核心不变量**：**新增一个技能 = 写一个 `@skill` 函数**。注册 / JSON schema /
MCP tool / manifest / discover() 全自动暴露。加技能**不改动** `server.py` /
`contracts.py` / `skill.py` / `rules.py` 等框架层 —— 这条约束改前务必确认。

## 常用命令

```bash
conda env update -n flowmind -f environment.yml          # 装/更新依赖（environment.yml 是依赖真源）
conda run -n flowmind pip install --no-deps "simple-lama-inpainting>=0.1.2"  # stale metadata 见 environment.yml 注释
conda run -n flowmind pip install -e . --no-deps          # 包本体 + entry points
conda run -n flowmind ruff check src                      # lint（必须通过，唯一质量门）
for f in examples/*_demo.py; do PYTHONPATH=$PWD/src conda run -n flowmind python "$f"; done  # demo 冒烟（本仓库无单测）
conda run -n flowmind mcp-base-gpu                        # 单端口入口（MCP + 任务 REST，默认 8001）
conda run -n flowmind flowmind-mcp                        # stdio 调试入口
```

**Python 3.12**（conda env `flowmind`）。**依赖真源 = `environment.yml`**；
`pyproject.toml` 只保留包元数据 / entry points / ruff 配置。
**不用 Makefile / Docker / pytest**（tests/ 已删除，验证靠 demo + 真实 invoke）。
worktree 中验证一律 `PYTHONPATH=<worktree>/src conda run -n flowmind ...`。

## 本地模型（GPU 化，2026-09）

硬件：NVIDIA P104-100（8GB，Compute Capability 6.1 / Pascal）。本地优先（GPU 可行域内），
LLM / 云 ASR/OCR/TTS 走云 API（dashscope / Anthropic 兼容协议）：

| 能力 | 本地后端（默认优先） | 云回落 | 开关（flowmind.config.toml [localizer]） |
|---|---|---|---|
| ASR | faster-whisper（CTranslate2，int8） | dashscope qwen-audio | `asr_backend` = local/cloud/auto |
| OCR 字幕定位 | RapidOCR（onnxruntime，CPU） | dashscope qwen3.5-ocr | `ocr_backend` = local/cloud/auto |
| 字幕擦除 | simple-lama（big-lama） | delogo 滤镜 | `erase_backend` = auto/local/delogo |
| 配音 | qwen-tts（Qwen3-TTS 0.6B 零样本克隆） | dashscope 声音复刻 | `tts_backend` = auto/local/cloud |
| BGM 人声分离 | demucs（htdemucs） | — | `bgm_vocal_sep` |
| 字幕向量化 | —（远程 BGE HTTP 服务，见下） | — | `infra.vectorize` / FLOWMIND_VECTORIZE |

关键约束与约定：

- **Pascal 6.1 限制**：torch 锁 `2.5.1+cu121`（cu124+ 构建剔除 Pascal，勿升）；
  `torchaudio==2.5.1+cu121` 必须在 `qwen-tts` 之后重新钉回（qwen-tts 会拉错版）。
- **显存预算 ~7.5G**：whisper ~1.5G + Qwen3-TTS ~4G + LaMa ~1.5G。两道闸串行：
  TaskManager `workers=1`（任务间）+ `tasks/gpu.gpu_lane()` 信号量（sync invoke 与任务线程之间）。
  模型懒加载另有 `model_cache_guard()` 双检锁。
- **auto 语义**：本地库可导入即用本地，否则回落云；两端都不可用显式报错，
  实际 backend 体现在 report 字段与 ReasoningChain 文本中，**不静默降级**。
- **模型缓存与下载**：`HF_HOME=/srv/data/models`、`TORCH_HOME=/srv/data/models/torch`、
  代理 `http://127.0.0.1:7890`（均已写入 `~/.bashrc`）。模型实例进程内缓存
  （`_local_asr._models` / `_local_ocr._ocr_engine` / `_inpaint`）。
- **BGE 嵌入走远程 HTTP 服务**（`skills/_bge_embed.py`）：TEI `/embed` 与
  OpenAI `/v1/embeddings` 双形状自适应；地址 env `FLOWMIND_EMBEDDING_BASE_URL`
  → config `infra.embedding_base_url` → 默认 `http://127.0.0.1:31997`。

## 架构（大图）

数据流（两条通道共用技能层）：

```
MCP 客户端 ──/mcp (Streamable HTTP)──┐
                                     ├─ invoke() → @skill 函数 → SkillResult 信封
HTTP 客户端 ──/api/v1/tasks (REST)───┘        │
   POST 202 ──────────────────────→ TaskManager.submit（PG queued → GPU 串行执行）
   GET 轮询 / download ←─────────── TaskStore（PG）+ MQTT 事件 + Milvus 向量
```

分层（`src/flowmind/`）：

- **`contracts.py`** —— 对外契约层：`SkillResult[T]` 信封 / `ReasoningChain` 四段式链 /
  `ReliabilityMetrics` / `TraceContext` / `SkillOutput[T]`。**改这里 = 改对外 API**。
- **`skill.py`** —— 融合点：`@skill` 装饰器 + `_REGISTRY` + `invoke()`。技能函数只返回
  轻量 `SkillOutput`；`invoke()` 统一套 `SkillResult` 信封（trace / latency / 错误兜底）。
- **`discover.py` + `manifest.py`** —— Agent 自助发现：input + output 完整 JSON Schema
  一次暴露，Agent 不读源码即可调用。
- **`errors.py`** —— `ErrorCode`（NOT_FOUND/VALIDATION/INTERNAL）+ 失败四分类
  （environment/video/transient/unknown）+ `is_retriable()`。
- **`config.py`** —— `FlowmindConfig` = `LocalizerConfig`（业务参数，带通用默认）+
  `InfraConfig`（基础设施）。**配置源优先级全仓库统一：env → config.toml → 内置默认**
  （12-factor：容器部署 env 注入即可覆盖）。
- **`server.py`** —— FastMCP（**v1**，`mcp>=1.27,<2`）遍历注册表动态登记 MCP tool。
  `_make_tool` 靠设置 `__annotations__` 驱动 schema 推断 —— v1 特定技巧，勿升 v2。
- **`server_http.py`** —— **单端口唯一入口**（8001）：组合 MCP（`/mcp`）+ REST 路由 +
  中间件（CORS + `AuthPlaceholderMiddleware` 鉴权占位）。`.env` 加载在此发生。
- **`server_rest.py`** —— 发现 API：`GET /api/v1/manifest[/id]`（custom_route 模式）。
- **`server_tasks.py`** —— 任务 REST：`POST /api/v1/tasks`（202/429 背压）/
  `GET /api/v1/tasks/{id}` / `GET .../download?file=`（basename 白名单防穿越）/
  `GET /api/v1/health`（pg/mqtt/milvus 尽力检查，失败不 500）。
- **`tasks/`** —— 任务引擎：
  - `store.py` —— TaskStore：PG 持久层。PgBouncer 事务模式适配（每次操作短连接、
    autocommit 单语句、不用 advisory lock / 服务端 prepared statement）；幂等建表；
    启动恢复 `recover_running()`。
  - `manager.py` —— TaskManager：单 GPU worker 线程池 + 协作式取消（阶段边界
    CancelledError；cancel 同时 CAS 落终态）+ 终态分类（终态写入带
    queued/running CAS 守卫，first-writer-wins 不互相覆盖）+ worker 最外层
    异常兜底 + TTL GC（删 workdir 与 outputs/ 不删行）+ 孤儿目录清扫。
    惰性单例 `get_task_manager()`。
  - `events.py` —— TaskEventPublisher：MQTT `mcp-base-gpu/tasks/{id}/events`，
    QoS=1，终态 retain；未配置/失败一律静默降级纯落库（通知通道绝不阻断任务）。
  - `vectors.py` —— Milvus `localize_segments`（768 维 HNSW/COSINE，按 task_id 幂等
    upsert）+ `health_status()`。
  - `gpu.py` —— `gpu_lane()` / `model_cache_guard()` 双道闸。
- **`skills/`** —— 7 个 `@skill`（localize_submit/status/retry/cancel/download/search/video）
  + 12 个 helper（`_cloud_asr` / `_local_asr` / `_bge_embed` / `_media` / `_inpaint` 等）。
  **耦合注意**：`_local_asr` 用 `_cloud_asr` 的 `ASRError`；`_local_cr` 用 `_cloud_ocr`
  私有函数；`_inpaint` import `_media` —— 这几对必须同进退。
  **输入沙箱**：提交通道的本地路径必须位于 `data_dir/uploads/` 内（URL 不受限），
  收口在 `localize_submit._split_paths`（MCP 与 REST 两通道共用）；
  `FLOWMIND_ALLOW_ANY_PATH=1` 仅限本地测试放行（生产禁设）。鉴权实装前的
  隔离层：防任意路径探测、产物外带、覆盖写源目录。

### 任务 REST 状态码约定

`202` 受理（`task_ids` 列表；队列中途满 → 202 + warning 部分受理，transient 可重提）；
`429` 队列满（TaskQueueFull，一个都没受理；errors.py 无独立错误码，HTTP 层按异常类型
映射）；`400` 非 JSON；`422` 入参校验（含本地路径沙箱越界 / uploads 未就绪 / 全部
扩展名被拒）；`404` 任务/产物不存在。健康探针恒 200。

## 关键约定

- **本地优先 + 云回落**：ASR / OCR / 擦除 / 配音走本地模型（Pascal 6.1 域内），
  翻译 LLM / 云 ASR/OCR/TTS 回落走云 API。后端开关 local/cloud/auto。
  生产路径无任何可用后端时必须显式报错，**绝不静默降级出假结果**。
- **语言**：注释 / 文档字符串 / 日志 / 提交信息用**中文**；标识符用**英文**。
- **提交格式**：`<type>: <中文描述>`，type ∈ `feat/fix/docs/refactor/test/chore`。
- **错误永不静默**：所有失败经 `SkillResult(ok=False, error=...)` 或 `degraded=True`
  返回结构化结果，绝不吞异常、不返回半成品。`invoke()` 是统一执行点。
- **不留代码 TODO 给下游开发者**：可调项全部实现并带通用默认，走 config。
- **`trace_id` 贯穿**每次调用（REST 任务通道以 task_id 即 trace_id 贯穿）。
- **验证靠真实运行**：无单测，改完跑 `examples/*_demo.py`（8 个，全 PASS 为准）
  + 直接 `invoke("<id>", args)` 看 envelope。
- **API key 永不进 toml / commit**：`AI_LLM_API_KEY` / `AI_SPEECH_API_KEY` 只从
  env / gitignored `.env` 读（经 `skills/_secrets.get_api_key`）。集群凭证
  （`RAK_PG_*` 等）同样不入库。
- **错误消息脱敏**：失败路径不放完整异常详情或内部 host / 凭证。

## 失败返回的两种契约

`localize_*` 全系走 **degraded SkillOutput 模式**（**不是** raise）：
```python
r = invoke("localize_status", {"task_ids": ["..."]})
r.ok is True              # ← 不论成功失败
r.metrics.degraded is True
r.data.failure_category   # "environment" / "video" / "transient" / "unknown"
r.data.retriable          # True iff transient
r.error is None
```

例外：`localize_submit` 队列满且一个都没受理时 **raise TaskQueueFull**（invoke 兜底为
ok=False / INTERNAL + 「稍后重试」message，即 429 语义；REST 层按异常类型映射 429）。

调试时先确认技能是哪一类，再看对应字段。

## 验证（demo 冒烟）

```bash
for f in examples/*_demo.py; do PYTHONPATH=$PWD/src conda run -n flowmind python "$f"; done
```

8 个 demo：`localize_video / submit / status / retry / cancel / download / search /
api_server`。前 7 个用 mock 流水线/内存 fake manager（不依赖 GPU / 集群），
`api_server_demo` 起真实 uvicorn 测任务 REST 通道（400/422/429/部分受理/穿越防护）。
真实 GPU / 集群链路的任务生命周期冒烟单独做（起服务 + MCP 探针 + REST 提交）。

## 贡献新技能

1. `src/flowmind/skills/<name>.py` 写一个 `@skill` 函数返回 `SkillOutput`。
2. `src/flowmind/skills/__init__.py` 按字母序追加 import（`@skill` 重复 id 抛 ValueError）。
3. 可调参数加到 `config.py` 并纳入 `FlowmindConfig`。
4. `examples/<name>_demo.py` 加 demo（happy / 默认 / 错误三段式）。
5. `ruff check src` 全绿 + demo 跑通才 commit（`<type>: <中文描述>`）。

## MCP 端到端调试

```bash
conda run -n flowmind mcp-base-gpu          # 前台起单端口服务
# MCP 探针（python + mcp 库，streamablehttp_client → list_tools 应恰好 7 个 localize_*）
# REST 探活：curl http://127.0.0.1:8001/api/v1/health
```
