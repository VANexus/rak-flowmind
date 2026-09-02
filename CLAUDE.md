# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

`rak-flowmind` 是一个**对任意 Agent 友好**的 Python Skill SDK，通过 MCP **和 Google A2A** 双协议暴露，提供「能被任意 Agent 优雅调度的技能」的框架契约。**FlowMind 既是 MCP 服务端，也是 A2A Agent** —— `invoke()` 是两套协议的共用入口，技能层零改动即可同时服务 MCP tool call 和 A2A Task 委托。

**核心不变量**：**新增一个技能 = 写一个 `@skill` 函数**。注册 / JSON schema / MCP tool / manifest / discover() 自动暴露全自动。加技能**不改动** `server.py` / `contracts.py` / `skill.py` / `rules.py` / `__init__.py` 之外的契约 / 框架层 —— 这条约束改前务必确认。

`README.md` 顶部的 `🤖 FRESH AGENT DEPLOYMENT PROTOCOL` 段是新 Agent 第一次拿到这个 repo 该走的 5 步 startup（自我对话 + 自动部署 + MCP 配置）。**Agent 进来先读那段**，不是读 CLAUDE.md。

## 常用命令

```bash
conda env update -n flowmind -f environment.yml          # 装/更新依赖（environment.yml 是依赖真源）
conda run -n flowmind pip install -e . --no-deps          # 包本体 + entry points（依赖全由 environment.yml 管）
conda run -n flowmind ruff check src                      # lint（必须通过）
for f in examples/*_demo.py; do conda run -n flowmind python "$f"; done  # demo 冒烟（本仓库无单测）
conda run -n flowmind flowmind-mcp                        # 启动 MCP 服务器（stdio 传输）
conda run -n flowmind flowmind-init                       # 9 步对话式初始化向导（用户跑）
```

**Python 3.12**（conda env `flowmind`）。**依赖真源 = `environment.yml`**（uv 已废弃，uv.lock 已删除）；`pyproject.toml` 只保留包元数据 / entry points / ruff 配置。**不用 Makefile / Docker / n8n / pytest**（tests/ 已删除，验证靠 demo + 真实 invoke）。

## 本地模型（GPU 化，2026-09）

硬件：NVIDIA P104-100（8GB，Compute Capability 6.1 / Pascal）。架构从「云优先」升级为「**本地优先（GPU 可行域内）+ LLM/TTS/生图继续云 API**」：

| 能力 | 本地后端（默认优先） | 云回落 | 开关（flowmind.config.toml） |
|---|---|---|---|
| ASR | faster-whisper（CTranslate2，int8） | dashscope qwen-audio | `localizer.asr_backend` = local/cloud/auto |
| OCR 字幕定位 | RapidOCR（onnxruntime，CPU） | dashscope qwen3.5-ocr | `localizer.ocr_backend` = local/cloud/auto |
| FAQ 向量召回 | bge-small-zh-v1.5（sentence-transformers，GPU FP32） | 无（回落双路 BM25+TF-IDF） | `feishu_kb.embed_backend` = auto/on/off |
| LLM / TTS / 生图 | —（不做本地） | LongCat / dashscope / ciyuansky | 维持云 API |

关键约束与约定：

- **Pascal 6.1 限制**：torch 锁 `2.5.1+cu121`（cu124+ 构建剔除 Pascal，勿升）；vLLM 不可用；CTranslate2 下限 CC 6.0，int8 走 dp4a。
- **auto 语义**：本地库可导入即用本地，否则回落云；两端都不可用显式报错。实际使用的 backend 体现在 report 字段（`asr_backend` / `ocr_backend`）与 ReasoningChain 文本中，**不静默降级**。
- **模型缓存与下载**：统一缓存 `HF_HOME=/srv/data/models` + `MODELSCOPE_CACHE=/srv/data/models`，HF 下载走本机代理 `http://127.0.0.1:7890`——均已写入 `~/.bashrc`（hf-mirror 会 302 回被墙主站，勿用）。新 shell 直接 import 库即生效，无需额外配置。
- 模型实例进程内缓存（`_local_asr._models` / `_local_embed._models` / `_local_ocr._ocr_engine`），测试通过 monkeypatch `available()` / `encode()` 打桩，不打真实模型。

## 架构（大图）

数据流：`Agent → (MCP tool call 或 A2A Task) server.py / a2a/server.py → invoke() / run_orchestrator() → skill 函数 → SkillResult → 业务结果 / 错误信封`。

分层（`src/flowmind/`，传输无关核心 + 薄 MCP/A2A 双协议层）：

- **`contracts.py`** —— 对外契约层：`SkillResult[T]` 信封 / `ReasoningChain` 四段式链 / `ReliabilityMetrics` / `TraceContext` / `SkillError` / `SkillOutput[T]`。**改这里 = 改对外 API**，必须 bump version 并走 PR。
- **`skill.py`** —— 融合点：`@skill` 装饰器 + `_REGISTRY` + `invoke()`。技能函数只返回轻量 `SkillOutput`；`invoke()` 统一套 `SkillResult` 信封（注入/透传 `trace_id`、填 `latency_ms`、把三类失败兜底成结构化错误）。`SkillSpec` 在注册时**自动捕获** `output_model`（从 `SkillOutput[T]` 返回注解）和 `description`（从函数 docstring 第一段）。
- **`discover.py` + `manifest.py`** —— Agent 自助发现：`discover()` / `field_names()` 把 input + output 完整 JSON Schema 一次暴露，Agent **不再需要读源码**就能拿到 `r.data.foo` 该叫什么。
- **`errors.py`** —— 错误分类（不放在 `contracts.py` 守不变量）：`ErrorCode` enum + `_classify_exception()` + `is_retriable()`。把异常归到 `environment` / `video` / `transient` / `unknown` 四类。
- **`interactive.py`** —— 对话式可交互初始化：`run_interactive_init(ask_fn)` 逐项问用户 9 个偏好；CLI 入口 `flowmind-init`。
- **`config.py`** —— 配置层：`FlowmindConfig` / `InventoryConfig` / `FeishuKbConfig` / `MarketingImageConfig` / `LocalizerConfig` / `OrchestratorConfig`；`load_config` / `save_config` / `get_config` / `reload_config` / `init_for_user`。可调项只经 config 暴露，**带通用默认**；个性化由终端用户对话写 `flowmind.config.toml`（gitignored）。
- **`vl_client.py`** —— 视频本地化后端 HTTP 封装（含请求分类）。
- **`server.py`** —— FastMCP（**v1**，`mcp>=1.27,<2`）遍历注册表动态登记 MCP tool。`_make_tool` 靠设置 `__annotations__` 驱动 schema 推断 —— v1 特定技巧。
- **`orchestrator/`** —— A2A 编排器：`planner.py` / `executor.py` / `recovery.py` / `summarizer.py` / `graph.py`。LLM 规划技能调用序列，调 `invoke()` 执行，错误恢复决策，LLM 汇总结果。
- **`a2a/`** —— A2A 协议层：`agent_card.py`（Agent Card 构建）/ `types.py`（A2A ↔ FlowMind 类型映射）/ `server.py`（Task 端点 + Starlette app）。
- **`server_http.py`** —— HTTP 服务器：Starlette 挂载 MCP（Streamable HTTP，`/mcp`）+ A2A（`/.well-known/agent.json` + `/a2a`）双协议，单端口 8001。
- **`skills/`** —— 14 个 `@skill` 注册在 `__init__.py`：3 个纯计算（`inventory_risk` / `feishu_kb_search` / `marketing_image_gen`）+ 6 个 HTTP 依赖的 `localize_*` + 5 个内容创作 `content_*`（`content_idea_design` / `content_copywrite` / `content_hot_topics` / `content_audit` / `content_image_gen`，密钥走 `LONGCAT_API_KEY` / `CIYUANSKY_API_KEY` env）。每个技能文件第一段 docstring 会被 `SkillSpec.description` 自动捕获。

### `feishu_kb_search` 关键能力（PR #6 + PR #7 合入后）

- **113 条企业 FAQ seed**（`feishu_kb_seed.json`，覆盖 8 份企业 docx 解析产物）—— 由 `scripts/build_seed_from_docx.py` 一次性重建，**不进入运行时依赖**。
- **Hard-gate 防话题外**：中文 query 走"意图分类置信度=0" + `FeishuKbConfig.min_top1_score`（默认 0.015）双门；EN/TH 跳过关键词 gate（跨语言关键词不适用），仅走分数 gate。任何 path 下 `top_k=[]` → `metrics.degraded=True` + `agent_reply_hint` 透传"暂未收录"文案。
- **中英泰三语支持（zero-LLM）**：`_detect_language()` 基于 Unicode 范围判 `zh/en/th/other`；`_CROSS_LANG_SYNONYMS`（~200 项）把 EN/TH 领域词桥接到中文 FAQ；`_phrase_match_bonus` 解决 BM25 长 answer bias。
- **Agent 结构化字段**（无 API key 时下游 agent 也能读）：`user_language` / `translation_required` / `translation_directive`（`{source, target, rule}`）。`agent_reply_hint` 末尾追加 `[Language-MANDATORY]` 强约束翻译层。
- **严格忠于 KB**：`_agent_reply_hint` 改为"直接引用 Top-1 原文"，禁止 LLM 整合 / 补充 / 推测。

### A2A 集成（2026-08）

FlowMind 同时是 Google A2A Agent：

- **Agent Card**：`GET /.well-known/agent.json` 暴露分组能力（content/video/data），A2A 客户端自发现入口。
- **Task 端点**：`POST /a2a`（JSON-RPC：`tasks/send` / `tasks/get` / `tasks/cancel`）。
- **编排器**：`src/flowmind/orchestrator/` —— Planner → Executor → Recovery → Summarizer 四节点，LLM 规划技能调用序列。
  - `planner.py` —— 调 LLM 生成 `{steps, cot}` 计划。
  - `executor.py` —— 调 `invoke()` 执行单步技能。
  - `recovery.py` —— 根据 `SkillResult.retriable` 决定 retry / skip / fail。
  - `summarizer.py` —— 调 LLM 汇总步骤结果为最终输出。
  - `graph.py` —— 组装四节点，暴露 `run_orchestrator(goal, skill_group, include_reasoning)`。
- **CoT**：按需暴露（默认仅结果，`include_reasoning=true` 返回完整推理链），复用 `ReasoningChain` 契约。
- **配置**：`OrchestratorConfig`（LLM key/base_url/model 可配，默认 LongCat；`max_plan_steps` / `max_retries_per_step` 可调）。
- **技能层零改动**：`invoke()` 是 MCP 和 A2A 共用入口，`@skill` 函数不感知调用来源。
- **HTTP 服务器**：`src/flowmind/server_http.py` —— Starlette 挂载 MCP（Streamable HTTP，`/mcp`）+ A2A（`/.well-known/agent.json` + `/a2a`）双协议，单端口 8001。

## 关键约定

- **本地优先 + 云回落（GPU 化，2026-09 起，最高原则）**：ASR / OCR / 向量嵌入走本地模型（faster-whisper / RapidOCR / bge-small，P104-100 8GB Pascal 6.1），LLM / TTS / 生图继续云 API。后端开关走 config（local/cloud/auto，auto = 本地可用即用本地，无库回落云，两端都不可用显式报错）。确定性 mock 后端仅作测试基建（须显式指定），生产路径无任何可用后端时必须显式报错，**绝不静默降级出假结果**。
- **语言**：注释 / 文档字符串 / 日志 / 提交信息用**中文**；标识符（变量/函数/类）用**英文**。
- **提交格式**：`<type>: <中文描述>`，type ∈ `feat/fix/docs/refactor/test/chore`。
- **错误永不静默**：所有失败经 `SkillResult(ok=False, error=...)` 或 `degraded=True` 返回结构化结果，绝不吞异常、不返回半成品。`invoke()` 是这条铁律的统一执行点。
- **不留代码 TODO 给下游开发者**：可调项全部实现并带通用默认，走 config；定制只发生在终端用户对话初始化。
- **`trace_id` 贯穿**每次调用（透传优先，缺失则 `new_trace()` 生成）。
- **DSI（周转天数）无动销（`sales_30d==0`）时取 `None`**，避免 `Infinity` 破坏 JSON 序列化。
- **验证靠真实运行**：本仓库无单测（tests/ 已删），改完跑 `examples/*_demo.py` + 直接 `invoke("<id>", args)` 看 envelope。
- **API key 永不进 toml / commit**：视频本地化 `ALLIN_API_KEY`、营销生图 `ALLIN_API_KEY` 都只从环境变量读。代码里只有 `*_key_env: str = "ALLIN_API_KEY"` 这种 env var 名字。
- **错误消息脱敏**：失败路径（`api_message` / `causal_analysis` / `warning`）不放完整异常详情或 `api_base` URL —— Agent 拿到 result 后能据此决策，但不泄漏内部 host / 凭证。

## 失败返回的两种契约

5 个 `localize_*` 的错误走 **degraded SkillOutput** 模式（**不是** raise）：
```python
r = invoke("localize_batch", {...})
r.ok is True              # ← 不论成功失败
r.metrics.degraded is True
r.data.failure_category   # "environment" / "video" / "transient" / "unknown"
r.data.retriable          # True iff transient
r.error is None
```

`inventory_risk` / `feishu_kb_search` / `marketing_image_gen` 走**普通 raise 模式**：
```python
r.ok is False
r.error.code             # "VALIDATION" / "NOT_FOUND" / "INTERNAL"
r.metrics.degraded is False
r.data.failure_category is None
```

调试 / 判断结果结构时**先看 skill 是哪一类**（degraded SkillOutput vs raise），再看对应字段。

## 验证（demo 冒烟）

本仓库**没有单测**（tests/ 已删除，用户决定：验证靠真实运行）：

```bash
for f in examples/*_demo.py; do conda run -n flowmind python "$f"; done
```

每个 demo 脚本（`examples/<skill>_demo.py`）第一行都跑 `discover()` —— 这是真实字段名的来源。改完技能 = 跑对应 demo；改动框架层 = 跑全部 demo。

## 贡献新技能

1. `src/flowmind/skills/<name>.py` 写一个 `@skill` 函数返回 `SkillOutput`。
2. `src/flowmind/skills/__init__.py` 加一行 `from flowmind.skills import <name>  # noqa: F401`。
3. 可调参数加到 `config.py` 的 `XxxConfig` 类 + 纳入 `FlowmindConfig`。
4. `examples/<name>_demo.py` 加 demo（happy / 默认 / 错误三段式），通过 `invoke("<id>", args)` 走 envelope 层。
5. `ruff check src` 全绿 + demo 跑通才 commit。
6. 提交格式 `<type>: <中文描述>`，type ∈ `feat/fix/docs/refactor/test/chore`。

具体配方 + 反例见 `.claude/skills/flowmind-test-skill/SKILL.md`（必读）和 `flowmind-onboard` skill。

## Agent / 用户工具

- **CLI 向导**：`conda run -n flowmind flowmind-init`（用户跑 9 步问 9 个偏好）
- **Agent 对话式**：`from flowmind.interactive import run_interactive_init; run_interactive_init(ask_fn=my_llm_ask_fn)`
- **Schema 发现**：`from flowmind import discover, field_names`
- **MCP 起服务**：`nohup conda run -n flowmind flowmind-mcp > /tmp/flowmind-mcp.log 2>&1 &`
- **真打 allin-api**（视频本地化 / 营销生图）：`export ALLIN_API_KEY="sk-..."` 后 backend 自动选真；无 key 自动 fallback mock。`examples/marketing_image_gen_real.py` 是真打集成 demo。

## 仓库特有目录

- `.claude/skills/flowmind-onboard/` —— Agent 第一次进 repo 必读
- `.claude/skills/flowmind-test-skill/` —— 端到端测试 skill
- `examples/` —— 13 个 demo 脚本 + 1 个真打集成示例 + 1 个 A2A 集成 demo
- `.test-runs/` —— gitignored，端到端测试报告输出位置