# AGENTS.md —— Claude Code / AI Agent 开发指南

> 给**使用 Claude Code（或类似 AI 编程助手）参与本仓库开发**的协作者看。
> 这份文档假设你熟悉 Claude Code 但不熟悉本项目代码结构。
> 与 `CLAUDE.md`（项目不变量 + 架构）互补；本文专注**开发工作流**。

## 30 秒上手

```bash
conda env update -n flowmind -f environment.yml   # 装依赖（environment.yml 是依赖真源）
conda run -n flowmind pip install --no-deps "simple-lama-inpainting>=0.1.2"  # stale metadata 见 environment.yml 注释
conda run -n flowmind pip install -e . --no-deps  # 包本体 + entry points
conda run -n flowmind ruff check src              # lint（必须通过）
for f in examples/*_demo.py; do
  PYTHONPATH=$PWD/src conda run -n flowmind python "$f"
done                                              # 跑 8 个 demo 冒烟
```

如果遇到任何概念不清楚，先看本文件对应章节，再问。

> **定位**（2026-09 收敛后）：`mcp-base-gpu` = 视频本地化 MCP over HTTP SaaS 服务端，
> 单端口 8002 双通道（`/mcp` 轻技能 + `/api/v1/tasks` 长任务）。
> **无单测 / 无 Makefile / 无 pytest**——验证靠 `examples/*_demo.py` 冒烟 + 真实调用。
> GPU（P104-100，Pascal 6.1）与基础设施约定见 `CLAUDE.md`「本地模型」「架构」段。
> worktree 中验证一律 `PYTHONPATH=<worktree>/src conda run -n flowmind ...`。

## 仓库速览

```
src/flowmind/
├── contracts.py       # 对外契约（SkillResult / ReasoningChain / ...）── 改这里 = 改对外 API
├── rules.py           # 规则求值器（四段式链的「触发规则/数据证据」自动产出）
├── config.py          # FlowmindConfig = LocalizerConfig + InfraConfig（env → toml → 默认）
├── skill.py           # @skill 装饰器 + invoke() 入口 ── 融合点
├── manifest.py        # build_manifest() ── Agent 视角的能力清单
├── server.py          # FastMCP v1 薄壳 ── 把 _REGISTRY 暴露成 MCP tool
├── server_http.py     # 单端口唯一入口（8002）：MCP + REST 路由 + CORS/鉴权占位中间件
├── server_rest.py     # 发现 API：GET /api/v1/manifest[/id]
├── server_tasks.py    # 任务 REST：POST/GET /api/v1/tasks、download、health
├── tasks/
│   ├── store.py       # TaskStore：PG 持久层（PgBouncer 事务模式，幂等建表，启动恢复）
│   ├── manager.py     # TaskManager：GPU 串行执行 + 协作取消 + 终态分类 + TTL GC
│   ├── events.py      # MQTT 事件（mcp-base-gpu/tasks/{id}/events，终态 retain）
│   ├── vectors.py     # Milvus 字幕向量库（localize_segments，768 维 HNSW/COSINE）
│   └── gpu.py         # gpu_lane() / model_cache_guard() 双道闸
└── skills/
    ├── __init__.py    # import 各技能触发 @skill 注册（按字母序）
    ├── localize_submit.py     # 批量提交（TaskQueueFull → 429 语义）
    ├── localize_status.py     # 参考：轻量只读（demo 惯用替身）
    ├── localize_retry.py / localize_cancel.py / localize_download.py
    ├── localize_search.py     # Milvus 语义检索
    ├── localize_video.py      # 流水线本体（ASR→OCR→译→擦→TTS→混→向量化）
    └── _*.py          # 12 个 helper（_cloud_asr/_local_asr/_bge_embed/_media/_inpaint…）
examples/              # 8 个 demo（冒烟验证的唯一手段，本仓库无单测）
docs/                  # 设计文档
```

## 典型任务与工作流

### 任务 1：修一个 bug

1. 写一个最小复现脚本（或直接调 `invoke("<id>", args)`）重现问题
2. 修源码 → 复现脚本确认修复
3. `conda run -n flowmind ruff check src` 通过再 commit
4. 提交格式 `<type>: <中文描述>`，type ∈ `feat/fix/docs/refactor/test/chore`

### 任务 2：加一个新技能

1. 在 `src/flowmind/skills/<name>.py` 写 `@skill` 函数（优先参考 `localize_status.py`
   —— 轻量只读；失败语义用 degraded SkillOutput 模式，见 CLAUDE.md）
2. `src/flowmind/skills/__init__.py` 按字母序追加 import（`@skill` 重复 id 抛 ValueError）
3. 可调参数：`config.py` 加 Config 类 + 纳入 `FlowmindConfig`
4. `examples/<name>_demo.py` 加 demo（三段式：happy / 默认 / 错误）
5. `ruff check src` 全绿 + demo 跑通 → commit

**铁律**：
- ❌ 不改 `server.py` / `contracts.py` / `manifest.py` / `skill.py`（除非改对外契约）
- ❌ 不留 `# TODO` 给下游开发者（可调项走 config + 通用默认）
- ❌ 不吞异常（任何失败必须走 `SkillResult(ok=False, error=...)` 或 `degraded=True`）

### 任务 3：理解一段陌生代码

1. 用 `LSP` 工具查定义（goToDefinition / hover）
2. 看 `CLAUDE.md`「架构」段对每个文件的角色说明
3. 跑 `examples/*_demo.py` 看实际输出
4. 直接 `invoke("<id>", args)` 打一遍看 envelope（trace / metrics / error）

### 任务 4：升级某个依赖

```bash
# 1) 改 environment.yml（依赖真源）：加/升级 <pkg>
# 2) conda env update -n flowmind -f environment.yml   # 重装
for f in examples/*_demo.py; do PYTHONPATH=$PWD/src conda run -n flowmind python "$f"; done
conda run -n flowmind ruff check src
```

**依赖真源是 `environment.yml`（conda）**。GPU 相关版本（torch cu121）有 Pascal 6.1
约束：`torch==2.5.1+cu121` 不可升；`torchaudio` 钉版必须在 `qwen-tts` 之后；
`nvidia-cublas-cu12` / `nvidia-cudnn-cu12` 已钉 torch 配套版（conda env update 的
pip 子进程带 -U，裸名称会被升到 12.9/9.25 破坏 CUDA 12.1 组合）；
`simple-lama-inpainting` 不在 pip 段（stale metadata `pillow<10` 在 py3.12 无解），
须按文件头序列 `--no-deps` 单独补装（均见 environment.yml 内注释）。

### 任务 5：合并一个 PR（maintainer 视角）

1. `gh pr view <N>` 看改动
2. `gh pr view <N> --json files` 看共享文件（`config.py` / `skills/__init__.py`）冲突风险
3. **没有 CI 的项目**（本仓库当前）：本地 fetch PR 分支 → 跑 ruff + demo → `git merge --no-ff`
4. 共享文件冲突：保留**双方新增**，import 按字母序

## 调试技巧

### 失败信号 → 检查方向

| 信号 | 看哪里 |
|---|---|
| `ok=False, error.code=NOT_FOUND` | `invoke()` 入口 / `_REGISTRY` / 是否漏注册 |
| `ok=False, error.code=VALIDATION` | 技能入参 BaseModel 的 Field 约束 |
| `ok=False, error.code=INTERNAL` + 「稍后重试」 | `TaskQueueFull`（队列满背压）——正常语义，非 bug |
| `degraded=True` | 技能自己判定降级 → 看 `degradation_reason` / `failure_category` |
| REST `429 {"error":"queue_full"}` | `max_pending_tasks` 水位满；查 `tasks.store.count_pending` |
| REST `404 {"error":"unknown_task"}` | task_id 不存在（或已被 GC 只删了 workdir——DB 行保留，应是 id 错） |
| health `components.pg=error` | PG 不可达：`FLOWMIND_PG_DSN` / `RAK_PG_*`（source rak .env） |
| health `components.mqtt=disabled` | `FLOWMIND_MQTT_HOST` / config `infra.mqtt_host` 均空（事件降级纯落库，非故障） |
| MCP 工具列表没出现 | `_make_tool` 的 `__annotations__` 注入没生效 / server 启动失败 |
| torch `libcudart.so.13` 报错 | qwen-tts 把 torchaudio 拉错版了——按 environment.yml 钉回 `2.5.1+cu121` |
| 入参 schema 多一层 `inp` | FastMCP v1 的固有行为，升 v2 可去（需重验 `_make_tool` 反射） |

### 单步追踪

```bash
# 加断点（任意源码位置）
import pdb; pdb.set_trace()

# 看 trace_id 贯穿与 envelope
PYTHONPATH=$PWD/src conda run -n flowmind python -c "
import flowmind.skills
from flowmind.skill import invoke
r = invoke('localize_status', {'task_ids': ['no-such-task']})
print(r.ok, r.metrics.degraded, r.data.failure_category)
"
```

### 端到端调试（单端口服务）

```bash
conda run -n flowmind mcp-base-gpu                 # 前台起（MCP + REST 同端口 8002）
curl http://127.0.0.1:8002/api/v1/health           # 组件状态
curl http://127.0.0.1:8002/api/v1/manifest          # 7 技能清单
# MCP 探针：python + mcp 库 streamablehttp_client → list_tools 应恰好 7 个 localize_*
```

## 千万别做（Anti-patterns）

| 行为 | 为什么错 |
|---|---|
| 在 `server.py` 加新 tool 注册代码 | 违反"加技能不动 server"不变量 |
| 修改 `SkillResult` 信封字段 | 对外契约变更，所有 Agent 都要适配 |
| 写代码 TODO 留给用户 | 违反"不留 TODO 给下游"约定；可调项走 config |
| 在技能函数里 `try/except: pass` | 违反"错误永不静默"铁律 |
| 放宽 TaskManager `workers=1` 或 `gpu_lane` 信号量 | 单卡 8GB 显存预算 ~7.5G，并发即 OOM |
| 升级 torch / cu121 钉版、调整 environment.yml 钉版顺序 | Pascal 6.1 硬约束；torchaudio 必须在 qwen-tts 之后；cublas/cudnn 已钉 torch 配套版；simple-lama 不在 pip 段（--no-deps 补装） |
| 跳过 `ruff check src` 直接 commit | lint 是合并前唯一的质量门 |
| 把 `flowmind.config.toml` / `.env` / 集群凭证提交进 git | 用户私有配置 + 安全红线 |

## 提交前 Checklist

- [ ] `conda run -n flowmind ruff check src` 全绿
- [ ] 若改了技能行为：对应 `examples/*_demo.py` 跑通（8 个全 PASS 为准）
- [ ] 提交信息 `<type>: <中文描述>`
- [ ] 若改了 `environment.yml`：确认已 `conda env update` 且 demo 通过、torch 栈未破坏
- [ ] 若改了 `config.py` / `skills/__init__.py`：留意 merge conflict hotspot

## 给 Claude Code 的额外提示

- **优先用 LSP** 查定义 / 重构（rename / go-to-impl / find-refs），比 grep 准
- **改架构前先读 `CLAUDE.md`**「关键约定」段 —— 那里有不变量
- **大改动先给方案**让用户确认再下手
- **改完跑 `examples/*_demo.py`** —— demo 跑通 = 技能没被破坏（最快冒烟测试）

## 🤖 第一次拿到这个项目（FRESH AGENT）

如果你刚被用户部署到这个 repo：**先读 README.md 顶部的「🤖 如果你是一个 AI Agent
第一次读到这个文件」段**——装依赖 → 起服务 → health 探活 → MCP 接入三步，
字段 schema 一律 `GET /api/v1/manifest/<skill_id>` 查询，不猜不读源码。
架构与不变量见 `CLAUDE.md`。
