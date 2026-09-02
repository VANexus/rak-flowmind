# AGENTS.md —— Claude Code / AI Agent 开发指南

> 给**使用 Claude Code（或类似 AI 编程助手）参与本仓库开发**的协作者看。
> 这份文档假设你熟悉 Claude Code 但不熟悉本项目代码结构。
> 与 `CLAUDE.md`（项目不变量 + 架构）互补；本文专注**开发工作流**。

## 30 秒上手

```bash
conda env update -n flowmind -f environment.yml   # 装依赖（environment.yml 是依赖真源）
conda run -n flowmind pip install -e . --no-deps  # 包本体 + entry points
conda run -n flowmind ruff check src              # lint（必须通过）
for f in examples/*_demo.py; do conda run -n flowmind python "$f"; done  # 跑 demo 冒烟
```

如果遇到任何概念不清楚，先看本文件对应章节，再问。

> **环境注意**（2026-09 GPU 化升级后）：依赖真源已从 uv 迁到 **conda（environment.yml）**，`uv.lock` 已删除；
> **测试目录已删除**（用户决定：不做单测，验证靠 `examples/*_demo.py` 冒烟 + 真实调用），
> 本仓库**没有 Makefile / pytest**（历史文档中的 `make test` / `pytest` 均已失效，用上面的命令）。
> GPU（P104-100，Pascal 6.1）本地模型约定见 `CLAUDE.md`「本地模型」段：ASR/OCR/向量嵌入本地优先，
> LLM/TTS/生图走云 API，后端开关 local/cloud/auto 在 `flowmind.config.toml`。

## 仓库速览

```
src/flowmind/
├── contracts.py       # 对外契约（SkillResult / ReasoningChain / ...）── 改这里 = 改对外 API
├── rules.py           # 规则求值器（四段式链的「触发规则/数据证据」自动产出）
├── config.py          # FlowmindConfig + 各技能 Config ── 新技能的可调参数都加这里
├── skill.py           # @skill 装饰器 + invoke() 入口 ── 融合点
├── manifest.py        # build_manifest() ── Agent 视角的能力清单
├── server.py          # FastMCP v1 薄壳 ── 把 _REGISTRY 暴露成 MCP tool
└── skills/
    ├── __init__.py    # import 各技能触发 @skill 注册（加新技能就追加一行）
    ├── inventory_risk.py         # 参考：纯确定性
    ├── marketing_image_gen.py    # 参考：确定性 mock 后端
    └── feishu_kb_search.py       # 参考：BM25+TF-IDF，引入外部依赖
examples/              # 可跑 demo（冒烟验证的唯一手段，本仓库无单测）
docs/                  # 设计文档 / 集成指南 / 技能开发配方
scripts/setup.sh       # 一键 setup（依赖 + demo + 配置）
```

## 典型任务与工作流

### 任务 1：修一个 bug

1. 写一个最小复现脚本（或直接调 `invoke("<id>", args)`）重现问题
2. 修源码 → 复现脚本确认修复
3. `conda run -n flowmind ruff check src` 通过再 commit
4. 提交格式 `<type>: <中文描述>`，type ∈ `feat/fix/docs/refactor/test/chore`

### 任务 2：加一个新技能

**最小路径**（详细配方见 `docs/skill-authoring-guide.md`）：

1. 在 `src/flowmind/skills/<name>.py` 写 `@skill` 函数，遵循**现有技能模板**（优先复制 `inventory_risk.py` —— 最简）
2. 在 `src/flowmind/skills/__init__.py` 末尾追加 `from flowmind.skills import <name>  # noqa: F401`
3. 若有可调参数：在 `src/flowmind/config.py` 加一个 `XxxConfig` 类 + 纳入 `FlowmindConfig`
4. 在 `examples/<name>_demo.py` 加 demo（沿用三段式：happy / 默认 / 错误）—— demo 就是本仓库的「测试」
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
for f in examples/*_demo.py; do conda run -n flowmind python "$f"; done  # 验证不破坏
conda run -n flowmind ruff check src                                     # 验证风格
```

**依赖真源是 `environment.yml`（conda）**。不要改 `pyproject.toml` 的 dependencies —— 那只是元数据兜底，改了 environment.yml 不 `conda env update` 等于没改。GPU 相关版本（torch cu121）有 Pascal 6.1 约束，见文件头注释。

### 任务 5：合并一个 PR（maintainer 视角）

1. `gh pr view <N>` 看改动 + CI
2. `gh pr view <N> --json files` 看共享文件（`config.py` / `skills/__init__.py`）冲突风险
3. `gh pr diff <N>` 看实际改动
4. **没有 CI 的项目**（本仓库当前）：本地 `git fetch origin pull/<N>/head:pr-<N>` → 切到分支跑 ruff + demo → 本地合并（`git merge --no-ff`）→ 推送
5. 共享文件冲突：保留**双方新增**，import 按字母序

## 调试技巧

### 失败信号 → 检查方向

| 信号 | 看哪里 |
|---|---|
| `ok=False, error.code=NOT_FOUND` | `invoke()` 入口 / `_REGISTRY` / 是否漏注册 |
| `ok=False, error.code=VALIDATION` | 技能入参 BaseModel 的 Field 约束 |
| `ok=False, error.code=INTERNAL` | 技能函数内部异常 → 看 traceback（`error.details`） |
| `degraded=True` | 技能自己判定降级（非失败）→ 看 `degradation_reason` |
| MCP 工具列表没出现 | `_make_tool` 的 `__annotations__` 注入没生效 / server 启动失败 |
| 入参 schema 多一层 `inp` | FastMCP v1 的固有行为，升 v2 可去 |

### 单步追踪

```bash
# 加断点（任意源码位置）
import pdb; pdb.set_trace()

# 看真实 trace_id 是否贯穿
conda run -n flowmind python -c "
import flowmind.skills
from flowmind.skill import invoke
r = invoke('inventory_risk', {'items': [{'sku':'A','on_hand':10,'unit_cost':1,'sales_30d':1}]})
print(r.trace.trace_id, r.error, r.metrics.latency_ms)
"
```

### MCP 端到端调试

```bash
# 起一个 stdio MCP server，前台跑
conda run -n flowmind flowmind-mcp

# 另开终端，用 MCP 客户端连
conda run -n flowmind python /tmp/probe_mcp.py    # 见 agent-integration.md 里的 probe 脚本
```

## 千万别做（Anti-patterns）

| 行为 | 为什么错 |
|---|---|
| 在 `server.py` 加新 tool 注册代码 | 违反"加技能不动 server"不变量 |
| 修改 `SkillResult` 信封字段 | 对外契约变更，所有 Agent 都要适配 |
| 写代码 TODO 留给用户 | 违反"不留 TODO 给下游"约定；可调项走 config |
| 在技能函数里 `try/except: pass` | 违反"错误永不静默"铁律 |
| 跳过 `ruff check src` 直接 commit | lint 是合并前唯一的质量门 |
| 把 `flowmind.config.toml` 提交进 git | 它是 gitignored 的用户私有配置 |

## 提交前 Checklist

- [ ] `conda run -n flowmind ruff check src` 全绿
- [ ] 若是新技能 / 改了技能行为：对应 `examples/*_demo.py` 跑通
- [ ] 提交信息 `<type>: <中文描述>`
- [ ] 若改了 `environment.yml`：确认已 `conda env update` 且 demo 通过
- [ ] 若改了 `config.py` / `skills/__init__.py`：留意 merge conflict hotspot

## 给 Claude Code 的额外提示

- **优先用 LSP** 查定义 / 重构（rename / go-to-impl / find-refs），比 grep 准
- **改架构前先读 `CLAUDE.md`** 「关键约定」段 —— 那里有不变量
- **大改动进 plan mode** 让用户先看方案再下手
- **改完跑 `examples/*_demo.py`** —— demo 跑通 = 技能没被破坏（最快冒烟测试）
- **`@skill` 重复 id** 会抛 `ValueError`，加新技能前 `grep -r "id=\"" src/flowmind/skills/`

## 🤖 第一次拿到这个项目（FRESH AGENT）

如果你刚被用户部署到这个 repo（用户给了 GitHub 链接或 zip）：

**不要立刻跑命令**。**先读 README.md 顶部的 `🤖 如果你是一个 AI Agent 第一次读到这个文件` 段** —— 那里有完整的部署协议。

**协议要点（5 步）**：
0. 一句话自我介绍 + 告诉用户你要做什么（不要让他手动跑命令）
1. **只问 1 个问题**：「你的项目主要场景？」（决定要不要 init video localization config）
   - **不要问** Agent 平台 —— 用户发给你就是选了你
   - **不要问** 要不要 MCP —— Agent 接 SDK 就必须装
2. 按答案自动跑（视频本地化调 `run_interactive_init(ask_fn=...)`；其他什么都不配）
3. `conda env update -n flowmind -f environment.yml` + `conda run -n flowmind pip install -e . --no-deps` + `examples/*_demo.py` 验证
4. 起 MCP server + **自动检测 Agent 平台**（`~/.claude` / `~/.cursor` / `~/.config/cline`）写 stdio 配置
5. 给用户交付摘要