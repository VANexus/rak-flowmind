---
name: flowmind-onboard
description: Onboard any Agent (Claude Code / Claude Desktop / Cursor / Cline / OpenClaw / generic Python developer / MCP client) to the FlowMind Skill SDK for the first time. Use this skill when the user says "熟悉一下这个项目", "这是什么项目", "项目结构", "怎么看代码", "this repo", "first time", "where is X", "怎么接入", "how do I start", "agent 怎么连", "MCP 怎么配", or any first-touch question about the SDK. This skill is the **startup playbook** for a fresh Agent — it prescribes the exact sequence (connect → discover → schema → init user → call) and points at the right file/command instead of dumping everything.
---

# flowmind-onboard

**新 Agent 接入 SDK 的 5 步 startup playbook**。

不管你是 Claude Code / Claude Desktop / Cursor / Cline / OpenClaw 还是直接 Python import，下面这 5 步走完就是「上岗」。

## 30 秒极速版

```bash
make install && make test && make demo
```

绿灯就说明环境 + 技能都 OK。完了之后读本文「5 步 startup」段。

---

## 5 步 startup（适用于任何 Agent）

> 哲学：**先 discover（不要硬编码字段名）→ 再调 → 失败看 category**。这是 v0.3 之后的官方 onboarding 路径。

### Step 1：CONNECT —— 确认 SDK 可达

**Python 直连：**
```python
import flowmind.skills  # noqa  触发 @skill 自动注册
from flowmind import registry
assert len(registry()) >= 8, "期望 8 个 skill 注册"
```

**MCP 客户端（Claude Desktop / Cursor / Cline / OpenClaw）：**
按 `flowmind-mcp-setup` skill 配 stdio 连接。配完 MCP 客户端会自动 list tools —— 如果看到 8 个 tool 就 OK。

> 如果 import 报错或 MCP 工具列表为空 → 触发 `flowmind-mcp-setup`（修环境）/ 看 `CLAUDE.md`「关键约定」。

### Step 2：DISCOVER —— 拿到所有 skill 的完整契约

```python
from flowmind import discover, field_names

# 列出所有 skill
for s in discover():
    print(f"{s['id']} v{s['version']} — {s['description']}")
    # → 8 个技能 + 每个的 description（从 docstring 提）
```

```python
# 看单个 skill 的完整 input + output JSON Schema
info = discover("inventory_risk")
# info["input_schema"]   入参字段 + 类型 + 必填
# info["output_schema"]  出参字段 + 类型（这就是 r.data 里能取什么）
# info["description"]    业务语义（为什么用 / 什么时候用）
```

```python
# 看 r.data 下面的嵌套字段路径（避免猜 r.data.band vs r.data.summary.level_counts）
for path, names in field_names("inventory_risk").items():
    print(f"{path}: {names}")
# → data.items[].sku / data.summary.level_counts / ...
```

**这一步完成前，不要写任何调用代码或测试断言。** 没 discover 就开始写 = 猜 schema（v0.3 之前最大的痛点）。

### Step 3：INIT_USER —— 引导用户配置偏好（首次必做）

如果用户是新用户（没有 `flowmind.config.toml`），Agent 应该**分步问**而不是一次塞 9 个参数。两种走法：

#### 走法 A：Agent 驱动对话（推荐 —— 用户体验最好）

```python
from flowmind.interactive import run_interactive_init

# ask_fn 是「问用户一个问题，返回答案」的 callable
# Agent 实现：调 LLM 生成对话问题给用户，把用户回答返回
cfg = run_interactive_init(ask_fn=my_llm_ask_fn)
# → 9 个问题逐个问（目标语言 / 源语言 / TTS / 字号 / 位置 / 文件后缀）
# → 每题给默认 + 解释 + 选项
# → 用户直接按 Enter 用默认也行
# → 写 flowmind.config.toml + reload_config
```

#### 走法 B：CLI 跑（用户自己设）

```bash
uv run flowmind-init
# → 同样的 9 步向导在终端里问
```

#### 走法 C：一次性传参（脚本/测试用）

```python
from flowmind.config import init_for_user

cfg = init_for_user(
    target_lang="th",
    enable_tts=True,
    tts_voice="th-TH-NiwatNeural",
    subtitle_font_size=28,
)
```

**注意**：5 个 `localize_*` 技能会读这套偏好；其他 3 个 skill（`inventory_risk` / `feishu_kb_search` / `marketing_image_gen`）不走这套，是独立配置。

### Step 4：CALL —— 第一次 invoke

```python
from flowmind import invoke

r = invoke("inventory_risk", {
    "items": [{"sku": "A", "on_hand": 50, "sales_30d": 30, "unit_cost": 80.0}],
})

print(r.ok)                # True / False
print(r.metrics.latency_ms)
print(r.trace.trace_id)    # 永远非空（铁律）
print(r.data.summary)      # 业务数据（用 Step 2 拿到的 schema 读）
```

### Step 5：HANDLE —— 失败/降级响应

按 `SkillResult` 类型**两种契约**分支处理（这是 v0.3 的硬规则）：

```python
if not r.ok:
    # 纯计算类失败（inventory_risk / feishu_kb_search / marketing_image_gen）
    if r.error.code == "VALIDATION":
        # 入参校验失败 —— 改入参或告诉用户
    elif r.error.code == "NOT_FOUND":
        # 未知 skill_id —— typo 了
    else:
        # INTERNAL —— skill 内部 bug
        log(r.error.message)

elif r.metrics.degraded:
    # HTTP 依赖类失败（5 个 localize_*）—— 不抛异常，走 degraded SkillOutput
    cat = r.data.failure_category  # "environment" / "video" / "transient" / "unknown"
    if r.data.retriable:
        retry()                     # transient 类可重试
    elif cat == "video":
        fix_input()                 # 检查视频路径 / 格式
    elif cat == "environment":
        check_vl_service()          # VL 后端不通，先修环境
    # else: unknown —— 看 warning 字段
else:
    # 正常成功
    use(r.data)
```

**关键**：5 个 `localize_*` 失败时 `r.ok=True` + `r.metrics.degraded=True` + `r.data.failure_category` —— **不要断言 `r.ok is False`**，那是纯计算类的失败模式。

---

## 项目一句话定位

`rak-flowmind` = 对任意 Agent 友好的 Python Skill SDK，通过 MCP 暴露。**技能底座**，提供可被任意 Agent 优雅调度的技能框架。

**核心不变量**：加技能 = 写 `@skill` 函数。注册 / JSON schema / MCP tool / manifest 全自动。不动 `server.py` / `contracts.py` / `manifest.py`。

## 仓库速览（v0.3）

```
src/flowmind/
├── __init__.py           # 顶层导出 discover / field_names / invoke / registry
├── contracts.py          # SkillResult / SkillError / ReasoningChain / SkillOutput
├── skill.py              # @skill 装饰器 + invoke() ─ 融合点
├── rules.py              # Rule + evaluate_rules()
├── config.py             # FlowmindConfig + LocalizerConfig + init_for_user + save_config
├── errors.py             # ErrorCode enum + _classify_exception + is_retriable
├── discover.py           # discover() / field_names() ─ Agent 自助发现
├── manifest.py           # build_manifest() ─ MCP 发现清单
├── server.py             # FastMCP v1 薄壳
├── vl_client.py          # video-localizer HTTP 封装
└── skills/
    ├── inventory_risk.py        # 纯计算：库销比风险分级
    ├── feishu_kb_search.py      # 纯计算：FAQ 检索
    ├── marketing_image_gen.py   # 纯计算：营销生图
    ├── localize_batch.py        # HTTP 依赖：批量本地化
    ├── localize_status.py       # HTTP 依赖：状态查询
    ├── localize_download.py     # HTTP 依赖：产物下载
    ├── localize_retry.py        # HTTP 依赖：失败重提
    └── localize_cancel.py       # HTTP 依赖：取消任务
```

## 看哪里找什么

| 我想…… | 看哪里 |
|---|---|
| 5 步 startup 完整流程 | **本文 Step 1–5** |
| 加一个新技能 | 触发 `flowmind-add-skill` 或读 `docs/skill-authoring-guide.md` |
| 把 SDK 接进 Claude Desktop / Cursor / OpenClaw | 触发 `flowmind-mcp-setup` 或读 `docs/agent-integration.md` |
| 处理一个 PR | 触发 `flowmind-handle-pr` |
| 测试 skill 端到端 | 触发 `flowmind-test-skill` 或读 `examples/<skill>_demo.py` |
| 看一个 skill 实际输出 | `make demo` 或 `uv run python examples/<skill>_demo.py` |
| 了解 MCP 暴露了哪些工具 | `flowmind-mcp-setup` |
| 改 SkillResult / ReasoningChain 等契约层 | 先读 `CLAUDE.md`「关键约定」+ 找 maintainer 确认 |
| 升级依赖 | `uv add <pkg>`（不要手改 pyproject.toml 后忘 uv sync） |
| 提交代码 | `<type>: <中文描述>`，type ∈ `feat/fix/docs/refactor/test/chore` |

## 第一步该跑的命令

```bash
make install     # 装依赖
make test        # 183 个测试必须全绿
make demo        # 看 8 个技能实际输出
cat CLAUDE.md    # 读项目不变量 + 测试铁律
```

## 关键约定速记

- **语言**：注释 / 文档字符串 / 日志 / 提交信息 → 中文；标识符 → 英文
- **错误永不静默**：失败经 `SkillResult(ok=False, error=...)` 或 `degraded=True`
- **两种失败契约**：纯计算类（`r.ok=False + r.error.code`）vs HTTP 依赖类（`r.ok=True + r.metrics.degraded + r.data.failure_category`）
- **`trace_id` 贯穿**每次调用
- **DSI 无动销时取 None**
- **TDD + 两层测试**：先 pytest，再 `flowmind-test-skill`

## 已知局限

- **入参 schema 多一层 `inp` 包装**（FastMCP v1 行为，升 v2 可去）
- **无 CI**：合并 PR 前本地跑 `make check` + `flowmind-test-skill`
- **远程依赖 mock**：marketing_image_gen 用确定性 mock 后端；5 个 localize_* 用 `vl_client.py` 调真实 video-localizer（生产前要起 VL 后端）
- **个性化配置走 `flowmind.config.toml`**（gitignored）

## 进一步

- 完整架构与不变量 → `CLAUDE.md`
- 加技能 7 步配方 → `flowmind-add-skill` / `docs/skill-authoring-guide.md`
- MCP 接入 → `flowmind-mcp-setup` / `docs/agent-integration.md`
- 处理 PR → `flowmind-handle-pr`
- 测试 skill → `flowmind-test-skill`