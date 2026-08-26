# FlowMind Skill SDK

对任意 Agent 友好的技能底座。技能一次定义即成为 MCP 工具，
自带四段式因果推理链、可靠性指标与 trace_id，并可由终端用户对话初始化个性化配置。

---

## 🤖 如果你是一个 AI Agent 第一次读到这个文件 —— STOP

**你是被用户部署到本项目的 Agent（Claude Code / Cursor / Cline / OpenClaw / 任何 MCP 客户端）。**

不要立刻跑 `uv run` / `make install` / 写代码。本节是**给你看的部署协议**，按下面的顺序与用户对话，再执行。

### 协议第 0 步：用一句话自我介绍并告诉用户你要做什么

> "Hi，我是 [agent name]。我刚拿到了 FlowMind Skill SDK —— 这是给 AI Agent 用的 Python 技能框架。我会帮你部署 + 跑通验证 + 装好 MCP（你既然发给我，就是要用 MCP），再问 1 个项目偏好。准备好了吗？"

### 协议第 1 步：通过对话采集 1 件事

**只问 1 个**（别再问 Agent 平台 —— 用户发给你就说明选了你；别问要不要 MCP —— Agent 接 SDK 就必须装）。一次问完，不要拆。

> **"你的项目主要场景是哪个？这决定要不要配视频本地化偏好。"**

| 选项 | 走法 |
|---|---|
| **A. 视频本地化（5 个 localize_*）** | 调 `run_interactive_init(ask_fn=...)` 让 SDK 引导用户答 9 个偏好 |
| **B. 库存 / FAQ / 营销 / 综合** | 直接走 demo 即可，不需要 init config |

### 协议第 2 步：自动跑（不问）

#### A. 视频本地化分支

```python
from flowmind.interactive import run_interactive_init

# Agent 驱动：ask_fn 由你实现（把每个 prompt 转成对用户的对话问题）
# 默认实现：用户按 Enter 接受默认
cfg = run_interactive_init(ask_fn=my_chat_ask_fn)
# → 9 个问题逐个问（目标语言 / 源语言 / TTS / 字号 / 位置 / 文件后缀）
# → 用户每题直接按 Enter 用默认也行
# → 自动写 flowmind.config.toml + reload
```

#### B. 其他分支

什么都不用配 —— `inventory_risk` / `feishu_kb_search` / `marketing_image_gen` 全部带通用默认，直接用。

### 协议第 3 步：装依赖 + 跑测试 + 跑 demos（不问）

```bash
# 1. 装依赖
uv sync --extra dev

# 2. 跑 14 个技能的全部测试（必须全绿）
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -p asyncio

# 3. 跑全部 demo 看真实输出
for f in examples/*_demo.py; do
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python "$f"
done

# 4. 发现能力（schema 驱动，避免猜字段）
uv run python -c "
from flowmind import discover
for s in discover():
    print(s['id'], '-', s['description'])
"
```

任何一步报错 → 读 CLAUDE.md「关键约定」段排查。

### 协议第 4 步：起 MCP server（必装，不问）

用户既然发给你这个 Agent，**就是要走 MCP**。直接装：

```bash
# 后台启动（stdio 传输）
nohup uv run flowmind-mcp > /tmp/flowmind-mcp.log 2>&1 &
```

**自动检测 Agent 平台并写配置**（不要问）：

```python
# 检测当前 Agent 平台
import os
from pathlib import Path

home = Path.home()
if (home / ".claude").exists():
    # Claude Code / Claude Desktop
    config_path = home / ".claude" / "mcp_servers.json"   # 路径按平台调整
elif (home / ".cursor").exists():
    config_path = home / ".cursor" / "mcp.json"
elif (home / ".config" / "cline" / "cline_mcp_settings.json").exists():
    config_path = home / ".config" / "cline" / "cline_mcp_settings.json"
else:
    config_path = None  # 让用户手动粘贴 stdio config

mcp_config = {
    "mcpServers": {
        "flowmind": {
            "command": "uv",
            "args": ["--directory", str(Path(__file__).parent), "run", "flowmind-mcp"],
        }
    }
}
```

把这段 JSON 写到对应路径（或显示给用户让他们粘贴）。

### 协议第 5 步：交付摘要

> "部署完成。你现在可以：
> - **MCP 工具**：在我（[agent name]）里能看到 14 个工具 —— `inventory_risk` / `feishu_kb_search` / `marketing_image_gen` / 5 个 `localize_*` / 5 个 `content_*`
> - **直接 Python**：`from flowmind import invoke, discover` 就能用
> - **继续开发**：让我帮你加新技能（触发 flowmind-add-skill）或处理 PR（flowmind-handle-pr）
>
> 任何问题直接问我。"

---

## 人类用户视角（人类读 README，不是 Agent）

如果你**不是** Agent 在读这段，跳过上面 🤖 段。往下看：

## 安装

```bash
uv sync --extra dev
```

## 作为 MCP 服务器运行

```bash
uv run flowmind-mcp   # stdio 传输
```

## 直接调用（非 MCP）

```python
import flowmind.skills  # 触发技能注册
from flowmind.skill import invoke

result = invoke("inventory_risk", {"items": [
    {"sku": "A", "on_hand": 100, "unit_cost": 2.0, "sales_30d": 60},
]})
print(result.ok, result.data.summary, result.reasoning)
```

## 可用技能

- `inventory_risk` —— 库销比/库存风险分析。输入 SKU 列表，输出风险分级、
  资金占用汇总与四段式推理链。完整清单见 `python -c "import flowmind.skills, json; from flowmind.manifest import build_manifest; print(json.dumps(build_manifest(), ensure_ascii=False, indent=2))"`。
- `feishu_kb_search` —— 飞书 FAQ 检索
- `marketing_image_gen` —— 营销生图（多平台/多风格）
- `localize_batch` / `localize_status` / `localize_cancel` / `localize_download` / `localize_retry` —— 视频本地化 5 步编排
- `content_idea_design` —— 内容选题思路设计（xhs/wechat/douyin）
- `content_copywrite` —— 平台化文案生成（小红书种草 / 公众号长文 / 抖音口播）
- `content_hot_topics` —— 平台热点雷达（聚合热榜，API 不可达降级种子）
- `content_audit` —— 平台规则审计（规则库扫描 + LLM 复核）
- `content_image_gen` —— 平台比例 AI 配图（3:4 / 16:9 / 9:16）

## Web / HTTP 消费（flowmind-mcp-http）

MCP 默认 stdio；供 Web 前端（如 cross-dashboard）等 HTTP 客户端消费时，
以 **Streamable HTTP** 传输启动（技能与密钥全部留在 flowmind，客户端零密钥）：

```bash
# 后台启动（Streamable HTTP，默认 http://127.0.0.1:8001/mcp）
FLOWMIND_MCP_HOST=127.0.0.1 FLOWMIND_MCP_PORT=8001 nohup uv run flowmind-mcp-http > /tmp/flowmind-mcp-http.log 2>&1 &
```

## 给 Agent 的初始化剧本（AGENT INIT PLAYBOOK）

> 本节面向**装载此包的 Agent**。首次为某用户使用本 SDK 前，请执行以下步骤，
> 通过对话把技能阈值调整为「用户自己的默认」。不要让开发者改代码——
> 一切定制都写入 `flowmind.config.toml`。

**步骤 1：检查是否已初始化**

```python
from flowmind.config import is_initialized
if not is_initialized():
    ...  # 执行步骤 2、3
```

**步骤 2：与用户对话采集业务画像**

依次询问（用户不答则用括号内通用默认）：
- 经营类目与结算货币？（`currency="USD"`）
- 期望的健康周转天数上限？（`dsi_healthy_max=60`）
- 关注/预警的周转天数分界？（`dsi_watch_max=90`、`dsi_warn_max=120`）
- 视为「断货风险」的低周转天数？（`dsi_low=15`）
- 单 SKU「资金占用过高」的金额阈值？（`capital_high=100000`）

**步骤 3：写入用户专属配置**

```python
from flowmind.config import FlowmindConfig, InventoryConfig, save_config
save_config(FlowmindConfig(inventory=InventoryConfig(
    currency="CNY",          # ← 用步骤 2 采集到的值替换
    dsi_healthy_max=45,
    dsi_watch_max=75,
    dsi_warn_max=100,
    dsi_low=10,
    capital_high=80000,
)))
```

此后所有技能调用将自动采用用户设定的阈值（`flowmind.config.toml`），
未设定项回落通用默认。用户可随时要求重新初始化以调整。

**或者用对话式向导（推荐）**：

```bash
uv run flowmind-init      # CLI 9 步向导
```

或 Agent 调：

```python
from flowmind.interactive import run_interactive_init
cfg = run_interactive_init(ask_fn=my_llm_ask)  # Agent 驱动对话
```