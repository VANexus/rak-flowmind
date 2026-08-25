# 贡献指南（CONTRIBUTING）

欢迎为 **FlowMind Skill SDK** 贡献技能与改进！本项目的核心设计目标是：**新增一个技能 = 写一个 `@skill` 函数**，无需改动 server / 契约层 / 初始化骨架。

## 快速开始

```bash
uv sync --extra dev          # 安装运行时 + 开发依赖
uv run pytest                # 全量测试
uv run ruff check src tests  # lint
```

## 核心约定（务必遵守）

- **语言**：注释、文档字符串、日志、提交信息用**中文**；变量/函数/类名用**英文**。
- **提交格式**：`<type>: <中文描述>`，type ∈ `feat`/`fix`/`docs`/`refactor`/`test`/`chore`。
- **错误永不静默**：任何失败都通过 `SkillResult(ok=False, error=...)` 或 `degraded=True` 返回结构化结果，绝不吞异常、绝不返回半成品。
- **可自定义项只经 config 暴露**：不要把「某个用户的默认值」硬编码进代码。阈值等放进技能的 config 模型（带通用默认），由终端用户对话初始化覆盖。
- **TDD**：先写失败测试（RED），再实现（GREEN），小步提交。

## 如何新增一个技能

### 1. 建技能文件 `src/flowmind/skills/<your_skill>.py`

```python
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.contracts import Evidence, ReasoningChain, SkillOutput
from flowmind.rules import Rule, evaluate_rules
from flowmind.skill import skill


class MyInput(BaseModel):
    """技能入参（pydantic 校验 → 自动 MCP inputSchema）。"""
    value: float = Field(ge=0)


class MyReport(BaseModel):
    """技能业务载荷。"""
    verdict: str


@skill(id="my_skill", name="我的技能", version="0.1.0")
def my_skill(inp: MyInput) -> SkillOutput[MyReport]:
    """一句话说明这个技能做什么。"""
    # 1) 计算指标  2) 用 rules 求值 → 自动得到 触发规则 + 数据证据
    # 3) 组装四段式因果推理链  4) 返回 SkillOutput（框架会套上 SkillResult 信封）
    chain = ReasoningChain(
        conclusion="结论",
        causal_analysis="因果推理",
        risk_note="风险提示",
    )
    return SkillOutput(data=MyReport(verdict="ok"), reasoning=[chain], sample_size=1)
```

> **要点**：技能函数返回轻量 `SkillOutput`（业务数据 + 推理链）。框架的 `invoke()` 会统一注入 `trace_id`、计时填充可靠性指标、并把异常兜底为结构化错误 —— 你**不用**自己写这些管道。四段式推理链的「触发规则/数据证据」两段应由 `rules.evaluate_rules()` 自动产出，不要手写。

### 2. 注册（触发 `@skill`）

在 `src/flowmind/skills/__init__.py` 追加一行：

```python
from flowmind.skills import your_skill  # noqa: F401
```

就这样 —— `server.py`（MCP 暴露）、`manifest.py`（能力清单）会**自动**发现并暴露它，无需改动。

### 3. 可自定义阈值 → 放进 config

若技能有可调参数，在 `src/flowmind/config.py` 加一个配置模型（带通用默认），并纳入 `FlowmindConfig`；在技能里用 `load_config()` 读取。个性化由终端用户对话初始化（见 `README.md` 的「Agent 初始化剧本」）写入 `flowmind.config.toml`。

### 4. 写测试

在 `tests/test_<your_skill>.py` 里，优先通过 `invoke("<id>", args)` 做端到端断言：验证 `ok`、业务数据、四段式链四要素齐全、以及边界（非法入参 → `VALIDATION` 结构化错误）。

## 提交 PR 前的检查清单

- [ ] `uv run pytest` 全绿，输出干净（无告警）
- [ ] `uv run ruff check src tests` 通过
- [ ] 新技能已在 `skills/__init__.py` 注册
- [ ] 无 `# TODO` 遗留给下游开发者；可调项已走 config
- [ ] 提交信息符合 `<type>: <中文描述>`

## 对外契约（供任意 Agent 消费）

每次技能调用返回统一的 `SkillResult` 信封：`ok` / `skill` / `version` / `trace`（含 `trace_id`）/ `data`（业务载荷）/ `reasoning`（四段式因果推理链）/ `metrics`（可靠性指标）/ `error`（结构化错误）。这正是「对 Agent 友好」的来源 —— 保持它稳定。
