# FlowMind A2A 集成设计

> 日期：2026-08-26
> 状态：已批准
> 方案：A（a2a-sdk + LangGraph）

## 1. 背景与目标

### 当前状态
FlowMind 是 Python Skill SDK，通过 MCP（stdio + Streamable HTTP）暴露 14 个技能。现有架构是星型：所有 Agent → FlowMind。技能层（`@skill` + `invoke()`）是稳定的核心，对外契约是 `SkillResult[T]` 信封。

### 愿景
FlowMind 定位为**统一工具总线**，未来生态中存在多种形态的 Agent：Web 用户 Agent、机器人 Agent（具身智能）、小程序 Agent、网站 Agent。这些 Agent 之间需要 A2A（Agent-to-Agent）通信，FlowMind 作为能力节点接入 A2A 网络。

### 目标
1. FlowMind 自身成为一个 **A2A Agent**（发布 Agent Card，接收/响应 A2A Task）
2. 引入 **LLM 驱动的编排器**（LangGraph），把自然语言任务拆解为技能调用序列
3. 技能层**零改动**——A2A + 编排器是新增外层，包在现有 `invoke()` 外面

## 2. 设计决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 协议 | Google A2A（现成开放协议） | 互操作性，第三方 A2A Agent 可直接对接 |
| 实现 | 官方 `a2a-sdk`（Python） | 协议合规有保障，不造轮子 |
| 编排 | LangChain/LangGraph | 复杂工作流（条件分支、并行、重试）现成可用 |
| FlowMind 角色 | 自身 = 1 个 A2A Agent，技能不变 | 不引入多余 Agent，"子 agent"只是 LangGraph 工作流步骤 |
| 拓扑 | 混合：A2A 直连（轻量协商）+ FlowMind（工具调用） | FlowMind 是能力节点 + 信任锚点，不是消息路由器 |
| Agent Card | 分组暴露能力（内容/视频/数据分析） | 外部能路由决策，但不暴露底层技能细节 |
| 编排 LLM | 可配置，默认 LongCat | 复用现有 LLM 基础设施，可升级 |
| CoT | 按需暴露（默认结果，可拉推理链） | 节省带宽，调试/审计时可获取 |
| 做不了时 | 部分处理 + 报告做不了的 | 诚实默认，不做路由转发 |
| A2A 端点 | 同 HTTP 服务器新端点（8001 端口） | 单进程部署，运维简单 |

## 3. 架构

### 3.1 组件图

```
┌─────────────────────────────────────────────────────────┐
│  外部 A2A Agent（边缘 Web Agent / 机器人 Agent / ...）    │
└──────────────────────┬──────────────────────────────────┘
                       │ A2A Protocol (JSON-RPC over HTTP)
                       ▼
┌─────────────────────────────────────────────────────────┐
│  FlowMind A2A Agent（同端口 8001）                        │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │ A2A 协议层       │  │ LangGraph 编排器              │  │
│  │ (a2a-sdk)       │→│  LLM 规划 → 技能调用序列      │  │
│  │ Agent Card      │  │  CoT 生成                     │  │
│  │ Task 生命周期   │  └──────────┬───────────────────┘  │
│  └─────────────────┘             │                       │
│                                  ▼                       │
│                        ┌──────────────────┐              │
│                        │ 技能层（不变）    │              │
│                        │ @skill × 14      │              │
│                        │ invoke()         │              │
│                        └──────────────────┘              │
└─────────────────────────────────────────────────────────┘
                       │
                       │ MCP（同端口 /mcp，不变）
                       ▼
              现有 MCP Agent 消费者
```

### 3.2 数据流

1. **发现**：外部 Agent `GET /.well-known/agent.json` → 拿到 Agent Card（分组能力描述 + 端点 URL）
2. **委托**：外部 Agent `POST /a2a`（`tasks/send`）→ 自然语言目标 + 可选 skill 分组选择
3. **接收**：A2A 协议层接收 Task → 交给 LangGraph 编排器
4. **规划**：编排器调 LLM（LongCat）→ 拆解成技能调用序列（结构化 JSON 计划）
5. **执行**：逐步调用 `invoke(skill_id, params)` → 收集每步结果
6. **恢复**：单步失败时 Recovery 决策（重试一次 / 跳过 / 降级标记）
7. **汇总**：LLM 生成最终产物 + CoT 摘要
8. **返回**：通过 A2A 返回结果（按需附带完整 CoT）

### 3.3 关键原则
- **技能层零改动**：`@skill` 装饰器、`invoke()`、`SkillResult` 契约全部不变
- **开闭原则**：对扩展开放（新增 A2A 入口 + 编排层），对修改封闭（技能层不动）
- **复用**：`invoke()` 是技能执行唯一入口（MCP 和 A2A 共用），`ReasoningChain` 是 CoT 唯一格式

## 4. 模块设计

### 4.1 A2A 协议层（`src/flowmind/a2a/`）

```
a2a/
├── __init__.py
├── agent_card.py      # Agent Card 生成（分组能力描述）
├── server.py          # A2A Task 端点（tasks/send, tasks/get, tasks/sendSubscribe, tasks/cancel）
└── types.py           # A2A ↔ FlowMind 类型映射
```

**Agent Card**（`agent_card.py`）：
- 从注册表动态生成分组能力描述
- 分组映射：
  - `content` → content_idea_design, content_copywrite, content_audit, content_image_gen, content_hot_topics, content_crawler_suite, content_web_fetch, content_wechat_*, content_xhs_*, crawler_*, marketing_image_gen
  - `video` → localize_batch, localize_video, localize_status, localize_download, localize_retry, localize_cancel
  - `data` → inventory_risk, feishu_kb_search
- 暴露端点 URL、streaming 能力、认证方式

**Task 端点**（`server.py`）：
- 基于 `a2a-sdk` 的 `A2AServer` / `DefaultRequestHandler`
- Task 状态映射：A2A 的 `submitted → working → input-required → completed/failed/canceled` 直接透传
- `degraded` 处理：FlowMind 的 `degraded=True` 映射到 A2A `completed` + 标记字段

**类型映射**（`types.py`）：
- A2A `Task` → FlowMind 编排请求（提取 skill 分组 + 自然语言目标）
- 编排结果 → A2A `Task` 响应（产物 + 状态 + 可选 CoT）

### 4.2 编排器（`src/flowmind/orchestrator/`）

```
orchestrator/
├── __init__.py
├── graph.py           # LangGraph 图定义（Planner → Executor → Recovery → Summarizer）
├── planner.py         # LLM 规划节点（调 LLM 生成技能调用计划）
├── executor.py        # 执行节点（调 invoke()）
├── recovery.py        # 恢复决策（重试/跳过/降级）
├── summarizer.py      # 汇总节点（LLM 生成最终产物 + CoT）
└── prompts.py         # LLM prompt 模板
```

**图结构**（`graph.py`）：
```
Task 输入
   │
   ▼
[Planner Node] --LLM--> 技能调用计划 [{skill, params, reason}, ...]
   │
   ▼
[Executor Node] --invoke()--> 逐步执行
   │
   ├── 成功 → 下一步
   ├── 失败 → [Recovery Node]（重试/跳过/降级）
   │
   ▼
[Summarizer Node] --LLM--> 最终产物 + CoT 摘要
   │
   ▼
Task 输出
```

**Planner**（`planner.py`）：
- 输入：自然语言目标 + 可选 skill 分组
- 调用 LLM（复用 `_llm_client.llm_json`）生成结构化计划
- 输出格式：
  ```json
  {
    "steps": [
      {"skill": "content_idea_design", "input": {...}, "reason": "先定选题方向"},
      {"skill": "content_copywrite", "input": {...}, "reason": "基于选题写文案"}
    ],
    "cot": "用户需要一篇小红书，我计划先定选题再写文案..."
  }
  ```
- 约束：最多 `max_plan_steps` 步（默认 5），防止 LLM 规划过长

**Executor**（`executor.py`）：
- 调用 `invoke(skill_id, params)` 执行每步
- 收集 `SkillResult`（含 `reasoning`、`metrics`、`data`）
- 失败时抛错交给 Recovery

**Recovery**（`recovery.py`）：
- 决策逻辑：
  - `retriable=True`（transient 错误）→ 重试一次
  - `retriable=False` → 跳过该步 + 标记 degraded
  - 规划阶段失败 → 直接报错，不假装执行

**Summarizer**（`summarizer.py`）：
- 输入：所有步骤的执行结果
- 调用 LLM 生成最终产物 + CoT 摘要
- 输出：结构化结果（产物 + reasoning chain）

**Prompts**（`prompts.py`）：
- Planner system prompt：注入可用技能列表 + 分组描述 + 输出格式约束
- Summarizer system prompt：注入步骤结果 + 要求生成用户友好摘要

### 4.3 配置扩展（`config.py`）

新增 `OrchestratorConfig`：

```python
class OrchestratorConfig(BaseModel):
    """A2A 编排器配置。"""
    llm_key_env: str = "LONGCAT_API_KEY"      # 复用现有 LLM key
    llm_base_url: str = "https://api.longcat.chat/anthropic"
    llm_model: str = "LongCat-2.0"
    max_plan_steps: int = 5                   # 防止 LLM 规划过长
    max_retries_per_step: int = 1             # 单步重试次数
    enable_streaming: bool = True             # 是否推送实时状态更新
```

纳入 `FlowmindConfig`：
```python
class FlowmindConfig(BaseModel):
    ...
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
```

### 4.4 HTTP 服务器扩展（`server_http.py`）

- 复用现有 FastMCP 实例（8001 端口）
- 新增 A2A 端点：
  - `GET /.well-known/agent.json` → Agent Card
  - `POST /a2a` → A2A Task 处理（tasks/send, tasks/get, tasks/sendSubscribe, tasks/cancel）
- 实现方式：在 FastMCP 的 Starlette 应用上挂载额外路由（FastMCP v1 基于 Starlette）

### 4.5 新增依赖（`pyproject.toml`）

```toml
# A2A 协议层
"a2a-sdk>=0.1.0",
# 编排层
"langchain>=0.3",
"langchain-core>=0.3",
"langgraph>=0.2",
```

## 5. CoT 暴露机制

### 默认行为
- 只返回最终产物 + 简短摘要
- A2A Task 状态：`completed` + 产物数据

### 按需暴露
- A2A 请求带 `metadata.include_reasoning=true`
- 返回完整 CoT：
  - Planner 推理（为什么选这些技能、步骤顺序）
  - 每步执行结果（skill 调用 + 中间产物）
  - Summarizer 推理（如何汇总成最终产物）

### 数据结构
复用现有 `ReasoningChain`：
```python
ReasoningChain(
    conclusion="最终结论",
    triggered_rules=[RuleHit(...)],
    evidence=[Evidence(...)],
    causal_analysis="因果推理",
    risk_note="风险提示",
    confidence=0.95,
)
```

## 6. 错误处理

| 场景 | 行为 | A2A 状态 |
|------|------|----------|
| 单步技能失败（retriable） | 重试一次 | 保持 working |
| 单步技能失败（非 retriable） | 跳过 + degraded 标记 | completed + degraded |
| 任务完全无法处理 | 返回原因 | failed |
| 任务部分可完成 | 返回已完成部分 + "以下做不了" | completed + degraded |
| LLM 规划失败 | 直接报错 | failed |
| 请求参数无效 | 返回校验错误 | failed |

## 7. 测试策略

### 7.1 单元测试
- `agent_card.py`：分组映射正确性、Agent Card JSON 格式
- `planner.py`：LLM 输出解析、计划格式验证、`max_plan_steps` 截断
- `recovery.py`：重试/跳过/降级决策逻辑
- `types.py`：A2A ↔ FlowMind 类型双向映射

### 7.2 集成测试
- Mock LLM → 验证完整编排流（plan → execute → summarize）
- Mock `invoke()` → 验证 A2A 端到端（Task 提交 → 编排 → 返回）
- 错误注入 → 验证 Recovery 路径

### 7.3 A2A 协议合规
- 使用 `a2a-sdk` 的客户端测试库验证 Task 生命周期
- 验证 Agent Card 可被标准 A2A 客户端发现

### 7.4 端到端测试
- `flowmind-test-skill` 风格：A2A 入口 → 编排 → 技能 → 结果
- 新增 `examples/a2a_demo.py`：happy / 默认 / 错误三段式

## 8. 实现计划概览

> 注：详细实现计划由 writing-plans skill 生成。

### Phase 1：A2A 协议层
1. 新增 `a2a/` 模块（agent_card, server, types）
2. 配置扩展（OrchestratorConfig）
3. HTTP 服务器扩展（挂载 A2A 路由）
4. 测试：Agent Card 生成 + Task 端点基础功能

### Phase 2：编排器
1. 新增 `orchestrator/` 模块（graph, planner, executor, recovery, summarizer, prompts）
2. 接入现有 `invoke()` + `_llm_client.llm_json`
3. 测试：完整编排流（mock LLM）

### Phase 3：集成与端到端
1. A2A 协议层 ↔ 编排器对接
2. CoT 暴露机制
3. 端到端测试 + demo 脚本
4. 文档更新

## 9. 风险与缓解

| 风险 | 缓解 |
|------|------|
| `a2a-sdk` 版本不稳定 | 锁定版本范围，关注 changelog |
| LangGraph 学习曲线 | 先用最简图结构，逐步加复杂逻辑 |
| LLM 规划输出格式不可靠 | 严格 JSON schema 校验 + 重试 + 降级 |
| A2A 协议升级 | 依赖 `a2a-sdk` 抽象，升级 SDK 即可 |
| 编排 LLM 成本 | `max_plan_steps` 限制 + 可配置模型 |
