# rak-flowmind 对外契约快照（P0）

> 依据：**ECO-ADR-0029 §2「契约不变量（重构不得破坏）」**。
> 阶段：P0（契约快照 + 冒烟基线）。状态：**快照已落盘、可被脚本发现**。
> 机器可读真源：[`contracts.snapshot.json`](./contracts.snapshot.json)（32KB，JSON）。
> 本文是人读说明，**不重复 JSON 的每个字段**，只回答「哪条不变量、真源在哪个文件的哪个字段、怎么改」。

## 为什么有这份快照

ADR-0029 把 rak-flowmind 从「单进程单卡」重构为 GPU 微服务集群（P1 镜像化 → P2 控制面/数据面
分离 → P3 按显存分组拆 worker → P4 多卡多节点）。**重构期对外契约冻结、消费方零改动**。
在没有快照的情况下拆 worker = 蒙眼改契约；本目录把 6 条不变量变成机器可读、**可定位到文件与字段**、
可被 CI/脚本比对的产物。

## 怎么用

```bash
# 比对：当前代码 vs 落盘快照；不一致退 1 并逐条打印「哪个字段变了」
PYTHONPATH=$PWD/src conda run -n flowmind python scripts/contract_check.py

# 有意变更契约后重生成（必须与消费方双侧同步 + 另立 ADR，见文末）
PYTHONPATH=$PWD/src conda run -n flowmind python scripts/contract_check.py --write

# 打印当前提取的快照（供 P1–P4 阶段间 diff）
PYTHONPATH=$PWD/src conda run -n flowmind python scripts/contract_check.py --json
```

冒烟基线入口是 [`../../scripts/smoke_examples.py`](../../scripts/smoke_examples.py)（`examples/*_demo.py`
的可重复执行入口，含 offline / needs-key / needs-gpu 分类与 known-fail 机制）。

## 6 条不变量 → 快照段 → 代码真源

| ADR-0029 §2 | 快照段 | 内容（当前基线） | 代码真源（改动落点） | 消费方 |
|---|---|---|---|---|
| ① MCP wire + 工具全名 `{prefix}__{tool}` + `tools/list`/`inputSchema` 形状 | `INV1_mcp_wire_and_tools` | **21 个工具**（localize 7 / dsp 8 / edit 5 / packaging 1）；每工具 id/name/version/category/tags/**input_props/required/output_props**；`{prefix}__{tool}`：缺省 prefix=`video_localizer`，dsp 独立 backend prefix=`dsp` | `src/flowmind/skills/__init__.py#FAMILIES`（族表=清单）<br>`src/flowmind/server.py#_make_tool`（tool 名=schema 来源）<br>`src/flowmind/config.py#FederationBackendConfig.prefix` | go-kernel 网关（聚合与路由）、cross-dashboard |
| ② `SkillResult` / `ReasoningChain` / `manifest` 响应形状 | `INV2_response_shapes` | 8 个契约模型字段集：`SkillResult`(8)/`SkillOutput`(6)/`ReasoningChain`(6)/`RuleHit`(4)/`Evidence`(5)/`ReliabilityMetrics`(5)/`TraceContext`(4)/`SkillError`(4)，各含 `required`；manifest entry 键集（id/name/version/description/category/tags/input_schema/output_schema/reliability_profile 及其内层键） | `src/flowmind/contracts.py#<模型名>`<br>`src/flowmind/discover.py#skill_entry`<br>`src/flowmind/manifest.py#build_manifest` | 所有 MCP 客户端 / Agent |
| ③ REST `/api/v1/tasks` 状态机、状态码、轮询语义 | `INV3_rest_tasks_contract` | **9 条路由**（path+methods+**status_codes**+**error 码**，逐条从处理函数体提取）；6 态状态机 + terminal/non-terminal；轮询行投影 12 键 + `include=args` 选择性回传；health 恒 200（`ok`/`degraded`，组件 pg/mqtt/milvus） | `src/flowmind/server_tasks.py`（含 `#health`）<br>`src/flowmind/server_rest.py`<br>`src/flowmind/tasks/__init__.py#STATUS_*`<br>`src/flowmind/tasks/store.py#_row_to_dict` | cross-dashboard BFF、go-kernel 透传 |
| ④ PG `flowmind.tasks` / `editing_projects` / `editing_assets` 列集（向前兼容，不删列） | `INV4_pg_tables` | `flowmind.tasks` **13 列** + 4 索引；`flowmind.editing_projects` **7 列**；`flowmind.editing_assets` **7 列** | `src/flowmind/tasks/store.py#_DDL`<br>`src/flowmind/editing/store.py#_DDL` | 自有 `flowmind` schema；任务真相 |
| ⑤ 联邦注册 MQTT topic/QoS/retain + `mcp.federation_backends` | `INV5_federation_contract` | topic `mcp-base-gpu/federation/{register,heartbeat,unregister}`，QoS **1/0/1**；任务事件 `mcp-base-gpu/tasks/{task_id}/events` QoS 1、**仅终态 retain**、payload 6 键（+可选 `steps`）；注册 payload 7 键；`mcp.federation_backends` **10 列** | `src/flowmind/federation/mqtt.py#TOPIC_*`<br>`src/flowmind/tasks/events.py#_TOPIC_PREFIX`<br>`src/flowmind/federation/pg.py#_DDL_BACKENDS`（与 go-kernel `internal/pgstore` **同 DDL，改动双侧同步**） | go-kernel 联邦注册表 + 进度消费 |
| ⑥ 消费方（go-kernel 联邦、cross-dashboard）零改动可用 | `INV6_consumers_zero_change` | 零改动项清单：工具全名 / 响应形状 / 提交-轮询-下载语义 / MQTT topic 与终态 retain；两个 backend（`video_localizer`:8002、`dsp`:8003） | `README.md`（MCP 客户端接入）<br>`docs/architecture/greenfield/design.md` §12 | go-kernel、cross-dashboard |

### 快照的机械性质（为什么可信）

- 全部从**代码提取**，不手写：工具/schema 走 `manifest.build_manifest()`，字段集走 pydantic
  `model_fields`，DDL 列集与路由状态码走源码正则。每条 record 自带 `locator`（`文件#符号`），
  差异天然可定位。
- 列表按集合比对（顺序不是契约），字段按名+类型比。
- 有意简化（`shortcut:` 注释已在脚本内标注）：**不连 PG、不起 HTTP 服务**验证运行时，
  **不比 `$defs` 逐字节**。升级触发见脚本头。

### 不在快照内（已知边界）

- **MCP JSON-RPC 帧本身**（wire 层）由 SDK 定义，靠 `pyproject.toml` 钉版 `mcp>=1.27,<2` 保护；ADR §16 明确 P0–P4 不升 FastMCP v2。
- 工具**执行语义**（返回内容、降级行为）——由 `examples/*_demo.py` 冒烟覆盖，不属快照。
- `inputSchema` 的 `$defs` 展开细节与 FastMCP v1 的「多一层 `inp`」行为（既有实况，见 `AGENTS.md` 调试表）。
- 部署面（systemd/conda 常驻、无镜像、无 K8s 清单）——P1 起另议。

## 改契约的正确顺序（ADR-0029：契约变更须另立 ADR）

1. 确认这是**有意**变更（不是漂移）→ 与消费方（go-kernel / cross-dashboard）**双侧同步**。
2. 另立 ADR 记录；P0–P2 阶段改动工具集为**明确排除项**（ADR §6）。
3. 改代码 → `python scripts/contract_check.py --write` 更新快照 → `--write` 的 diff 作为 ADR 附件。
4. 若只是重构（不得破坏契约）：`contract_check.py` 必须保持**绿色**，这是 P1–P4 的合并门禁。
