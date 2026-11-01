# rak-flowmind 绿地架构待办（冻结仓）

- 日期：2026-11-02
- 状态：仅登记未完成事项；不解除维护冻结，不构成实施许可
- 配套：`design.md`、`architecture-update.md`
- 规则：任务 ID 稳定；每项验收必须有可复核证据；未获逐项许可前只允许文档、只读核验和离线快照

## 1. 优先级与证据规则

- P0：冻结、契约、GPU 串行、身份和任务真相；P1：经许可的隔离验证；P2：持续治理。
- 证据以真实 demo/invoke、契约快照、GPU/重启故障、权限负例和 PG/MQTT 对账为准；不以文档齐全或进程存活单独判定完成。
- 依赖未满足或许可范围不明确时不得开始；生态 G0-G7 总计划不能替代本仓逐项许可。
- 文件级迁移、DDL、依赖升级、部署形态变化均必须在许可中列出精确文件、配置、数据和回退，不能以“实现绿地架构”概括授权。

## 2. FLOW-G0：冻结与基线

- [ ] `FLOW-G0-001`（P0；依赖：无）建立维护冻结与历史授权例外清单。验收：`AGENTS.md`、`CLAUDE.md`、ECO-ADR-0011/0013 的范围解释一致；明确历史“功能迭代期”表述不能扩大为概括解冻；每项未来工作标注“文档/只读/需许可”。
- [ ] `FLOW-G0-002`（P0；依赖：`FLOW-G0-001`）冻结契约快照：SkillResult、ReasoningChain、TraceContext、manifest、MCP `tools/list` 与 `inputSchema`（含 `inp` 包裹）、REST 状态码、任务状态常量、`flowmind.tasks` 列集、MQTT topic/QoS/retain、联邦 payload。验收：快照可定位到具体文件和字段，差异能被测试或脚本发现。
- [ ] `FLOW-G0-003`（P0；依赖：`FLOW-G0-001`）核验当前部署事实：rak-core 上是否存在 `flowmind-mcp` systemd user unit、监听地址、环境变量、GPU 设备映射、模型卷、数据卷、日志和重启策略。验收：只读证据齐全；无法核验项显式标“待核验”，不以 argocd ConfigMap 注释推断全部运行细节。
- [ ] `FLOW-G0-004`（P1；依赖：`FLOW-G0-002`、`FLOW-G0-003`）核验文档漂移：README 与 `:8080`/`:8090` 的历史表述、demo 数量 18 与 22 的差异、`EditingConfig.reviewable_stages` 与本地流水线 `_pending_gate` 的差异、`allow_file_store_fallback` 的生产默认风险。验收：形成漂移清单；只做文档校正时也不得改变运行契约。
- [ ] `FLOW-G0-005`（P0；依赖：无）建立 `flowmind`/`mcp`/`cybercut` 数据访问矩阵。验收：明确 `flowmind.tasks`、`editing_projects`、`editing_assets` 的 owner 与读写路径；`mcp` 仅最小联邦写；无跨 schema 外键；无对 `auth` 或产品表的业务直查。

## 3. FLOW-G1：逐项许可与隔离前置

- [ ] `FLOW-G1-001`（P0；依赖：`FLOW-G0-001` 至 `FLOW-G0-005`）为拟实施的最小切片取得逐项明确许可。验收：许可列出文件、配置、数据、部署、允许变更、明确排除项、回退方式和停止条件；本文档与生态计划均不计为许可。
- [ ] `FLOW-G1-002`（P0；依赖：`FLOW-G1-001`、生态 G1-002/004）准备独立空 PG、测试 MQTT、测试 Service 和非生产输入/产物。验收：无旧生产写凭据；无真实敏感媒体；单卡资源与静态回退可核验；不连接旧生产数据库、PVC 或对象存储。
- [ ] `FLOW-G1-003`（P1；依赖：`FLOW-G1-002`）核验功能包镜像与部署候选。验收：若走服务化，镜像构建、GPU 资源、模型卷、数据卷、探针、启动命令、许可证与回退均有证据；若继续 host 常驻，记录 systemd unit、环境注入和可重复重建方式。当前仓库无功能包 Dockerfile，镜像能力标“待核验”。

## 4. FLOW-G2：Agent 护栏与部署形态

- [ ] `FLOW-G2-001`（P0；依赖：`FLOW-G1-002`、生态 G2-005/006/007）将 GPU 执行、TaskManager、PG、模型卷、任务数据卷和正式产物列入运维 Agent L2 排除清单。验收：自动重启/扩缩/回滚均被拒绝；L1 只能经人工 PR；L3 有独立审批和审计。
- [ ] `FLOW-G2-002`（P1；依赖：`FLOW-G1-002`、生态 G2-003/004）设计单活 Service、GPU 调度、探针、资源和模型卷策略，但不得创建清单。验收：方案明确 replicas=1、workers=1、gpu_lane 不放宽；readiness 与 liveness 分离；可选 MQTT/Milvus 不触发批量重启；静态回退可用。
- [ ] `FLOW-G2-003`（P0；依赖：`FLOW-G2-001`）验证“多副本/多进程禁止”能被实际门禁表达。验收：Agent、CI 或运维检查能拒绝 HPA/KEDA、副本扩容、第二 worker、API/worker 拆分；说明该门禁本身不自动授权实施。

## 5. FLOW-G3：数据与通知

- [ ] `FLOW-G3-001`（P0；依赖：`FLOW-G1-002`、生态 G3-001/002）在空 schema 验证 flowmind/mcp 最小角色、PgBouncer 事务模式和连接预算。验收：不能访问 `auth`、CyberCut 或产品域；无 advisory lock、无服务端 prepared statement、无跨 schema FK；池等待超时有显式错误。
- [ ] `FLOW-G3-002`（P0；依赖：`FLOW-G1-002`、`FLOW-G3-001`）验证任务状态机与恢复。验收：queued→running CAS、终态 first-writer-wins、取消与完成并发、running→interrupted、queued 重入队、TTL 只删工作目录不删行；不宣称断点续跑。
- [ ] `FLOW-G3-003`（P0；依赖：`FLOW-G1-002`、`FLOW-G3-001`）验证 MQTT 通知与 PG 真相分离。验收：注册/进度失败降级、retain 终态、重复/丢失/乱序不改变 PG 终态；Service 寻址不采用 MQTT payload 地址；联邦 MQTT/PG 双通道失败时静态回退可用。
- [ ] `FLOW-G3-004`（P1；依赖：`FLOW-G3-001`）验证编辑项目存储回落边界。验收：生产配置禁用文件回落；开发回落时 `storage=file` 可见；PG 失败不静默伪装成功；项目与素材租户字段有效。

## 6. FLOW-G4：单活 MCP 功能包切片

- [ ] `FLOW-G4-001`（P0；依赖：`FLOW-G2-002`、`FLOW-G3-002`、`FLOW-G3-003`、生态 G4-006）经许可部署或复用单活功能包，并由 go-kernel 调用。验收：MCP tools/list、tools/call、REST、trace、静态 Service 回退通过；网关扩容未增加 GPU 并发；实际部署形态与许可一致。
- [ ] `FLOW-G4-002`（P0；依赖：`FLOW-G4-001`）执行并发提交、背压、取消、重试、终态 CAS、GPU OOM 和 TTL 演练。验收：任一时刻只有一个 GPU 执行任务；429/422/404 语义正确；OOM 失败显式；无第二 worker。
- [ ] `FLOW-G4-003`（P0；依赖：`FLOW-G4-001`）执行进程重启与恢复演练。验收：running→interrupted、queued 重入队、无多 TaskManager 互杀、无误称断点续跑；产物与 PG 行行为符合现有契约。
- [ ] `FLOW-G4-004`（P0；依赖：`FLOW-G4-001`）执行双租户、无凭证、身份头伪造、路径沙箱和日志脱敏测试。验收：跨租户与不存在同形；无凭证 fail-closed；下载 basename 白名单有效；degraded/错误不含内部敏感信息。

## 7. FLOW-G5：编辑审批与 CyberCut 边界

- [ ] `FLOW-G5-001`（P1；依赖：`FLOW-G4-001`）验证 `backend=flowmind` 与 `backend=cybercut` 的审批闸差异。验收：明确本地流水线实际只在 storyboard/pre_render 停靠；CyberCut 后端支持 copy/script/storyboard/pre_render 映射；审批等待不占 GPU；approve/revise/reject 与租户边界通过。
- [ ] `FLOW-G5-002`（P0；依赖：`FLOW-G4-002`）验证 CyberCut 自有 NodeAttempt claim、Approval 和正式产物隔离。验收：无跨 schema FK；`executor_ref` 仅软引用；FlowMind TTL 不清理 CyberCut 正式产物；FlowMind 不成为 CyberCut 默认执行器。
- [ ] `FLOW-G5-003`（P1；依赖：`FLOW-G4-001`）回归 cross-dashboard、标准 MCP 客户端和现有技能 demo。验收：工具全名、schema、SkillResult、REST 状态码、MQTT 通知和真实 backend 报告无未批准变化；demo 实际数量先核验。

## 8. FLOW-G6：正式部署门禁

- [ ] `FLOW-G6-001`（P0；依赖：FLOW-G2 至 FLOW-G5 全部、生态 G6-001/002）另行取得用户对正式服务化或 host 常驻变更、数据初始化、GPU 节点变更和产物搬迁的明确许可。验收：许可与操作清单逐项一致；冻结例外不扩大到未列内容。
- [ ] `FLOW-G6-002`（P0；依赖：`FLOW-G6-001`）按批准范围实施并验收生产单活功能包。验收：生产 MCP/REST、单卡串行、恢复、审计、静态回退和旧版本回退通过；无契约擅改；无未批准副本或依赖变化。
- [ ] `FLOW-G6-003`（P0；依赖：`FLOW-G6-002`）验证生产数据保护与审计。验收：`flowmind.tasks` 与编辑项目数据按批准策略保护；旧业务数据不迁移不自动清空；备份/恢复或明确的不备份边界有记录。

## 9. FLOW-G7：持续治理而非自动解冻

- [ ] `FLOW-G7-001`（P2；依赖：`FLOW-G6-002`）复核运行文档、历史端口/部署表述和实际 Service，但不据此解除冻结。验收：文档与 `:8002`/`:8090` 职责一致；host 或集群形态描述与事实一致；任何后续变更仍需明确许可。
- [ ] `FLOW-G7-002`（P2；依赖：`FLOW-G6-002`）定期复核契约快照、GPU 串行、Agent 排除、数据 ownership 和 CyberCut 边界。验收：无副本/依赖/契约漂移；未授权差异保持为零。
- [ ] `FLOW-G7-003`（P2；依赖：`FLOW-G7-002`）复核运维 Agent L2 排除是否仍被执行。验收：任何自动重启、扩缩、回滚或 GPU/数据面操作仍被拒绝并留审计。

## 10. 阻断项

| 阻断项 | 被阻断任务 | 解除证据 |
|---|---|---|
| 维护冻结未逐项解除 | 所有实施任务 | 用户对具体文件、配置、数据、部署和回退的明确许可 |
| 当前部署形态未核验 | `FLOW-G0-003`、`FLOW-G2-002`、`FLOW-G4-001` | systemd unit 或服务化镜像/清单的只读证据齐全 |
| 功能包无镜像且无 K8s 清单 | 服务化候选 | 镜像构建、GPU/卷/探针/回退方案通过评审 |
| 多进程恢复与 claim 未解决 | 任何副本扩容 | 不得以扩容绕过；必须另行设计与许可，本文不提供 |
| 编辑审批实现与配置声明不一致 | `FLOW-G5-001` | 差异经评审并决定“文档校正”或“契约变更”，不得静默修改 |
| demo 数量与端口文档漂移 | `FLOW-G0-004`、`FLOW-G5-003` | 实际清单与运行证据核对完成 |
| 生产文件存储回落默认开启 | `FLOW-G3-004`、`FLOW-G6-002` | 生产显式禁用并有配置/测试证据 |
| PG/MQTT/GPU 故障语义未演练 | 正式部署 | `FLOW-G3`、`FLOW-G4` 证据齐全 |
