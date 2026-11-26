# rak-flowmind 绿地架构更新与迁移说明（冻结仓）

- 日期：2026-11-02
- 状态：冻结仓的目标边界说明；不授权实施
- 冻结声明：本文不解除维护冻结，不授权服务化、重构、契约/依赖修改、数据初始化、产物搬迁或集群变更
- 配套：`design.md`、`todo.md`

## 1. 更新实质

本次文档更新不是“把 systemd 常驻改成 K8s Deployment”，也不是“为 SaaS 并发目标放宽 GPU 串行”。它只做三件事：

1. 把当前真实运行事实写清：`:8002` 双通道、PG 任务真相、MQTT 通知、单卡串行、启动恢复、联邦双通道、host/集群部署差异。
2. 把与 go-kernel、CyberCut、平台运维和 K8s 寻址的边界写成可检查的约束。
3. 把未来可能实施的文件级范围、验证、回退和停止条件登记为待评审项，不写成已批准变更。

旧版文档把“稳定 K8s Service 暴露单活功能包”写得过近于目标事实。当前仓库没有 Python 功能包镜像或 K8s Deployment；当前事实是 rak-core 上 conda/systemd 常驻、go-kernel 通过 ConfigMap 静态地址调用。本文已校正这一表述。

## 2. 当前事实与目标候选

| 维度 | 当前事实 | 目标候选（未批准） | 边界 |
|---|---|---|---|
| 服务入口 | 单端口 `:8002`，`/mcp` + `/api/v1/tasks` | 保持同一入口与契约 | 不改 FastMCP v1、不改 wire、不改状态码 |
| 部署形态 | conda/systemd 常驻 rak-core，`0.0.0.0:8002`；无功能包 K8s 清单 | 稳定 Service 指向单活实例；replicas=1、workers=1 | 不写“已服务化”；镜像/清单/GPU 资源待核验 |
| 网关寻址 | go-kernel 通过 ConfigMap `FLOWMIND_BASE_URL` 指向 mesh IP 静态调用 | 静态 Service/CoreDNS 或受控配置 | MQTT 不是地址真源；网关扩容不增 GPU 并发 |
| GPU | P104-100 8GB；`workers=1` + `gpu_lane()` + 模型锁 | 保持单卡串行 | 不放宽并发、不增加 GPU worker、不做多进程 claim |
| 任务真相 | `flowmind.tasks`；状态机与 CAS | 保持 | MQTT 通知不覆盖终态；TTL 不删 DB 行 |
| 启动恢复 | running→interrupted；queued 重入队 | 保持 | 不得多进程同时初始化；不承诺断点续跑 |
| 联邦注册 | MQTT + PG 双通道；默认关；30s 心跳/90s stale | 保留契约 | 不强制迁 JetStream；不把 payload 地址当路由 |
| 数据 | 自有 `flowmind`；`mcp` 最小写；CyberCut 独立 schema | 服务私有边界保持 | 无跨 schema FK；无第二身份源 |
| 编辑审批 | 默认 backend=cybercut；flowmind 本地流水线可切回 | 保持现有审批语义 | 本地流水线实际只在 storyboard/pre_render 停靠，属实现差异 |
| 部署副本 | 一个 API 进程内含 MCP/REST/TaskManager | 单活 Service | 副本扩容被禁止，原因见 §4 |

## 3. 阶段更新

### G0：冻结、契约与部署事实

盘点维护冻结和历史例外，建立 SkillResult/manifest/MCP/REST/状态机/MQTT/PG/GPU/CyberCut 契约快照；只读核验 systemd unit、环境注入、GPU 映射、模型卷和静态网关地址。退出条件是每个后续项都能判断“是否越权、是否需要许可、回退是什么”。本阶段不修改运行系统。

### G1：逐项许可与隔离

由用户对最小切片的精确文件、配置、数据和部署范围作出明确许可，再准备空 PG、测试 MQTT、测试 Service 和非生产媒体。未获许可或隔离不成立时停止，不能以生态 G0-G7 总计划替代许可。

### G2：单活部署与 Agent 护栏

只设计单活 Service、GPU 调度、模型卷、探针和资源策略；把执行器、PG、GPU、模型卷、任务数据卷和正式产物排除在 L2 外。退出条件是任何自动扩缩/重启都不能破坏在途任务，可选依赖不会引发批量重启。

### G3：数据、状态机与通知验证

在空环境验证 flowmind/mcp 最小角色、PgBouncer、连接预算、任务 CAS、恢复语义、MQTT 降级、retain 终态与 PG 对账。任何 schema、topic、权限或状态机扩大都停止并回到许可评审。

### G4：单活功能包垂直切片

经许可部署或复用单活功能包，由 go-kernel 调用，执行 MCP/REST、并发提交、背压、取消、重试、GPU OOM、进程重启、双租户和路径安全测试。退出条件是单卡串行、恢复语义和租户边界均有真实证据，未出现第二个执行器。

### G5：编辑审批与 CyberCut 隔离

验证 `backend=flowmind` 与 `backend=cybercut` 的审批差异，回归 cross-dashboard、标准 MCP 客户端和现有 demo；验证 CyberCut 自己的 claim、Approval 和正式产物不被 FlowMind TTL/恢复破坏。发现契约变化即停止，不以消费者同步修改来掩盖未授权破坏。

### G6：正式部署门禁

只有 G2-G5 证据完整且用户另行明确批准正式服务化或 host 常驻变更、数据初始化、GPU 节点变更和产物搬迁后才可实施。生产只按批准清单切换单活形态，验收 MCP/REST、GPU 串行、恢复、审计、静态回退和旧版本回退；不夹带依赖升级或多副本实验。

### G7：持续复核

复核运行文档、实际 Service、契约快照、GPU 道闸、Agent 排除、数据 ownership 和 CyberCut 边界。文档校正不等于解冻；任何新功能、重构或退役仍按根 `AGENTS.md` 和用户许可流程处理。

## 4. 为什么不能直接扩副本

1. `TaskManager` 构造时对所有 `running` 执行全局恢复；第二个进程会把第一个进程的在途任务标为 `interrupted`。
2. `queued` 重入队没有跨进程 claim/租约/fencing；两个进程会重复执行同一批 queued 任务。
3. GPU 单卡预算只允许一条推理道闸；副本增加的是 OOM 与竞态，不是吞吐。
4. 模型缓存、TTL 工作目录、进程内单例和事件发布器均按单进程设计。
5. CyberCut 正式产物的搬运依赖单实例时序；多实例会放大重复搬运和误清理。

因此“目标 Service”的候选是单活 Service，不是多副本 Deployment；go-kernel 的多副本目标不能推导为 flowmind 多副本目标。

## 5. 兼容与回退

### 5.1 契约与消费者

- 保持 `/mcp`、`/api/v1/tasks`、manifest、工具全名、SkillResult、状态机、错误/degraded、MQTT topic/QoS/retain 和 federation payload 不变。
- FastMCP、go-sdk、torch/CUDA 或其他依赖升级必须独立评审和许可，不能与部署迁移组成一个不可回退变更。
- 认证只按 ECO-ADR-0011 既定形状演进；始终从已校验 Bearer 派生租户；不得为兼容旧客户端放行自填 `X-Rak-*` 或无凭证全量任务访问。
- 现有任务进度 MQTT 继续保留，不以 JetStream 迁移作为绿地准入条件。

### 5.2 执行与寻址

- 首选回退是恢复上一已验证的单活进程/镜像、原配置和静态 Service 后端，不是临时增加第二个 TaskManager 或 GPU worker。
- 多副本或 API/worker 拆分一旦造成重复恢复、重复执行或 GPU 并发，立即停止并返回单活。
- MQTT/PG 发现故障时使用最后受控配置或静态 Service；任务状态仍以 PG 为准，不能从 retained MQTT 消息重建并覆盖任务行。
- 进程重启后的 running→interrupted 和 queued 重入队是现有兼容语义；回退验证必须检查任务与产物，不只检查进程存活。

### 5.3 数据与产物

- 不在部署切换中修改 `flowmind.tasks`、`editing_projects`、`editing_assets`，不建立跨 schema FK，不把 CyberCut 表并入 flowmind；任何 DDL 需单独许可和向后兼容方案。
- 正式产物与 flowmind TTL 目录分离；回退不得删除 CyberCut 正式产物、数据库审计行或模型缓存；实际删除仍需逐项批准。
- 旧数据不备份、不迁移的生态裁决不清空现有任务/产物，也不授权初始化生产 schema；新生产数据保护另行验收。

## 6. 文档与 ADR 影响

| 文档/ADR | 影响 |
|---|---|
| 根 `AGENTS.md`、`CLAUDE.md` | 历史解冻/功能迭代表述不得被用来扩大本次范围；按仓尾维护冻结、具体 ADR 例外和用户逐项许可保守执行 |
| `README.md` | 现有 `:8002` 功能包、GPU 和 API 契约仍是事实；其中 go-kernel `:8080` 与当前 `:8090` 的漂移后续校正；demo 数量 18/22 待核验 |
| `docs/concurrency.md` | “提交/轮询可并发、GPU 执行串行”继续有效；不能用 SaaS 并发目标推翻 workers=1 |
| `docs/agent-video-editing.md` | 审批状态机和 edit_review 语义继续有效；本地流水线与配置声明的审批阶段差异需评审 |
| CyberCut ADR-0001 | FlowMind 仅提供公开 MCP 技能、trace 信封和兼容任务；核心契约与 TaskManager 边界继续有效 |
| CyberCut ADR-0002 | NodeAttempt 原子 claim、租约、独立 worker 和正式产物归 CyberCut；不得回退为 FlowMind 多进程 TaskManager |
| ECO-ADR-0011 | 特定鉴权冻结例外不等于永久解冻；Bearer、下游换签、身份头和 MCP wire 继续有效 |
| ECO-ADR-0013 | `flowmind`/`mcp` schema 分离和最小授权继续有效；统一物理库不扩大数据访问 |
| ECO-ADR-0021 | go-kernel 保持能力网关，flowmind 作为功能包；不要求功能包承接普通业务或 SDUI |
| ECO-ADR-0022 | 既有 MQTT 联邦/进度主题保留；JetStream 新事件目标不强制改写当前任务契约 |
| go-kernel greenfield 文档 | 其多副本仅作用于网关数据面，明确不构成 flowmind GPU 或服务副本扩容依据 |

## 7. 停止条件

未获逐项许可、出现第二个 GPU 执行者或 TaskManager、任务重复/误恢复、任何未批准契约/依赖/schema 变化、跨租户或路径越权、MQTT 被用作地址或任务真相、CyberCut 正式产物进入 TTL 区、生产文件存储回落未禁用，或 Agent 可自动操作 GPU/数据面时，必须停止。

不得通过关闭认证、放宽显存并发、删除任务行、修改消费者契约或把 host 常驻描述成 K8s 已服务化来绕过门禁。

## 8. 验收原则

- 当前事实、目标候选、未核验项和禁止项在同一文档中可区分。
- 每个实施项有精确文件/配置/数据/部署范围和回退；未许可差异为零。
- 单卡串行、任务 CAS、恢复语义、MQTT/PG 对账、租户 fail-closed 和 CyberCut 隔离有真实证据。
- 不以“文档齐全”“进程存活”“Pod Running”单独判定完成。
