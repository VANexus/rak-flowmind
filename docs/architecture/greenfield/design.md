# rak-flowmind 绿地目标架构设计（冻结仓）

- 日期：2026-11-02
- 状态：冻结仓的目标边界说明；仅文档；不解除维护冻结，不构成代码、配置、契约、数据或部署变更许可
- 裁决依据：ECO-ADR-0011、0013、0020、0021、0022，仓内 CyberCut ADR-0001、ADR-0002
- 核验范围：本仓源码、脚本、示例、迁移和文档；`argocd-apps` 中与 flowmind 相关的只读部署引用
- 表述纪律：本文中的“目标”“候选”“建议”均不是已批准变更；任何实施项必须取得用户对具体范围的逐项许可

## 1. 文档定位

本文回答三个问题，且只回答这三个问题：

1. 在冻结纪律下，rak-flowmind 当前真实运行语义和不可变契约是什么。
2. 与 go-kernel、CyberCut、平台运维和 K8s 寻址的边界在哪里。
3. 若未来获准实施，哪些文件、契约和部署面会受影响，以及最小验证和回退是什么。

本文不替代 `README.md`、`AGENTS.md`、`CLAUDE.md`、`docs/concurrency.md`、`docs/agent-video-editing.md` 和 CyberCut ADR；这些文件仍是仓内事实和契约的来源。

## 2. 冻结边界

维护冻结是当前事实。`AGENTS.md` 顶部的历史功能迭代表述、ECO-ADR-0011/0013 中的特定例外，均不能被解释为持续、概括或自动解冻。

允许的当前工作：

- 只读核验源码、契约快照、示例、迁移和运行文档。
- 编写或修订本文档类内容，明确标注目标、候选和未核验项。
- 为未来评审准备文件级范围、风险、验证和回退清单。

必须另行取得逐项许可的工作：

- 修改 `contracts.py`、`skill.py`、`manifest.py`、`server*.py`、`auth.py`、`tasks/*`、`skills/__init__.py` 的既有语义。
- 新增或删除公开技能、改变工具全名、prefix、MCP wire、REST 状态码、SkillResult、ReasoningChain 或 manifest 形状。
- 修改 `flowmind.tasks`、`flowmind.editing_projects`、`flowmind.editing_assets` 的 DDL、索引、状态机或 TTL 语义。
- 修改 MQTT topic、QoS、retain、联邦 payload 或 PG `mcp.federation_backends` 的契约。
- 修改 GPU 并发、`workers=1`、`gpu_lane()`、模型缓存锁、依赖钉版或 CUDA 版本。
- 把进程拆成 API/worker、增加副本、增加 HPA/KEDA、增加第二 TaskManager 或第二 GPU 执行者。
- 初始化、清空或迁移任务数据，删除模型缓存、工作目录或任何正式产物。

## 3. 当前事实基线

| 项目 | 已核验事实 | 证据 |
|---|---|---|
| 服务入口 | 单端口 `:8002`，同时提供 `/mcp` Streamable HTTP 与 `/api/v1/tasks` 长任务 REST | `server_http.py`、`server_tasks.py` |
| MCP 协议 | FastMCP v1；工具经网关聚合为 `{prefix}__{tool}`，如 `video_localizer__*`、`dsp__*` | `server.py`、`README.md` |
| 任务状态 | `queued`、`running`、`succeeded`、`failed`、`cancelled`、`interrupted` | `tasks/__init__.py` |
| 任务真相 | PostgreSQL `flowmind.tasks` 是任务持久化真相 | `tasks/store.py` |
| 进度通知 | MQTT `mcp-base-gpu/tasks/{task_id}/events`；终态 retain；失败降级为纯落库 | `tasks/events.py` |
| 执行并发 | `TaskManager workers=1`，叠加 `gpu_lane()` 信号量；模型缓存初始化另有双检锁 | `tasks/manager.py`、`tasks/gpu.py` |
| 启动恢复 | 遗留 `running` 标 `interrupted`；遗留 `queued` 重新入队；流水线不可断点续跑 | `manager.py`、`store.py` |
| GPU 硬件 | P104-100 8GB、Pascal 6.1；显存预算约 7.5G；torch 必须保持 cu126 钉版 | `README.md`、`AGENTS.md` |
| 联邦注册 | MQTT + PG 双通道，默认关闭，`FLOWMIND_FEDERATION_REGISTER=1` 开启 | `federation/*` |
| 数据域 | 自有 `flowmind` schema；联邦注册仅最小写 `mcp` schema | ECO-ADR-0013、`store.py` |
| 编辑后端 | `EditingConfig.backend` 默认 `cybercut`，可选 `flowmind` | `config.py` |
| 当前部署 | Python 功能包以 conda/systemd 常驻 rak-core，监听 `0.0.0.0:8002`；go-kernel 通过 ConfigMap 静态地址调用 | `argocd-apps/apps/go-kernel/rak-flowmind-backend.yaml` 注释 |

待核验项：

- 集群内是否真实存在名为 `flowmind-mcp` 的 systemd user unit，以及其环境变量、GPU 设备映射和模型卷挂载方式。仓库只提供部署引用和注释，未提供该 unit 文件。
- 本仓没有 Python 功能包的容器镜像构建文件；仅 `deploy/ocr-sidecar/` 有 Dockerfile。任何“迁入 K8s Deployment”的设想都尚未核验镜像、探针和 GPU 资源声明。
- `README.md` 验证段写 18 个 demo，`AGENTS.md` 写 22 个 demo；实际可执行 demo 数量待核验。
- `README.md` 与部分历史文档仍写 go-kernel `:8080`，当前集群清单使用 `:8090`；以 `argocd-apps` 和端口裁决定义为准，本地文档漂移待后续校正。

## 4. 职责边界

| 组件 | 负责 | 明确不负责 |
|---|---|---|
| rak-flowmind | 已登记技能实现、`@skill` 注册、JSON schema、manifest、SkillResult/trace 信封；`/mcp` 轻技能；`/api/v1/tasks` 提交、查询、取消、重试、下载与背压；单卡任务串行、恢复、TTL；联邦注册与任务通知；自有 `flowmind` 数据域 | 第二网关、普通业务 BFF、工作流引擎、DAG 编排、多进程 GPU 调度、身份权威、长期事件日志 |
| go-kernel | MCP 联邦、工具命名空间、认证派生、Service 路由、限流/熔断、用量归因 | 不承接普通业务聚合；不把 MQTT 当作地址真源；不因网关扩容改变 GPU 并发 |
| rak-auth | 用户、租户、scope、凭证和身份主数据权威 | 不承担任务状态、技能注册或 GPU 执行 |
| CyberCut | 自有 Project/Run/NodeRun/NodeAttempt、claim、审批、局部重跑、正式产物、REST/SSE 控制面 | 不复用 `flowmind.tasks` 作为 NodeRun；不把 TaskManager 当作默认执行器 |
| K8s Service/CoreDNS | 网关到功能包的实际请求寻址 | 不存储任务状态；不让 MQTT payload 成为唯一寻址依据 |
| EMQX/MQTT | 联邦注册、心跳、注销和任务进度通知 | 不承担 MCP tools/call、服务地址真源、任务状态真相或领域事件日志 |

## 5. 双通道 API 与入站身份

`/mcp` 与 `/api/v1/tasks` 共用同一认证边界和同一进程：

- 入站只信任已校验 `Authorization: Bearer` 派生的 subject/tenant；`X-Rak-Tenant`、`X-Rak-Principal` 只作诊断，不决定任务归属。
- 鉴权支持 JWT（`FLOWMIND_KERNEL_JWKS_URL`）和 HMAC（`FLOWMIND_AUTH_SECRET`）两条路径，先验签再解载荷。
- 启用鉴权后，无有效凭证必须 fail-closed；跨租户与不存在任务保持同形返回，避免任务枚举。
- `/api/v1/health`、`/api/v1/manifest` 的匿名豁免只暴露健康与能力元数据，不暴露任务参数、产物或租户信息。
- 生产不得使用空认证密钥的 dev 降级；密钥只经受控环境或 Secret 注入，不进入 toml、镜像、日志或 Git。

REST 语义保持现有契约：

| 端点 | 语义 |
|---|---|
| `POST /api/v1/tasks` | 受理返回 `202`；队列水位满返回 `429`；入参或步骤选择非法返回 `422` |
| `GET /api/v1/tasks/{task_id}` | 查询任务；未知或跨租户返回 `404` |
| `DELETE /api/v1/tasks/{task_id}` | 取消；已终态返回 `409` |
| `POST /api/v1/tasks/{task_id}/retry` | 复制原 args 创建新任务；原失败/取消/中断可重提，成功不重提 |
| `GET /api/v1/tasks/{task_id}/download` | 只允许该任务 `output_paths` 的 basename 白名单，禁止路径穿越 |
| `GET /api/v1/health` | 版本与组件状态；PG 是硬依赖，MQTT 未配置显示 disabled |

## 6. 任务状态机与 CAS

状态转换：

```text
queued -> running -> succeeded
                  -> failed
                  -> cancelled
running/queued --进程重启--> interrupted
queued --重新入队--> running
```

关键语义：

1. 提交准入使用单语句软水位：全局 `max_pending_tasks` 与单租户 `max_pending_per_tenant`。READ COMMITTED 下允许 ±N 的软性超越，背压由队列和单 worker 串行提供。
2. `queued -> running` 使用 CAS 守卫；取消可能在取走与置 running 之间生效，worker 必须让位，不能让取消任务复活。
3. 终态 first-writer-wins；cancel 与 worker 完成并发时，后写方 CAS 返回 false 并幂等让位，不覆盖已落终态。
4. 失败分类：`ok=False` 或技能级 `failure_category` 按业务失败落 `failed`；`degraded` 但无 `failure_category` 可按现有语义落 `succeeded`。
5. 重试不是恢复原任务：它复制原 `args_json`，创建新 `task_id`，原行保留供审计。
6. 任务行保留；TTL 只删 `data_dir/tasks/<id>/` 与 `data_dir/outputs/<id>/`，不删除数据库行。

## 7. GPU 单卡串行与 TaskManager 恢复

### 7.1 双道闸

- 第一道：TaskManager `workers=1`，任务间串行。
- 第二道：`gpu_lane()` 为 `Semaphore(1)`，覆盖同步 invoke 直连与任务线程并发的 GPU 推理阶段。
- 第三道：`model_cache_guard()` 防止并发首次加载同一模型。
- 云 ASR/TTS、纯网络阶段不占 GPU 槽；本地 ASR、LaMa 擦除、本地 TTS 必须进入 GPU 槽。
- 显存预算约 7.5G；任何放宽 `workers`、信号量、模型缓存锁或并行加载都可能 OOM，不属于可调优项。

### 7.2 启动恢复

- 进程启动无条件执行 `recover_running()`：把该库所有 `running` 任务标为 `interrupted`。
- 随后 `_requeue_orphans()` 读取 `queued` 行并重新投递 worker；行不重新 INSERT。
- 由于恢复是对全局任务表的操作，且没有可靠的多进程 claim/fencing，多个进程同时初始化会互相把在途任务标为中断，并可能重复执行 queued 任务。
- 流水线不可断点续跑；Pod/进程重启后不得宣称任务无中断或媒体流水线可恢复。

### 7.3 单进程约束

- 默认且唯一安全的运行形态是单活进程：一个 API 进程内包含 MCP、REST、TaskManager、GC 和事件发布器。
- 不允许 API 多副本、standby、API/worker 拆分、独立第二 worker 或按技能族拆多个共享同一 GPU 的执行进程。
- 任何多进程方案必须先解决全局恢复副作用、任务 claim、租约、fencing、单卡互斥和产物归属，并另行取得用户许可；本文不提供该方案。

## 8. editing 审批闸

### 8.1 两种后端

- `EditingConfig.backend` 默认 `cybercut`；`flowmind` 为显式切回的本地流水线后端。
- 默认审批阶段 `default_review_stages=["storyboard"]`；`reviewable_stages=["copy","script","storyboard","pre_render"]`。
- `review_stages=[]` 表示全自动；`None` 使用配置默认。

### 8.2 flowmind 本地流水线

阶段顺序：

```text
understand -> copy -> script -> storyboard -> render
```

- 阶段产物落库后，若该闸在 `review_stages` 且未在 `approved`，项目置 `awaiting_review` 并正常结束本次任务，不占用 GPU worker。
- 当前 `_pending_gate` 实际只在 `storyboard` 完成后停靠 `storyboard` 或 `pre_render`；`copy`、`script` 虽在 `reviewable_stages`，但本地流水线当前未在此停靠。这是实现与配置声明的差异，标记待核验，不在本文授权修改。
- `edit_review` 动作：
  - `approve`：把闸记入 `approved`，提交新的 `edit_pipeline` 续跑任务。
  - `revise`：`feedback` 必填；非 `pre_render` 闸把该阶段置 `pending` 并注入 `_feedback`；`pre_render` 只记档并续跑渲染。
  - `reject`：项目落 `failed`，`error=review_rejected[...]`，不续跑。
- 只有 `awaiting_review` 且 `project.stage` 与请求闸一致的项目可动作；跨租户与不存在项目保持同形错误。
- 审批续跑是新任务，复用任务引擎的进度、取消和事件语义；审批等待本身不占 GPU。

### 8.3 CyberCut 后端

- `flowmind/editing/cybercut_backend.py` 把 `copy/script/storyboard/pre_render` 映射到 CyberCut 的 `creative.brief`、`creative.script`、`creative.storyboard`、`edit.plan`。
- 审批由 CyberCut `cybercut.approvals` 记录：`expected_revision` 是乐观并发守卫，基于旧产物的过期决定会冲突。
- `approve` 调 CyberCut `orchestrator.approve`；`revise` 调 `orchestrator.reject` 并创建新尝试；`reject` 请求取消 Run。
- FlowMind 的 `ProjectRecord` 是对外投影，CyberCut 的 Project/Run/NodeRun/NodeAttempt/ArtifactRevision 仍归 CyberCut 所有。

## 9. 数据所有权与存储

| 对象 | 所有者 | 访问边界 |
|---|---|---|
| `flowmind.tasks` | rak-flowmind | 任务增删查改、进度、步骤快照、终态；其他服务不得直接读写 |
| `flowmind.editing_projects` | rak-flowmind | 剪辑项目聚合根；tenant 字段隔离 |
| `flowmind.editing_assets` | rak-flowmind | 素材库索引；tenant 字段隔离 |
| `mcp.federation_backends` | go-kernel 注册表；flowmind 最小写 | 功能包只写自身注册所需的最小字段，不读用量或工具表作为业务依据 |
| `cybercut.*` | CyberCut | 自有 Project/Run/NodeRun/NodeAttempt/Approval/ArtifactRevision；FlowMind 仅软引用 |

约束：

- 不建立跨 schema 外键；CyberCut `executor_ref` 对 `flowmind.tasks.task_id` 仅作软引用，且仅在 `executor_kind=flowmind_task` 时填写。
- 不读取 `auth`、CyberCut 或产品域业务表；身份只使用稳定 subject/tenant 引用。
- PgBouncer 事务模式兼容：短连接或进程内池、autocommit 单语句、无 advisory lock、无服务端 prepared statement、无 LISTEN/NOTIFY。
- `TaskStore` 连接池 `min=2, max=8`，借连等待上限 5 秒；池等待超时显式抛错，不静默返回空。
- `ProjectStore` 的 PG 失败文件回落只适用于开发/demo；生产必须显式禁用，且回落时项目记录暴露 `storage=file`，不能伪装成 PG。
- 正式产物与 TTL 临时产物分离：CyberCut 正式产物必须位于 CyberCut 自己的正式产物路径/存储，不能依赖 flowmind TTL 清理。

## 10. MQTT 通知与 PG 真相分离

### 10.1 任务通知

- topic：`mcp-base-gpu/tasks/{task_id}/events`；payload 含 `task_id`、`stage`、`pct`、`message`、`status`、`ts`，可选 `steps`。
- QoS 1；终态 retain；发布失败只记日志，不抛异常、不阻塞任务。
- 重复、乱序或延迟的 MQTT 消息不得覆盖 PG 终态；终态发布以数据库 CAS 成功为前提。
- 监控不得把 MQTT connected 或 retained 终态等同于任务成功；任一状态异常需与 PG 行对账。

### 10.2 联邦通知

- topic：`mcp-base-gpu/federation/register`（QoS 1）、`heartbeat`（QoS 0，30 秒）、`unregister`（QoS 1）。
- payload 含 `backend_id`、`version`、`url`、`transport`、`prefix`、`capabilities`、`ts`；心跳与注销含 `backend_id`、`ts`。
- PG `mcp.federation_backends` 与 MQTT 双通道；注册/心跳/注销失败只记日志，不影响任务主流程。
- 不使用 LWT：异常死亡走网关 90 秒无心跳标 offline 的路径，工具保留、可恢复；优雅停机显式注销。
- 联邦 payload 中的地址不作为 go-kernel 目标寻址真源。实际调用应由预先登记的 K8s Service/CoreDNS 或受控静态配置完成。
- 既有任务进度 MQTT 不强制迁移 JetStream；如新增领域事件，必须另行裁决 owner、版本、幂等和 topic。

## 11. 部署形态与禁止副本扩容

### 11.1 当前事实

- Python 功能包当前以 conda/systemd 常驻 rak-core，监听 `0.0.0.0:8002`；go-kernel 在 `core-api` 通过 ConfigMap `go-kernel-federation` 的 `FLOWMIND_BASE_URL` 静态调用该地址。
- 本仓没有功能包镜像、K8s Deployment、Service、HPA 或 KEDA 清单；不得把“未来服务化”写成已存在。
- go-kernel 当前为单副本 Rollout，原因是内存注册表和 MCP 会话不跨副本；其 greenfield 多副本目标独立于 flowmind，不构成 flowmind 扩容依据。

### 11.2 目标候选（未批准）

若未来获准服务化，候选形态是：

- 一个稳定 K8s Service 指向**一个单活功能包实例**；副本数保持 1，GPU worker 保持 1。
- 调度到经核验的 P104-100 GPU 节点；固定模型缓存卷和任务数据卷；显式 requests/limits 与 GPU 资源约束。
- readiness 只接入可服务实例，检查 PG 等硬依赖；liveness 只检查进程活性，不因可选 MQTT/Milvus 故障批量重启。
- 不使用 HPA/KEDA 提高 GPU 执行并发；不把 API 和 TaskManager 拆成多个会同时初始化执行器的进程。
- MQTT/PG 发现不可用时，静态 Service 是回退；MQTT 不承载 tools/call 或真实 Pod 地址。

以上是待评审候选，不是当前部署事实，也不授权创建清单或镜像。

### 11.3 不得副本扩容的原因

1. `recover_running()` 是进程级全局副作用，多进程会互相把在途任务标为中断。
2. 没有跨进程 claim/租约/fencing；重复投递 queued 任务会造成重复执行。
3. GPU 单卡显存预算只允许一条推理道闸，副本数不增加吞吐，只增加 OOM 和竞态。
4. 模型缓存、TTL 工作目录和进程内单例均不是多副本安全设计。
5. 产物归属和 CyberCut 正式产物复制依赖单实例时序；多实例会放大重复搬运与误清理风险。

## 12. 与 go-kernel 的边界

- go-kernel 数据面可多副本是其自身目标，但所有副本仍调用同一个 flowmind 单活执行面；网关扩容不提高 GPU 吞吐。
- 工具呈现为 `{prefix}__{remote_tool}`；一个 backend 一个 prefix；新技能族若需要独立 prefix，必须作为独立 backend 部署，不能由平台层绕过。
- flowmind 不解析或信任调用方自填身份头；网关派生身份必须落到已校验 token。
- 网关静态后端配置是发现失败时的确定性回退；MQTT 注册/心跳只提示变化，不成为地址真源。
- 网关停止时，普通产品 API 不依赖它；MCP 调用不可用，但 flowmind 已在执行的任务和 PG 任务真相不受影响。

## 13. 与 CyberCut 的边界

- CyberCut 拥有自己的编排、DAG、Project/Run/NodeRun/NodeAttempt、claim、审批、局部重跑和正式产物。
- FlowMind 只提供公开 MCP 技能、trace 信封、既有 `localize_*` 兼容能力和可复用基础设施面。
- CyberCut 内部 NodeHandler 不注册进 FlowMind `_REGISTRY`；注册即对外暴露，隔离靠“不注册”。
- CyberCut 使用自己的 `cybercut` schema；不把 `flowmind.tasks` 当 NodeRun/NodeAttempt，不建跨 schema 外键。
- `executor_kind=flowmind_task` 时，`executor_ref` 仅软引用 task_id；适配器必须在基座任务成功后立即把产物复制到 CyberCut 正式产物目录，不能等待 TTL。
- FlowMind 的 editing 审批闸与 CyberCut 的 Approval 是两层：前者是面向 MCP 客户端的公开投影，后者是 CyberCut 的权威记录；冲突时以 CyberCut 的 `expected_revision` 和状态机为准。

## 14. 故障矩阵

| 故障 | 影响 | 必须保持的语义 | 禁止的处置 |
|---|---|---|---|
| PG 不可达 | 提交/查询/poll 失败；health `components.pg=error`；任务硬依赖中断 | liveness 不因 PG 抖动批量重启；恢复后按现有状态读取；错误显式可见 | 用 MQTT 或内存状态伪装任务成功；自动删除任务行 |
| PgBouncer 池饱和 | 借连等待 5 秒后 TaskStoreError | 错误显式返回；连接预算可观测 | 无限重试、绕过 pooler 直连生产、放宽 pool 无预算 |
| MQTT 不可达 | 进度不实时；联邦 MQTT 通道失效 | 任务继续执行，PG 落库；静态 Service 回退；恢复后事件幂等 | 阻塞任务主流程；把 retained 消息当作任务真相 |
| GPU OOM | 当前任务失败 | 失败显式落终态；不自动连续重试；需人工判断后新任务重提 | 自动增加副本/worker；放宽并发；静默降级成假成功 |
| 进程/Pod 重启 | running→interrupted；queued 重入队 | 恢复语义准确；不宣称断点续跑 | 第二个 TaskManager 同时初始化；直接扩副本 |
| 两个执行进程 | 互相中断在途任务、重复执行 | 立即停止并回到单活 | 以数据库行锁之外未评审机制临时补丁 |
| go-kernel 停止 | MCP 能力不可用 | 普通业务与已在跑任务不受影响；静态回退配置保留 | 让普通业务串行依赖网关；改写任务真相 |
| 联邦发现失效 | 网关可能看不到新变化 | 静态 Service/最后受控配置可用；stale 工具保留 | MQTT payload 地址成为唯一寻址依据 |
| CyberCut 后端失败 | edit_* 项目失败或停在可见错误 | 项目 `failed` 落库、错误可见；不生成假产物 | 把失败包装成 succeeded；删除正式产物 |
| 任务数据盘满 | 工作目录和产物写入失败 | 错误显式；TTL 与审计行保留 | 自动清空数据库或他人目录 |
| 鉴权密钥轮换不一致 | 入站 401 | fail-closed；不降级为匿名全量 | 关闭鉴权以便排障 |
| Milvus/embedding 不可用 | 向量检索/写入失败 | 主任务按现有降级语义；degraded 原因可见 | 把可选依赖故障升级为全服务 liveness 重启 |

## 15. 可观测性

- `trace_id` 在 MCP/REST/技能调用中贯穿；任务通道以 `task_id` 关联；步骤快照和终态事件可关联。
- 指标至少覆盖：提交数、429 背压、pending 水位、任务终态、任务时长、GPU lane 等待、取消/重试、PG 错误与池等待、MQTT 降级、TTL GC、联邦注册/心跳。
- 日志不得记录 raw token、API Key、签名密钥、未脱敏用户媒体内容或完整敏感参数。
- 健康检查区分进程活性、协议就绪和权限就绪；`/api/v1/health` 的组件状态不能替代业务验收。
- 审计与监控需能对账 MQTT retained 终态与 PG 任务行；重复/乱序通知不得覆盖终态 CAS。
- 运维 Agent 对 flowmind 只允许 L0 只读诊断、人工审批的 L1 PR 和 L3 人工控制；GPU、PG、模型卷和任务执行不进入 L2 自动重启、扩缩或回滚。

## 16. 测试与验收

在冻结下可先做只读或离线验证；真实运行验证也需明确许可。

| 层 | 内容 | 验收证据 |
|---|---|---|
| 静态 | `ruff check src`；契约字段、状态常量、topic、表列集快照 | lint 通过；快照差异可定位到具体契约 |
| 冒烟 | 现有 `examples/*_demo.py`；实际数量先核验 | 全部 demo 通过；失败项有明确原因 |
| MCP/REST | `tools/list`、`tools/call`、提交/查询/取消/重提/下载/health | 工具全名、schema、状态码、错误/degraded、trace 一致 |
| GPU | 并发提交 + 轮询；单卡显存与 lane 等待 | 任一时刻只有一个 GPU 执行任务；无第二个 worker |
| 恢复 | 进程重启、queued 重入队、running→interrupted、TTL | 任务与产物行为符合现有语义；不做断点续跑承诺 |
| 租户/安全 | 双租户、无凭证、伪造身份头、路径穿越、非法下载 | 全部 fail-closed；日志无敏感值 |
| 通知 | MQTT 断连、重连、重复、乱序、retain | PG 终态不被覆盖；静态回退可用 |
| CyberCut | 审批、Revise、Reject、正式产物隔离 | CyberCut 自有 claim/审批/产物不被 FlowMind TTL 或恢复破坏 |

不新增 pytest、CI、FastMCP v2 或工作流框架作为本设计的实施前提；若未来引入，必须单独评审。

## 17. 文件级迁移与变更清单（仅在逐项获准后）

下表是未来评审的范围索引，不是当前变更计划；未获许可前所有差异保持为零。

| 文件/目录 | 当前角色 | 若获准可能涉及 | 前置条件 |
|---|---|---|---|
| `src/flowmind/contracts.py` | SkillResult、ReasoningChain、TraceContext、错误信封 | 仅在契约变更时修改；任何字段变化需消费者同步 | 用户逐项许可；契约快照评审 |
| `src/flowmind/server.py`、`server_http.py`、`server_rest.py`、`server_tasks.py` | MCP/REST 入口与路由 | 仅在新增公开端点或认证边界变化时修改 | 路由/状态码/安全评审 |
| `src/flowmind/auth.py` | Bearer 验证与租户上下文 | 仅认证形状变化；不得放行自填身份头 | ECO-ADR-0011 双侧评审 |
| `src/flowmind/tasks/*` | 任务存储、状态机、GPU、事件、TTL | 仅状态机/并发/恢复语义变更 | 单卡与恢复演练通过；用户许可 |
| `src/flowmind/editing/*` | 剪辑流水线与 CyberCut 桥接 | 审批闸实现差异、存储回落、租户边界 | 编辑契约评审；不影响既有项目 |
| `src/flowmind/federation/*` | MQTT/PG 联邦注册 | topic、payload、QoS、表结构变化 | 与 go-kernel 双侧同步评审 |
| `src/flowmind/skills/__init__.py` | 族表与注册 | 新增/删除公开技能族 | 工具全名、prefix、manifest 评审 |
| `migrations/cybercut/*.sql` | CyberCut schema 迁移 | 新增 DDL 或约束 | 向后兼容方案；单独许可 |
| `deploy/ocr-sidecar/` | 唯一既有 Docker 落点 | 仅在 OCR 侧车变更时修改 | 不挡主服务启动；单独评审 |
| 任务 DDL（代码内 `TaskStore._DDL`） | 幂等建表与索引 | 表/索引/列变化 | 迁移与回滚方案；用户许可 |
| K8s 清单（不在本仓） | 当前无功能包清单 | 若服务化，清单应归部署仓并经评审 | 镜像、GPU、卷、探针、回退齐备 |

## 18. 停止条件

出现以下任一情况必须停止并回到文档评审：

- 未取得逐项许可就修改代码、配置、契约、数据或部署。
- 出现第二个 GPU 执行者、第二个 TaskManager 或副本扩容。
- 任务被重复执行、误恢复或终态被 MQTT 通知覆盖。
- 任何未批准的字段、状态码、topic、schema、依赖或工具全名变化。
- 跨租户、路径穿越、无凭证全量访问或 Secret 泄露。
- MQTT 被用作服务地址真源或任务状态真相。
- CyberCut 正式产物进入 FlowMind TTL 目录或恢复流程。
- 运维 Agent 可自动重启、扩缩、回滚或操作 GPU/数据面。

## 19. 非目标

- 不解除维护冻结，不把本文或生态 G0-G7 任务视为实施许可。
- 不把 flowmind 改造成第二网关、普通业务 BFF、通用工作流引擎或多进程 GPU 调度平台。
- 不新增多 GPU、多进程 claim、GPU 并行或自动故障切换。
- 不升级 FastMCP、torch/CUDA、go-sdk 或依赖钉版，不重写 SkillResult、REST、manifest、MQTT topic 或 federation 契约。
- 不将既有任务进度强制迁往 JetStream，不让 MQTT 成为 Service 寻址或 MCP 数据面。
- 不把 CyberCut 的编排/claim/审批/产物塞入 FlowMind TaskManager，不建立跨 schema 外键或第二身份源。
