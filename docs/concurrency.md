# flowmind 并发模型(SaaS 后端)

> 2026-10-31 建立。目标:作为 SaaS 后端扛海量并发提交与轮询。
> **物理约束:GPU 单卡(P104 8GB),任务执行串行不可放宽**(`manager.py` workers=1,
> AGENTS.md 明令)。模型围绕它设计,而非对抗它。

## 总纲

```
海量提交(毫秒级原子准入, 持久队列, 租户公平) → GPU 串行执行(队列即产品)
→ 海量轮询(池化读 + args 瘦身, 上层网关/BFF 再缓存)
```

| 路径 | 并发模型 | 关键实现 |
|---|---|---|
| 提交(REST/MCP) | 无锁原子准入, 全局+租户双水位 | `store.create_task_admitted`(单条 SQL, READ COMMITTED 软水位) |
| 执行 | 单 worker + gpu_lane 双道闸 | `manager.py` workers=1, `gpu.py` Semaphore(1) |
| 队列 | PG 即队列, 进程重启不丢 | 启动 `recover_running`(running→interrupted) + `_requeue_orphans`(queued 重新入队) |
| 轮询 | 连接池借还(信号量排队) + args 不回传 | `store._lease` ThreadedConnectionPool(2,8);`get_task(include_args=False)` |
| 进度 | 写合并(stage 变或 ≥2s) | `manager._on_progress` `_PROGRESS_MIN_INTERVAL`;MQTT 照发 |
| 探针 | 5s 全量缓存(auth-exempt 面收口) | `server_tasks._health_cache` |
| 线程 | anyio to_thread 上限 100(默认 40) | `server_http` lifespan;`FLOWMIND_THREAD_POOL_TOKENS` 可调 |
| MQTT | 进程级单客户端 + 启动预热 | `events.get_event_publisher().warmup()` |

## 关键决策记录

1. **水位是软上限**:READ COMMITTED 下并发提交存在 ±N(N=瞬时并发)的超越。
   GPU 单 worker 下 pending 只是背压信号而非硬不变量,接受软超越换取零额外
   往返、无应用层锁。硬保护由「队列排队 + worker 串行」提供。
2. **queued 持久化 vs running 中断**:流水线不可断点续跑,running 重启标
   interrupted(调用方 retry);queued 重新入队(202 受理不再静默丢失)。
3. **租户公平**:全局 100 / 单租户 20(`localizer.max_pending_per_tenant`),
   一个租户的批量提交不再把其他租户挤成 429。
4. **连接池 + 信号量**:池 2..8;借还经 BoundedSemaphore 排队(≤5s),风暴下
   礼貌等待而非 500。

## 与上下游的分工

- **go-kernel(网关)**:状态轮询 1s TTL 缓存 + singleflight、每租户限流、
  连接复用——见 go-kernel 仓库 TODO/README。
- **cross-dashboard(BFF)**:任务状态 Redis 缓存(2s)+ in-flight 合并、
  配额缓存(30s)——绝大多数轮询不穿透到本服务。

## 流水线步骤（SaaS 前端契约，2026-10-31）

前端需要「看到每一步的中间进度」和「选择只跑哪几步」。实现分四层：

| 层 | 文件 | 职责 |
|---|---|---|
| 注册表 | `src/flowmind/pipeline_steps.py` | 九阶段的 id/中文名/进度区间/**依赖**/可跳过性/产物键/跳过后果（唯一真源） |
| 选择解析 | 同上 `resolve_selection()` | `steps` 白名单 / `skip_steps` 黑名单 → 启用集合；**必需步骤**或**硬依赖**被剔除 → 422 |
| 运行跟踪 | `StepTracker` + `TaskContext.step_cb` | 每步 pending/running/done/skipped/failed + 起止时间 + 耗时 + 实际后端 |
| 对外暴露 | `GET /api/v1/skills/{id}/steps`；manifest 的 `stage_schema`；任务行 `steps`；MQTT 事件 `steps` | 前端零硬编码步骤名 |

**关键语义**
- `steps`（白名单）与 `skip_steps`（黑名单）可单用或同用；`extract_audio`/`asr` 必需。
- 跳过 `ocr` 时 `erase` 必须一并跳过（硬依赖），否则 422——**绝不静默跑半截**。
- 软依赖（如 `tts` 依赖 `translate`）缺失时降级：译文回落原文，仍可配音。
- 任务终态（成功/失败/取消）时，未执行步骤统一收尾为 skipped/failed，不留 running 假态。

**跨仓链路**：前端 `components/skills/StepSelector.tsx` → BFF `skills-gw`（按技能查联邦目录得到 backend_id，再走网关**按后端代理**）→ flowmind `GET /api/v1/skills/{id}/steps`。提交时 `skip_steps` 随 args 传入，逐步状态随任务行返回。
