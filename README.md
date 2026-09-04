# mcp-base-gpu

视频本地化 **MCP over HTTP SaaS 服务端**：把「中文视频 → 多语言配音/字幕版本」的
完整 GPU 流水线包装成 7 个 MCP 工具 + 一条异步任务 REST 通道。技能逻辑、GPU 与
密钥全部留在服务端，任意 MCP 客户端 / HTTP 客户端零密钥消费。

- **单端口双通道**：`/mcp`（Streamable HTTP）承载轻量同步调用，`/api/v1/tasks`
  承载分钟级长任务（提交 202 → 轮询 → 流式下载）。
- **任务引擎**：PostgreSQL 持久化（重启不丢）、MQTT 实时进度推送（终态 retain）、
  单卡 GPU 串行执行（防 OOM）、TTL 自动清理。
- **字幕语义检索**：每个任务的 ASR 分段向量化入 Milvus，跨任务「找讲过某句话的视频片段」。

---

## 🤖 如果你是一个 AI Agent 第一次读到这个文件

不要急着跑命令。按下面三步接入：

1. **装依赖**（依赖真源 = `environment.yml`）：

   ```bash
   conda env update -n flowmind -f environment.yml
   conda run -n flowmind pip install --no-deps "simple-lama-inpainting>=0.1.2"
   conda run -n flowmind pip install -e . --no-deps
   conda run -n flowmind ruff check src          # lint 必须全绿
   ```

2. **起服务 + 探活**：

   ```bash
   conda run -n flowmind mcp-base-gpu            # 单端口 8001（后台加 nohup）
   curl http://127.0.0.1:8001/api/v1/health      # {"status":"ok",...}
   ```

3. **接入 MCP**（Streamable HTTP，端点 `http://127.0.0.1:8001/mcp`）：
   工具清单以 `tools/list` 为准（7 个 `localize_*`），字段 schema 用
   `GET /api/v1/manifest/<skill_id>` 查询——**不要猜字段、不要读源码**。

架构细节与不变量见 `CLAUDE.md`；开发工作流见 `AGENTS.md`。

---

## 7 个工具（MCP tools/call 与部分 REST 共用同一技能层）

| 工具 | 语义 | 通道 |
|---|---|---|
| `localize_submit` | 批量提交视频本地化异步任务（每视频一个 task，PG 落库 queued） | MCP / REST |
| `localize_status` | 查询任务状态与进度（stage / pct / error） | MCP / REST |
| `localize_retry` | 重试失败任务（沿用原参数重新入队） | MCP |
| `localize_cancel` | 协作式取消（queued 直接落终态；running 在阶段边界生效） | MCP |
| `localize_download` | 列出已完成任务产物 + 下载 URL | MCP / REST |
| `localize_search` | 字幕语义检索（自然语言 → 768 维向量 → Milvus 余弦检索） | MCP |
| `localize_video` | 本地化流水线本体（由任务引擎串行执行，Agent 一般不直连） | MCP |

## 两条通道的分工

- **MCP tools/call**：轻技能同步调用（状态查询 / 检索 / 重试 / 取消），
  SkillResult 信封（`ok` / `data` / `reasoning` / `metrics` / `trace`）。
- **REST /api/v1/tasks**：分钟级 GPU 长任务专用——提交立即 202（不阻塞），
  轮询 `GET /api/v1/tasks/{task_id}`，完成后 `GET .../download?file=<name>`
  流式取产物（basename 白名单校验，防路径穿越）。

```bash
# 提交（body 与 localize_submit 同形状）
curl -X POST http://127.0.0.1:8001/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"videos": ["/data/demo.mp4"], "target_lang": "en"}'
# → 202 {"task_ids": ["..."], "accepted": 1, ...}

# 轮询 / 下载
curl http://127.0.0.1:8001/api/v1/tasks/<task_id>
curl -o out.mp4 "http://127.0.0.1:8001/api/v1/tasks/<task_id>/download?file=demo_sub.mp4"
```

状态码：`202` 受理（队列中途满 → 202 + warning 部分受理）；`429` 队列满背压；
`422` 入参校验；`404` 任务/产物不存在；健康探针恒 200（组件状态见 `components`）。

## 异步任务语义

- **状态机**：`queued → running → succeeded | failed | cancelled`；
  服务重启时遗留 `queued/running` 标为 `interrupted`（启动恢复）。
- **终态分类**：`succeeded` = 产物就绪（degraded 的空结果任务也算成功）；
  `failed` = 入参/内部/技能级失败（`error` 列有原因）。
- **进度推送**：MQTT 主题 `mcp-base-gpu/tasks/{task_id}/events`，QoS=1，
  终态消息 retain（新订阅者连接即见最终状态）；MQTT 未配置时自动降级纯 PG 落库。
- **TTL 清理**：终态任务工作目录超 `task_ttl_seconds` 回收（DB 行保留供审计）。

## 字幕语义检索（Milvus）

任务成功后，ASR 分段经 BGE（`BAAI/bge-base-zh-v1.5`，768 维）嵌入写入
Milvus collection `localize_segments`（HNSW + COSINE）。`localize_search`
把自然语言 query 嵌入后检索，支持按 task_id 限定范围；向量化是增值步骤，
失败仅降级 warning，不影响本地化主产出。开关：`FLOWMIND_VECTORIZE=0` 关闭。

## 配置

配置源优先级（全仓库统一）：**环境变量 → `flowmind.config.toml` → 内置默认**。
模板见 `.env.example`（云 key + 基础设施地址）；可调参数见 `config.py`
（`LocalizerConfig` 业务参数 / `InfraConfig` 基础设施）。密钥绝不进 toml / commit。

## 基础设施依赖

| 组件 | 用途 | 开发机 | 集群内部（svc 短名示例） |
|---|---|---|---|
| PostgreSQL + PgBouncer | 任务存储（事务模式，短连接快进快出） | mesh `RAK_PG_*` | `pgbouncer.agentic.svc:6432/mcp_base_gpu` |
| EMQX (MQTT) | 任务进度事件（明文 1883 / TLS 可选） | mesh `RAK_MQTT_HOST` | `emqx.agentic.svc:1883` |
| Milvus | 字幕分段向量库 | mesh NodePort | `http://milvus.agentic.svc:19530` |
| BGE 嵌入服务 | 字幕向量化（TEI / OpenAI 双形状自适应） | 本机 `127.0.0.1:31997` | 集群内注入服务地址 |

未配置的基础设施按语义降级（MQTT → 纯落库；Milvus/嵌入 → 跳过向量化）；
PG 是任务引擎硬依赖，缺失显式报错。

## GPU 部署约定

- **硬件**：NVIDIA P104-100（8GB，Pascal 6.1）。torch 锁 `2.5.1+cu121`
  （cu124+ 已剔除 Pascal，勿升级；`torchaudio` 必须在 `qwen-tts` 之后重新钉回配套版）。
- **模型缓存**：`HF_HOME=/srv/data/models`、`TORCH_HOME=/srv/data/models/torch`、
  下载代理 `http://127.0.0.1:7890`（写入 `~/.bashrc`，新 shell 生效）。
- **显存预算**：whisper ~1.5G + Qwen3-TTS ~4G + LaMa ~1.5G ≈ 7.5G——
  任务引擎 workers=1 串行 + `gpu_lane()` 信号量双道闸，**绝不放宽并发**。
- **系统依赖**：`sudo apt install sox fonts-noto-cjk ffmpeg`。

## 鉴权扩展点

单入口 `server_http.py` 挂有 `AuthPlaceholderMiddleware`（当前 no-op）：
对接既有登录授权后端时，在该中间件校验 `Authorization` 头、解析
`tenant_id` 写入 `request.state`——MCP 与 REST 两条通道经同一入口，
鉴权策略一处生效（`tasks` 表已预留 `tenant_id` 列）。

## 验证

本仓库无单测（用户决策：验证靠真实运行）：

```bash
conda run -n flowmind ruff check src                       # lint（唯一质量门）
for f in examples/*_demo.py; do
  PYTHONPATH=$PWD/src conda run -n flowmind python "$f"    # 8 个 demo 全 PASS
done
```

## License

见 [LICENSE](LICENSE)。
