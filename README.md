# mcp-base-gpu

**MCP over HTTP SaaS 服务端**：缺省加载两族技能——把「中文视频 → 多语言配音/字幕版本」
的完整 GPU 流水线包装成 **localize 族 7 个工具**，把抖音短视频成片（配音 → 时间线切镜 →
图生视频 → 字幕/封面/贴片 → 合成渲染）包装成 **dsp 族 8 个工具**；共一条异步任务 REST 通道。
技能逻辑、GPU 与密钥全部留在服务端，任意 MCP 客户端 / HTTP 客户端零密钥消费。

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
   conda run -n flowmind mcp-base-gpu            # 单端口 8002（后台加 nohup）
   curl http://127.0.0.1:8002/api/v1/health      # {"status":"ok",...}
   ```

3. **接入 MCP**（Streamable HTTP，端点 `http://127.0.0.1:8002/mcp`）：
   工具清单以 `tools/list` 为准（缺省 15 个：`localize_*` 7 + `dsp_*` 8；可用
   `FLOWMIND_SKILL_FAMILIES` 只载一族），字段 schema 用
   `GET /api/v1/manifest/<skill_id>` 查询——**不要猜字段、不要读源码**。

架构细节与不变量见 `CLAUDE.md`；开发工作流见 `AGENTS.md`。
环境分三层（主应用 / OCR 侧车容器 / 按可用性调用），侧车是**可选层、不挡启动**，见下文「OCR 侧车」段。

---

## localize 族 7 个工具（MCP tools/call 与部分 REST 共用同一技能层）

| 工具 | 语义 | 通道 |
|---|---|---|
| `localize_submit` | 批量提交视频本地化异步任务（每视频一个 task，PG 落库 queued） | MCP / REST |
| `localize_status` | 查询任务状态与进度（stage / pct / error） | MCP / REST |
| `localize_retry` | 重试失败任务（沿用原参数重新入队） | MCP |
| `localize_cancel` | 协作式取消（queued 直接落终态；running 在阶段边界生效） | MCP |
| `localize_download` | 列出已完成任务产物 + 下载 URL | MCP / REST |
| `localize_search` | 字幕语义检索（自然语言 → 768 维向量 → Milvus 余弦检索） | MCP |
| `localize_video` | 本地化流水线本体（由任务引擎串行执行，Agent 一般不直连） | MCP |

## dsp 成片技能族（8 个工具 · **独立 backend**）

抖音短视频成片链路的 8 个原子技能（契约真源在 cross-dashboard 仓的
`docs/architecture/2026-10-11-dsp-backend-skill-contract.md`）。**算力全下沉云厂商**
（豆包语音 / 火山 AI MediaKit / 图生视频厂商），本后端只做编排、参数装配、幂等与结果托管。

| 工具 | 语义 | 降级 |
|---|---|---|
| `tts_generate` | 口播配音（火山 · 豆包语音；**DAG 起点**，出音轨 + **逐句实测时间戳** = 时长真源） | abort |
| `timeline_build` | 素材对齐 · **切镜收口** · 时间线锁定（**唯一时间线写者**；按配音逐句实测锁 EDL 并按实测并/拆镜，无配音则按估算兜底） | abort |
| `image_to_video` | **图生视频**（逐镜产出视频底 · 契约 §7.5 第 1 步；可关开关） | degrade |
| `bgm_match` | 背景音乐（**当前无选曲接口 → 一律如实降级为空轨**） | degrade |
| `subtitle_build` | 字幕 SRT（依配音逐句时间戳；只出参数不入画） | degrade |
| `cover_pick` | 封面取帧（图片源：指定那张图 + 标题位，标题 L1） | degrade |
| `cart_sticker` | 小黄车贴片（**仅带货形态**；商品链接属平台组件不烧进画面） | degrade |
| `render_compose` | 合成渲染（火山 AI MediaKit；**唯一渲染、DAG 汇点**）；有片段时叠加**运镜**（§7.5 第 2 步） | abort |

> **时长真源 = 配音逐句实测**（2026-10-13 改判，ADR-069）：DAG 是 **`tts → timeline`**——
> 配音先出音轨与逐句时间戳，`timeline` 再用 `_dsp_align` 把句子归属到各镜（**文配精确优先**、
> 对不上才按**该镜文案的可念字数**为权重连续切分并**如实告警「归属为推断」**），轨严格连续
> ⇒ **画面与配音逐帧对齐**（`totalMs` 即音频总长）。无口播镜吃「停顿 / 尾部静默」，
> 不够才取显式常量并标注**非实测**。`shots[].durationMs` 因此只是**估算兜底**
> （配音开关关掉时用，标 `durationSource: "estimated"`）。
> **旧「甲案·时间线前移」（配音适配已锁定画面）已推翻**——那要求配音服从给定时段，
> 而厂商不保证该精度（详见 ADR-069）。
>
> **镜头由文案自己定 · 切镜收口**（2026-10-13 改判，ADR-070；**六个内置模板已整块删除**）：
> 镜头的「有几个、切在哪」不再由模板锁死——文案按「视觉信息点」写成镜头单元（镜数由内容定），
> `timeline` 在**实测轴**上做**节奏收口**（`_dsp_align.consolidate_shots`：实测 < 1.5s 的镜
> 并入相邻镜、> 6s 的镜在其**内部一分为二**——**先取最近的句边界，句内没有句边界就下沉到词边界**，
> 镜数落回 2~10），切点**永远落在实测 token（句 / 词）边界**上（词边界同为实测值，画面与配音仍逐帧对齐）。
> 收口会重排镜 id（`s1..sK`），故**终版镜头的文案一并落进 `edl.shots[].copy`**——下游
> （I2V 提示词等）一律读 EDL，不再回查下发入参。词级 token 由 `tts_generate` 的
> `voSegments[].words[]` 透出（ADR-072）。
>
> **转场**（ADR-070）：`segments[i].transition` = 该镜**末尾**桥接 i→i+1 的转场（**符号名**），
> 厂商表达 = 加在**前一个**元素上的 `extra: [{type:"transition", source:<资源 ID>, duration}`。
> **只允许「非交叠」**（音轨是一整条实测音频、时间线严格连续，交叠会移动画面或改总时长）；
> `_dsp_mediakit.TRANSITION_VENDOR` **当前为空**（厂商资源 ID 表不在本仓）⇒ **未登记即如实回落
> 硬切并告警**，**绝不编造 ID**；拿到 ID 表后只填那一张表。

> **图生视频（`image_to_video`）**：运镜 `apply-camera-motion` **只接受视频**，故「让静态图
> 动起来」需先有视频底——本技能逐镜把素材图 + **画面提示词**（`edl.shots[].visualPrompt`；缺则
> 回落口播文案并如实告警，ADR-073）喂图生视频模型（**独立第三方** `api.chenxingai.vip`，与火山
> **不共享**端点 / 密钥），片段落 `{cachePrefix}/{runId}/clips/`；`render_compose` 再对片段加
> 运镜并拼成成片。**某镜失败即该镜回落静图**（如实标注）。
>
> **首尾帧桥 + 下发前降采样**（ADR-074）：① 逐镜把**下一镜的图**当**尾帧**一起下发
> （`images = [本镜图, 下一镜的图]`，第 2 位 = 尾帧；尾帧取「其后**最近一张有图的镜**」，
> 末镜无尾帧）——聚合端点**没有**指南首尾帧用的 `content.role`，故这是**该接口面上最接近的
> 落法**，**语义待真跑验证**（若按「多参考图」处理 ⇒ 退化为无桥，不产生错画面）；
> ② 素材图超 I2V **输出**分辨率（`videogen_size` = `720x1280`）的，**下发前**等比缩到以内
> （**只缩不放**，已在分辨率内则原图直用）→ 派生件落 `{cachePrefix}/{runId}/i2v_input/…`，
> 失败**回落原图 + 如实告警、不丢镜**（指南：输入超标易出**指纹状纹理**）。两行为各配 env
> 开关（默认开）：`FLOWMIND_DSP_VIDEOGEN_BRIDGE_NEXT_IMAGE` / `FLOWMIND_DSP_VIDEOGEN_DOWNSCALE_INPUT`。

**为什么是独立 backend**：网关把每个 backend 的工具聚合为 `{prefix}__{remote_tool}`，
而 **prefix 每 backend 只有一个**（`video_localizer` 已被 localize 占用）。故 dsp 以
**独立进程 / 部署**注册为 `prefix=dsp`，注册表里只应有 dsp 八项（契约 §6
「`dsp__` 前缀下无多余工具」）——由 `skills/__init__.py` 的**族筛选**
（env `FLOWMIND_SKILL_FAMILIES`，缺省 = 全部族）落地：

```bash
# dsp 功能包（与 localize 的 :8002 并存；注册出 dsp__timeline_build … 八个）
FLOWMIND_SKILL_FAMILIES=dsp \
FLOWMIND_MCP_PORT=8003 \
FLOWMIND_FEDERATION_REGISTER=1 \
FLOWMIND_FEDERATION_BACKEND_ID=dsp \
FLOWMIND_FEDERATION_PREFIX=dsp \
  mcp-base-gpu
```

**新增依赖**：`boto3`（产物对象存储 S3/MinIO 兼容；**懒加载**——不装它不影响其它技能）。
产物落云存储 prefix（`{cachePrefix}` / `{outputPrefix}`），密钥经 `DspConfig` 的
**env 名**解析（`S3_ACCESS_KEY` / `DSP_MEDIAKIT_API_KEY` / `DSP_TTS_API_KEY` /
`DSP_VIDEOGEN_API_KEY`），**绝不落 toml / commit**；未配置即如实降级或抛结构化错误，
**绝不返回假产物**。

## 两条通道的分工

- **MCP tools/call**：轻技能同步调用（状态查询 / 检索 / 重试 / 取消），
  SkillResult 信封（`ok` / `data` / `reasoning` / `metrics` / `trace`）。
- **REST /api/v1/tasks**：分钟级 GPU 长任务专用——提交立即 202（不阻塞），
  轮询 `GET /api/v1/tasks/{task_id}`，完成后 `GET .../download?file=<name>`
  流式取产物（basename 白名单校验，防路径穿越）。

```bash
# 提交（body 与 localize_submit 同形状；本地路径必须位于 data_dir/uploads/ 内）
curl -X POST http://127.0.0.1:8002/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"videos": ["https://cdn.example.com/demo.mp4"], "target_lang": "en"}'
# → 202 {"task_ids": ["..."], "accepted": 1, ...}

# 轮询 / 下载
curl http://127.0.0.1:8002/api/v1/tasks/<task_id>
curl -o out.mp4 "http://127.0.0.1:8002/api/v1/tasks/<task_id>/download?file=demo_sub.mp4"
```

状态码：`202` 受理（队列中途满 → 202 + warning 部分受理）；`429` 队列满背压；
`422` 入参校验；`404` 任务/产物不存在；健康探针恒 200（组件状态见 `components`）。

## 异步任务语义

- **状态机**：`queued → running → succeeded | failed | cancelled | interrupted`；
  服务重启时遗留 `queued/running` 标为 `interrupted`（启动恢复）。
  终态写入带 CAS 状态守卫（仅 queued/running 可写）：cancel 与任务完成
  并发时 first-writer-wins，终态绝不互相覆盖（cancelled 也不会被 retry
  重复执行）。
- **终态分类**：`succeeded` = 产物就绪（degraded 的空结果任务也算成功）；
  `failed` = 入参/内部/技能级失败（`error` 列有原因）。
- **产物路径语义**：本地路径输入 → 产物落源文件旁 `<stem>_localized.mp4`；
  URL 输入（SaaS 主路径）→ 产物落 `data_dir/outputs/<task_id>/output.mp4`，
  经 `output_paths` 白名单由 download 端点寻址，与工作目录同 TTL 清理。
- **输入沙箱**：提交通道（MCP `localize_submit` 与 REST `POST /api/v1/tasks`）
  的本地路径必须位于 `data_dir/uploads/` 内（URL 不受限）；`uploads` 不存在
  或路径越界 → `422`/VALIDATION。这是鉴权实装前的隔离层：防任意服务器
  路径探测、产物外带与 `ffmpeg -y` 覆盖写源目录。`FLOWMIND_ALLOW_ANY_PATH=1`
  仅限本地测试放行。
- **进度推送**：MQTT 主题 `mcp-base-gpu/tasks/{task_id}/events`，QoS=1，
  终态消息 retain（新订阅者连接即见最终状态；终态落库后残余进度自动停发，
  不污染 retain）；MQTT 未配置时自动降级纯 PG 落库。
- **TTL 清理**：终态任务工作目录与 `data_dir/outputs/<task_id>/` 超
  `task_ttl_seconds` 回收（DB 行保留供审计）。

## 字幕语义检索（Milvus）

任务成功后，ASR 分段经 BGE（`BAAI/bge-base-zh-v1.5`，768 维）嵌入写入
Milvus collection `localize_segments`（HNSW + COSINE）。`localize_search`
把自然语言 query 嵌入后检索，支持按 task_id 限定范围。向量化是增值步骤：
- **未配置**（`FLOWMIND_MILVUS_URI` 与 `FLOWMIND_EMBEDDING_BASE_URL` 均空，
  内置默认即未配置）：流水线尾部静默跳过（记 info，`data.vectorized=false`），
  `localize_search` 返回结构化错误——不刷 warning；
- **调用失败**（已配置但服务不可达/出错）：降级 warning，不影响本地化主产出。
- 开关：`FLOWMIND_VECTORIZE=0` 显式关闭（`data.vectorized=null`）。

## 配置

配置源优先级（全仓库统一）：**环境变量 → `flowmind.config.toml` → 内置默认**。
模板见 `.env.example`（云 key + 基础设施地址 + 任务治理参数）；可调参数见
`config.py`（`LocalizerConfig` 业务参数 / `InfraConfig` 基础设施）。密钥绝不进 toml / commit。

部署时需关注的治理参数（env 覆盖，详见 `.env.example`）：

| 环境变量 | 默认 | 语义 |
|---|---|---|
| `FLOWMIND_DATA_DIR` | `~/flowmind-data` | 任务工作目录基准（`uploads/` `tasks/` `outputs/` 根；部署必配绝对路径） |
| `FLOWMIND_MAX_PENDING_TASKS` | `100` | 待处理上限（queued+running，超出 429 背压） |
| `FLOWMIND_TASK_TTL_SECONDS` | `3600` | 终态任务 workdir/outputs 的 GC 保留周期（DB 行保留供审计） |

## 基础设施依赖

| 组件 | 用途 | 开发机 | 集群内部（svc 短名示例） |
|---|---|---|---|
| PostgreSQL | 任务存储（短连接快进快出；统一库 rak 的 flowmind schema，ECO-ADR-0013） | mesh `RAK_PG_*` | `pg-np.database.svc:5432/rak` |
| EMQX (MQTT) | 任务进度事件（明文 1883 / TLS 可选 / 认证可选） | mesh `RAK_MQTT_HOST` | `emqx.agentic.svc:1883` |
| Milvus | 字幕分段向量库 | 自建（如 mesh NodePort） | `http://milvus.agentic.svc:19530` |
| BGE 嵌入服务 | 字幕向量化（TEI / OpenAI 双形状自适应） | 自建（如本机 127.0.0.1:31997） | 集群内注入服务地址 |

未配置的基础设施按语义降级（MQTT → 纯落库；Milvus/嵌入 → 向量化显式禁用，
流水线尾部静默跳过、`localize_search` 显式报错）；PG 是任务引擎硬依赖，
缺失显式报错。MQTT 认证：`FLOWMIND_MQTT_USERNAME/PASSWORD`（空 = 匿名）。

## OCR 侧车（本机 dev 的第三层 · 容器 · **不挡启动**）

dsp「生图质检 · 文字核查」要一台**独立于生图模型**的 OCR 引擎，它跑在本机容器里，**不属于本服务**：
本服务不 import 它、不依赖它、也不因它没起而拒绝启动——`src/flowmind/ocr_sidecar.py` **不是** `@skill`，
不进 `skills/__init__.py` 的 `FAMILIES`，也不碰 `server*.py`（故本服务不必为它拉起 15 个技能模块的依赖）。

三层边界（决策记录在 cross-dashboard `ADR.md#ADR-100`；这条通道本身是 `#ADR-097`）：
① 主应用只依赖自己的运行时（`environment.yml`）；② 侧车跑自己的容器（`deploy/ocr-sidecar/`，
只发布到宿主 `127.0.0.1`）；③ **侧车可用才跑 OCR，不可用 ⇒ 消费方如实记 `not_run`（不假报通过，也不挡启动）**。

```bash
python scripts/ocr_sidecar_dev.py            # 起（先探活：已在跑就如实说，绝不重复起第二个）
python scripts/ocr_sidecar_dev.py --status   # 看：在跑 ⇒ 报引擎/模型；没跑 ⇒ 如实说没跑 + 为什么
python scripts/ocr_sidecar_dev.py --down     # 停
```

- **端口只有一处真源** `OCR_HOST_PORT`（默认 `8765`，与 compose 同一个变量）。⚠️ `docker compose`
  只读**项目目录**（compose 文件所在的 `deploy/ocr-sidecar/`）的 `.env`，**不读仓库根的 `.env`**
  ⇒ 脚本把解析出的端口**显式传进 compose**，两处不会各写一个数。
- **真正连它的地址不在本仓**：cross-dashboard `.env` 的 `AI_OCR_API_URL`（默认 `http://127.0.0.1:8765`）。
  两边对不上 ⇒ 那边连不上 ⇒ 那一项如实记没跑成——**不会有人替你猜**。
- **容器 ≠ 原生**：容器那条路是默认；`rapidocr-sidecar --port 8123` /
  `PYTHONPATH=src python -m flowmind.ocr_sidecar` 是原生那条路，留着当**对账基准**
  （2026-10-16 实测：同一张图两条路逐行文字 / 四点框 / 置信度三样全同）。它吃**本机 env** 的
  `rapidocr` + `onnxruntime`（后者必须显式装——rapidocr 不拉它；缺了是**引擎初始化才炸**，
  `available()` 那个探针探不到）。
- ⚠️ **别顺手清依赖**：`environment.yml` 里的 `rapidocr` 服务的是**字幕条定位**
  （`skills/_local_ocr.py`，`localize_video` 的 `ocr_backend=local/auto`），与这条侧车是两条路
  ——**它不是为了侧车装的**。

## MCP 联邦网关（Go go-kernel）

仓库旁挂载了一个 Go 联邦网关 `go-kernel/`（独立 go module，占用 **:8080**，
与 Python 功能包 **:8002** 共存）：把本仓库的 7 个 `localize_*` 工具（localize 族）
+ 网关内置 `dify__*` 工具聚合为**单一 `/mcp` 端点**对外；dsp 族以**独立 backend**
（`prefix=dsp`，见上「dsp 成片技能族」）另行注册。

- **独立仓库**：`go-kernel/` 是独立 git 仓库（VANexus/go-kernel），不随
  本仓库分发（外层 `.gitignore` 已忽略该目录）——需单独 `git clone` 到
  本仓库根旁，源码与文档中的 `go-kernel/...` 路径引用才可达。

- **端口约定**：Python 功能包 :8002（FastMCP + 任务 REST）＝网关后端；
  Go 网关 :8080（`/mcp`、`/api/v1/tasks` 透传、`/api/v1/federation/*`、
  `/metrics`）＝对客户端的唯一入口。客户端连 :8080，不直连 :8002。
- **动态注册**：本仓库以 `FLOWMIND_FEDERATION_REGISTER=1` 启动后，经
  MQTT（`mcp-base-gpu/federation/{register,heartbeat,unregister}`）+ PG
  （`federation_backends` 表）双通道自注册，30s 心跳 / 90s 无心跳标
  offline / 优雅停机注销；默认关，不影响现有部署。
- **代理语义**：工具名 `{prefix}__{remote_tool}`（如
  `video_localizer__localize_status`）；后端断连工具保留（stale），调用
  返回 `BACKEND_UNAVAILABLE / BACKEND_TIMEOUT` 结构化错误，重连自动恢复。
- **协议契约改动必须双侧同步**：`src/flowmind/federation/` ↔
  `go-kernel/internal/{mqttclient,federation,pgstore}`，详见
  [go-kernel/README.md](./go-kernel/README.md)（topic 契约表、指标清单、
  K8s 部署）。

## GPU 部署约定

- **硬件**：NVIDIA P104-100（8GB，Pascal 6.1）。torch 锁 `2.5.1+cu121`
  （cu124+ 已剔除 Pascal，勿升级；`torchaudio` 必须在 `qwen-tts` 之后重新钉回配套版）。
- **模型缓存**：`HF_HOME=/srv/data/models`、`TORCH_HOME=/srv/data/models/torch`、
  下载代理 `http://127.0.0.1:7890`（写入 `~/.bashrc`，新 shell 生效）。
- **显存预算**：whisper ~1.5G + Qwen3-TTS ~4G + LaMa ~1.5G ≈ 7.5G——
  任务引擎 workers=1 串行 + `gpu_lane()` 信号量双道闸，**绝不放宽并发**。
- **系统依赖**：`sudo apt install sox fonts-noto-cjk ffmpeg`。

## 调用方鉴权（RakAuthMiddleware · ECO-ADR-0011）

单入口 `server_http.py` 挂有 `RakAuthMiddleware`（纯 ASGI），对 MCP 与 REST 两条通道一并生效：

- **token**：`Authorization: Bearer <base64url(JSON{sub,tid,exp})>.<hex(HMAC-SHA256)>`，
  **先验签再解 JSON**（`flowmind/auth.py::verify_token`）；签名密钥 `FLOWMIND_AUTH_SECRET`
  与 go-kernel 的 `KERNEL_TRUSTED_SECRET` 同值。该格式与 cross-dashboard 的
  `lib/server/rak/token.ts`、go-kernel 逐字节对齐（跨语言 token 向量对账）。
- **豁免前缀**：`/api/v1/health`、`/api/v1/manifest` 匿名可达（`AuthConfig.exempt_prefixes`）；
  密钥为空则鉴权整体关闭（开发态）。
- **归属 fail-closed**：`auth.tenant_scope()` 三态——未启用鉴权不过滤 / 已鉴权看本租户 /
  **启用鉴权但上下文无凭证 = 拒绝一切任务访问**（防后台脚本「看全部」）。
  `X-Rak-Tenant` 头只作诊断，**归属只认已校验 token**；`tasks` 表带 `tenant_id` 列落归属。

## 验证

本仓库无单测（用户决策：验证靠真实运行）：

```bash
conda run -n flowmind ruff check src                       # lint（唯一质量门）
for f in examples/*_demo.py; do
  PYTHONPATH=$PWD/src conda run -n flowmind python "$f"    # 18 个 demo 全 PASS
done
```

## License

见 [LICENSE](LICENSE)。
