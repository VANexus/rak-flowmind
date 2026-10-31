# Agent 智能视频剪辑（edit_*）— 架构设计 V1.2

> 状态：**设计已定稿，随实现同步维护**（v1.2 = 2026-10-30 韧性/流程完善，
> 变更见文末 §12）。
> 范围：仅 FlowMind（:8002 侧）。go-kernel / rak-auth 只预留接口，不实装、不改动；
> 端口约定以生态架构图为准但本文不绑定（用户指示：端口问题搁置）。
> 基座：mcp-base-gpu（flowmind）。全部能力 `@skill` 化，注册即自动进 manifest，
> 经联邦网关聚合后前端/Agent 可直接消费。
> 执行后端：`editing.backend = "cybercut"`（默认，PR#23 桥接）或 `"flowmind"`
> （内置流水线，本文主体；真实 LLM 创意层目前仅在 flowmind 后端生效）。

## 1. 定位

剪映式一站式智能剪辑的**纯后端 Agent 服务**：

- 投喂原始素材（视频 / 图片 / 音频），Agent 自动完成
  **素材理解 → 文案 → 脚本编排 → 分镜 → 剪辑渲染 → 成片**；
- 每一步产出结构化中间产物 + 四段式推理链反馈；
- **全自动为主**，逐阶段可插人工审批（approve / revise 带新提示词 / reject）；
- 批量出片；无 GUI，接口交给前端（cross-dashboard 等）。

## 2. 全流水线（6 阶段）

```
understand   素材理解   ffprobe 元数据 + 场景切分 + ASR 转写(可选) + LLM 标签
   ↓
copy         文案       LLM：标题/钩子/口播文案/CTA（基于素材理解摘要）
   ↓
script       脚本编排   LLM：结构化脚本（段落/口播/画面需求/时长预算/字幕）
   ↓
storyboard   分镜       LLM：脚本段 → 镜头序列（素材片段匹配/裁剪点/转场/字幕样式/BGM）
   ↓
render       剪辑渲染   ffmpeg：拼接/图片序列 Ken Burns/转场 xfade/字幕烧录/TTS 口播(可选)
   ↓
done         成片       mp4 + SRT + timeline.json + 全部中间产物 JSON
```

每个阶段产物落盘 + 落库；审批开关开的阶段在产出后暂停（`awaiting_review`）。

## 3. 技能面（对外接口，v1.2 共 5 个）

全部走统一 `SkillResult` 信封；长任务走 TaskManager（202 受理 → 进度事件 → 终态）。

### 3.1 `edit_project` — 创建剪辑项目（长任务）

入参 `EditProjectInput`：

| 字段 | 类型 | 说明 |
|---|---|---|
| assets | list[str] | 素材路径/URL（视频/图片/音频混合；本地路径受 uploads 沙箱约束，同 localize_submit） |
| theme | str | 创作主题/要求（“科技感产品宣传片”“探店 vlog 配口播”） |
| style | str\|None | 风格关键词（节奏/情绪/配色），进 LLM 提示词 |
| target_duration_s | float\|None | 目标时长；None=LLM 按素材自定（上限 config.editing.max_duration_s） |
| aspect_ratio | str\|None | 输出画幅 `16:9`/`9:16`/`1:1`；None=按首个视频素材 |
| voiceover | bool | 是否 TTS 口播（默认 False；True 需配置 TTS key，走 dashscope） |
| subtitle | bool | 是否烧录字幕（默认 True） |
| review_stages | list[str]\|None | 需人工审批的阶段；None=config 默认 `["storyboard"]`；`[]`=全自动。copy/script/storyboard 闸在对应阶段产出后停靠，pre_render 闸在进入渲染前停靠（v1.2 起全部生效） |
| extra_brief | str\|None | 追加提示词（进所有 LLM 阶段） |

出参 `EditProjectReport`：`project_id`（=task_id）、`assets_accepted/rejected`、
`review_stages`、`next`（提示前端“轮询 edit_status / 事件订阅”）。

**批量**：assets 多条 = 单项目多素材；**多项目批量** = 循环提交（背压语义复用 429）。

### 3.2 `edit_status` — 项目状态与产物

入参：`project_ids: list[str]`。出参每项目：

| 字段 | 说明 |
|---|---|
| project_id / task_id | 同一 id（项目任务即引擎任务，进度/取消/下载全复用） |
| state | `running` / `awaiting_review` / `succeeded` / `failed` / `cancelled` / `interrupted` |
| stage | 当前阶段（understand…render）+ 各阶段 `done/pending/failed` |
| review | awaiting_review 时给出：待审阶段、产物摘要、Agent 决策理由（推理链） |
| artifacts | 中间产物清单（copy.json / script.json / storyboard.json / …） |
| outputs | 成片产物（mp4/srt/timeline.json + 下载通道同 localize_download 语义） |

### 3.3 `edit_review` — 审批动作（人工复查回路）

入参：

| 字段 | 说明 |
|---|---|
| project_id | 项目 id |
| stage | 待审阶段名 |
| action | `approve`（放行下一阶段）/ `revise`（重跑本阶段）/ `reject`（终止项目） |
| feedback | revise/reject 时的反馈文本（进该阶段 LLM 提示词，**只重跑本阶段及之后**） |

revise 语义：该阶段产物作废 → 带 feedback 重跑 → 若其后还有审批点继续暂停。

### 3.4 `edit_assets` — 素材库（v1 轻量）

入参 `action: list|register`：登记素材元数据/标签（understand 阶段自动登记），
按 `type/tags` 过滤列出。语义向量检索预留（Milvus，v2 接 BGE）。

### 3.5 `edit_retry` — 失败/取消项目断点续跑（v1.2）

入参：`project_id`。语义对齐 `localize_retry`：

- `failed` / `cancelled`（终态）→ 定位第一个未 done 阶段置 pending、清 error、
  提交新 `edit_pipeline` 任务续跑——done 阶段与中间产物**全复用**；
- `succeeded` / `running` / `awaiting_review` → 拒绝（degraded）；
  审批等待请走 `edit_review`；
- 仅 flowmind 后端支持；cybercut 后端的节点重试由其执行器内部处理。

## 4. 审批状态机

```
                    ┌──────────── reject ───────────┐
                    ▼                               ▼
created → running(stage_i) → [stage_i 有审批?] ─no→ running(stage_i+1) …→ succeeded
                    │              │yes
                    │              ▼
                    │      awaiting_review(stage_i)
                    │       ├─ approve  → running(stage_i+1)
                    │       ├─ revise(feedback) → running(stage_i_i 重跑)
                    │       └─ reject → failed(review_rejected)
                    └─ 失败/取消 → failed / cancelled（引擎终态复用）
```

- 状态随任务落库（PG 表 `editing_projects`，JSONB payload；PG 不可达回落
  `data_dir/editing/*.json` 文件存储，demo/独立部署可用）；
- 审批等待**不占用 GPU worker**（任务在阶段边界安全退出并落 awaiting 态，
  resume 由 edit_review 重新驱动）；
- 失败/取消（引擎终态）→ `edit_retry` 断点续跑（v1.2）：done 阶段与产物全复用，
  从第一个未完成阶段继续；
- 租户隔离：读写经 `auth.tenant_scope()`，跨租户同形不可见（复用引擎语义）。

## 5. 中间产物 Schema（前端二次编辑的数据面）

全部 JSON 落 `data_dir/editing/<project_id>/`：

- `understand.json` — `assets[]: {id, kind(video/image/audio), path, duration_s, width,
  height, fps, shots[]: {start,end,score}, asr[]: {begin,end,text}, tags[], summary}`
- `copy.json` — `{title, hook, body_sections[], cta, full_narration, style_notes}`
- `script.json` — `{sections[]: {id, purpose, narration, visual_brief, duration_s, subtitle_text}}`
- `storyboard.json` — `{shots[]: {id, section_id, source_asset, source_in, source_out,
  duration_s, scale/pan(图片 Ken Burns), transition, subtitle_text, note}, bgm: {...}}`
- `timeline.json` — render 直接消费的最终时间线（同 storyboard + 绝对时间轴 + 混音增益）
- `*.srt` — 字幕；`output.mp4` — 成片

> 前端拿到 timeline.json + 中间产物即可做二次编辑（改分镜重渲、换段落重生成）。

## 6. 事件与进度（前端实时反馈）

复用任务引擎事件通道（MQTT `mcp-base-gpu/tasks/{id}/events`，终态 retain）：
`progress_cb(stage, pct, message)` 逐阶段上报；message 携带
`{"stage": "storyboard", "event": "awaiting_review"}` 语义（JSON 串）。
前端亦可轮询 `edit_status`。**预留**：SSE/WebSocket 适配层（go-kernel WS 透传）。

## 7. 渲染执行（render 阶段 v1 能力）

- 视频素材：按 storyboard 区间裁剪 → 三段式渲染（逐镜归一 → xfade 链/concat
  → 字幕烧录+音轨）；镜头时长量化到帧网格，时间线/转场折算/实际段长同网格
  （offset 恒等式，防尾镜截断——离线渲染回归 `examples/edit_render_check.py` 兜底）
- 图片素材：Ken Burns（zoompan，单帧输入 d=N + `-frames:v` 封底；下游不重复过
  fps 滤镜——实测会丢帧）
- 转场：xfade（fade / none=1 帧 epsilon 硬切，逐镜声明；全 none 走 concat）
- 字幕：SRT → `subtitles` filter 烧录（`subtitle=False` 跳过生成与烧录）；
  Windows 盘符冒号转义 + `-vf` 必须在输入之后
- 口播：`voiceover=True` 时 dashscope TTS；音轨 `apad` 垫底 + `-shortest`，
  口播短于成片不再截断视频
- 产物编码 libx264 + aac；渲染后抽 preview/ 自检帧；失败显式走 degraded/failed，
  绝不静默

**不在 v1 渲染范围（预留后端适配器）**：BGM 自动选段混音（storyboard.bgm_note
只记意向）、超分、声音克隆（RVC）、扩散重绘。timeline.json 设计上可插入这些
处理节点。

## 8. LLM 接入（创意层）

- 协议：Anthropic 兼容 `/v1/messages`（httpx 直调；LongCat / MiniMax 兼容端点，
  模式同 `_llm_translate`）；key 经 `_secrets.get_api_key("AI_LLM_API_KEY")`，
  **无 key 显式报错不降级**；
- 每阶段独立系统提示词 + 结构化 JSON 输出（pydantic 校验，解析失败带原文入 error）；
- **两层韧性（v1.2，实测 MiniMax 教训）**：
  - 瞬时失败（网络/408/429/5xx/空 content/`stop_reason=max_tokens`）指数退避
    重试（`llm_max_attempts`，默认 5），尊重 `Retry-After`；配额型 429 重试耗尽
    升级 environment——understand 的 vision 随之**快速失败**（同一 key 的
    copy/script 必然也失败，盲剪只会浪费整条流水线），transient 类仍降级盲剪；
  - 解析/schema 失败再做一轮多轮对话**自纠**（报错+原输出喂回，要求只输出修正
    JSON）；storyboard 归一化兜住同义词转场/顶层 list/超限值（钳制与模型一致）；
- `extra_brief`/`feedback` 注入对应阶段提示词；storyboard 素材目录超预算
  （>150k 字符）按重要度逐档裁剪 content_notes；
- Vision（v1.1 已实装）：抽帧 → image block → 每帧 {t_s, scene, subtitle,
  importance} 内容笔记，分镜靠它"看见"素材；失败显式记 `vision_degraded`。

## 9. 配置（`config.editing`，全走通用默认）

见 `EditingConfig`：模型名/超时/默认审批阶段/最大时长/默认画幅/字幕字体/
场景切分阈值/抽帧数/TTS 音色/口播语速/存储回落开关等。

## 10. go-kernel / 前端预留（不实装）

- 新技能注册后自动进 manifest → 联邦注册开启即以 `{prefix}__edit_*` 出现在
  网关 `/mcp`——**零网关改动**；
- 长任务/下载复用既有 `/api/v1/tasks` REST 透传通道；
- 审批 `edit_review` 是普通工具调用，经 MCP 即达（无需新 REST 路由）；
- 租户：调用链 Bearer → RakAuthMiddleware → task tenant 归属（已实装）。

## 11. v1 交付清单

1. `src/flowmind/editing/` 包：models / store / pipeline / llm / media（analyze 并入 llm.describe_frames）
2. 技能：`edit_project` / `edit_status` / `edit_review` / `edit_assets` / `edit_pipeline`（引擎入口）/ `edit_retry`（v1.2）
3. `config.py`：`EditingConfig` 纳入 `FlowmindConfig`
4. demo：`examples/edit_pipeline_demo.py`（mock 流水线：全自动 + 审批 revise→approve + copy 闸 + edit_retry + 错误路径）
5. 渲染回归：`examples/edit_render_check.py`（零 LLM，真实 ffmpeg 断言成片时长/自检帧）
6. ruff 全绿 + demo PASS + 真实 invoke 冒烟

## 12. v1.1 → v1.2 变更记录（2026-10-30）

- **LLM 调用层**：429/408 退避重试 + Retry-After + 配额墙快速失败；max_tokens
  截断显式报错；schema 自纠一轮；storyboard 归一化防御（顶层 list/非 dict）；
  speed/freeze 钳制与模型上限统一（0.25~4 / 0~3）；素材目录载荷预算守卫。
- **流程**：copy/script 审批闸生效 + pre_render 闸改渲染前停靠（续跑可命中）；
  新增 `edit_retry` 断点续跑；`subtitle=False` 生效；移除死旋钮
  `config.editing.transition`；storyboard section_id 漂移置空告警 + 预算偏离告警；
  understand 阶段 vision 配额类失败快速失败。
- **渲染**：修复 kenburns 双重 fps 滤镜丢帧（60→59）与 xfade offset 舍入越界
  两大截断根因（镜头时长量化到帧网格 + Pass2 去冗余归一 + offset 向下取整）；
  口播 `apad` 垫底不再截断视频；`scene_shots` 超时归类 MediaError；
  `_resolve_format` 兑现"None=按首个视频素材画幅就近映射"。
- **存储/工具**：`list_projects` 租户过滤下推（PG WHERE / 文件先滤后截）；
  `examples/edit_render_check.py` 离线渲染回归；demo 扩至 5 场景。
