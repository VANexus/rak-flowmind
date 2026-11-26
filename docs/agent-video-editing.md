# Agent 智能视频剪辑（edit_*）— 架构设计 V1

> 状态：**设计已定稿，随实现同步维护**。
> 范围：仅 FlowMind（:8002 侧）。go-kernel / rak-auth 只预留接口，不实装、不改动；
> 端口约定以生态架构图为准但本文不绑定（用户指示：端口问题搁置）。
> 基座：mcp-base-gpu（flowmind）。全部能力 `@skill` 化，注册即自动进 manifest，
> 经联邦网关聚合后前端/Agent 可直接消费。

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

## 3. 技能面（对外接口：基础 4 个 + OpenFilm 可选 2 个）

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
| review_stages | list[str]\|None | 需人工审批的阶段；None=config 默认 `["storyboard"]`；`[]`=全自动 |
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
按 `type/tags/text` 过滤列出。语义向量检索预留（Milvus，v2 接 BGE）。

### 3.5 OpenFilm 可选后端（`editing.backend="openfilm"`）

OpenFilm 是**可选**的创作/Studio/渲染后端，默认关闭；CyberCut 与旧 FlowMind
流水线保持不变。产品边界与完整操作步骤见
[openfilm-integration.md](openfilm-integration.md)。

- `edit_project`：创建 `data_dir/editing/<id>/openfilm` 工作区（空白
  `film.html` + `AGENTS.md` + 复制到 `assets/` 的初始素材），项目落
  `awaiting_agent`，**不提交**任何创作任务（FlowMind 无内置编程 Agent）。
- 外部编程 Agent 在工作区写 `film.html` / 页面 / `assets/`（非 FlowMind 运行时）。
- `edit_openfilm_studio`（新）：只收 `project_id`，启动/复用 Studio，返回本机
  `studio_url` 供外部浏览器打开；不传路径/命令。
- `edit_openfilm_render`（新）：只收 `project_id`；有 `pre_render` 闸时先落
  `awaiting_review`，否则预检 Node/OpenFilm/FFmpeg 后提交 `edit_pipeline` 任务；
  渲染进度/取消/终态复用任务引擎，成片固定 `<project_dir>/output.mp4`。
- `edit_status`：OpenFilm 项目额外返回 `backend/workspace/studio_url/progress/
  render/feedback/outputs/stages/next`；`edit_review` 的 approve/revise/reject
  对 `pre_render` 闸复用（revise 退回 `awaiting_agent`，不渲染）。
- 依赖可选：普通 CyberCut/旧流水线用户无需安装 Node/OpenFilm/FFmpeg；OpenFilm
  项目在缺依赖时得到显式环境错误，不静默回落。
- 旧记录路由：优先 `ProjectRecord.backend`；无该字段时按 `input.cybercut` /
  `input.openfilm` 桥接字段判断；**两者都没有的升级前记录保守回落 `flowmind`**，
  不跟随全局 `editing.backend`，避免全局切 OpenFilm 后误路由旧项目。详见
  [openfilm-integration.md §6.1](openfilm-integration.md#61-旧项目的后端路由兼容规则)。

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

- 视频素材：按 storyboard 区间裁剪 → concat（统一分辨率/帧率 pad+fps filter）
- 图片素材：序列成片，Ken Burns（zoompan）简版
- 转场：xfade（fade/slide 两档，v1）
- 字幕：SRT → `subtitles` filter 烧录（Windows 字体名走 config，防中文豆腐块）
- 口播：`voiceover=True` 时 dashscope TTS（复用 `_cloud_tts`），amix 混音保 BGM
- 产物编码 libx264 + aac；失败显式走 degraded/failed，绝不静默

**不在 v1 渲染范围（预留后端适配器）**：去字幕水印（ProPainter 链路，独立任务线）、
超分、声音克隆（RVC）、扩散重绘。timeline.json 设计上可插入这些处理节点。

## 8. LLM 接入（创意层）

- 协议：Anthropic 兼容 `/v1/messages`（LongCat，模式同 `_llm_translate`）；
  key 经 `_secrets.get_api_key("AI_LLM_API_KEY")`，**无 key 显式报错不降级**；
- 每阶段独立系统提示词 + 结构化 JSON 输出（pydantic 校验，解析失败带原文入 error）；
- `extra_brief`/`feedback` 注入对应阶段提示词；
- Vision（抽帧打标）预留 image block，v1 默认文本标签（ASR+元数据）。

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

1. `src/flowmind/editing/` 包：models / store / pipeline / llm / media / analyze
2. 技能：`edit_project` / `edit_status` / `edit_review` / `edit_assets`（注册进 skills/__init__.py）
3. `config.py`：`EditingConfig` 纳入 `FlowmindConfig`
4. demo：`examples/edit_pipeline_demo.py`（mock 流水线：全自动 + 审批回路 + revise 三段式）
5. ruff 全绿 + demo PASS + 真实 invoke 冒烟
