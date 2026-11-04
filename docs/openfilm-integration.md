# OpenFilm 能力接入（FlowMind 主产品）

> 状态：**可选后端，默认关闭**。默认仍是 CyberCut（`editing.backend = "cybercut"`），
> 旧 FlowMind 流水线（`"flowmind"`）也保留；切换到 `"openfilm"` 才启用本文能力。
> OpenFilm 版本基线：`openfilm@0.2.7`（CLI 的 `open` / `render` 行为以该版本源码为准）。

## 1. 产品边界（已定，不随实现漂移）

```
用户 / 上层 Agent
      │  只认 FlowMind 的 project_id 与 edit_* 技能
      ▼
FlowMind（主产品：项目记录、状态机、审批、任务引擎、成片管理）
      │  edit_project(openfilm) 建工作区
      │  edit_openfilm_studio   打开 Studio（外部浏览器窗口）
      │  edit_openfilm_render   提交渲染任务
      ▼
OpenFilm 项目工作区（data_dir/editing/<project_id>/openfilm）
      ▲
      │  外部编程 Agent（Codex / Claude Code / 其它）按 AGENTS.md 创作
      │  film.html + 页面 + assets/（FlowMind 不内置编程 Agent 运行时）
```

- **FlowMind 是任务入口、项目状态与成片管理方**；OpenFilm 不是需要用户另行管理的
  平行产品。
- **外部编程 Agent 负责创作 OpenFilm 项目内容**。FlowMind 只创建工作区、复制初始
  素材、调用 OpenFilm Studio/渲染、汇报状态与成片，不内置、也不调用任何内置
  “创作 Agent”。
- **Studio 在外部窗口打开**：`edit_openfilm_studio` 启动/复用本机 Studio 并返回
  `studio_url`，由调用方在浏览器打开。默认不擅自拉起系统浏览器
  （`FLOWMIND_OPENFILM_NO_BROWSER=0` 可交还 OpenFilm CLI 的默认行为）。
- **OpenFilm 依赖可选**：不安装 Node/OpenFilm/FFmpeg 不影响 CyberCut 与旧流水线；
  只是 OpenFilm 项目在 Studio/渲染时会得到显式的环境错误。

## 2. 安装与配置（可选）

运行 OpenFilm 需要：

- Node **>= 22.2**（OpenFilm CLI 的 engines；本仓库不捆绑 Node）
- OpenFilm CLI（`npm i -g openfilm`，或指向源码/局部安装的
  `packages/openfilm/bin/openfilm.mjs`）
- **FFmpeg + FFprobe**（渲染必需；Studio 的视频画面/波形也需要）
- 首次运行会下载 headless Chromium（约 100 MB，OpenFilm 自身行为）

FlowMind 不把它们放进 Python 依赖；请按需安装。配置优先级与其他模块一致：
**环境变量 → `flowmind.config.toml` → 内置默认**。

### 2.1 环境变量（`.env`）

```dotenv
# OpenFilm CLI：可指向 openfilm.mjs（用 Node 执行）或原生可执行文件/命令名
FLOWMIND_OPENFILM_CLI=
# 可选：显式指定 Node（CLI 是 .mjs 时）
FLOWMIND_OPENFILM_NODE=
# 可选：只注入 OpenFilm 子进程的 ffmpeg/ffprobe bin 目录（不改进程全局 PATH）
# 未配置时回退 FLOWMIND_MEDIA_BIN_DIR，再回退系统 PATH
FLOWMIND_OPENFILM_FFMPEG_BIN_DIR=
# 1（默认）= 不自动拉起系统浏览器；Studio 服务照常启动并返回地址
FLOWMIND_OPENFILM_NO_BROWSER=1
```

### 2.2 `flowmind.config.toml`

```toml
[editing]
# 默认 cybercut；显式切到 OpenFilm 才启用本文能力
backend = "openfilm"

[editing.openfilm]
# 示例：npm 局部安装 / 源码 checkout 的 bin/openfilm.mjs
cli = "C:/tools/openfilm/node_modules/openfilm/bin/openfilm.mjs"
# node = "C:/Program Files/nodejs/node.exe"
ffmpeg_bin_dir = "C:/ffmpeg/bin"
open_timeout_s = 300       # 首次 open 可能下载 Chromium
render_timeout_s = 3600
render_fps = 30
no_browser = true
# 外部 Agent 完成项目后先审片再渲染；[] = 全自动
default_review_stages = ["pre_render"]
```

> `backend` 是全局的（与 CyberCut/FlowMind 的既有语义一致）。切换后**新建**项目
> 走 OpenFilm；已有项目记录仍按自己的 `backend` 字段/桥接字段读取，不会把旧项目
> 当 OpenFilm 处理。

CLI 解析顺序：配置的 `cli` → PATH 上的 `openfilm`。显式 `.mjs` 用 Node 执行并校验
Node >= 22.2；POSIX 原生可执行文件直接以参数数组启动。

Windows 上 `npm i -g openfilm` 在 PATH 里通常是 `openfilm.cmd` shim：FlowMind
会读取 shim、解析其中的 `openfilm.mjs` 入口，再以 `node <mjs>` 的参数数组启动，
**不会**通过 `cmd.exe` / `shell=True` 执行。若 shim 结构无法解析，错误会提示把
`FLOWMIND_OPENFILM_CLI` 直接指向 `openfilm` 包的 `bin/openfilm.mjs`。

## 3. 从 FlowMind 使用的完整步骤

下面以 MCP/`invoke` 的入参为例；REST 长任务通道仍只用于 `edit_pipeline` 渲染任务，
项目创建/Studio/审批走技能调用。

### 步骤 1：创建 FlowMind 项目（自动建 OpenFilm 工作区）

```json
{
  "theme": "20 秒产品发布片，干净、科技感",
  "style": "clean tech",
  "target_duration_s": 20,
  "aspect_ratio": "16:9",
  "assets": ["<data_dir>/uploads/logo.png"],
  "review_stages": ["pre_render"]
}
```

调用 `edit_project`。返回字段：

| 字段 | 含义 |
|---|---|
| `project_id` | FlowMind 项目 id，后续所有操作只用它 |
| `state` | `awaiting_agent`（等待外部 Agent 创作） |
| `task_id` | 空串：OpenFilm 创建阶段**不提交**任何创作任务 |
| `backend` | `openfilm` |
| `workspace` | 工作区绝对路径 `data_dir/editing/<id>/openfilm` |
| `entry_file` | `<workspace>/film.html`（已生成空白片） |
| `agent_brief` | 给外部编程 Agent 的下一步说明 |
| `review_stages` | 默认 `["pre_render"]`；传 `[]` 则全自动 |

工作区内容：

```
data_dir/editing/<project_id>/
├── project.json（按存储配置可能在 PG）
├── openfilm/
│   ├── film.html              # OpenFilm 入口（空白片，外部 Agent 改写）
│   ├── flowmind-project.json  # FlowMind 元数据（主题/素材/审批配置）
│   ├── AGENTS.md              # 外部 Agent 创作说明
│   ├── assets/                # edit_project 已复制的初始素材
│   └── .film/                 # OpenFilm 自己创建（Studio 打开后）
└── output.mp4                 # 渲染成片（成功后）
```

### 步骤 2：外部编程 Agent 在工作区创作

把 `workspace` 交给 Codex / Claude Code 等外部编程 Agent，要求它：

1. 阅读 `<workspace>/AGENTS.md` 与 OpenFilm 官方
   [MANUAL](https://github.com/openfilm/openfilm/blob/main/packages/openfilm/MANUAL.md)；
2. 只在 `<workspace>` 内写 `film.html`、页面 `*.html`、`assets/`；
3. 不要改 `flowmind-project.json` / `FLOWMIND_REVIEW.json`；
4. 完成后交给 FlowMind 的 `edit_openfilm_render(project_id)`。

FlowMind 不运行该 Agent，也不为它提供模型/工具；这是**外部**步骤。

### 步骤 3：（可选）打开 OpenFilm Studio

调用 `edit_openfilm_studio`，入参只有 `{"project_id": "<id>"}`。

- 返回 `opened=true`、`studio_url`（本机地址）、`project_path`；
- 在**外部浏览器窗口**打开 `studio_url` 即可预览/手动编辑；
- Studio 的单实例在本机复用；同一台机器上的多个 FlowMind 项目会在 Studio 页面间切换；
- 人工在 Studio 的改动直接落到同一工作区，外部 Agent 下次读写能看到；
- 若 OpenFilm/Node/Studio 不可用，返回 `degraded=true` + `degradation_reason`，
  `opened=false`，不会静默成功。

### 步骤 4：渲染

调用 `edit_openfilm_render`，入参只有 `{"project_id": "<id>"}`。

- 若项目 `review_stages` 含 `pre_render` 且尚未放行：项目进入
  `awaiting_review`，返回 `task_id=null`，不会启动渲染；
- 否则：做 Node/OpenFilm/FFmpeg 预检 → 提交 `edit_pipeline` 任务 →
  返回渲染 `task_id`；任务引擎负责进度、取消、终态；
- 渲染输出固定为 `<project_dir>/output.mp4`（先写临时文件，成功后原子替换）；
- 失败/取消**不会留下半截成片**，也不会把旧成片冒充成本次结果。
- **同一项目最多一个活跃渲染领取/任务**：`prepare_render` 在项目级跨进程锁内
  读-检查-领取，claim 持久化到项目记录；并发第二次调用显式失败（degraded
  validation），不会重复提交。取消/失败后 claim 被清理，可重新调用。
- 每次渲染的临时文件带 UUID（不是 PID），同进程并发任务也不会争用；成功后
  原子替换 `output.mp4`，失败/取消清理自己的临时文件。
- `edit_openfilm_render` 的响应从**最新项目记录**读取：异步排队时
  `state=running / render_status=queued / output_path=null`；inline/极快完成时
  返回 `state=succeeded / render_status=succeeded / output_path=<mp4>`，字段一致。

### 步骤 5：审批（复用既有 `edit_review`）

`edit_status` 看到 `state=awaiting_review`、`stage=pre_render` 时：

| 动作 | 效果 |
|---|---|
| `edit_review approve` | 记入 `approved`，提交渲染任务；`task_id` 返回 |
| `edit_review revise` + `feedback` | 反馈写入工作区 `FLOWMIND_REVIEW.md/.json`；项目回到 `awaiting_agent`，**不渲染**；外部 Agent 改稿后再次调用 `edit_openfilm_render` |
| `edit_review reject` + `feedback` | 项目 `failed`（`review_rejected`），不再渲染 |

### 步骤 6：查询状态与成片

调用 `edit_status`，入参 `{"project_ids": ["<id>"]}`。OpenFilm 项目额外返回：

| 字段 | 说明 |
|---|---|
| `backend` | `openfilm` |
| `workspace` / `entry_file` | 工作区与入口文件绝对路径 |
| `studio_url` | 最近一次成功的 Studio 地址（可能为空） |
| `progress` | 当前渲染任务的进度（0–100，来自任务引擎） |
| `render` | `status` / `task_id` / `task_status` / `task_progress` / `task_error` / `bytes` / `finished_at` 等 |
| `stages` | `authoring` / `pre_render` / `render` 三阶段状态 |
| `feedback` | `pre_render` 的 revise 反馈历史 |
| `artifacts` | 工作区相对产物清单（`openfilm/film.html`、页面、素材等，限制在工作区内） |
| `outputs` | `{"output.mp4": "<project_dir>/output.mp4"}`（成片存在时） |
| `next` | 当前状态下的下一步提示 |
| `review` | `awaiting_review` 时的闸门、反馈、摘要与操作提示 |

`state` 取值：`awaiting_agent` → `awaiting_review`（若开闸）→ `running` →
`succeeded` / `failed` / `cancelled`。与既有项目状态机一致，没有另立一套。

## 4. 取消、错误与降级语义

| 情况 | FlowMind 行为 |
|---|---|
| 缺 Node | `edit_openfilm_render`/`studio` 返回 `degraded=true`，`degradation_reason=environment`；项目记录失败原因 |
| 缺 OpenFilm CLI | 同上，提示安装 `openfilm` 或配置 `FLOWMIND_OPENFILM_CLI` |
| 缺 FFmpeg/FFprobe | 渲染预检显式失败（不会启动一个注定失败的 CLI），错误含安装/目录配置提示 |
| 超时 | 杀整棵子进程树（Windows `taskkill /T`，POSIX 进程组），项目 `failed`，`error` 含超时 |
| 非零退出码 | 项目 `failed`，`error` 含退出码与 stderr 尾部（截断，不刷屏） |
| CLI 退出 0 但无输出 | 项目 `failed`，显式“未产出非空 output.mp4” |
| 渲染取消 | 任务引擎 `cancel` → 子进程树终止 → 临时成片删除 → 项目 `cancelled` |
| 工作区/输出越界 | `validation` 错误；路径解析在 FlowMind 项目目录内校验 |
| Studio 返回非本机地址 | 拒绝写入状态（防把 FlowMind 当任意 URL 代理） |
| 项目不是 OpenFilm 后端 | 显式 `validation` 错误，提示用 `backend=openfilm` 创建 |

取消渲染：`edit_status` 返回的 `render.task_id` 可交给既有 `localize_cancel`
（任务引擎的通用取消入口）；项目状态随 worker 协作取消落 `cancelled`。

## 5. 安全与工作区约束

- **对外 API 只有 `project_id`**：`edit_openfilm_studio` / `edit_openfilm_render`
  的入参 schema 不含 `path` / `command` / `args` / `output`，调用方无法传任意
  命令或任意文件路径。
- **工作区固定**：`<data_dir>/editing/<project_id>/openfilm`；素材复制到其
  `assets/`；成片固定 `<project_dir>/output.mp4`。所有路径 resolve 后校验
  仍在 FlowMind 项目目录内（含符号链接/`..`/绝对路径）。
- **素材来源**：`edit_project` 的初始素材仍走既有 `data_dir/uploads/` 沙箱与
  扩展名白名单；OpenFilm 不绕过该收口。
- **子进程**：参数数组 + `shell=False`；CLI 命令由适配器按白名单构造
  （`open <workspace>`、`render --fps N --out <内部路径>`）；FFmpeg bin 目录只
  注入本次子进程的 `PATH`，不改全局环境。
- **Studio 地址**：只接受 `http(s)://` 且 host 为 `localhost` / `127.0.0.1` / `::1`
  的地址；返回地址写入项目状态。
- **外部 Agent 的边界**：FlowMind 校验自己的 API 与产物路径，但外部编程 Agent
  是独立进程，理论上可访问工作区外文件；产品约定它只写工作区。若需要强隔离，
  应在部署层给外部 Agent 单独的沙箱/容器与只读挂载。

## 6. 与既有后端的关系

| backend | 创作 | Studio | 渲染 | 默认 |
|---|---|---|---|---|
| `cybercut` | CyberCut DAG（创意层默认 Fake） | 无 | CyberCut render | **是** |
| `flowmind` | 旧 FlowMind LLM 流水线 | 无 | ffmpeg | 可切回 |
| `openfilm` | **外部编程 Agent** | OpenFilm Studio（外部浏览器） | OpenFilm CLI（Chromium+ffmpeg） | 可选 |

- OpenFilm 不替换任何现有后端；`load_config().editing.backend` 默认仍是
  `cybercut`，未安装 OpenFilm 的普通用户不受影响。
- OpenFilm 项目与 CyberCut 项目共享 ProjectStore / 任务引擎 / 租户隔离 / 审批语义，
  不另建平行项目库。

### 6.1 旧项目的后端路由兼容规则

项目记录可能早于 `backend` 字段。`resolve_backend()` 的规则是**优先项目自身、
绝不回退当前全局设置**：

| 项目记录 | 路由 |
|---|---|
| `backend` 为 `cybercut` / `flowmind` / `openfilm` | 按记录值 |
| 无 `backend`，但 `input.cybercut` 存在 | `cybercut` |
| 无 `backend`，但 `input.openfilm` 存在 | `openfilm` |
| 无 `backend`，也无两种桥接字段 | **`flowmind`（保守默认）** |

因此把全局 `editing.backend` 切到 `openfilm`，不会让升级前创建的待处理/运行中
项目误入 OpenFilm。代价是极少数“CyberCut 项目已落库、但 worker 尚未写入
`input.cybercut` 桥接字段”的旧记录会按旧 FlowMind 处理——这是有意的保守取舍；
若这类项目需要继续，请在升级前确保其已进入 worker/写入桥接字段。

## 7. 排障与验证

```bash
# 环境自检（不启动 Studio、不写项目）
PYTHONPATH=src python -c "from flowmind.editing.openfilm_backend import check_ready; print(check_ready())"

# 技能注册与 schema
PYTHONPATH=src python -c "
import flowmind.skills
from flowmind.skill import registry
print([s for s in registry() if 'openfilm' in s])
"

# 测试（适配器 + 技能 + CyberCut 回归）
PYTHONPATH=src python -m pytest tests/test_openfilm_backend.py tests/test_openfilm_skills.py tests/test_cybercut_backend.py -q
```

OpenFilm 官方命令（在项目工作区内手动排障时）：

```sh
openfilm open <workspace>       # 打开/复用 Studio，打印 project/Studio 地址
openfilm look                   # 接触表 + 帧一致性检查
openfilm render --out <file>    # FlowMind 内部实际调用的形态
openfilm doctor                 # 环境报告
```
