# examples —— 不需要 MCP 客户端也能体验技能

每个脚本都是**自包含**的最小可运行示例（mock 全部外部依赖，无需
GPU / PG / MQTT / Milvus / API key）：

```bash
PYTHONPATH=$PWD/src conda run -n flowmind python examples/<name>_demo.py
```

## 18 个 demo 一览

| 脚本 | 演示能力 | 你将看到 |
|---|---|---|
| [`localize_submit_demo.py`](./localize_submit_demo.py) | 批量提交本地化任务 | 3 视频受理、扩展名预检分桶、队列中途满 partial、全满 429 背压 |
| [`localize_status_demo.py`](./localize_status_demo.py) | 批量状态查询 | 并发轮询、stalled 卡住判定、per-task 404 → not_found、store 读失败 → INTERNAL |
| [`localize_download_demo.py`](./localize_download_demo.py) | 产物清单 + 下载 URL | happy path 产物寻址、空结果/未完成/不存在 → degraded、404 分类 |
| [`localize_retry_demo.py`](./localize_retry_demo.py) | 失败任务重提 | 复制原 args 重提、running/succeeded 拒绝重提、队列满 → ok=False |
| [`localize_cancel_demo.py`](./localize_cancel_demo.py) | 协作式取消 | queued 直落终态、running 阶段边界生效、终态幂等、不存在 → video |
| [`localize_search_demo.py`](./localize_search_demo.py) | 字幕语义检索（BGE + Milvus） | 向量命中、task_id 过滤、空库空态、服务不可用 → ok=False |
| [`localize_video_demo.py`](./localize_video_demo.py) | 本地化流水线本体 | mock 全链路（ASR→翻译→擦除→克隆 TTS→合成）、无 key 显式 degraded、文件不存在 |
| [`api_server_demo.py`](./api_server_demo.py) | 单端口 REST 任务通道冒烟 | health/manifest 发现、POST 202 → 轮询 → 流式下载、422/429 错误语义 |
| [`auth_token_demo.py`](./auth_token_demo.py) | 调用方凭证校验（RakAuthMiddleware） | Bearer 验签、豁免前缀匿名可达、tenant_scope 三态 fail-closed、`X-Rak-Tenant` 被忽略 |
| [`federation_register_demo.py`](./federation_register_demo.py) | 联邦自注册（PG+MQTT 双通道） | register QoS1 payload 契约、单通道降级仍完成注册、双通道全挂静默降级、心跳与优雅注销 |
| [`dsp_timeline_build_demo.py`](./dsp_timeline_build_demo.py) | dsp 素材对齐 · **切镜收口** · 时间线锁定 | 实测逐句归属到镜头、比例兜底如实告警「归属为推断」、无口播镜吃停顿/尾部静默、收口并镜/拆镜、转场符号落 EDL、非法镜头时长 → VALIDATION |
| [`dsp_tts_generate_demo.py`](./dsp_tts_generate_demo.py) | dsp 口播配音（豆包 · **DAG 起点**） | 无上游依赖也跑通、`text_prompt` 纯文案、出音轨 + **逐句实测时间戳**、预计超 120s / 无 shots → 明确报错 |
| [`dsp_image_to_video_demo.py`](./dsp_image_to_video_demo.py) | dsp 图生视频（逐镜视频底 · §7.5 第 1 步） | 直链**转存**为不可变 key、未配密钥/缺锁定时间线/单镜出片失败各自如实降级、`maxShots` 成本闸 |
| [`dsp_bgm_match_demo.py`](./dsp_bgm_match_demo.py) | dsp 背景音乐（**占位降级**） | 契约 §3.3：选曲接口未给 → 一律 degraded，成片空轨继续 |
| [`dsp_subtitle_build_demo.py`](./dsp_subtitle_build_demo.py) | dsp 字幕 SRT | 依配音逐句时间戳出 SRT、时间点越界**如实报出**、无时间戳如实降级 |
| [`dsp_cover_pick_demo.py`](./dsp_cover_pick_demo.py) | dsp 封面取帧 | 默认取第一镜的图、标题未确认即 degraded（不阻塞）、无图不臆造 |
| [`dsp_cart_sticker_demo.py`](./dsp_cart_sticker_demo.py) | dsp 小黄车贴片 | 仅带货形态、商品锚点手工档、纯内容形态/缺锚点/越界各自如实降级 |
| [`dsp_render_compose_demo.py`](./dsp_render_compose_demo.py) | dsp 合成渲染（MediaKit） | EDL→`track[][]`、封面走 `output.cover` 抽帧、**24h 签名 URL 转存**为不可变 key、缺密钥/空轨道报错 |

## 真 ffmpeg 复核（按需，不进回归底线）

[`dsp_trim_realcheck.py`](./dsp_trim_realcheck.py) 与上面 18 个 demo **分工相反**：demo 一律
**离线自包含**（mock 掉全部外部依赖，故 CI 里任何机器都绿），而它**要求真 ffmpeg**——用合成片段
把 dsp 里**依赖 ffmpeg 的原语**真跑一遍，并用 ffprobe 复核输出：

```bash
PYTHONPATH=$PWD/src python examples/dsp_trim_realcheck.py
# ffmpeg 没进全局 PATH 时，把 bin 目录当第一个参数传入（脚本会写进 FLOWMIND_MEDIA_BIN_DIR）：
PYTHONPATH=$PWD/src python examples/dsp_trim_realcheck.py /path/to/ffmpeg/bin
```

ffmpeg 的解析与生产**同一套**：`_media` 先看 `.env` 的 `FLOWMIND_MEDIA_BIN_DIR`（本机 ffmpeg 常是
「按需下载、不进全局 PATH」），没有才按 `PATH`。覆盖 `_media.clip_to_window`（超长真裁 / 不足真
`tpad` 定格补齐 / 等长与关闭补齐的原样返回 / 坏文件抛 `MediaError`）与 `_dsp_qc.sample_frames`
（10%/50%/90% 三帧真抽出 PNG、时长未知只抽首帧、坏文件抛错），共 **28 条断言**。**两处都找不到
ffmpeg 即自行跳过（exit 0）**——故**不**进「一键全跑 demo」回归底线；它是**按需**复核，把
「裁切/抽帧机制是否真按设计工作」从口头断言变成可复跑的产物。

## OCR 侧车自检（按需，不进回归底线）

[`ocr_sidecar_realcheck.py`](./ocr_sidecar_realcheck.py) 是 `flowmind.ocr_sidecar`（cross-dashboard
文字核查的本机侧车，`ADR-097` 后端半）的自检：真起侧车 → `POST /ocr` → 断言**逐行文字 / 四点框 /
置信度**三段式与 `/health` 字段形状，外加几条**不许把失败冒充成「读过了但没字」**的负向哨兵
（坏字节 ⇒ 502、非图片 Content-Type ⇒ 415、空白图 ⇒ 0 行——**0 行是合法答案，不是错误**）。
它要求解释器里真有 `rapidocr` 引擎（`environment.yml` 自 2026-10-16 修漂移后钉的正是新包
`rapidocr>=3.10`），缺引擎即 `check(ocr_lines.available())` **当场失败**；装好引擎后约 10 秒跑完。

```bash
PYTHONPATH=$PWD/src python examples/ocr_sidecar_realcheck.py
```

⚠️ 它起的是**进程内**侧车（`port=0`，系统随机分配），与 `deploy/ocr-sidecar/` 那个容器**无关**
——容器那条路由 `python scripts/ocr_sidecar_dev.py` 起停（见仓库根 README「OCR 侧车」段）；
它验的也正是**原生那条路**，即与容器对账时的基准。

## 🚀 Agent 开箱即用：`discover()`

每个 demo 第一步都跑 `discover()` / `field_names()`，让 Agent 自动发现技能
的字段 —— 不再靠「猜 schema」。

```python
from flowmind import discover, field_names

# 看全部技能
for skill in discover():
    print(f"{skill['id']}: {skill['description']}")

# 看某个技能的 input + output 完整 schema
info = discover("localize_status")
print(info["input_schema"])   # JSON Schema
print(info["output_schema"])  # JSON Schema

# 拿到 data 字段名（含嵌套），避免 r.data.foo vs r.data.report.foo 猜错
for path, names in field_names("localize_status").items():
    print(f"{path}: {names}")
```

`discover()` 把 input_schema、output_schema、description 全部暴露给 Agent ——
这是「开箱即用」的核心契约。

## 一键全跑（回归底线）

```bash
for f in examples/*_demo.py; do
  echo "════ $f ════"
  PYTHONPATH=$PWD/src conda run -n flowmind python "$f"
done
```

18 个 demo 全 PASS 是改动合并前的回归底线（本仓库无单测，demo 即冒烟测试）。
（`api_server_demo` / `auth_token_demo` 需 `uvicorn` / `starlette`；`localize_video_demo`
需 GPU 侧依赖——在未装齐的裸 Python 环境里它们会各自报缺依赖，与技能层改动无关。）

## demo 都做了什么

每个 demo 都覆盖 3 类用例：

1. **discover() 字段发现** —— 让 Agent / 人类立即看到 `data.foo` 应该是什么
2. **Happy path** —— 正常输入 + 完整输出（业务载荷 + 四段式推理链）
3. **错误路径** —— 故意触发入参错 / 环境错 / 服务端临时错，看 `failure_category`
   分类 + Agent 下一步动作建议

## MCP 客户端接入

不想用 demo 脚本？直接接 MCP 客户端（Streamable HTTP，端点
`http://127.0.0.1:8002/mcp`）或 REST 任务通道，见仓库根 `README.md`。

## 加新 demo 的规范

新增能力时，建议同时在本目录加一个 `<name>_demo.py`，沿用三段式结构
（discover + happy + 错误），mock 外部依赖、自包含可跑，让评审 / 用户
30 秒看懂这个能力能干嘛。
