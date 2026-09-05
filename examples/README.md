# examples —— 不需要 MCP 客户端也能体验技能

每个脚本都是**自包含**的最小可运行示例（mock 全部外部依赖，无需
GPU / PG / MQTT / Milvus / API key）：

```bash
PYTHONPATH=$PWD/src conda run -n flowmind python examples/<name>_demo.py
```

## 9 个 demo 一览

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
| [`federation_register_demo.py`](./federation_register_demo.py) | 联邦自注册（PG+MQTT 双通道） | register QoS1 payload 契约、单通道降级仍完成注册、双通道全挂静默降级、心跳与优雅注销 |

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

9 个 demo 全 PASS 是改动合并前的回归底线（本仓库无单测，demo 即冒烟测试）。

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
