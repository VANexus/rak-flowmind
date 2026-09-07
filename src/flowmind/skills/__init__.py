"""技能包：导入各技能以触发 @skill 注册。

mcp-base-gpu 主线仅注册视频本地化技能（localize_*）。

224a194 合并修复（2026-09-07）：合并曾把 35 个不存在或已按 f72fb2d 裁剪决策
移除的技能模块写入 import 列表，导致 `import flowmind.skills` 直接 ImportError、
服务无法启动。本列表收敛为当前实际存在且可完整加载的模块：
- content_hot_boards 依赖的 _content_common.py 已被 f72fb2d 裁剪，该技能
  待依赖补全后再恢复注册；
- 其余 content_* / b2b_* / alibaba_* / crawler_* / tiktok_* 等待源码分支
  合入后再逐个恢复，禁止幽灵 import 入库。
"""
from flowmind.skills import (  # noqa: F401
    localize_cancel,
    localize_download,
    localize_retry,
    localize_search,
    localize_status,
    localize_submit,
    localize_video,
)
