"""技能包：导入各技能以触发 @skill 注册。

mcp-base-gpu 视频本地化链路 7 技能（任务引擎语义 + 检索）：
localize_submit（批量提交）/ localize_status / localize_retry /
localize_cancel / localize_download（任务治理）/
localize_search（字幕语义检索）/ localize_video（同步流水线本体）。
"""
from flowmind.skills import (  # noqa: F401  按字母序
    localize_cancel,
    localize_download,
    localize_retry,
    localize_search,
    localize_status,
    localize_submit,
    localize_video,
)
