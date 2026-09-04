"""技能包：导入各技能以触发 @skill 注册。

mcp-base-gpu 裁剪后仅保留视频本地化 6 技能（localize_*）。
"""
from flowmind.skills import (
    localize_batch,  # noqa: F401
    localize_cancel,  # noqa: F401
    localize_download,  # noqa: F401
    localize_retry,  # noqa: F401
    localize_status,  # noqa: F401
    localize_video,  # noqa: F401
)
