"""项目 .env 加载：技能层读云 key 的统一入口。

约定（CLAUDE.md）：API key 永不进 toml / commit——.env 已 gitignored，
key 只落在这里。任何技能需要 key 时经 get_api_key(env_name) 读取：
先查进程环境变量，再回落 .env（先工作区根统一 .env，后项目根 .env 兜底）。

env 文件的**加载实现**收敛在 ``flowmind.config.ensure_env_loaded``（唯一真源）——
config 各字段也要按「env 优先」解析，两处各存一份会漂移。本模块只保留
``get_api_key`` 这一读取面。
"""
from __future__ import annotations

import os

from flowmind.config import ensure_env_loaded


def get_api_key(env_name: str) -> str | None:
    """读取 API key：进程环境变量优先，其次项目 .env。空白值视为未设置。"""
    ensure_env_loaded()
    val = os.environ.get(env_name)
    return val.strip() or None if val else None
