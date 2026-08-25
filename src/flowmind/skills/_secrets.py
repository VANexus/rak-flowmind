"""项目 .env 加载：技能层读云 key 的统一入口。

约定（CLAUDE.md）：API key 永不进 toml / commit——.env 已 gitignored，
key 只落在这里。任何技能需要 key 时经 get_api_key(env_name) 读取：
先查进程环境变量，再回落项目根 .env。
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
import os

_loaded = False

# 项目根 = src/flowmind/skills/ 上溯三级
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV_PATH = _PROJECT_ROOT / ".env"


def _ensure_loaded() -> None:
    global _loaded
    if not _loaded:
        load_dotenv(_ENV_PATH)
        _loaded = True


def get_api_key(env_name: str) -> str | None:
    """读取 API key：进程环境变量优先，其次项目 .env。空白值视为未设置。"""
    _ensure_loaded()
    val = os.environ.get(env_name)
    return val.strip() or None if val else None
