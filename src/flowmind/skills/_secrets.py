"""项目 .env 加载：技能层读云 key 的统一入口。

约定（CLAUDE.md）：API key 永不进 toml / commit——.env 已 gitignored，
key 只落在这里。任何技能需要 key 时经 get_api_key(env_name) 读取：
先查进程环境变量，再回落 .env（先工作区根统一 .env，后项目根 .env 兜底）。
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
import os

_loaded = False

# 项目根 = src/flowmind/skills/ 上溯三级
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
# 工作区根 = 项目根上一级（统一密钥文件所在）
_WORKSPACE_ROOT = _PROJECT_ROOT.parent


def _ensure_loaded() -> None:
    global _loaded
    if not _loaded:
        # 先加载工作区根统一 .env（功能域命名，唯一权威密钥文件），
        # 再兜底加载项目 .env——load_dotenv 默认不覆盖已加载变量，根 .env 优先。
        load_dotenv(_WORKSPACE_ROOT / ".env")
        load_dotenv(_PROJECT_ROOT / ".env")
        _loaded = True


def get_api_key(env_name: str) -> str | None:
    """读取 API key：进程环境变量优先，其次项目 .env。空白值视为未设置。"""
    _ensure_loaded()
    val = os.environ.get(env_name)
    return val.strip() or None if val else None
