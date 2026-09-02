"""TikHub 情报 skills 共用：取 client + 统一错误映射（真实数据，绝不 mock）。"""
from __future__ import annotations

from flowmind.config import get_config
from flowmind.skills._tikhub_client import TikHubClient, TikHubError, new_client_from_config


def intel_client() -> TikHubClient:
    """按当前 config 组装 TikHubClient；缺 key 时抛 TikHubError(environment)。"""
    cfg = get_config().keyword_trend
    client = new_client_from_config(cfg)
    if not client.api_key:
        raise TikHubError(
            "情报数据走 TikHub 但未配置 AI_TRENDS_API_KEY；请在工作区 .env 填写趋势 API Key。",
            category="environment", retriable=False,
        )
    return client


def fail_fields(exc: Exception) -> dict:
    """把 TikHubError/其它异常映射为 degraded 字段。"""
    if isinstance(exc, TikHubError):
        category = exc.category
        retriable = exc.retriable
    elif isinstance(exc, (ValueError, KeyError, TypeError)):
        # 入参缺失/结构不符：调用方改参数即可，不可重试、与环境无关
        category = "invalid_argument"
        retriable = False
    else:
        category = "environment"
        retriable = False
    return {
        "degraded": True,
        "failure_category": category,
        "retriable": retriable,
        "warning": f"情报数据不可用（{category}）：{str(exc).strip()} 请按上述原因修复配置/网络后重试。",
    }
