"""TikHub 响应磁盘缓存 + 免费窗口投机学习（stdlib sqlite3，零新依赖）。

设计决策（用户拍板 2026-09-02）：
- **默认重复请求完全不免费**：所有端点保守模式——本地缓存 soft_ttl 内直接回缓存，
  过期后真实外呼（正常计费）。
- **只投机学习的才走积极请求**：每次真实响应 best-effort 扫描响应头
  （x-cache / x-cache-expiration / x-cache-ttl 等候选键），一旦发现该端点
  带「服务端缓存窗口」证据，把端点升级为 speculative：soft_ttl 过期后的外呼
  大概率落在 TikHub 自身缓存窗口内（免费命中），我们白拿最新数据。
  未发现任何 cache 头的端点永远保守——零多花钱风险。

磁盘 schema：
- entries(key PRIMARY KEY, payload, fetched_at, endpoint)  响应缓存
- learned_policies(endpoint PRIMARY KEY, free_window_s, learned_at, evidence)  学习结果

并发：模块级实例注册表（同 db_path 单例）保证 in-flight 锁去重跨技能生效；
sqlite 开 WAL + busy_timeout，多线程安全。
"""
from __future__ import annotations

import contextvars
import json
import sqlite3
import threading
import time
from pathlib import Path

# ── 缓存元信息（ContextVar：每次 MCP 工具调用独立上下文，线程安全） ──

_cache_meta: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "tikhub_cache_meta", default=None
)


def get_last_cache_meta() -> dict | None:
    """返回当前调用上下文里最近一次 TikHub 请求的缓存元信息。

    形如 ``{"mode": "local"|"local_fallback"|"speculative"|"live",
    "hit": bool, "age_s": float}``；未启用缓存时为 None。
    """
    return _cache_meta.get()


def _set_cache_meta(mode: str, *, hit: bool, age_s: float) -> None:
    _cache_meta.set({"mode": mode, "hit": hit, "age_s": round(age_s, 1)})


# ── 响应头探测（best-effort；TikHub 未承诺头格式） ──

# 命中类：值 ∈ _HIT_VALUES 即视为「该响应来自服务端缓存」
_HIT_HEADER_KEYS = ("x-cache", "x-cache-status", "x-cache-hit")
# 数值窗口类：直接给出秒数
_TTL_HEADER_KEYS = ("x-cache-ttl", "x-cache-max-age")
# 到期时间戳类：epoch 秒/毫秒或 ISO 串
_EXPIRE_HEADER_KEYS = ("x-cache-expiration", "x-cache-expiration-at", "x-cache-expire")
_HIT_VALUES = {"hit", "true", "1", "yes", "cached", "fresh"}

# 学习窗口下限（防止把过期时间误读成 1s 后）与上限
_MIN_LEARNED_WINDOW_S = 60.0


def _parse_epoch(v: str) -> float | None:
    """宽容解析时间戳：epoch 秒（10 位）/毫秒（13 位）；失败返回 None。"""
    s = str(v).strip()
    if not s.replace(".", "").isdigit():
        return None
    try:
        n = float(s)
    except ValueError:
        return None
    if n >= 1e12:  # 毫秒
        n /= 1000.0
    if n < 1e9:  # 不是合理 epoch（1970+32年）
        return None
    return n


class TikHubCache:
    """TikHub 响应磁盘缓存 + 政策引擎（local / speculative / live）。"""

    def __init__(
        self,
        *,
        db_path: str,
        default_soft_ttl_s: float = 1800.0,
        max_free_window_s: float = 21600.0,
        overrides: dict[str, float] | None = None,
    ):
        self.default_soft_ttl_s = max(0.0, float(default_soft_ttl_s))
        self.max_free_window_s = max(60.0, float(max_free_window_s))
        self.overrides = dict(overrides or {})
        p = Path(db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(p), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS entries (
              key TEXT PRIMARY KEY,
              payload TEXT NOT NULL,
              fetched_at REAL NOT NULL,
              endpoint TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS learned_policies (
              endpoint TEXT PRIMARY KEY,
              free_window_s REAL NOT NULL,
              learned_at REAL NOT NULL,
              evidence TEXT NOT NULL
            );
            """
        )
        self._conn.commit()
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ── 基础存取 ──

    @staticmethod
    def make_key(method: str, path: str, json_body: dict | None, params: dict | None) -> str:
        """缓存键 = method + path + 规范化参数（键排序，忽略顺序差异）。"""
        norm = json.dumps(
            {"m": method.upper(), "p": path, "b": json_body or {}, "q": params or {}},
            sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        )
        return norm

    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def get(self, key: str) -> tuple[dict, float] | None:
        """返回 (payload, age_s)；无缓存返回 None。payload 损坏时静默清除。"""
        row = self._conn.execute(
            "SELECT payload, fetched_at FROM entries WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row[0])
        except ValueError:
            self._conn.execute("DELETE FROM entries WHERE key = ?", (key,))
            self._conn.commit()
            return None
        return payload, max(0.0, time.time() - float(row[1]))

    def put(self, key: str, endpoint: str, payload: dict) -> None:
        self._conn.execute(
            "INSERT INTO entries(key, payload, fetched_at, endpoint) VALUES(?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET payload=excluded.payload, "
            "fetched_at=excluded.fetched_at, endpoint=excluded.endpoint",
            (key, json.dumps(payload, ensure_ascii=False), time.time(), endpoint),
        )
        self._conn.commit()

    # ── 政策 ──

    def soft_ttl(self, endpoint: str) -> float:
        """本地缓存直回窗口（该窗口内绝不外呼）。per-endpoint override > 默认。"""
        if endpoint in self.overrides:
            return max(0.0, float(self.overrides[endpoint]))
        return self.default_soft_ttl_s

    def learned_window(self, endpoint: str) -> float | None:
        """已学习到的 TikHub 服务端免费缓存窗口（秒）；未学习返回 None。"""
        row = self._conn.execute(
            "SELECT free_window_s FROM learned_policies WHERE endpoint = ?", (endpoint,)
        ).fetchone()
        return float(row[0]) if row else None

    def decide(self, endpoint: str, age_s: float) -> str:
        """缓存命中时的决策：local（直接回缓存）| speculative | live（外呼）。"""
        if age_s < self.soft_ttl(endpoint):
            return "local"
        window = self.learned_window(endpoint)
        if window is not None and age_s < window:
            return "speculative"
        return "live"

    # ── 投机学习 ──

    def learn_from_headers(self, endpoint: str, headers) -> dict | None:
        """从真实响应头学习该端点的服务端缓存窗口；学到就持久化。

        返回 ``{"free_window_s": float, "evidence": str}`` 或 None（无证据）。
        优先级：数值 TTL 头 > 到期时间戳头 > 纯 hit 头（窗口未知，取上限保守值）。
        """
        flat = {str(k).lower(): str(v).strip() for k, v in dict(headers or {}).items()}
        evidence = ""
        window: float | None = None

        for k in _TTL_HEADER_KEYS:
            v = flat.get(k, "")
            if v.replace(".", "").isdigit():
                w = float(v)
                if w >= _MIN_LEARNED_WINDOW_S:
                    window = w
                    evidence = f"{k}={v}"
                    break

        if window is None:
            now = time.time()
            for k in _EXPIRE_HEADER_KEYS:
                exp = _parse_epoch(flat.get(k, ""))
                if exp is not None and exp > now:
                    window = min(self.max_free_window_s, max(_MIN_LEARNED_WINDOW_S, exp - now))
                    evidence = f"{k}={flat.get(k, '')}"
                    break

        if window is None:
            for k in _HIT_HEADER_KEYS:
                if flat.get(k, "").lower() in _HIT_VALUES:
                    # 纯 hit 无窗口信息：按上限记（反正 speculative 外呼最坏也只是
                    # 一次本来就要发的 live 请求，不多花一分钱）
                    window = self.max_free_window_s
                    evidence = f"{k}={flat.get(k, '')}"
                    break

        if window is None:
            return None

        window = min(self.max_free_window_s, max(_MIN_LEARNED_WINDOW_S, window))
        self._conn.execute(
            "INSERT INTO learned_policies(endpoint, free_window_s, learned_at, evidence) "
            "VALUES(?,?,?,?) ON CONFLICT(endpoint) DO UPDATE SET "
            "free_window_s=excluded.free_window_s, learned_at=excluded.learned_at, "
            "evidence=excluded.evidence",
            (endpoint, window, time.time(), evidence),
        )
        self._conn.commit()
        return {"free_window_s": window, "evidence": evidence}


# ── 模块级实例注册表：同 db_path 单例（锁去重跨技能生效） ──

_INSTANCES: dict[str, TikHubCache] = {}
_INSTANCES_GUARD = threading.Lock()


def get_cache(
    db_path: str,
    *,
    default_soft_ttl_s: float,
    max_free_window_s: float,
    overrides: dict[str, float] | None = None,
) -> TikHubCache:
    with _INSTANCES_GUARD:
        cache = _INSTANCES.get(db_path)
        if cache is None:
            cache = TikHubCache(
                db_path=db_path,
                default_soft_ttl_s=default_soft_ttl_s,
                max_free_window_s=max_free_window_s,
                overrides=overrides,
            )
            _INSTANCES[db_path] = cache
        return cache
