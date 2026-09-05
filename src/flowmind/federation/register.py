"""功能包自注册编排：进程启动时向联邦注册表声明自身元数据并维持心跳。

流程（server_http.main 内装配，开关 FLOWMIND_FEDERATION_REGISTER 默认关）：
1. 组装注册元数据（backend_id / prefix / url / version）；
2. PG upsert（federation_backends，幂等）+ MQTT publish register（QoS1）
   ——双通道尽力而为，任一成功即完成注册；
3. 启动心跳守护线程（federation/heartbeat.py）；
4. 优雅注销：主路径 uvicorn lifespan shutdown（SIGTERM 下 atexit 不
   执行——uvicorn 停机完成后 re-raise 信号直接杀进程，见
   server_http._wrap_streamable_lifespan 根因注释）+ atexit 兜底
   （CTRL+C / 程序内退出）：unregister QoS1 + PG offline。

version 读取顺序：importlib.metadata（pip 安装态，权威）→ pyproject.toml
（源码态）→ "0.0.0"。PG / MQTT 不可达仅日志降级，绝不阻塞服务启动。
"""
from __future__ import annotations

import atexit
import logging
import tomllib
from pathlib import Path

from flowmind.federation.heartbeat import HeartbeatWorker
from flowmind.federation.mqtt import FederationPublisher, now_iso
from flowmind.federation.pg import FederationPGStore

logger = logging.getLogger(__name__)

_DEFAULT_TRANSPORT = "streamable-http"


def _resolve_version() -> str:
    """包版本（importlib.metadata 优先，pyproject.toml 兜底，最终 0.0.0）。"""
    try:
        from importlib.metadata import version

        return version("mcp-base-gpu")
    except Exception:  # noqa: BLE001  未安装态继续回落 pyproject
        pass
    try:
        # 层级：__file__ = <repo>/src/flowmind/federation/register.py →
        # parents[0]=federation/ [1]=flowmind/ [2]=src/ [3]=<repo>/，
        # pyproject.toml 在仓库根（parents[3]）——源码态部署（PYTHONPATH
        # 直指 src/ 或 editable 安装）版本解析都靠它兜底。
        root = Path(__file__).resolve().parents[3]  # <repo>/（pyproject.toml 所在）
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        return str(data["project"]["version"])
    except Exception:  # noqa: BLE001  元数据读取失败不阻塞注册
        return "0.0.0"


class FederationRegistrar:
    """联邦自注册编排器（PG + MQTT 双通道注册、心跳维持、优雅注销）。"""

    def __init__(self, *, backend_id: str, prefix: str, port: int = 8002,
                 url: str = "", heartbeat_interval: float = 30.0):
        # url 空 = 本机可达地址兜底（集群部署经 FLOWMIND_FEDERATION_URL 注入
        # 外部可达地址，如 http://<pod-ip>:8002/mcp）
        self._backend_id = backend_id
        self._prefix = prefix
        self._url = url.strip() or f"http://127.0.0.1:{port}/mcp"
        self._version = _resolve_version()
        self._interval = heartbeat_interval
        self._pg = FederationPGStore()
        self._mqtt = FederationPublisher()
        self._worker: HeartbeatWorker | None = None
        self._stopped = False

    @property
    def backend_id(self) -> str:
        return self._backend_id

    @property
    def url(self) -> str:
        return self._url

    def registration_payload(self) -> dict:
        """register 消息体（与 Go 侧 RegisterPayload JSON 契约对齐）。"""
        return {
            "backend_id": self._backend_id,
            "version": self._version,
            "url": self._url,
            "transport": _DEFAULT_TRANSPORT,
            "prefix": self._prefix,
            "capabilities": {},  # 能力标签预留（阶段 3 鉴权/路由策略用）
            "ts": now_iso(),
        }

    def start(self) -> bool:
        """执行注册并启动心跳。返回注册是否至少经一条通道成功。"""
        payload = self.registration_payload()
        ok_pg = self._pg.upsert_backend(
            backend_id=self._backend_id,
            version=self._version,
            url=self._url,
            transport=_DEFAULT_TRANSPORT,
            prefix=self._prefix,
            capabilities="{}",
        )
        ok_mqtt = self._mqtt.publish("register", payload)

        if not (self._pg.enabled or self._mqtt.enabled):
            logger.warning("联邦注册双通道均未配置——跳过心跳线程"
                           "（配置 FLOWMIND_PG_DSN 或 FLOWMIND_MQTT_HOST 后启用）")
            return False
        # 双通道瞬时不可达也启动心跳：paho 后台自动重连 + PG 每拍重试，
        # 通道恢复后下个心跳周期自动补上注册语义
        self._worker = HeartbeatWorker(
            self._mqtt, self._pg,
            backend_id=self._backend_id, interval=self._interval)
        self._worker.start()
        # atexit 兜底（lifespan 主路径见 server_http._wrap_streamable_lifespan；
        # stop 幂等，双触发无害）
        atexit.register(self.stop)
        logger.info("联邦自注册完成 backend_id=%s url=%s version=%s"
                    "（pg=%s mqtt=%s）",
                    self._backend_id, self._url, self._version, ok_pg, ok_mqtt)
        return ok_pg or ok_mqtt

    def stop(self) -> None:
        """优雅注销（幂等；atexit 钩子自动调用）。"""
        if self._stopped:
            return
        self._stopped = True
        if self._worker is not None:
            self._worker.stop(reason="shutdown")
        self._mqtt.close()


def start_federation(port: int = 8002) -> FederationRegistrar | None:
    """按配置启动联邦自注册（server_http.main 装配入口）。

    开关：``FLOWMIND_FEDERATION_REGISTER=1`` 显式开启（默认 0——现有
    demo 与独立部署行为完全不变）。未开启或启动异常返回 None，服务
    照常独立运行（联邦能力绝不阻塞主流程）。
    """
    from flowmind.config import get_config

    cfg = get_config().federation
    if not cfg.enabled:
        logger.info("联邦自注册未开启（FLOWMIND_FEDERATION_REGISTER=0）")
        return None
    try:
        registrar = FederationRegistrar(
            backend_id=cfg.backend_id,
            prefix=cfg.prefix,
            port=port,
            url=cfg.url,
            heartbeat_interval=cfg.heartbeat_interval,
        )
    except Exception as exc:  # noqa: BLE001  联邦能力绝不阻塞服务启动
        logger.warning("联邦自注册初始化失败（服务继续独立运行）: %s", exc)
        return None
    try:
        registrar.start()
    except Exception as exc:  # noqa: BLE001
        logger.warning("联邦注册执行异常（服务继续独立运行）: %s", exc)
    return registrar
