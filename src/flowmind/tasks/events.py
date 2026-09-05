"""MQTT 任务事件发布器（paho-mqtt）——进度/状态实时推送。

契约：
- 主题：``mcp-base-gpu/tasks/{task_id}/events``
- payload（JSON）：{"task_id", "stage", "pct", "message", "status", "ts"}
- 终态（succeeded/failed/cancelled/interrupted）消息 retain=True，
  新订阅者（前端/Agent）连接即见最终状态。
- QoS=1：至少一次（paho 断连期间 QoS>0 消息入队，重连后补发）。

降级铁律：**发布失败只记日志，绝不抛异常、绝不阻塞任务主流程**
（任务可靠性由 PG 落库保证，MQTT 是尽力而为的通知通道）。

连接策略（非阻塞）：
- 惰性初始化：首次 publish 才建客户端；connect_async + loop_start，
  首次最多等 2s 确认连接，超时视为暂不可达——后台线程继续自动重连，
  后续 publish 的 QoS=1 消息由 paho 排队补发。
- 未配置（FLOWMIND_MQTT_HOST / config ``infra.mqtt_host`` 均空）→ 永久禁用，零开销。
- 认证（可选）：FLOWMIND_MQTT_USERNAME / FLOWMIND_MQTT_PASSWORD
  （或 config ``infra.mqtt_username/mqtt_password``）；username 空 = 匿名连接。
- TLS：FLOWMIND_MQTT_USE_TLS / config ``infra.mqtt_use_tls`` 开启时
  ``tls_set()``（默认系统 CA；EMQX 明文 1883 部署保持 False）。
- 首次失败记 warning，之后降为 debug（成功后复位），不刷屏。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone

from flowmind.tasks import TERMINAL_STATUSES

logger = logging.getLogger(__name__)

_TOPIC_PREFIX = "mcp-base-gpu/tasks"


def _resolve_broker() -> tuple[str, int, bool, str, str] | None:
    """broker 解析（配置源顺序：env → config.toml → None）。

    返回 (host, port, use_tls, username, password)；host 未配置返回 None
    （发布器永久禁用）。username/password 为空 = 匿名连接（EMQX 未开
    认证时的默认形态）。
    """
    from flowmind.config import get_config

    host = (os.environ.get("FLOWMIND_MQTT_HOST")
            or os.environ.get("RAK_MQTT_HOST")
            or get_config().infra.mqtt_host or "").strip()
    if not host:
        return None
    raw_port = (os.environ.get("FLOWMIND_MQTT_PORT")
                or os.environ.get("RAK_MQTT_PORT") or "")
    try:
        port = int(raw_port) if raw_port.strip() else get_config().infra.mqtt_port
    except ValueError:
        port = 1883
    raw_tls = os.environ.get("FLOWMIND_MQTT_USE_TLS", "").strip().lower()
    if raw_tls in ("1", "true", "yes"):
        use_tls = True
    elif raw_tls in ("0", "false", "no"):
        use_tls = False
    else:
        use_tls = get_config().infra.mqtt_use_tls
    username = (os.environ.get("FLOWMIND_MQTT_USERNAME", "").strip()
                or get_config().infra.mqtt_username.strip())
    # 密码不 strip（口令含首尾空格属合法值，与 store.py RAK_PG_APP_PASS 同口径）
    password = (os.environ.get("FLOWMIND_MQTT_PASSWORD", "")
                or get_config().infra.mqtt_password)
    return host, port, use_tls, username, password


class TaskEventPublisher:
    """任务事件 MQTT 发布器（线程安全；失败静默降级为纯落库）。"""

    def __init__(self, host: str | None = None, port: int | None = None,
                 use_tls: bool = False, username: str = "", password: str = ""):
        if host is not None:
            self._broker = ((host, port or 1883, use_tls) if host else None)
            self._auth = (username, password) if username else None
        else:
            resolved = _resolve_broker()
            self._broker = None if resolved is None else resolved[:3]
            self._auth = None if resolved is None or not resolved[3] else (
                resolved[3], resolved[4])
        self._client = None
        self._lock = threading.Lock()
        self._connected = threading.Event()
        self._warned = False  # 首次失败 warning，之后 debug；成功后复位
        self._enabled = self._broker is not None
        if not self._enabled:
            logger.info("MQTT 未配置（FLOWMIND_MQTT_HOST / config infra.mqtt_host 均空）"
                        "——任务事件降级为纯 PG 落库")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def status(self) -> str:
        """健康探针用：disabled（未配置）/ connected / connecting。"""
        if not self._enabled:
            return "disabled"
        return "connected" if self._connected.is_set() else "connecting"

    def _get_client(self):
        """惰性建连（双检锁）。connect_async 非阻塞，loop_start 后自动重连。"""
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            import paho.mqtt.client as mqtt

            host, port, use_tls = self._broker  # type: ignore[misc]
            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"flowmind-tasks-{os.getpid()}",
                protocol=mqtt.MQTTv311,
            )
            if self._auth:
                client.username_pw_set(self._auth[0], self._auth[1])
            if use_tls:
                client.tls_set()  # 默认系统 CA（ssl.default_ca_certs）
            client.reconnect_delay_set(min_delay=1, max_delay=30)
            client.on_connect = self._on_connect
            client.on_disconnect = self._on_disconnect
            client.connect_async(host, port, keepalive=60)
            client.loop_start()
            self._client = client
            # 首次等待连接确认（2s）；超时不阻塞任务——后续消息排队补发
            self._connected.wait(timeout=2.0)
            return client

    def _on_connect(self, client, *_args, **_kw) -> None:
        self._connected.set()
        logger.info("MQTT 已连接 %s:%s", self._broker[0], self._broker[1])

    def _on_disconnect(self, client, *_args, **_kw) -> None:
        self._connected.clear()
        logger.warning("MQTT 断开（paho 自动重连中）")

    def publish(self, task_id: str, *, status: str, stage: str = "",
                pct: float = 0.0, message: str = "") -> bool:
        """发布一条任务事件。终态 retain=True。失败返回 False（绝不抛）。"""
        if not self._enabled:
            return False
        payload = json.dumps({
            "task_id": task_id,
            "stage": stage,
            "pct": round(float(pct), 2),
            "message": message,
            "status": status,
            "ts": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False)
        topic = f"{_TOPIC_PREFIX}/{task_id}/events"
        retain = status in TERMINAL_STATUSES
        try:
            client = self._get_client()
            info = client.publish(topic, payload, qos=1, retain=retain)
            if info.rc != 0:  # mqtt.MQTT_ERR_SUCCESS == 0
                raise RuntimeError(f"publish rc={info.rc}")
            self._warned = False
            return True
        except Exception as exc:  # noqa: BLE001  通知通道失败绝不外泄
            if not self._warned:
                self._warned = True
                logger.warning("MQTT 事件发布失败（降级为纯落库）: %s", exc)
            else:
                logger.debug("MQTT 事件发布失败: %s", exc)
            return False

    def close(self) -> None:
        """停 loop、断连接（进程退出时调用；失败静默）。

        与 _get_client 的写路径共用 _lock：读-停-置空原子化，防 close 与
        惰性建连并发时把刚建的 client 置空、或 stop 了他人正在用的 loop。
        """
        with self._lock:
            if self._client is None:
                return
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
