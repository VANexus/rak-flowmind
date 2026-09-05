"""联邦协议 MQTT 发布器（paho-mqtt）——register / heartbeat / unregister。

契约（与 Go 网关侧 mqttclient.go 联邦主题逐字对齐，改动必须双侧同步）：
- ``mcp-base-gpu/federation/register``    QoS1（至少一次，新后端必须送达）
- ``mcp-base-gpu/federation/heartbeat``   QoS0（高频容忍丢失，30s 间隔）
- ``mcp-base-gpu/federation/unregister``  QoS1（至少一次，优雅退出必须送达）

payload（JSON，与 Go RegisterPayload 对齐）：
- register：{"backend_id", "version", "url", "transport", "prefix",
  "capabilities", "ts"}
- heartbeat / unregister：{"backend_id", "ts"}（unregister 另带 "reason"）

降级铁律：与 tasks/events.py 同口径——发布失败只记日志，绝不抛异常。
连接策略：惰性建连 + connect_async 非阻塞 + paho 自动重连；首次最多等
2s 确认连接，超时视为暂不可达（后台继续重连，QoS1 消息排队补发）。

不使用 LWT（will_set）的决策记录：异常死亡（kill -9 / 断网）必须走
「90s 无心跳 → offline」的心跳超时路径（网关 stale 标记语义，工具保留、
可自动恢复）；若挂 LWT 会在 broker 检测到连接断开时立即代发 unregister，
触发网关物理摘除——与上述生命周期设计冲突。优雅退出由本模块 stop()
显式发布 unregister（atexit 钩子）。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

TOPIC_REGISTER = "mcp-base-gpu/federation/register"
TOPIC_HEARTBEAT = "mcp-base-gpu/federation/heartbeat"
TOPIC_UNREGISTER = "mcp-base-gpu/federation/unregister"

# action → (topic, qos)；QoS 契约见模块 docstring
_ACTIONS: dict[str, tuple[str, int]] = {
    "register": (TOPIC_REGISTER, 1),
    "heartbeat": (TOPIC_HEARTBEAT, 0),
    "unregister": (TOPIC_UNREGISTER, 1),
}


class FederationPublisher:
    """联邦协议 MQTT 发布器（线程安全；失败静默降级）。"""

    def __init__(self) -> None:
        # broker 解析与 tasks/events.py 完全同源（FLOWMIND_MQTT_HOST →
        # RAK_MQTT_HOST → config infra.mqtt_host；单点维护避免口径漂移）
        from flowmind.tasks.events import _resolve_broker

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
            logger.info("MQTT 未配置（FLOWMIND_MQTT_HOST / RAK_MQTT_HOST / "
                        "config infra.mqtt_host 均空）——联邦注册降级为仅 PG 通道")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def publish(self, action: str, payload: dict, *, wait: bool = False) -> bool:
        """发布一条联邦消息。失败返回 False（绝不抛）。

        wait=True 时阻塞至 QoS1 消息确认或超时（unregister 优雅退出用，
        atexit 场景必须确保离场消息送达）。
        """
        spec = _ACTIONS.get(action)
        if spec is None:
            logger.warning("未知联邦动作 %r，消息未发送", action)
            return False
        if not self._enabled:
            return False
        topic, qos = spec
        body = json.dumps(payload, ensure_ascii=False)
        try:
            client = self._get_client()
            info = client.publish(topic, body, qos=qos)
            if info.rc != 0:  # mqtt.MQTT_ERR_SUCCESS == 0
                raise RuntimeError(f"publish rc={info.rc}")
            if wait and qos > 0:
                info.wait_for_publish(timeout=3.0)
            self._warned = False
            return True
        except Exception as exc:  # noqa: BLE001  联邦通道失败绝不外泄
            if not self._warned:
                self._warned = True
                logger.warning("联邦 MQTT %s 发布失败（降级）: %s", action, exc)
            else:
                logger.debug("联邦 MQTT %s 发布失败: %s", action, exc)
            return False

    def close(self) -> None:
        """停 loop、断连接（进程退出时调用；失败静默）。

        与 _get_client 的写路径共用 _lock：读-停-置空原子化，防 close 与
        惰性建连并发时把刚建的 client 置空、或停了他人正在用的 loop。
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

    # ── 内部 ──

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
                client_id=f"flowmind-federation-{os.getpid()}",
                protocol=mqtt.MQTTv311,
            )
            if self._auth:
                client.username_pw_set(self._auth[0], self._auth[1])
            if use_tls:
                client.tls_set()  # 默认系统 CA
            client.reconnect_delay_set(min_delay=1, max_delay=30)
            client.on_connect = self._on_connect
            client.on_disconnect = self._on_disconnect
            client.connect_async(host, port, keepalive=60)
            client.loop_start()
            self._client = client
            # 首次等待连接确认（2s）；超时不阻塞——QoS1 消息排队补发
            self._connected.wait(timeout=2.0)
            return client

    def _on_connect(self, client, *_args, **_kw) -> None:
        self._connected.set()
        broker = self._broker or ("?", 0)
        logger.info("联邦 MQTT 已连接 %s:%s", broker[0], broker[1])

    def _on_disconnect(self, client, *_args, **_kw) -> None:
        self._connected.clear()
        logger.warning("联邦 MQTT 断开（paho 自动重连中）")


def now_iso() -> str:
    """协议时间戳（UTC ISO8601；Go 侧 time.Time 按 RFC3339 解析）。"""
    return datetime.now(timezone.utc).isoformat()
