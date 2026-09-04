"""联邦心跳守护线程——每 30s 发布 heartbeat（QoS0），停机时 unregister。

生命周期协议（与 Go 网关侧 federation/health 语义对齐）：
- 心跳：daemon 线程每 ``interval``（默认 30s）经 MQTT（QoS0）+ PG 双通道
  刷新——任一通道成功即视为在线；网关 90s 无心跳才标 offline（容忍
  单次丢失与短暂网络抖动）。
- 优雅注销：``stop()`` 发布 unregister（QoS1，wait=True 确保送达）+
  PG 置 offline。挂接点（stop 幂等，双触发无害）：
  - lifespan shutdown：uvicorn 优雅停机（含 SIGTERM）主路径——
    uvicorn 停机完成后会恢复默认 handler 并 re-raise 信号，进程被
    信号直接终止、atexit 不执行（根因记录见
    server_http._wrap_streamable_lifespan）；
  - atexit：兜底——CTRL+C（SIGINT 走 KeyboardInterrupt 异常路径）
    与非 uvicorn 宿主的正常退出；不单独覆盖 SIGTERM handler 的原因：
    uvicorn 已安装自己的信号处理做优雅停机，重复注册会互相干扰；
  - 显式调用：嵌入方（server_http）主动停止联邦时调用。
- 异常死亡（kill -9 / 断电）：无 unregister——网关走 90s 心跳超时 →
  offline（stale 标记，工具保留），恢复后自动复活。刻意不用 MQTT LWT，
  决策记录见 federation/mqtt.py 模块 docstring。
"""
from __future__ import annotations

import logging
import threading

from flowmind.federation.mqtt import FederationPublisher, now_iso
from flowmind.federation.pg import FederationPGStore

logger = logging.getLogger(__name__)


class HeartbeatWorker:
    """心跳守护线程（start/stop 二段式；线程安全）。"""

    def __init__(self, publisher: FederationPublisher, pg: FederationPGStore, *,
                 backend_id: str, interval: float = 30.0):
        self._publisher = publisher
        self._pg = pg
        self._backend_id = backend_id
        self._interval = max(1.0, float(interval))
        self._stop = threading.Event()
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="federation-heartbeat", daemon=True)

    @property
    def interval(self) -> float:
        return self._interval

    def start(self) -> None:
        self._thread.start()
        logger.info("联邦心跳线程已启动（backend_id=%s interval=%.0fs MQTT=%s PG=%s）",
                    self._backend_id, self._interval,
                    self._publisher.enabled, self._pg.enabled)

    def stop(self, *, reason: str = "shutdown") -> None:
        """停心跳并优雅注销（幂等；atexit 与显式调用并发安全）。"""
        if self._stopped.is_set():
            return
        self._stopped.set()
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)  # daemon 线程，超时不强杀
        # 优雅离场：MQTT unregister（QoS1 确认送达）+ PG 置 offline
        self._publisher.publish(
            "unregister",
            {"backend_id": self._backend_id, "reason": reason, "ts": now_iso()},
            wait=True,
        )
        self._pg.set_offline(self._backend_id)
        logger.info("联邦注销完成（backend_id=%s reason=%s）", self._backend_id, reason)

    # ── 内部 ──

    def _run(self) -> None:
        # Event.wait 兼作定时器与停止信号：stop() 置位后立即唤醒，无延迟退出
        while not self._stop.wait(self._interval):
            self._beat()

    def _beat(self) -> None:
        ok_mqtt = self._publisher.publish(
            "heartbeat", {"backend_id": self._backend_id, "ts": now_iso()})
        ok_pg = self._pg.update_heartbeat(self._backend_id)
        if ok_mqtt or ok_pg:
            logger.debug("联邦心跳已发送（mqtt=%s pg=%s）", ok_mqtt, ok_pg)
        else:
            logger.warning("联邦心跳双通道均失败（网关将在 90s 后标记 offline）")
