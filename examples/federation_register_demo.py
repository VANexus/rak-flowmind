"""federation 联邦自注册 e2e 冒烟 —— PG + MQTT 双通道注册 / 心跳 / 优雅注销。

运行：PYTHONPATH=<repo>/src conda run -n flowmind python examples/federation_register_demo.py

联邦链路（FLOWMIND_FEDERATION_REGISTER=1 开启，默认关）经 start_federation
走完整装配 → 注册 → 心跳 → 注销生命周期：
1. 默认关：未设开关时 start_federation 返回 None（server_http 据此跳过
   lifespan 包裹——默认关闭路径与无联邦部署严格等价）；
2. happy：双通道注册（PG upsert + MQTT register QoS1）→ 心跳周期消息
   （QoS0）→ stop 优雅注销（unregister QoS1 + PG offline）+ 幂等；
3. degraded：单通道存活仍完成注册（PG 未配置仅 MQTT / MQTT 未配置仅 PG）；
4. error：双通道全不可达（未配置态 / 配置但连接全败）→ 静默降级——
   不抛异常、不阻塞、无成功消息（联邦绝不影响主流程）；
5. 版本契约：_resolve_version 非 "0.0.0"（pip 安装态读 importlib.metadata，
   源码态回落 pyproject.toml——register.py parents[3] 仓库根修复）。

mock 方式：patch flowmind.federation.register 模块级 FederationPGStore /
FederationPublisher 符号为内存 fake（本文件内实现），不依赖真实集群；
配置经 env（FLOWMIND_FEDERATION_*）+ reload_config() 注入，与部署路径同源。
"""

from __future__ import annotations

import os
import re
import time

import flowmind.config as cfg_mod
from flowmind.federation import (
    TOPIC_HEARTBEAT,
    TOPIC_REGISTER,
    TOPIC_UNREGISTER,
    register as fed_register,
)

# RFC3339 契约（Go 侧 time.Time 按 RFC3339 解析；now_iso() 输出带时区 ISO8601）
_RFC3339 = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")

# QoS 契约（与 federation/mqtt.py _ACTIONS 逐字对齐，改动必须双侧同步）
_QOS = {"register": 1, "heartbeat": 0, "unregister": 1}


def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


class _FakePGStore:
    """FederationPGStore 内存替身（记录调用；enabled/fail 两维控制通道态）。"""

    def __init__(self, *, enabled: bool = True, fail: bool = False) -> None:
        self.enabled = enabled      # False = 未配置（永久禁用语义）
        self._fail = fail           # True = 配置但连接/写入全败（瞬时不可达）
        self.calls: list[tuple[str, str]] = []  # (op, backend_id)

    def _gate(self) -> bool:
        return not (self._fail or not self.enabled)

    def upsert_backend(self, *, backend_id: str, **_kw) -> bool:
        if self._gate():
            self.calls.append(("upsert", backend_id))
        return self._gate()

    def update_heartbeat(self, backend_id: str) -> bool:
        if self._gate():
            self.calls.append(("heartbeat", backend_id))
        return self._gate()

    def set_offline(self, backend_id: str) -> bool:
        if self._gate():
            self.calls.append(("offline", backend_id))
        return self._gate()


class _FakePublisher:
    """FederationPublisher 内存替身（记录 (action, topic, qos, payload)）。"""

    def __init__(self, *, enabled: bool = True, fail: bool = False) -> None:
        self.enabled = enabled
        self._fail = fail
        self.messages: list[tuple[str, str, int, dict]] = []

    def publish(self, action: str, payload: dict, *, wait: bool = False) -> bool:
        if self._fail or not self.enabled:
            return False
        topics = {
            "register": TOPIC_REGISTER,
            "heartbeat": TOPIC_HEARTBEAT,
            "unregister": TOPIC_UNREGISTER,
        }
        self.messages.append((action, topics[action], _QOS[action], dict(payload)))
        return True

    def close(self) -> None:  # 停机路径调用，no-op
        pass


_ORIG_PG = fed_register.FederationPGStore
_ORIG_PUB = fed_register.FederationPublisher


def _install(pg: _FakePGStore, pub: _FakePublisher) -> None:
    """patch register 模块级类符号为 fake 工厂（Registrar 构造时命中）。"""
    fed_register.FederationPGStore = lambda: pg
    fed_register.FederationPublisher = lambda: pub


def _set_federation_env(on: bool) -> None:
    """经 env 注入联邦配置（与部署路径同源）+ 强制重读配置缓存。"""
    if on:
        os.environ["FLOWMIND_FEDERATION_REGISTER"] = "1"
        os.environ["FLOWMIND_FEDERATION_BACKEND_ID"] = "video_localizer"
        os.environ["FLOWMIND_FEDERATION_PREFIX"] = "video_localizer"
        os.environ.pop("FLOWMIND_FEDERATION_URL", None)  # 空 → 127.0.0.1:8002 兜底
        os.environ["FLOWMIND_FEDERATION_HEARTBEAT_INTERVAL"] = "1"
    else:
        for key in ("FLOWMIND_FEDERATION_REGISTER", "FLOWMIND_FEDERATION_BACKEND_ID",
                    "FLOWMIND_FEDERATION_PREFIX", "FLOWMIND_FEDERATION_URL",
                    "FLOWMIND_FEDERATION_HEARTBEAT_INTERVAL"):
            os.environ.pop(key, None)
    cfg_mod.reload_config()


def _msgs(pub: _FakePublisher, action: str) -> list[tuple[str, str, int, dict]]:
    return [m for m in pub.messages if m[0] == action]


def main() -> None:
    # ── 0) 默认关：不设 FLOWMIND_FEDERATION_REGISTER → 不装配联邦 ──
    section("0) 默认关（无 FLOWMIND_FEDERATION_REGISTER）→ start_federation 返回 None")
    _set_federation_env(on=False)
    _install(_FakePGStore(), _FakePublisher())
    check(fed_register.start_federation(port=8002) is None,
          "默认关应返回 None（server_http 据此跳过 lifespan 包裹，主流程零变化）")
    print("  ✓ 未启用 → None → 无注销回调 → 不包裹 lifespan（启动路径严格等价）")

    # ── 1) Happy：双通道注册 → 心跳 → 优雅注销（payload/topic/QoS 契约全断言）──
    section("1) Happy：PG upsert + MQTT register(QoS1) → 心跳(QoS0) → 注销(QoS1)")
    _set_federation_env(on=True)
    fake_pg, fake_pub = _FakePGStore(), _FakePublisher()
    _install(fake_pg, fake_pub)
    registrar = fed_register.start_federation(port=8002)
    check(registrar is not None, "开关开启应返回 registrar 实例")

    reg = _msgs(fake_pub, "register")
    check(len(reg) == 1, "register 消息恰一条")
    _, topic, qos, payload = reg[0]
    print(f"  topic   : {topic}")
    print(f"  qos     : {qos}")
    print(f"  payload : {payload}")
    check(topic == TOPIC_REGISTER, f"topic 应为 {TOPIC_REGISTER}")
    check(qos == 1, "register QoS 契约 = 1（至少一次，新后端必须送达）")
    check(payload["backend_id"] == "video_localizer", "payload.backend_id 契约")
    check(payload["prefix"] == "video_localizer", "payload.prefix 契约")
    check(payload["url"] == "http://127.0.0.1:8002/mcp",
          "url 空 → http://127.0.0.1:{port}/mcp 兜底")
    check(_RFC3339.match(payload["ts"]) is not None, "ts 应为 RFC3339（Go time.Time 可解析）")
    check(fake_pg.calls == [("upsert", "video_localizer")], "PG upsert 幂等注册一次")

    # 心跳周期消息（QoS0；interval=1s，等 1.4s 容忍线程调度）
    time.sleep(1.4)
    beats = _msgs(fake_pub, "heartbeat")
    print(f"  heartbeat 条数: {len(beats)}（interval=1s）")
    check(len(beats) >= 1, "心跳线程应按周期发布 heartbeat")
    check(all(b[2] == 0 for b in beats), "heartbeat QoS 契约 = 0（高频容忍丢失）")
    check(all(b[3]["backend_id"] == "video_localizer" and
              _RFC3339.match(b[3]["ts"]) for b in beats), "heartbeat payload 契约")

    # 优雅注销：unregister QoS1 + PG offline；stop 幂等
    n_before = len(fake_pub.messages)
    registrar.stop()
    registrar.stop()  # 双触发无害
    unreg = _msgs(fake_pub, "unregister")
    print(f"  unregister: {unreg[0][2:] if unreg else None}")
    check(len(unreg) == 1, "stop 幂等 → unregister 恰一条")
    check(unreg[0][1] == TOPIC_UNREGISTER and unreg[0][2] == 1,
          "unregister topic/QoS 契约 = 1（优雅退出必须送达）")
    check(unreg[0][3]["backend_id"] == "video_localizer"
          and unreg[0][3]["reason"] == "shutdown", "unregister payload 契约")
    check(("offline", "video_localizer") in fake_pg.calls, "PG 置 offline（行保留供审计）")
    check(len(fake_pub.messages) == n_before + 1, "stop 之外无额外消息")
    check(len(_msgs(fake_pub, "register")) == 1, "register 全程恰一条（幂等不重发）")

    # ── 2) Degraded：单通道存活仍完成注册 ──
    section("2a) Degraded：PG 未配置，仅 MQTT 通道 → 注册成功 + 心跳照常")
    fake_pg, fake_pub = _FakePGStore(enabled=False), _FakePublisher()
    _install(fake_pg, fake_pub)
    registrar = fed_register.start_federation(port=8002)
    check(registrar is not None, "单通道存活应完成装配")
    check(len(_msgs(fake_pub, "register")) == 1
          and _msgs(fake_pub, "register")[0][2] == 1, "MQTT register(QoS1) 完成注册")
    check(fake_pg.calls == [], "PG 未配置零写入（永久禁用语义）")
    time.sleep(1.2)
    check(len(_msgs(fake_pub, "heartbeat")) >= 1, "心跳经 MQTT 单通道维持在线")
    registrar.stop()

    section("2b) Degraded：MQTT 未配置，仅 PG 通道 → 注册成功")
    fake_pg, fake_pub = _FakePGStore(), _FakePublisher(enabled=False)
    _install(fake_pg, fake_pub)
    registrar = fed_register.start_federation(port=8002)
    check(registrar is not None, "单通道存活应完成装配")
    check(fake_pg.calls[:1] == [("upsert", "video_localizer")], "PG upsert 完成注册")
    check(fake_pub.messages == [], "MQTT 未配置零消息")
    registrar.stop()

    # ── 3) Error：双通道全不可达 → 静默降级（不抛异常、不阻塞） ──
    section("3a) Error：双通道均未配置 → 静默跳过，返回 False 语义")
    fake_pg, fake_pub = _FakePGStore(enabled=False), _FakePublisher(enabled=False)
    _install(fake_pg, fake_pub)
    registrar = fed_register.start_federation(port=8002)
    check(registrar is not None, "装配不抛（联邦绝不阻塞服务启动）")
    check(fake_pub.messages == [] and fake_pg.calls == [], "双通道未配置零副作用")
    registrar.stop()  # 幂等清理不抛
    print("  ✓ 未配置 → 无注册无心跳，进程照常")

    section("3b) Error：双通道配置但连接全败 → 不抛异常，心跳兜底待通道恢复")
    fake_pg, fake_pub = _FakePGStore(fail=True), _FakePublisher(fail=True)
    _install(fake_pg, fake_pub)
    registrar = fed_register.start_federation(port=8002)
    check(registrar is not None, "瞬时全败不抛、不阻塞（尽力而为重试语义）")
    check(fake_pub.messages == [] and fake_pg.calls == [], "全败时无成功消息/调用")
    time.sleep(1.2)  # 心跳线程兜底运行（paho 自动重连 + PG 每拍重试）
    registrar.stop()
    print("  ✓ 全不可达 → 静默降级，主流程不受影响")

    # ── 4) 版本契约：_resolve_version 非 "0.0.0" ──
    section("4) 版本契约：_resolve_version 应读到真实版本（非 0.0.0）")
    version = fed_register._resolve_version()
    print(f"  version: {version}")
    check(version != "0.0.0",
          "源码态应经 importlib.metadata / pyproject.toml（parents[3]）读到真实版本")

    # ── 清理：恢复被 patch 的类符号与 env（进程内整洁） ──
    fed_register.FederationPGStore = _ORIG_PG
    fed_register.FederationPublisher = _ORIG_PUB
    _set_federation_env(on=False)

    print("\n✅ federation_register_demo 全部通过")


if __name__ == "__main__":
    main()
