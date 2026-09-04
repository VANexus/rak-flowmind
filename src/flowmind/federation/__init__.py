"""flowmind.federation —— 功能包侧联邦自注册与心跳协议（任务 #14 阶段 2）。

功能包作为 MCP 后端加入联邦（Go 网关 go-kernel 动态发现）的接入面：
- 注册：进程启动时经 PG upsert + MQTT register（QoS1）双通道声明自身
  元数据（backend_id / prefix / url / version）；
- 心跳：daemon 线程每 30s 发布 heartbeat（QoS0）+ PG 刷新（双通道
  尽力而为）；网关 90s 无心跳标 offline（stale，工具保留）；
- 注销：优雅停机发布 unregister（QoS1）+ PG 置 offline（主路径
  uvicorn lifespan shutdown——SIGTERM 下 uvicorn 停机后 re-raise 信号
  杀进程、atexit 不执行；atexit 兜底 CTRL+C 等解释器正常退出路径）；
  异常死亡走网关心跳超时路径（无 LWT）。

降级铁律：本包任何失败（PG/MQTT 未配置或不可达）只记日志——联邦是
增值能力，绝不影响 7 个 localize_* 工具与任务 REST 主流程。默认关闭
（``FLOWMIND_FEDERATION_REGISTER=1`` 开启），现有 demo 行为不变。

模块：
- pg：注册表 PG 持久层（federation_backends 表，短连接 autocommit）
- mqtt：联邦协议 MQTT 发布器（register/heartbeat/unregister）
- register：FederationRegistrar 编排 + start_federation 装配入口
- heartbeat：HeartbeatWorker 守护线程

与 Go 网关侧的契约（主题/payload/QoS/DDL）改动必须双侧同步：
go-kernel/internal/{mqttclient,federation,pgstore}。
"""
from flowmind.federation.heartbeat import HeartbeatWorker
from flowmind.federation.mqtt import (
    TOPIC_HEARTBEAT,
    TOPIC_REGISTER,
    TOPIC_UNREGISTER,
    FederationPublisher,
)
from flowmind.federation.pg import FederationPGStore
from flowmind.federation.register import FederationRegistrar, start_federation

__all__ = [
    "FederationPGStore",
    "FederationPublisher",
    "FederationRegistrar",
    "HeartbeatWorker",
    "TOPIC_HEARTBEAT",
    "TOPIC_REGISTER",
    "TOPIC_UNREGISTER",
    "start_federation",
]
