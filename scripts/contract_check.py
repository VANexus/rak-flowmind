#!/usr/bin/env python
"""P0 契约快照 + 漂移检查（依据 ECO-ADR-0029 §2 的 6 条契约不变量）。

用途：把「重构期冻结的对外契约」变成**机器可读快照**，任何改动都能被
`contract_check.py` 当场发现，并定位到具体文件与字段。这是 P1–P4（镜像化 /
控制面数据面分离 / 拆 worker）的安全网——没有它，拆 worker 等于蒙眼改契约。

用法（须在 flowmind 环境里跑，包按 PYTHONPATH 引入）：

    PYTHONPATH=$PWD/src conda run -n flowmind python scripts/contract_check.py
        # 从代码提取快照 → 与 docs/contracts/contracts.snapshot.json 比对；不一致退 1
    PYTHONPATH=$PWD/src conda run -n flowmind python scripts/contract_check.py --write
        # 重新生成快照（仅在**确认契约有意变更**后使用，并与消费方双侧同步）
    PYTHONPATH=$PWD/src conda run -n flowmind python scripts/contract_check.py --json
        # 只打印当前快照 JSON（供 diff / P1-P4 阶段比对）

设计要点：快照**只从代码提取**，每条 record 带 ``locator``（``文件#符号``），
因此差异天然可定位到「哪个文件的哪个字段」。人读说明见 docs/contracts/contracts.md。

shortcut: DDL 列集 / 路由状态码用源码正则提取（不连 PG、不起 HTTP 服务），
升级触发=需要校验「线上已建表的真实列集」，那时应改为查 information_schema 对账。
shortcut: 只比对字段名 / 类型名 / 常量集合，不比 pydantic schema 逐字节
（后者含 $defs 顺序等噪声，极脆），升级触发=消费方要求 schema 严格等价。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SNAPSHOT_PATH = REPO / "docs" / "contracts" / "contracts.snapshot.json"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


def _columns(body: str, indent: int) -> list[str]:
    """从 CREATE TABLE 字段体里提列名（按行首缩进识别字段行）。"""
    return [m.group(1) for m in
            re.finditer(rf"^\s{{{indent}}}(\w+)\s+", body, re.M)]


# ── INV1：MCP wire + 工具全名 + tools/list 与 inputSchema 形状 ──────────────

def inv1_mcp_tools() -> dict:
    from flowmind.manifest import build_manifest
    from flowmind.skills import FAMILIES, selected_families

    skills = []
    for entry in build_manifest()["skills"]:
        inp = entry.get("input_schema") or {}
        out = entry.get("output_schema") or {}
        item = {
            "id": entry["id"],
            "name": entry["name"],
            "version": entry["version"],
            "category": entry["category"],
            "tags": entry["tags"],
            "input_props": sorted((inp.get("properties") or {})),
            "input_required": sorted(inp.get("required") or []),
            "output_props": sorted((out.get("properties") or {})),
        }
        if "stage_schema" in entry:
            item["stage_schema_ids"] = [s["id"] for s in entry["stage_schema"]]
        skills.append(item)

    from flowmind.config import FederationBackendConfig

    return {
        "locator": "src/flowmind/skills/__init__.py#FAMILIES",
        "families": {k: list(v) for k, v in FAMILIES.items()},
        "default_selection": list(selected_families()),
        "count": len(skills),
        "tool_ids": sorted(s["id"] for s in skills),
        "skills": skills,
        "tool_full_name": "{prefix}__{tool}",
        "prefix_default": {
            "locator": "src/flowmind/config.py#FederationBackendConfig.prefix",
            "value": FederationBackendConfig().prefix,
        },
    }


# ── INV2：SkillResult / ReasoningChain / manifest 响应形状 ────────────────

CONTRACT_MODELS = (
    "RuleHit", "Evidence", "ReasoningChain", "ReliabilityMetrics",
    "TraceContext", "SkillError", "SkillOutput", "SkillResult",
)


def inv2_response_shapes() -> dict:
    from flowmind import contracts as C

    models = {}
    for name in CONTRACT_MODELS:
        model = getattr(C, name)
        models[name] = {
            "locator": f"src/flowmind/contracts.py#{name}",
            "fields": {k: str(v.annotation).replace("typing.", "")
                       for k, v in model.model_fields.items()},
            "required": sorted(k for k, v in model.model_fields.items() if v.is_required()),
        }

    src = _read("flowmind/discover.py")
    entry_body = re.search(r"def skill_entry.*?return \{(.*?)\n    \}", src, re.S).group(1)
    return {
        "models": models,
        "manifest_entry": {
            "locator": "src/flowmind/discover.py#skill_entry",
            "keys": sorted(set(re.findall(r'"(\w+)":', entry_body))),
            "note": "manifest = {\"skills\": [entry, ...]}；有流水线的技能额外带 stage_schema",
        },
    }


# ── INV3：REST /api/v1/tasks 状态机、状态码、轮询语义 ─────────────────────

_ROUTE_RE = re.compile(r'custom_route\("([^"]+)", methods=\[([^\]]+)\]\)')
_STATUS_RE = re.compile(r"status_code=(\d{3})")
_ERROR_RE = re.compile(r'"error": "(\w+)"')


def _routes() -> list[dict]:
    """按源码顺序提路由，并把其处理函数体里的 status_code / error 码归到该路由。"""
    routes: list[dict] = []
    current: dict | None = None
    for rel in ("flowmind/server_tasks.py", "flowmind/server_rest.py"):
        for line in _read(rel).splitlines():
            m = _ROUTE_RE.search(line)
            if m:
                current = {
                    "path": m.group(1),
                    "methods": sorted(x.strip().strip('"') for x in m.group(2).split(",")),
                    "status_codes": [],
                    "error_codes": [],
                    "source": f"src/{rel}",
                }
                routes.append(current)
                continue
            if current is not None:
                current["status_codes"] += [int(x) for x in _STATUS_RE.findall(line)]
                current["error_codes"] += _ERROR_RE.findall(line)
    for r in routes:
        r["status_codes"] = sorted(set(r["status_codes"]))
        r["error_codes"] = sorted(set(r["error_codes"]))
    return routes


def inv3_rest_tasks() -> dict:
    from flowmind.tasks import (
        NON_TERMINAL_STATUSES,
        STATUS_CANCELLED,
        STATUS_FAILED,
        STATUS_INTERRUPTED,
        STATUS_QUEUED,
        STATUS_RUNNING,
        STATUS_SUCCEEDED,
        TERMINAL_STATUSES,
    )

    src = _read("flowmind/server_tasks.py")
    health = re.search(r"body = \{(.*?)\n        \}", src, re.S).group(1)
    return {
        "locator": "src/flowmind/server_tasks.py + src/flowmind/server_rest.py",
        "single_port": 8002,
        "server_version": re.search(r'_VERSION = "([^"]+)"', src).group(1),
        "routes": _routes(),
        "health": {
            "locator": "src/flowmind/server_tasks.py#health",
            "keys": sorted(set(re.findall(r'"(\w+)":', health))),
            "status_values": ["ok", "degraded"],
            "components": ["pg", "mqtt", "milvus"],
            "always_200": True,
        },
        "status_machine": {
            "locator": "src/flowmind/tasks/__init__.py#STATUS_*",
            "states": [STATUS_QUEUED, STATUS_RUNNING, STATUS_SUCCEEDED,
                       STATUS_FAILED, STATUS_CANCELLED, STATUS_INTERRUPTED],
            "terminal": sorted(TERMINAL_STATUSES),
            "non_terminal": sorted(NON_TERMINAL_STATUSES),
        },
        "polling": {
            "locator": "src/flowmind/server_tasks.py#task_get + tasks/store.py#_row_to_dict",
            "row_keys": sorted(re.findall(
                r'"(\w+)":',
                re.search(r"out = \{(.*?)\n    \}", _read("flowmind/tasks/store.py"), re.S).group(1))),
            "args_opt_in": "GET /api/v1/tasks/{id}?include=args",
        },
    }


# ── INV4：PG flowmind.tasks / editing_projects / editing_assets 列集 ──────

def inv4_pg_tables() -> dict:
    store = _read("flowmind/tasks/store.py")
    ddl = re.search(r"_DDL = \[(.*?)\n\]", store, re.S).group(1)
    tasks_cols = _columns(re.search(
        r"CREATE TABLE IF NOT EXISTS \{_SCHEMA_NAME\}\.\{_TABLE_NAME\} \((.*?)\n    \)",
        ddl, re.S).group(1), indent=8)

    editing = _read("flowmind/editing/store.py")
    eddl = re.search(r"_DDL = \[(.*?)\n\]", editing, re.S).group(1)
    names = re.findall(r'_TABLE_\w+ = "(\w+)"', editing)
    bodies = re.findall(
        r"CREATE TABLE IF NOT EXISTS \{_SCHEMA\}\.\{_TABLE_\w+\} \((.*?)\n    \)", eddl, re.S)
    tables = {f"flowmind.{n}": _columns(b, indent=8) for n, b in zip(names, bodies)}

    return {
        "locator": "src/flowmind/tasks/store.py#_DDL + src/flowmind/editing/store.py#_DDL",
        "policy": "列集向前兼容，不删列（ADR-0029 §2.4）",
        "flowmind.tasks": {
            "columns": tasks_cols,
            "indexes": sorted(re.findall(r"CREATE INDEX IF NOT EXISTS (\w+)", ddl)),
        },
        "editing": tables,
    }


# ── INV5：联邦注册 MQTT topic/QoS/retain + mcp.federation_backends 契约 ───

def inv5_federation() -> dict:
    pg = _read("flowmind/federation/pg.py")
    backends = _columns(re.search(
        r"_DDL_BACKENDS = f\"\"\"(.*?)\"\"\"", pg, re.S).group(1), indent=4)

    mqtt = _read("flowmind/federation/mqtt.py")
    register = _read("flowmind/federation/register.py")
    payload_body = re.search(
        r"def registration_payload.*?return \{(.*?)\n        \}", register, re.S).group(1)

    events = _read("flowmind/tasks/events.py")
    event_body = re.search(r"body: dict = \{(.*?)\n        \}", events, re.S).group(1)
    return {
        "mqtt": {
            "locator": "src/flowmind/federation/mqtt.py#TOPIC_*",
            "topics": dict(re.findall(r'^TOPIC_(\w+) = "([^"]+)"', mqtt, re.M)),
            "qos": {k: int(v) for k, v in
                    re.findall(r'"(\w+)": \(TOPIC_\w+, (\d)\)', mqtt)},
            "task_events": {
                "locator": "src/flowmind/tasks/events.py#_TOPIC_PREFIX",
                "template": "mcp-base-gpu/tasks/{task_id}/events",
                "qos": int(re.search(r"qos=(\d)", events).group(1)),
                "retain": "terminal statuses only",
                "payload_keys": sorted(set(re.findall(r'"(\w+)":', event_body))),
            },
        },
        "pg": {
            "locator": "src/flowmind/federation/pg.py#_DDL_BACKENDS",
            "table": "mcp.federation_backends",
            "columns": backends,
            "owner": "go-kernel 注册表；flowmind 只写自身一行（最小写）",
            "sync_note": "与 go-kernel/internal/pgstore 同 DDL，改动双侧同步",
        },
        "registration_payload_keys": sorted(set(re.findall(r'"(\w+)":', payload_body))),
    }


# ── INV6：消费方零改动（go-kernel 联邦 / cross-dashboard）────────────────

def inv6_consumers() -> dict:
    return {
        "locator": "README.md#MCP 客户端接入 + docs/architecture/greenfield/design.md §12",
        "consumers": ["go-kernel（MCP 联邦网关，:8090）", "cross-dashboard（BFF）"],
        "zero_change_contract": [
            "{prefix}__{tool} 工具全名",
            "SkillResult / manifest 响应形状",
            "REST /api/v1/tasks 提交-轮询-下载语义",
            "MQTT 任务进度 topic 与终态 retain",
        ],
        "backends": [
            {"name": "video_localizer", "port": 8002, "note": "缺省进程（localize+edit+packaging）"},
            {"name": "dsp", "port": 8003, "note": "独立进程/部署，prefix=dsp"},
        ],
    }


INVARIANTS = {
    "INV1_mcp_wire_and_tools": inv1_mcp_tools,
    "INV2_response_shapes": inv2_response_shapes,
    "INV3_rest_tasks_contract": inv3_rest_tasks,
    "INV4_pg_tables": inv4_pg_tables,
    "INV5_federation_contract": inv5_federation,
    "INV6_consumers_zero_change": inv6_consumers,
}


def build_snapshot() -> dict:
    return {
        "schema": "flowmind-contract-snapshot/1",
        "adr": "ECO-ADR-0029 §2（重构期对外契约冻结）",
        "invariants": {name: fn() for name, fn in INVARIANTS.items()},
    }


# ── 比对 ────────────────────────────────────────────────────────────────

def _flatten(obj, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        # 列表按「元素集合」比（顺序不是契约），元素是标量就整体比
        if all(not isinstance(x, (dict, list)) for x in obj):
            flat[prefix] = sorted(obj, key=str)
        else:
            for i, v in enumerate(obj):
                flat.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        flat[prefix] = obj
    return flat


def diff_snapshots(want: dict, got: dict) -> list[str]:
    """返回可定位的差异行（空 = 一致）。"""
    diffs: list[str] = []
    flat_want, flat_got = _flatten(want), _flatten(got)
    for key in sorted(set(flat_want) | set(flat_got)):
        a, b = flat_want.get(key, "<缺>"), flat_got.get(key, "<缺>")
        if a != b:
            diffs.append(f"{key}\n    快照: {a!r}\n    当前: {b!r}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description="P0 契约快照 / 漂移检查（ECO-ADR-0029 §2）")
    ap.add_argument("--write", action="store_true", help="重新生成快照文件")
    ap.add_argument("--json", action="store_true", help="只打印当前提取的快照 JSON")
    args = ap.parse_args()

    snapshot = build_snapshot()
    if args.json:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    text = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.write:
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(text, encoding="utf-8")
        n = len(snapshot["invariants"])
        print(f"已写入契约快照：{SNAPSHOT_PATH.relative_to(REPO)}（{n} 条不变量）")
        return 0

    if not SNAPSHOT_PATH.is_file():
        print(f"缺少快照文件：{SNAPSHOT_PATH.relative_to(REPO)}；先跑 --write 生成")
        return 1

    saved = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    diffs = diff_snapshots(saved, snapshot)
    if diffs:
        print(f"契约漂移 {len(diffs)} 处（快照 vs 当前代码）：\n")
        for d in diffs:
            print(f"  - {d}\n")
        print("判定：契约已漂移。若是**有意**变更：与消费方双侧同步 → 更新快照"
              "（--write）→ 另立 ADR；否则修回代码。")
        return 1

    print("契约一致：6 条不变量全部匹配快照 "
          f"（{SNAPSHOT_PATH.relative_to(REPO)}）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
