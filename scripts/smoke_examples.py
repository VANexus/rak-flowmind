#!/usr/bin/env python
"""P0 冒烟基线：`examples/*_demo.py` 的可重复执行入口（ECO-ADR-0029 §4「P0 退出证据」）。

本仓验证靠 demo 冒烟 + 真实调用（无 CI、无 Makefile）。历史做法是手敲
`for f in examples/*_demo.py; do ... done`，结果是：数量对不上（README 说 18、
AGENTS 说 22）、失败原因不可归因、依赖外部的 demo 会挡住整轮。本脚本把这三件事收口。

用法（须在 flowmind 环境里跑）：

    PYTHONPATH=$PWD/src conda run -n flowmind python scripts/smoke_examples.py
        # 跑全部 demo（含 needs-* 项；缺依赖的会被如实记 SKIP 而不是阻塞）
    PYTHONPATH=$PWD/src conda run -n flowmind python scripts/smoke_examples.py \
        --skip needs-ffmpeg,needs-key,needs-gpu
        # 纯离线底线（无 ffmpeg / 无密钥 / 无 GPU 的机器上可复现全绿）
    ... --list          # 只打印分类表与 known-fail
    ... --json          # 机器可读结果（P1–P4 阶段间比对）
    ... --keep-scratch  # 保留临时 HOME/TMPDIR（排障用）

分类（DEMO_REQUIRES，逐项显式列出——**这是「哪些 demo 需要真外部资源」的唯一真源**）：
``offline`` / ``needs-ffmpeg`` / ``needs-key`` / ``needs-gpu``。``--skip`` 只影响是否
执行，不影响报告——跳过项在汇总里始终可见。

环境归一（脚本级，**不改仓内 config.py 默认值**，依据见下）：

1. ``HOME`` / ``TMPDIR`` → 临时目录。原因：`LocalizerConfig.data_dir` 默认
   ``"~/flowmind-data"`` 且**无 env 覆盖**（.env.example 里的 FLOWMIND_DATA_DIR
   文档化了但 src 零读取，已登记为漂移）。HOME 不可写时 `localize_video` /
   `localize_quality` 会在 workdir mkdir 上抛 PermissionError。
2. 清 ``http(s)_proxy`` / 净化 ``no_proxy``（去掉 ``[::1]`` 这类带方括号的条目）。
   原因：httpx 0.28.1 把 no_proxy 条目当 URL 解析，``[::1]`` → ``InvalidURL:
   Invalid port: ':1]'``，`api_server_demo` 一启动就炸（2026-11-03 实测）。
   所有 demo 均为离线自包含（自 mock 云厂商/存储），不需要出网。

退出码：0 = 无 FAIL（known-fail 不计入）；1 = 有 FAIL。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"

OFFLINE = "offline"
NEEDS_FFMPEG = "needs-ffmpeg"
NEEDS_KEY = "needs-key"
NEEDS_GPU = "needs-gpu"

# demos 分类表（**真源**）：23 项 = `examples/*_demo.py` 实际数量
# （README 写 18 / AGENTS 写 22 / examples/README 写 19，均已漂移，见本 PR 修正）。
DEMO_REQUIRES: dict[str, tuple[str, str]] = {
    "api_server_demo.py": (OFFLINE, "子线程 uvicorn + 假 manager；不依赖 PG/MQTT/GPU/key"),
    "auth_token_demo.py": (OFFLINE, "TestClient 测鉴权中间件，纯内存"),
    "decide_demo.py": (OFFLINE, "决策模型全 mock"),
    "diarize_demo.py": (NEEDS_FFMPEG, "现场用 ffmpeg 合成正弦波 wav 并断言 rc==0"),
    "dsp_bgm_match_demo.py": (OFFLINE, "选曲接口未给 ⇒ 占位降级，无外部调用"),
    "dsp_cart_sticker_demo.py": (OFFLINE, "纯逻辑（商品锚点/越界判定）"),
    "dsp_cover_pick_demo.py": (OFFLINE, "取帧走注入假件"),
    "dsp_image_to_video_demo.py": (OFFLINE, "替换 _dsp_video_gen/_dsp_storage/media 假件 + 注入假 ffmpeg"),
    "dsp_render_compose_demo.py": (OFFLINE, "替换 MediaKit/S3 假件"),
    "dsp_subtitle_build_demo.py": (OFFLINE, "纯 SRT 逻辑"),
    "dsp_timeline_build_demo.py": (OFFLINE, "纯时间线/切镜逻辑"),
    "dsp_tts_generate_demo.py": (OFFLINE, "云配音走假件"),
    "edit_pipeline_demo.py": (NEEDS_FFMPEG, "FLOWMIND_EDIT_MOCK=1 下渲染走真 ffmpeg lavfi，缺则显式退出"),
    "federation_register_demo.py": (OFFLINE, "PG/MQTT 双通道走假件"),
    "localize_cancel_demo.py": (OFFLINE, "假 manager"),
    "localize_download_demo.py": (OFFLINE, "假 store"),
    "localize_quality_demo.py": (NEEDS_FFMPEG, "3A 现场用 ffmpeg 生成 wav 真跑 _mix_timed_audio"),
    "localize_retry_demo.py": (OFFLINE, "假 store/manager"),
    "localize_search_demo.py": (OFFLINE, "BGE/Milvus 走假件"),
    "localize_status_demo.py": (OFFLINE, "假 store"),
    "localize_submit_demo.py": (OFFLINE, "假 manager"),
    "localize_video_demo.py": (OFFLINE, "全链路 mock：云调用 / ffmpeg / 本地模型全替身，不占 GPU"),
    "packaging_dieline_demo.py": (OFFLINE, "纯几何（Shapely）；DXF/PDF 缺依赖时如实少产"),
}

# known-fail：环境/仓内已知原因**预期失败**的 demo。命中者不计入退出门禁，
# 但在汇总里**始终列出**（不允许静默）。格式：文件名 → (原因, 定位/后续动作)。
# 当前为空：P0 修完 3 个陈旧 demo（api_server skill_count / dsp_render_compose uploads /
# dsp_image_to_video 假件入口与 payload 形状）后，23 项要么 PASS 要么按分类 SKIP。
KNOWN_FAIL: dict[str, tuple[str, str]] = {}


def _child_env(scratch: Path, timeout_home: Path) -> dict:
    """给子进程的归一化环境（依据见模块 docstring）。"""
    env = dict(os.environ)
    env["HOME"] = str(timeout_home)
    env["TMPDIR"] = str(scratch / "tmp")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    for key in ("http_proxy", "https_proxy", "all_proxy",
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env.pop(key, None)
    for key in ("no_proxy", "NO_PROXY"):
        raw = env.get(key)
        if raw:
            kept = [p for p in raw.split(",") if p.strip() and not p.strip().startswith("[")]
            env[key] = ",".join(kept)
    env["PYTHONPATH"] = str(REPO / "src")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _discover() -> list[str]:
    return sorted(p.name for p in EXAMPLES.glob("*_demo.py"))


def _classify(name: str) -> tuple[str, str]:
    if name in DEMO_REQUIRES:
        return DEMO_REQUIRES[name]
    return OFFLINE, "未登记（新 demo 请补 DEMO_REQUIRES 分类与理由）"


def _run_one(name: str, timeout: int, env: dict) -> dict:
    cmd = [sys.executable, str(EXAMPLES / name)]
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, timeout=timeout,
                              capture_output=True, text=True)
        code, out = proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        code = -1
        out = f"TIMEOUT（>{timeout}s）\n" + (exc.stdout or "") + (exc.stderr or "")
    elapsed = time.monotonic() - started
    return {"demo": name, "exit_code": code, "seconds": round(elapsed, 1), "output": out}


def main() -> int:
    ap = argparse.ArgumentParser(description="examples/*_demo.py 冒烟基线（P0）")
    ap.add_argument("--skip", default="",
                    help="要跳过的分类，逗号分隔（offline|needs-ffmpeg|needs-key|needs-gpu）")
    ap.add_argument("--timeout", type=int, default=300, help="单个 demo 超时秒数（默认 300）")
    ap.add_argument("--json", action="store_true", help="只输出 JSON 结果")
    ap.add_argument("--list", action="store_true", help="只打印分类表")
    ap.add_argument("--keep-scratch", action="store_true", help="保留临时 HOME/TMPDIR")
    args = ap.parse_args()

    demos = _discover()
    skipped_cats = {c.strip() for c in args.skip.split(",") if c.strip()}
    unknown = skipped_cats - {OFFLINE, NEEDS_FFMPEG, NEEDS_KEY, NEEDS_GPU}
    if unknown:
        print(f"未知分类：{sorted(unknown)}；可用：{sorted({OFFLINE, NEEDS_FFMPEG, NEEDS_KEY, NEEDS_GPU})}")
        return 1

    if args.list:
        print(f"共 {len(demos)} 个 demo（examples/*_demo.py）\n")
        for name in demos:
            cat, why = _classify(name)
            mark = " [KNOWN-FAIL]" if name in KNOWN_FAIL else ""
            print(f"  {cat:<13} {name:<34} {why}{mark}")
        counts: dict[str, int] = {}
        for name in demos:
            counts[_classify(name)[0]] = counts.get(_classify(name)[0], 0) + 1
        print(f"\n分类计数：{counts}")
        print("needs-gpu=0 的依据：全部 demo 都把本地模型替换为不可用（_local_asr/_local_tts/"
              "_inpaint/_diarize.available），真实 GPU 链路另做「起服务 + MCP 探针」，见 CLAUDE.md。")
        return 0

    scratch = Path(tempfile.mkdtemp(prefix="flowmind-smoke-"))
    home = scratch / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = _child_env(scratch, home)

    results: list[dict] = []
    quiet = args.json
    try:
        for name in demos:
            cat, why = _classify(name)
            if cat in skipped_cats:
                results.append({"demo": name, "category": cat, "status": "SKIP",
                                "reason": f"{why}（--skip {cat}）"})
                if not quiet:
                    print(f"SKIP  {name}  [{cat}]")
                continue
            res = _run_one(name, args.timeout, env)
            res["category"] = cat
            if name in KNOWN_FAIL:
                res["status"] = "KNOWN-FAIL"
                res["reason"] = KNOWN_FAIL[name][0]
            else:
                res["status"] = "PASS" if res["exit_code"] == 0 else "FAIL"
            results.append(res)
            if not quiet:
                print(f"{res['status']:<10} {name:<34} {res['seconds']:>6.1f}s  rc={res['exit_code']}")
    finally:
        if not args.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)

    fails = [r for r in results if r["status"] == "FAIL"]
    known = [r for r in results if r["status"] == "KNOWN-FAIL"]
    skips = [r for r in results if r["status"] == "SKIP"]
    passes = [r for r in results if r["status"] == "PASS"]

    if args.json:
        print(json.dumps({"demos": len(demos), "pass": len(passes), "fail": len(fails),
                          "known_fail": len(known), "skip": len(skips), "results": results},
                         ensure_ascii=False, indent=2))
    else:
        print(f"\n{'=' * 62}\n  demo {len(demos)}：PASS {len(passes)} / FAIL {len(fails)}"
              f" / KNOWN-FAIL {len(known)} / SKIP {len(skips)}\n{'=' * 62}")
        for r in fails:
            print(f"\n── FAIL {r['demo']}（rc={r['exit_code']}）最后 15 行 ──")
            for line in r["output"].strip().splitlines()[-15:]:
                print(f"  {line}")
        for r in known:
            print(f"KNOWN-FAIL {r['demo']}：{r['reason']}")
        if fails:
            print("\n判定：冒烟未全绿。修复后重跑，或把「环境/仓内已知原因」项登记进 KNOWN_FAIL"
                  "（必须写清原因与定位，不允许静默）。")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
