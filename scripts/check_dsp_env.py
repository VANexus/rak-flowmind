"""dsp 接入前环境自检：只报「已配 / 未配」，**绝不打印密钥值**。

用途：把材料交过来之后，双方用同一条命令对账——你不用把密钥贴进对话，
我也无需读值即可确认哪些项还缺。

运行：
    PYTHONPATH=src python scripts/check_dsp_env.py

退出码：必需项齐备 0；有缺 1（可当接入前门禁用）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from flowmind.config import load_config
from flowmind.skills._secrets import get_api_key

_REPO = Path(__file__).resolve().parents[1]

# 必需项：(说明, env 名)；缺任一项，成片链路跑不到底
_REQUIRED_KEYS = [
    ("对象存储 access key", "S3_ACCESS_KEY"),
    ("对象存储 secret key", "S3_SECRET_KEY"),
    ("火山 MediaKit（渲染）", "DSP_MEDIAKIT_API_KEY"),
    ("火山豆包语音（配音）", "DSP_TTS_API_KEY"),
]

# 可选项：(说明, env 名)；只在开启对应能力时才需要，缺了不算链路不可用
_OPTIONAL_KEYS = [
    ("图生视频（第三方聚合 · 开启 videoGen 才需要）", "DSP_VIDEOGEN_API_KEY"),
]

# 非密钥配置：值可安全打印（照抄进 .env 即可）
_PLAIN_ITEMS = [
    ("FLOWMIND_SKILL_FAMILIES", "dsp"),
    ("FLOWMIND_MCP_PORT", "8003"),
    ("FLOWMIND_FEDERATION_REGISTER", "1"),
    ("FLOWMIND_FEDERATION_BACKEND_ID", "dsp"),
    ("FLOWMIND_FEDERATION_PREFIX", "dsp"),
]


def _mark(ok: bool) -> str:
    return "✅ 已配" if ok else "❌ 未配"


def main() -> int:
    cfg = load_config().dsp
    ws_env = _REPO.parent / ".env"
    proj_env = _REPO / ".env"

    print("─" * 62)
    print("  A) 密钥（只看「是否已配」，不读值）")
    print("─" * 62)
    missing = 0
    for label, env_name in _REQUIRED_KEYS:
        ok = bool(get_api_key(env_name))
        missing += 0 if ok else 1
        print(f"  {_mark(ok)}  {label:<22} env: {env_name}")

    print("  ── 可选（开关开启才需要）──")
    for label, env_name in _OPTIONAL_KEYS:
        ok = bool(get_api_key(env_name))
        print(f"  {_mark(ok)}  {label} env: {env_name}")

    ws_state = "（存在）" if ws_env.exists() else "（不存在）"
    proj_state = "（存在）" if proj_env.exists() else "（不存在）"
    print(f"\n  密钥落点：工作区根 {ws_env}{ws_state}  |  项目 {proj_env}{proj_state}")
    print("            （先加载工作区根 .env，再加载项目 .env；已存在的变量不被覆盖）")

    print("\n" + "─" * 62)
    print("  B) 非密钥配置（值可直接照抄）")
    print("─" * 62)
    print(f"  存储端点 FLOWMIND_DSP_S3_ENDPOINT = {cfg.storage_endpoint or '(空 → 未配)'}")
    print(f"  存储桶   FLOWMIND_DSP_S3_BUCKET   = {cfg.storage_bucket}（须与前端 /api/assets 同桶）")
    print(f"  存储区域 FLOWMIND_DSP_S3_REGION   = {cfg.storage_region or '(空 → 可空)'}")
    print(f"  明文传输 FLOWMIND_DSP_S3_SECURE   = {cfg.storage_secure}（MinIO 明文设 0）")
    # ADR-123：图生视频厂商要求参考素材是公网 http(s)；没有公网入口 ⇒ 逐镜跳过（回落静图）。
    from flowmind.skills import _dsp_storage as storage  # noqa: PLC0415  仅为判定公网入口
    public = cfg.storage_public_endpoint or '(空 → 未配)'
    print(f"  公网端点 FLOWMIND_DSP_S3_PUBLIC_ENDPOINT = {public}")
    print("  图生视频公网入口 = "
          + ("✅ 可用" if storage.public_presign_configured()
             else "❌ 没有（图生视频会逐镜 skipped / 回落静图，步骤标 degraded）"))
    print(f"  配音音色 DspConfig.tts_default_speaker = {cfg.tts_default_speaker or '(空 → L1 待确认)'}")
    print(f"  渲染队列 FLOWMIND_DSP_MEDIAKIT_QUEUE_ID = {cfg.mediakit_queue_id or '(空 → 可空)'}")
    for name, expect in _PLAIN_ITEMS:
        actual = os.environ.get(name, "")
        ok = actual == expect
        print(f"  {_mark(ok if actual else False)}  {name} = {actual or '(未设)'}"
              + ("" if ok or not actual else f"（建议 {expect}）"))

    print("\n" + "─" * 62)
    print("  C) 厂商接口文档（放仓内后我能逐参数纠偏实现）")
    print("─" * 62)
    pdfs = sorted(p.name for p in _REPO.rglob("*.pdf") if ".git" not in p.parts)
    if pdfs:
        for name in pdfs:
            print(f"  ✅ {name}")
    else:
        print("  ❌ 仓内未发现 PDF。建议放 docs/vendor/（多轨道剪辑 / 提交多轨道剪辑任务 API /")
        print("     视频添加运镜 / 豆包语音·音频生成），我再按真文档核参数。")

    print("\n  D) 网关（go-kernel，独立仓；用于确认 prefix=dsp 聚合语义）")
    kernel = _REPO.parent / "go-kernel"
    print(f"  {_mark(kernel.exists())}  {kernel}")

    print("\n" + "═" * 62)
    if missing:
        print(f"  尚有 {missing} 项必需密钥未配 → 成片链路会在对应步如实失败/降级")
    else:
        print("  必需密钥齐备 → 可进行真实调用冒烟（会打付费 API，需你确认后我再跑）")
    print("═" * 62)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
