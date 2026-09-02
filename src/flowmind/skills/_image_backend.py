"""图像生成后端抽象：可插拔的"出图"接口。

- MockBackend:基于 sha256 的确定性占位,无外部依赖,用于测试/离线场景。
- AllInApiBackend:调用 allin-api.com(OpenAI 兼容协议),模型 gpt-image-2。
- 公开输入/输出契约稳定,业务层只换 backend 类即可。

安全:AllInApiBackend 的 API key 只在 generate 时按 env var 名读取,绝不进
config 文件或 commit。调用方可注入 key 用于测试,但生产路径必须从环境变量走。
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import httpx


@dataclass
class GeneratedImage:
    """单张生成结果。url / local_path 二选一,base64 也作为 url 透传。"""
    index: int
    url: str
    local_path: str | None = None
    width: int = 0
    height: int = 0
    seed: int | None = None


class ImageBackend:
    """图像生成后端基类。子类实现 ``generate``。"""

    name: str = "base"

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        n: int,
        seed: int | None,
        save_dir: str | None,
    ) -> list[GeneratedImage]:
        raise NotImplementedError


class MockBackend(ImageBackend):
    """确定性占位后端:URL 由 sha256(prompt+seed+i) 派生,纯本地,可复现。

    安全:``save_dir`` 必须满足以下任一情况:
    - 为空 / None → 不生成 local_path（只返回 URL）
    - 绝对路径 + 不含 ``..`` 段 + 不指向敏感系统目录

    不满足时抛 ``ValueError``,避免 path-traversal（CWE-22）类风险。
    """

    name = "mock"

    def generate(self, *, prompt, negative_prompt, width, height, n, seed, save_dir):
        if seed is None:
            seed = int(_sha12(prompt), 16) & 0x7FFFFFFF
        # 校验 save_dir: 绝对路径 + 不含 .. 段 + 不指向系统目录
        safe_save_dir = _sanitize_save_dir(save_dir) if save_dir else None
        images: list[GeneratedImage] = []
        for i in range(n):
            img_seed = seed + i
            img_id = _sha12(prompt, negative_prompt, width, height, img_seed)
            url = f"https://flowmind.local/mock/{img_id}.png?w={width}&h={height}"
            local = f"{safe_save_dir}/{img_id}.png" if safe_save_dir else None
            images.append(GeneratedImage(
                index=i + 1,
                url=url,
                local_path=local,
                width=width,
                height=height,
                seed=img_seed,
            ))
        return images


def _sanitize_save_dir(save_dir: str) -> str:
    """校验 save_dir 防 path-traversal。

    - 必须是绝对路径
    - 不含 ``..`` 段（不允许相对路径跳转）
    - 不指向系统关键目录（/etc、/root、/var、/proc、/sys 等）

    返回规范化后的绝对路径。任何不合规直接 ``ValueError``。
    """
    from pathlib import Path
    p = Path(save_dir).expanduser()
    # 必须绝对 —— 相对路径无法判断安全边界
    if not p.is_absolute():
        raise ValueError(
            f"save_dir 必须是绝对路径，收到：{save_dir!r}。"
            f"为防 path-traversal，禁止相对路径。"
        )
    # 解析 .. 段和 symlink（strict=False 避免不存在报错）
    resolved = p.resolve(strict=False)
    # 检查 .. 段：解析后路径必须在原路径下
    try:
        resolved.relative_to(p)
    except ValueError:
        raise ValueError(
            f"save_dir 含 .. 路径跳转：{save_dir!r} → {resolved}。"
            f"为防 path-traversal，禁止使用 .. 段。"
        ) from None
    # 黑名单系统目录
    forbidden_prefixes = ("/etc", "/root", "/var", "/proc", "/sys", "/boot", "/dev")
    for prefix in forbidden_prefixes:
        if str(resolved).startswith(prefix + "/") or str(resolved) == prefix:
            raise ValueError(
                f"save_dir 指向系统敏感目录 {resolved}。为安全起见，禁止写入此处。"
            )
    return str(resolved)


class AllInApiBackend(ImageBackend):
    """生产后端:allin-api.com 兼容 OpenAI ``/v1/images/generations``。

    - 模型默认 ``gpt-image-2``,可在构造时覆盖。
    - API key 仅从环境变量读取(由调用方传入字符串,本类不直接读 env)。
    - gpt-image-2 不支持单独的 ``negative_prompt``,合并到 prompt 末尾。
    """

    name = "allin_api"

    def __init__(
        self,
        *,
        api_base: str = "https://allin-api.com",
        api_key: str,
        model: str = "gpt-image-2",
        timeout_s: float = 60.0,
        client: httpx.Client | None = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        # 测试时可注入 httpx.Client(配合 respx/transport=MockTransport)
        self._client = client

    def generate(self, *, prompt, negative_prompt, width, height, n, seed, save_dir):
        if not self.api_key:
            raise ValueError(
                "AllInApiBackend 收到空 API key。请检查环境变量 AI_IMAGE_API_KEY 是否设置。"
            )

        final_prompt = prompt
        if negative_prompt:
            # OpenAI 协议不直接支持 negative_prompt —— 合并到 prompt 末尾
            final_prompt = f"{prompt}\n\nAvoid: {negative_prompt}"

        body: dict = {
            "model": self.model,
            "prompt": final_prompt,
            "n": n,
            "size": f"{width}x{height}",
        }
        if seed is not None:
            # 部分实现支持 seed;不支持会被忽略,不会报错
            body["seed"] = seed

        url = f"{self.api_base}/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self._client is not None:
            resp = self._client.post(url, headers=headers, json=body)
        else:
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.post(url, headers=headers, json=body)
        if resp.status_code >= 400:
            # 提取 API 错误体里的具体原因（如 额度不足/无权使用模型），
            # 不用 raise_for_status（会把原因吞成笼统的 "403 Forbidden"）；
            # 错误消息脱敏：不回显完整 URL。
            detail = ""
            try:
                detail = str(resp.json().get("error", {}).get("message", ""))[:200]
            except Exception:
                detail = ""
            raise RuntimeError(
                f"生图 API 返回 HTTP {resp.status_code}：{detail or resp.reason_phrase}"
            )
        data = resp.json()

        items = data.get("data") or []
        if not items:
            raise RuntimeError(f"allin-api 返回空 data：{data}")

        images: list[GeneratedImage] = []
        for i, item in enumerate(items[:n]):
            img_url = item.get("url")
            if not img_url:
                b64 = item.get("b64_json")
                img_url = f"data:image/png;base64,{b64}" if b64 else ""
            images.append(GeneratedImage(
                index=i + 1,
                url=img_url,
                local_path=None,
                width=width,
                height=height,
                seed=(seed + i) if seed is not None else None,
            ))
        return images


# --- 工具 ---

def _sha12(*parts: object) -> str:
    """对多段输入算 sha256 取前 12 位。"""
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:12]


def resolve_api_key(env_var: str) -> str | None:
    """从环境变量读取 API key;找不到返回 None(由调用方决定如何兜底)。"""
    val = os.environ.get(env_var)
    return val.strip() or None if val else None


def select_backend(
    *,
    requested: str | None,
    cfg_allin_key_env: str,
    cfg_allin_base: str,
    cfg_allin_model: str,
    cfg_allin_timeout_s: float,
) -> ImageBackend:
    """根据 cfg 与入参 backend 选择后端。

    - ``backend="mock"`` → MockBackend（仅显式指定时可用，供测试/离线开发）
    - ``backend="allin_api"`` → AllInApiBackend(必须有 key,否则抛错)
    - ``backend="auto"`` 或 None → 有 key 用 allin_api；**无 key 直接抛错**，
      不再静默降级 mock（云优先原则：一切真实出图必须走云 API）
    """
    chosen = (requested or "auto").lower()

    if chosen == "mock":
        raise ValueError(
            "显式 mock 出图后端已禁用（云优先原则：一切营销生图 / SKU 图必须走真实云 API）。"
            "请在「设置 → B 端运营」配置 AI_IMAGE_API_KEY 后再重试。"
        )

    if chosen == "allin_api":
        api_key = resolve_api_key(cfg_allin_key_env) or ""
        return AllInApiBackend(
            api_base=cfg_allin_base,
            api_key=api_key,
            model=cfg_allin_model,
            timeout_s=cfg_allin_timeout_s,
        )

    if chosen == "auto":
        api_key = resolve_api_key(cfg_allin_key_env)
        if not api_key:
            raise ValueError(
                f"backend=auto 但未设置环境变量 {cfg_allin_key_env}。"
                f"云优先原则：真实出图必须走云 API；如需离线测试请显式传 backend='mock'。"
            )
        return AllInApiBackend(
            api_base=cfg_allin_base,
            api_key=api_key,
            model=cfg_allin_model,
            timeout_s=cfg_allin_timeout_s,
        )

    raise ValueError(f"未知 backend：{requested}")