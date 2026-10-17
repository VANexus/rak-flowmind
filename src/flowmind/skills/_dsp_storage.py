"""dsp 产物的**对象存储适配层**（S3/MinIO 兼容）。

契约 §2.4「目录 = 云存储（MinIO/S3）prefix」——中间产物落 ``cachePrefix``、
成片落 ``outputPrefix``；回传的 ``*Key`` 必须能被 cross-dashboard
``/api/assets/<key>`` 按**当前租户段**取到。故用 S3/MinIO（而非火山 TOS）：
cross-dashboard 的 ``files.ts`` 对 key 是 fail-closed 的（只认本租户段），
段名由调用方下发的 ``cachePrefix`` / ``outputPrefix`` 携带。

密钥纪律（AGENTS.md §3 同款）：端点 / 桶 / key 只经 ``DspConfig`` 的
**env 名**解析，密钥走 ``_secrets.get_api_key``；未配置即**显式报错**（不静默降级、
绝不返回假产物）。``boto3`` **懒加载**——不装它不影响包导入与其它技能。

可测性：本模块所有外部动作都是**模块级函数**，demo 通过替换
``_dsp_storage.put_bytes``（等）注入内存假实现，无需真对象存储。
"""
from __future__ import annotations

import ipaddress
from pathlib import Path
from urllib.parse import urlsplit

import requests

from flowmind.config import load_config
from flowmind.skills._secrets import get_api_key

# 常见产物扩展名 → Content-Type（对象存储写入用；未知回落二进制）
_CONTENT_TYPES: dict[str, str] = {
    ".json": "application/json",
    ".srt": "text/plain; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".mp4": "video/mp4",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class DspStorageError(Exception):
    """对象存储调用失败。category/retriable 语义与 errors.py 对齐。

    ``code`` 是**机器可读**的原因码（如 ``public_endpoint_unconfigured`` /
    ``public_endpoint_private`` / ``presign_failed``）——逐镜状态据此区分「前置条件不满足」
    与「厂商调用失败」，也避免调用方去解析中文错误文本。
    """

    def __init__(self, message: str, category: str = "unknown", retriable: bool = False,
                 code: str = ""):
        super().__init__(message)
        self.category = category
        self.retriable = retriable
        self.code = code


def content_type_for(key: str) -> str:
    """按扩展名猜 Content-Type（未知 = application/octet-stream）。"""
    return _CONTENT_TYPES.get(Path(key).suffix.lower(), "application/octet-stream")


def assert_key_shape(key: str) -> str:
    """对象 key 形状守卫：非空、无前导斜杠、无空段、无 ``..`` 回溯、无反斜杠。

    key 由本仓按 ``{prefix}/{runId}/…`` 拼出（§2.4），此处只做**形状**兜底；
    「是否本租户段」由 cross-dashboard 的 ``files.ts`` 侧 fail-closed 判定。
    """
    if not key or not key.strip():
        raise DspStorageError("对象 key 为空", category="video")
    if key.startswith("/"):
        raise DspStorageError(f"对象 key 不得以 / 开头：{key!r}", category="video")
    if "\\" in key or ".." in key.split("/"):
        raise DspStorageError(f"对象 key 含非法片段：{key!r}", category="video")
    if any(seg == "" for seg in key.split("/")):
        raise DspStorageError(f"对象 key 含空段：{key!r}", category="video")
    return key


def storage_configured() -> bool:
    """端点 + 桶 + 凭据是否齐备（调用方据此**如实降级**）。"""
    cfg = load_config().dsp
    if not cfg.storage_endpoint or not cfg.storage_bucket:
        return False
    return bool(get_api_key(cfg.storage_access_key_env)
                and get_api_key(cfg.storage_secret_key_env))


def _endpoint_url(endpoint: str, secure: bool) -> str:
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    return f"{'https' if secure else 'http'}://{endpoint}"


def _make_client(endpoint: str, secure: bool, addressing_style: str = ""):  # noqa: ANN202
    """按给定端点构造 S3/MinIO 客户端（懒加载 boto3；缺配置/缺依赖即显式报错）。

    **为什么端点必须是参数**：SigV4 把 ``Host`` 签进预签名 URL——给外部厂商的地址必须由
    **公网端点**自己签发；先按内部端点签、再把 hostname 换成公网域名，签名必然对不上（403）。
    故这里只接受**完整的目标端点**，调用方分成 `_client()`（内部）与 `_public_client()`（公网）。
    """
    cfg = load_config().dsp
    if not endpoint:
        raise DspStorageError(
            "未配置对象存储端点 FLOWMIND_DSP_S3_ENDPOINT（dsp 产物落云存储 prefix 必需）。",
            category="environment",
        )
    access = get_api_key(cfg.storage_access_key_env)
    secret = get_api_key(cfg.storage_secret_key_env)
    if not access or not secret:
        raise DspStorageError(
            f"未配置对象存储凭据 {cfg.storage_access_key_env} / {cfg.storage_secret_key_env}。",
            category="environment",
        )
    try:
        import boto3  # noqa: PLC0415  懒加载：不装 boto3 不影响包导入
        from botocore.config import Config as BotoConfig  # noqa: PLC0415
    except ImportError as exc:
        raise DspStorageError(
            "未安装 boto3（对象存储客户端）。environment.yml 已列 boto3，"
            "执行 `conda env update -n flowmind -f environment.yml` 后可用。",
            category="environment",
        ) from exc
    s3_opts = {"addressing_style": addressing_style} if addressing_style else None
    boto_cfg = BotoConfig(
        signature_version="s3v4",
        connect_timeout=cfg.storage_timeout_s,
        read_timeout=cfg.storage_timeout_s,
        retries={"max_attempts": 2},
        **({"s3": s3_opts} if s3_opts else {}),
    )
    return boto3.client(
        "s3",
        endpoint_url=_endpoint_url(endpoint, secure),
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name=cfg.storage_region or None,
        config=boto_cfg,
    )


def _client():  # noqa: ANN202  返回 boto3 S3 client（懒加载，类型不便标注）
    """**内部** S3/MinIO 客户端（本仓读写用；不用于给外部厂商签地址）。"""
    cfg = load_config().dsp
    return _make_client(cfg.storage_endpoint, cfg.storage_secure)


def _bucket() -> str:
    bucket = load_config().dsp.storage_bucket
    if not bucket:
        raise DspStorageError("未配置对象存储桶 FLOWMIND_DSP_S3_BUCKET", category="environment")
    return bucket


def put_bytes(key: str, data: bytes, content_type: str | None = None) -> str:
    """写入字节到对象存储，返回该 key。"""
    assert_key_shape(key)
    try:
        _client().put_object(
            Bucket=_bucket(), Key=key, Body=data,
            ContentType=content_type or content_type_for(key),
        )
    except DspStorageError:
        raise
    except Exception as exc:  # noqa: BLE001  boto 异常形态多变，统一分类
        raise DspStorageError(f"对象写入失败（{key}）：{exc}", category="transient",
                              retriable=True) from exc
    return key


def put_text(key: str, text: str, content_type: str | None = None) -> str:
    """写入 UTF-8 文本（EDL json / SRT），返回该 key。"""
    return put_bytes(key, text.encode("utf-8"), content_type=content_type)


def put_file(path: str, key: str, content_type: str | None = None) -> str:
    """把本地文件上传到对象存储（成片转存用），返回该 key。"""
    assert_key_shape(key)
    try:
        _client().upload_file(
            path, _bucket(), key,
            ExtraArgs={"ContentType": content_type or content_type_for(key)},
        )
    except DspStorageError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DspStorageError(f"对象上传失败（{key}）：{exc}", category="transient",
                              retriable=True) from exc
    return key


# ── 公网可达性（图生视频参考素材的前置条件，ADR-123）────────────────
# 厂商硬约束：参考素材必须是**公网可访问的 http(s) URL**（ADR-111）。本组函数只做**字面主机**
# 判定与**一次** HTTP 预读，不做 DNS 探测、也不承诺「厂商网络一定能读」——它解决的是
# 「本机/内网端点被当成公网地址签发」与「签名与公网 Host 不匹配」这两类**确定性**问题。

_INTERNAL_HOST_SUFFIXES = (".local", ".internal", ".svc", ".cluster.local", ".lan", ".home")


def public_url_host(url: str) -> tuple[bool, str]:
    """URL 主机是否**字面**公网；返回 ``(是否公网, 主机名)``。

    规则与前端 `dspEndpointPubliclyReachable()` 同口径：
    localhost / 回环 / 私网 IPv4 / 链路本地 / IPv6 私网 / 集群服务名 / 无点主机名一律非公网。
    **解析到内网的公网域名判不出**——交给厂商报错（不做 DNS 探测，避免把排查变成猜谜）。
    """
    try:
        host = (urlsplit(url).hostname or "").strip().lower()
    except ValueError:
        return False, ""
    if not host:
        return False, ""
    host = host.rstrip(".")  # FQDN 尾点
    if host in ("localhost", "::1", "0.0.0.0", "::") or host.endswith(".localhost"):
        return False, host
    if any(host.endswith(suffix) for suffix in _INTERNAL_HOST_SUFFIXES):
        return False, host
    # 无点的主机名（`minio` / `fm-minio`）是集群内服务名，不是公网域名
    if "." not in host and ":" not in host:
        return False, host
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True, host  # 有域名形态且不是 IP ⇒ 字面公网（可达性由预检/厂商负责）
    # is_global 同时排除 private / loopback / link-local / reserved / multicast /
    # unspecified / CGNAT（100.64/10）/ 文档网段等——比逐个 is_* 更不容易漏。
    if not ip.is_global:
        return False, host
    return True, host


def is_publicly_reachable_url(url: str) -> bool:
    """URL 主机是否**字面**公网（见 `public_url_host`）。"""
    return public_url_host(url)[0]


def public_presign_endpoint() -> str:
    """给外部厂商签发预签名 URL 的**公网端点**（空串 = 没有可用公网入口）。

    优先级：``FLOWMIND_DSP_S3_PUBLIC_ENDPOINT`` > 内部端点（**仅当它字面公网**）。
    **绝不**把本机 / 内网端点回落到「公网端点」——那会让技能假装具备公网访问能力。
    """
    cfg = load_config().dsp
    explicit = cfg.storage_public_endpoint.strip()
    if explicit:
        return _endpoint_url(explicit, cfg.storage_public_secure)
    internal = cfg.storage_endpoint.strip()
    if internal and is_publicly_reachable_url(_endpoint_url(internal, cfg.storage_secure)):
        return _endpoint_url(internal, cfg.storage_secure)
    return ""


def public_presign_configured() -> bool:
    """是否配了**可用**的公网预签名端点（空 / 私网 ⇒ False）。"""
    return bool(public_presign_endpoint())


def _public_client():  # noqa: ANN202  返回 boto3 S3 client（公网端点；懒加载）
    """构造**公网端点**的 S3 客户端（只用于签发对外预签名 URL）。"""
    cfg = load_config().dsp
    endpoint = public_presign_endpoint()
    if not endpoint:
        raise DspStorageError(
            "未配置公网对象存储端点 FLOWMIND_DSP_S3_PUBLIC_ENDPOINT（或内部端点不是公网地址）——"
            "图生视频厂商拉不到本机/内网素材。",
            category="environment", code="public_endpoint_unconfigured",
        )
    # 显式公网端点用 public_secure；沿用内部公网端点时用 internal secure（端点已是完整 URL 时二者都不参与）
    secure = cfg.storage_public_secure if cfg.storage_public_endpoint.strip() else cfg.storage_secure
    return _make_client(endpoint, secure, cfg.storage_public_addressing_style)


def presign_get_public(key: str, ttl_s: int | None = None) -> str:
    """为**外部厂商**签发公网可读的预签名 GET URL（图生视频参考素材）。

    **必须用公网端点自己签**：SigV4 把 Host 签进查询串，签完再替换 hostname 会 403。
    未配置公网端点 / 端点字面内网 ⇒ 抛 ``DspStorageError``（``code`` 说明原因），
    调用方据此**逐镜跳过**，不浪费厂商按镜计费的提交。

    有效期缺省仍是 ``DspConfig.presign_ttl_s``（契约 §7.3①：现签现用、不入库）。
    """
    assert_key_shape(key)
    cfg = load_config().dsp
    endpoint = public_presign_endpoint()
    if not endpoint:
        raise DspStorageError(
            "未配置公网对象存储端点 FLOWMIND_DSP_S3_PUBLIC_ENDPOINT（或内部端点不是公网地址）——"
            "图生视频厂商拉不到本机/内网素材。",
            category="environment", code="public_endpoint_unconfigured",
        )
    if not is_publicly_reachable_url(endpoint):
        raise DspStorageError(
            "公网对象存储端点字面指向本机/内网——厂商拉不到预签名素材。",
            category="environment", code="public_endpoint_private",
        )
    ttl = ttl_s or cfg.presign_ttl_s
    client = _public_client()
    try:
        return client.generate_presigned_url(
            "get_object", Params={"Bucket": _bucket(), "Key": key}, ExpiresIn=ttl)
    except DspStorageError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DspStorageError(
            f"公网预签名失败（{key}）：{exc}", category="transient",
            retriable=True, code="presign_failed") from exc
    finally:
        client.close()


def probe_public_read(url: str, timeout_s: float = 8.0) -> tuple[bool, str]:
    """**公网预检**：用预签名 URL 真读 1 字节，确认签名 / 反代 / 路径匹配。

    返回 ``(可读, 原因)``；原因**绝不包含 URL 查询串**（那里有临时访问凭证）。
    能提前拦住两类必然失败的付费提交：
      - 签名是一次性的、与 Host 绑定 ⇒ 反代改写 Host 会得到 403；
      - 公网入口没打通 ⇒ 连接失败 / 超时。

    ⚠️ 这是**后端此刻**的可读性，不等于厂商网络一定可达（DNS / 出口策略可不同）；
    故只作为「不浪费请求」的前置闸，单镜失败仍按 `skipped` 如实记录。
    """
    ok, host = public_url_host(url)
    if not ok:
        return False, f"主机 {host or '(空)'} 不是公网地址"
    try:
        resp = requests.get(
            url,
            headers={"Range": "bytes=0-0"},
            stream=True,
            timeout=timeout_s,
            allow_redirects=True,
        )
    except requests.exceptions.Timeout:
        return False, "预检请求超时（后端到公网端点不可达）"
    except requests.exceptions.RequestException as exc:
        # 不打印 URL / 查询串：预签名凭证只在有效期内暴露一次
        return False, f"预检请求失败（{type(exc).__name__}）"
    try:
        status = resp.status_code
        if status in (200, 206):
            return True, ""
        if status == 403:
            return False, "HTTP 403（签名与公网 Host 不匹配，或反向代理未透传 Host）"
        return False, f"HTTP {status}"
    finally:
        resp.close()


def presign_get(key: str, ttl_s: int | None = None) -> str:
    """为读取生成**内部端点**的预签名 URL（本仓自用；不用于给外部厂商）。

    现签现用、**不入库**（契约 §7.3①：对象 key 不可变语义不变）。
    ⚠️ 2026-10-17 起 ``render_compose`` **不再走这条**：本机 / 内网端点（如 ``127.0.0.1:9000``）
    厂商拉不到 ⇒ 改成「先上传 MediaKit，再用 ``mediakit://`` 引用」（见 ``get_bytes`` 与
    ``_dsp_mediakit.upload_media``）。
    ⚠️ 图生视频也**不再走这条**：参考素材必须用 ``presign_get_public`` 由**公网端点**签发
    （ADR-123；签完替换 hostname 会 403）。
    """
    assert_key_shape(key)
    ttl = ttl_s or load_config().dsp.presign_ttl_s
    try:
        return _client().generate_presigned_url(
            "get_object", Params={"Bucket": _bucket(), "Key": key}, ExpiresIn=ttl)
    except DspStorageError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DspStorageError(f"预签名失败（{key}）：{exc}", category="transient",
                              retriable=True) from exc


def get_bytes(key: str) -> bytes:
    """读出对象**原始字节**（渲染素材上传 MediaKit 用；契约 §7.3① 的本地上传支）。

    与 ``presign_get`` 的区别：不签发任何对外地址——素材由**本仓自己**送出去，
    故与对象存储的**公网可达性彻底解耦**（本机 dev 也能真渲染）。
    """
    assert_key_shape(key)
    try:
        obj = _client().get_object(Bucket=_bucket(), Key=key)
        return obj["Body"].read()
    except DspStorageError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DspStorageError(f"对象读取失败（{key}）：{exc}", category="transient",
                              retriable=True) from exc


def exists(key: str) -> bool:
    """对象是否已存在（``head_object``）。

    ADR-079 用它实现**按镜重生成**：`regenerateShotIds` 之外的镜头若已有确定性 key 的产物，
    直接**复用**（不重复计价、不重复出片）。读不通 / 未配置 ⇒ 返回 ``False``（调用方据此照常生成，
    不因探测失败而丢镜——**宁可多做一次，不可少出一镜**）。
    """
    if not key or not storage_configured():
        return False
    try:
        _client().head_object(Bucket=_bucket(), Key=key)
        return True
    except Exception:  # noqa: BLE001  不存在 / 无权限 / 网络抖动：一律按「没有」处理
        return False


def fetch_to_file(url: str, dest_path: str, timeout_s: float | None = None) -> str:
    """把 URL 内容流式落本地文件（MediaKit 成片转存第一步）。

    ``timeout_s`` 缺省 = ``DspConfig.storage_timeout_s`` 放大 20 倍（视频体积大）。
    """
    cfg = load_config().dsp
    timeout = timeout_s or max(cfg.storage_timeout_s * 20, 300.0)
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # **错误信息只带主机，不带 URL**：这条函数也会下载**公网预签名地址**（图生视频输入图），
    # URL 查询串里有临时凭证——不能随 warning / step note 落库或打印（ADR-123 ⑦）。
    host = urlsplit(url).hostname or "(未知主机)"
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
    except requests.exceptions.Timeout as exc:
        raise DspStorageError(f"下载超时（主机 {host}）", category="environment") from exc
    except requests.exceptions.RequestException as exc:
        raise DspStorageError(
            f"下载失败（主机 {host}，{type(exc).__name__}）", category="environment") from exc
    return str(dest)
