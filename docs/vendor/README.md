# docs/vendor — 厂商接口文档与**错误码分类真源**

本目录存放 dsp 成片链路**火山系两家上游**（豆包语音 / AI MediaKit）的**入仓原文档**（PDF），
以及据此整理的**错误码 → 失败分类对照表**。代码里的错误分类（`_dsp_tts.py` / `_dsp_mediakit.py`）
以本文件为本地真源；改分类 = 先改这里，再同步两侧 `_*_CODES` 集合。

> 注：链路里还有**第三家**上游——**图生视频**厂商（晨星 `api.chenxingai.vip`，见
> `skills/_dsp_video_gen.py`），其接口文档**未入仓**；该家错误分类走 `errors.py` 的通用规则
> （未登记者兜底 `video`），不按本目录的表 A / 表 B 处理。

## 文档清单

| 文件 | 内容 | 代码引用处 |
|---|---|---|
| `豆包语音_音频生成_1790593987.pdf` | 音频生成接口（请求头/体全字段、响应结构） | `skills/_dsp_tts.py` |
| `豆包语音_错误码查询_1789027400.pdf` | 豆包语音错误码 | 本文表 A |
| `AI MediaKit_多轨道剪辑_1788492356.pdf` | 多轨道剪辑参数模型 | `skills/_dsp_mediakit.py` |
| `AI MediaKit_提交多轨道剪辑任务 API_1788490836.pdf` | 提交/查询任务 API | `skills/_dsp_mediakit.py` |
| `智能处理_获取媒体上传地址 API_1785483287.pdf` | 本地上传：申请上传位（PUT 原始字节）→ `mediakit://{file_id}` 引用 | `skills/_dsp_mediakit.py`（`request_upload_slot` / `upload_media`） |
| `AIMediaKit_公共错误码_1775248816.pdf` | 火山公共错误码（字符串名） | 本文表 B |
| `智能处理_视频添加运镜_1785416057.pdf` | 运镜（`apply-camera-motion`，只吃视频） | `skills/dsp_render_compose.py`（运镜缺口告警） |

## ⚠️ 抽文工具选择（踩过的坑）

这些 PDF 用 CID 字体。**`pdftotext -layout` 会丢中文**（只剩英文/数字与空白），
**`pymupdf`（`import pymupdf` / `fitz`）能正常抽出中文**。核表请一律用 pymupdf：

```python
import pymupdf
d = pymupdf.open("docs/vendor/豆包语音_错误码查询_1789027400.pdf")
print("\n".join(pg.get_text() for pg in d))
```

（`queue_id` 等字段释义也只能这样拿到——`pdftotext` 抽该字段所在页同样会丢中文。）

## 失败分类语义（真源 = `src/flowmind/errors.py`）

| 分类 | 含义 | Agent 该做什么 |
|---|---|---|
| `environment` | 网络 / 服务不通、鉴权 / 配置问题 | **别重试**，先查环境或凭据 |
| `video` | 资源 / **输入**问题（超限、审核不过、参数错） | 改输入，不重试 |
| `transient` | 服务端**临时**故障 / 限流 | **可重试**（退避后） |

**豆包总则**：8 位码 **5 开头 = 服务端报错**（transient）、**4 开头 = 客户操作错误**
（默认 video；音色/模型/方言一类改判 environment）。代码即按此码段规则兜底。
**MediaKit**：码是字符串名，**未登记者兜底 video**。

---

## 表 A · 豆包语音错误码

错误经**顶层 `code`(int)** 返回——故客户端**先解析响应体判码，拿不到才回落 HTTP 状态**
（`_dsp_tts.py::synthesize`）。下表为「音频生成」主接口（`/api/v3/tts/create`）相关的码。

### 显式登记为 `environment`

| code | 文档释义（原因 → 解决） |
|---|---|
| `45001115` | 参考音色不存在 → 检查音色 ID 是否正确、是否在可用范围 |

### 显式登记为 `transient`

| code | 文档释义 |
|---|---|
| `45002000` | 超并发（请求并发数超限） |
| `55000000` | 服务端通用错误 → 「请稍后重试」 |
| `55001301`/`55001302`/`55001303`/`55001304` | DB 查询/插入/更新/删除失败（服务异常，可重试） |
| `55001305`/`55001306` | TOS 上传/下载失败（对象存储不可用/超时） |
| `55001307` | 音色克隆失败（下游失败/超时） |
| `55001309` | 合成失败，内部错误 → 重试 |
| `55001310` | 合成失败，合成音频未通过安审 → 重试 |
| `55001311` | 合成失败，未通过声纹检查 → 重试 |

### 兜底 `video`（**未显式登记**，靠码段规则落 4 开头 → video）

| code | 文档释义 |
|---|---|
| `45001001` | 参数校验失败（参考音频/图片大小超限）→ 检查输入参数合法性 |
| `45001116` | prompt 文本长度超限（> 3000）→ 检查输入 |
| `45001117` | 参考音频时长超限（> 30s）→ 检查参考音频长度 |
| `45001125` | 文本审核失败 → 检查输入文本 |
| `45001104` | 声纹检测未通过（敏感声纹/黑名单）→ 更换音频 |
| `45001127` | 音频审核失败（参考音频合法性） |
| `45001130` | 图片审核失败（参考图片合法性） |
| `45001131` | **参考音频下载失败** → 检查参考音频地址 |
| `45001132` | **参考图片下载失败** → 检查参考图片地址 |
| `40000020` | `InvalidPayload`：时长超 2min / Prompt 被大模型安全过滤 / 图片大小不符（1k–10M） |
| `40402002` | `TTSInvalidText:ReadableTextRatioTooLow` —— prompt 可读文本比例不符 |

> `40000020` 与 `45001131`/`45001132` 文档「解决方案」列写了「重试」，但**原因列都是输入问题**
> （参考地址错、prompt 过长/被过滤），故仍归 `video`（改输入）。此为本仓的取舍，非文档原话。

### 声音复刻（voice clone）接口码 —— 本契约不经此路，未登记（落 video）

`45001101`（音频上传失败）`45001102`（ASR 转写失败）`45001105` `45001107`（SpeakerID 未找到）
`45001108` `45001109` `45001110` `45001112` `45001113` `45001114` `45001122` `45001123`
`45001124` `45001126` `45001128`；另有 WS 专属 `45000000`/`45000001`/`45002001`。

> 其中 `45001107`（SpeakerID 未找到）语义同 `45001115`，若日后走复刻链路应移入 environment。

### 客户端本地前置校验（调用前即拦，不发请求）

- `text_prompt` 上限 **3000 字符** → `DspConfig.tts_max_chars`（`build_text_prompt` 抛错）
- 单次合成上限 **120 秒** → `MAX_SINGLE_MS`
- `speech_rate` ∈ `[-50, 100]` → `synthesize` 抛错

---

## 表 B · 火山 AI MediaKit 公共错误码

code 是**字符串名**（非数字），同一个名可能跨多个 HTTP 状态返回——故客户端
**按码名分类**（比按 HTTP 状态稳），拿不到码名才回落状态
（`_dsp_mediakit.py::_classify_mk_error`）。

### 显式登记为 `transient`

| 码名 | 文档释义 | HTTP |
|---|---|---|
| `UndefinedError` | 未知错误 | 500 |
| `InternalError` / `InternalServiceError` | 内部错误 | 500 / 502 |
| `InternalServiceTimeout` | 内部服务超时 | 504 |
| `FailToConnect` | 服务连接错误 | 502 |
| `ServiceUnavailable` / `ServiceUnavailableTemp` | 服务不可用 / 暂时不可访问 | 503 |
| `ServiceTimeout` | 服务超时 | 408 |
| `ServerIsBusy` | 服务器忙 | 503 |
| `FlowLimitExceeded` / `RequestLimitExceeded` | 请求过于频繁 / 被限流 | 429 |
| `ResourceUnavailable` | 资源不可用（如欠费等**临时**不可用状态） | 403 |
| `ResourceInUse` | 资源被占用（临时冲突） | 409 |

### 显式登记为 `environment`

| 码名 | 文档释义 | HTTP |
|---|---|---|
| `MissingAuthenticationToken` | 缺少身份认证信息（如 Auth） | 401 |
| `MissingSignature` | 缺少签名结果 | 401 |
| `MissingRequestInfo` | 缺少请求必要信息（Accesskey/service/region 等） | 400 |
| `InvalidClientTokenId` | 请求的 AK 不合法 | 401 |
| `SignatureDoesNotMatch` | 签名结果不正确 | 401 |
| `InvalidTimestamp` | 请求过期 / 签名时间来自未来 | 403 |
| `LackPolicy` | 子用户请求缺少权限 | 403 |
| `LackProjectPolicy` | 缺少项目权限 | 401 |
| `AccessDenied` | 子用户权限不支持当前操作 | 403 |
| `Forbidden` / `RequestForbidden` | 请求被禁止 | 403 |
| `OperationDenied` | 操作被拒绝、**未开通** | 403 |
| `Unauthorized` / `UnauthorizedOperation` | 未授权 | 401 |
| `QuotaInsufficient` / `ResourceInsufficient` / `Insufficient` | 配额 / 资源不足 | 409 |

### 兜底 `video`（**未显式登记**）

参数 / 资源不存在 / 方法不允许一类：`InvalidParameter` / `InvalidDomain` /
`MissingParameter` / `UnsupportedProtocol` / `UnsupportedRegion` / `UnsupportedOperation` /
`InvalidAction` / `InvalidVersion` / `InvalidActionOrVersion` / `ServiceNotFound` /
`NotFound` / `ResourceNotFound` / `MethodNotAllowed`。

> `Success`（HTTP 200）是**成功码**，`_mk_error` 显式排除，不当错误处理。
> 文档里无独立 `InUse` 码（「InUse 类」是分节标题），实际码是 `ResourceInUse`。

---

## 维护约定

1. **改分类**：先改本表，再同步代码里的 `_TTS_ENVIRONMENT_CODES` /
   `_TTS_TRANSIENT_CODES` / `_TTS_ENV_MESSAGE_KEYWORDS`（`_dsp_tts.py`）与
   `_MK_TRANSIENT_CODES` / `_MK_ENVIRONMENT_CODES`（`_dsp_mediakit.py`）。
2. **上游换文档**：把新 PDF 放本目录、更新「文档清单」，并用 **pymupdf** 复核。
3. 分类错了**不静默**——`DspTtsError` / `DspMediakitError` 都带 `category` + `retriable`，
   由技能信封如实透出。
