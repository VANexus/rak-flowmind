# 视频去字幕/水印 + 英文本地化流水线（本地 GPU，零 API 费用）

一条可在自己机器上批量跑的视频处理流水线：

- **A 管线（全自动）**：OCR 笔画级 mask → 全分辨率 ProPainter 视频修复（分块+重叠缝合）→ Real-ESRGAN x4 超分
  → 产出无烧录字幕、无固定水印的干净视频（本机交付为 2K 1440×2560，可配置到 4K）。
- **B 管线**：demucs 分离 BGM → faster-whisper 中文转写 → 人工/LLM 翻译 → **GPT-SoVITS 克隆原片人声**说英文
  → 按原句时槽排布 + 混 BGM → 烧录英文字幕 → 产出英文本地化视频（含外挂 SRT）。

实测样例：720×1280 / 82s 竖屏宣传片，A+B 全链路在 RTX 4080 Laptop 12GB 上约 7 小时（A 占 5.5h）。

## 目录结构

```
video-pipeline/
├── README.md                        本说明
├── docs/HANDOFF-视频处理任务.md      设计文档（方案、验收标准、执行结果与踩坑）
├── scripts/                         流水线脚本（见下方用法）
├── patches/
│   ├── propainter.patch             ProPainter 必打补丁（git diff 格式）
│   ├── inference_propainter_patched.py  补丁后的完整文件（对照用）
│   └── PROPAINTER-PATCHES.md        补丁说明
├── fetch_models.ps1                 权重/工具一键下载（模型不进 git，见下）
└── .gitignore
```

## 为什么模型权重不在仓库里

GitHub 单文件 100MB 硬限制、仓库建议 <1GB；本流水线涉及的权重：
ProPainter 检查点 ~190MB、GPT-SoVITS 预训练 ~4.3GB、faster-whisper/demucs 模型 ~2GB。
所以仓库只含**代码 + 文档 + 补丁 + 一键下载脚本**（`fetch_models.ps1` + 各项目官方安装器），
所有权重来源均为官方/HuggingFace 镜像链接，可随时复现。

## 环境要求

- Windows + NVIDIA GPU（实测 12GB 显存可跑全链路；显存更小需把 `run_pp_chunked.py` 的 `CHUNK` 调低）
- ffmpeg（建议 full build，放 PATH）
- Python 3.10 conda 环境 ×2：
  - **主环境**：`torch`(CUDA)、`torchvision`、`opencv-python`、`numpy`、`scipy`、`imageio[ffmpeg]`、`av`、
    `einops`、`thop`、`matplotlib`、`requests`、`faster-whisper`、`edge-tts`、`demucs`、`rapidocr-onnxruntime`、`onnxruntime-gpu`
  - **GPT-SoVITS 环境**：见 `scripts/install_gptsovits.ps1`（torch 2.11.0+cu126 + 官方 requirements）

## 一次性准备

```powershell
# 1. 克隆 ProPainter 并打补丁（官方脚本跑长视频必 OOM，补丁必打）
git clone https://github.com/sczhou/ProPainter.git work\ProPainter
git -C work\ProPainter apply patches\propainter.patch

# 2. 下载权重 + Real-ESRGAN 工具
.\fetch_models.ps1          # ProPainter ckpts → work\ProPainter\weights\；ncnn → work\realesrgan\

# 3. GPT-SoVITS（克隆 TTS）
git clone https://github.com/RVC-Boss/GPT-SoVITS.git work\GPT-SoVITS
# 用管理员以外权限运行（会装 torch cu126 + 官方模型到 work\GPT-SoVITS）
powershell -File work\GPT-SoVITS\install_custom.ps1

# 4. shared 版 ffmpeg DLL（torchcodec 解码参考音频需要；本机原 ffmpeg 是静态版）
#    fetch_models.ps1 已下载到 work\ffmpeg_shared\，tts_clone.py 启动服务时自动注入 PATH
```

> 所有脚本顶部有绝对路径常量（`WORK`/`ROOT` 等，指向本机 `D:\Desktop\obj114514\work`），
> 换机器先按实际路径改一遍。

## A 管线（去字幕+去水印+超分，全自动）

```powershell
$py = '<主环境>\python.exe'
$vid = 'D:\path\to\video.mp4'

# 1. OCR 笔画级掩码（~18min/82s视频；换视频先抽帧确认脚本顶部 BAND_Y0/Y1 与 WM 常量）
& $py scripts\gen_masks_ocr.py
# 2. 分块全分辨率 ProPainter（~10min/块@120帧；中断重跑自动续；输出 output/clean/*_clean720.mp4）
& $py scripts\run_pp_chunked.py $vid
# 3. Real-ESRGAN x4 → 2K/4K（-g 1 必须，核显慢 40 倍）
& $py scripts\run_superres.py 'output\clean\<名字>_clean.mp4'
```

## B 管线（克隆原声英文配音+字幕）

```powershell
$py  = '<主环境>\python.exe'
$pyg = '<GPT-SoVITS环境>\python.exe'

# 1. demucs 分离 BGM/人声  2. whisper 转写  3. 人工写 en.json  4. 截 3~10s 人声参考+中文原文
# 见 docs/HANDOFF §10 与 scripts 内注释；随后：
& $pyg scripts\tts_clone.py              # 5. 克隆生成英文配音（服务 9880）
& $py  scripts\trim_silence.py work\en\tts_clone
& $py  scripts\build_en_audio.py work\en\tts_clone   # 6. 时槽排布+混BGM+英文SRT
& $py  scripts\finish_video.py $vid                  # 7. 烧字幕封装（≥1440px 自动 nvenc）
```

## 关键踩坑（详见 docs/HANDOFF §10）

1. **提交内存 48.8GB 上限**：全分辨率块进程峰值 ~23GB，跑批时关掉浏览器/直播/聊天等大内存应用；
2. **Real-ESRGAN ncnn 默认选 Intel 核显**，必须 `-g 1`；
3. **GPT-SoVITS**：参考音频限 3~10s 且必须附中文原文；torchaudio.load 依赖 torchcodec → 需 shared ffmpeg DLL；
4. **4K 烧字幕用 hevc_nvenc**（libx264 4K 慢到不可接受）；libass 样式随分辨率自动等比缩放；
5. ProPainter 官方脚本整段推理必 OOM → 本仓库补丁 + 分块脚本解决（fp16、短剪辑=2、原地融合等）。

## 版权与用途

本仓库代码用于处理**自有视频**的本地化（去第三方水印能力仅用于合法用途）；
引用的开源项目（ProPainter、GPT-SoVITS、demucs、faster-whisper、Real-ESRGAN、RapidOCR 等）
均需遵守各自许可证；克隆他人音色须获授权。
