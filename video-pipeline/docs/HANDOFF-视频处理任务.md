# 任务交接文档 V2：视频去字幕水印 + 英文本地化

> **状态：方案已与用户敲定，可直接开工**（V1 为初版调研，本版已并入全部决策）
> 交接原因：前一个 agent 的会话环境故障（shell / glob / grep / 联网 全部不可用），无法执行任何操作。
> 本文档包含任务目标、已定决策、环境故障清单、最终技术方案与执行计划，新 agent **从 §7 执行计划开始**，无需重新调研选型。

---

## 1. 任务目标

用户需求原文：

> 工作区改为 obj114514，看看 github 上有没有开源项目，能把文件夹的视频字幕和水印去除生成一个新的视频，同时再生成一个视频字幕为英文字幕旁白改为英文旁白的。

补充需求（后续沟通中确认）：

> 这个配音是克隆的原视频声音还是额外的音纹？去除字幕后的那一块地方感觉有点马赛克，有没有更高级的处理办法，哪怕算力要求高一点，也要视频清楚。

拆解为**两条流水线**，对 `obj114514` 文件夹里的每个视频各出一个成品：

| # | 流水线 | 输入 | 输出 |
|---|---|---|---|
| A | 去字幕 + 去水印（极致无痕） | 文件夹内原视频 | `output/clean/*_clean.mp4`：画面无烧录字幕、无水印/logo，音轨不变，**超分到 2K/4K** |
| B | 英文本地化（克隆原声） | 原视频（或 A 的干净版） | `output/en/*_en.mp4 + *.en.srt`：英文字幕 + **原说话人音纹的英文旁白**，原 BGM 保留 |

**验收标准**：
- A：抽帧检查字幕带/水印区域无残影、无马赛克/涂抹感、帧间无闪烁；超分后整体清晰度不低于原片。
- B：英文字幕与语音时间轴对齐；配音音色**听感接近原说话人**；BGM 保留且音量平衡；翻译准确。
- 批量处理整个文件夹，单个视频失败记日志不中断整批。

---

## 2. ✅ 已敲定的决策（用户已拍板，不要重新询问）

| 决策项 | 结论 | 说明 |
|---|---|---|
| **画质档位** | **极致无痕档** | 用户原话"哪怕算力要求高一点，也要视频清楚"。OCR 笔画级 mask → ProPainter → 泊松融合 → 全片超分 → 复杂区扩散重建 |
| **算力** | **RTX 4080 Laptop 12GB** | 可跑全链路；扩散环节需 fp16 + 512~768 分辨率 + 分段；laptop 需插电+散热 |
| **预算** | **全部本地优先，零 API 费用** | 不用云 TTS/云克隆；唯一遗留见 §8 开放项（翻译 LLM） |
| **配音音色** | **本地 RVC 两步法（克隆原说话人）** | 本地英文 TTS 出旁白 → RVC 音色转换用原声参考转回原人声；声音不出机器 |
| **处理顺序** | 先 A 后 B | 先出干净版再做英文版，避免原字幕残留干扰 B 的转写/烧录 |

已评估但**被否决**的备选（记录备查，勿再推荐）：
- 云 API 克隆（ElevenLabs / dashscope 声音复刻）：跨语言自然度最高，但需上传原声到第三方 + key + 费用 → 用户选了本地优先。
- 标准英文神经声线（edge-tts 固定声线）：与原说话人无关 → 用户要克隆原声。

---

## 3. ⚠️ 环境故障清单（接手后第一件事，先探测再干活）

前一会话以下工具全部故障，**不要假设已修复**：

| 探测项 | 上次故障表现 | 期望 |
|---|---|---|
| `pwsh` 执行命令 | `spawn C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe ENOENT` | 能执行基本命令 |
| `glob` / `grep` | `ripgrep provider failure`（连 `*.md` 都搜不了） | 能列目录、搜内容 |
| `web_search` | 缺 `DEEPSEEK_API_KEY`（Web GUI → credentials/Models 页可配置） | 能联网搜索 |
| `web_fetch` | 所有域名报 `resolves to a non-public IP`，出站被整体拦截 | 能访问 GitHub |

**需要用户在 GUI/宿主侧配合（agent 自己做不了）**：
1. **手动把工作区切换到 `obj114514`** —— 工作区是 DeepSeek Harness GUI 设置，agent 无切换工具；当前会话工作区是 `D:\Desktop\rak-flowmind-main`（无关仓库，别在里面跑处理）。
2. **重启 DeepSeek Harness 进程** —— `ENOENT` 与 ripgrep 故障是宿主进程级问题，重启才能恢复。
3. （可选）配置 `DEEPSEEK_API_KEY` —— 影响联网搜资料；**不影响本任务**（方案已定，但 clone 项目仓库仍需网络）。

**若 shell 恢复但网络仍不通**：无法 `git clone` / 下载模型权重 → 任务不可行，明确告知用户，不要假装执行。

---

## 4. 目标工作区盘点（拿到访问权后立即做）

- 路径：`D:\Desktop\obj114514`（**未验证，上次连列目录都失败**）
- 盘点清单：
  - [ ] 视频数量、扩展名（mp4/mov/mkv…）、总大小
  - [ ] `ffprobe` 批量导出：分辨率 / 帧率 / 时长 / 音轨
  - [ ] 字幕形态：烧录位置（底部固定条 or 随布）、单行/双行
  - [ ] 水印形态：固定角落 logo（可 delogo 预处理）还是移动水印（纯 AI 修复）
  - [ ] 语音：语言（假定中文）、单人/多人、是否有 BGM
  - [ ] `nvidia-smi` 确认 4080 12GB 可见、驱动/CUDA 正常
  - [ ] 已有工具：ffmpeg / python / conda 是否可用

---

## 5. 最终技术方案

### 流水线 A：去字幕 + 去水印（极致无痕档）

```
① OCR 笔画级 mask：RapidOCR/PaddleOCR 逐帧定位文字像素
   → 只圈文字本体 + 膨胀 2~3px（关键！mask 面积小 = 涂抹痕迹小，
   这一步对画质的贡献不亚于换大模型）
② ProPainter 视频修复：光流 + 前后帧传播，时间一致防闪烁；720p 分段防 OOM
③ 泊松融合（Poisson blending）：修复区与周边接缝消除
④ 全片超分：BasicVSR++ 或 Real-ESRGAN → 2K/4K
   （直接打击"马赛克感"——超分重建修复区纹理，痕迹大幅淡化）
⑤ 复杂区域扩散重建（区域分治，只花在刀刃上）：
   仅人脸/复杂纹理帧用 AnimateDiff + SD1.5 ControlNet inpaint 重绘背景；
   简单背景（墙/天空）②已足够，不全片跑扩散
```

备选/兜底项目（联网核验后选用）：**ProPainter**（首选）、**E2FGVI**（低配兜底）、**IOPaint**（静态水印）、社区 `video-subtitle-remover` 类一键工具（开箱即用但需实测）、FFmpeg `delogo`（固定角落 logo 预处理，非 AI）。

参考命令骨架：
```bash
# 静态水印预处理（w:h:x:y 按实际位置调）
ffmpeg -i in.mp4 -vf "delogo=x=10:y=10:w=200:h=80" -c:a copy out_delogo.mp4
# ProPainter：官方 README 出 mask → 修复
# python scripts/propainter.py --video in.mp4 --mask mask.mp4 --output out_clean.mp4
# 超分（示例）
# python inference_video_vsr.py -i out_clean.mp4 -o out_2k.mp4
```

### 流水线 B：英文字幕 + 英文旁白（本地 RVC 两步法）

```
① demucs 分离：人声 / BGM 分开（BGM 后面要保留）
② whisperX 转写：中文 + 精确时间轴 → zh.srt
③ LLM 翻译：zh.srt → en.srt（保时间轴逐句翻译，见 §8 开放项）
④ 本地英文 TTS：fish-speech / CosyVoice2 用 en.srt 出英文旁白（标准英文声）
⑤ RVC 音色转换：用①分离出的原说话人纯净人声做参考，
   把④的英文输出转成原说话人音纹 → "原人说英文"
⑥ ffmpeg 混流：英文配音 + 原 BGM（音量平衡 amix）
   + 烧录英文字幕（force_style 调字号字体），同时导出 *.en.srt 外挂
```

参考命令骨架：
```bash
demucs -n htdemucs in.mp4 -o demucs_out                 # 分离 BGM
whisperX in.mp4 --model small --language zh --output_format srt
# LLM 逐句翻译 → en.srt（保时间轴）
# 英文 TTS → voice_en_raw.wav
# RVC：infer-rvc-cli -i voice_en_raw.wav -f 0 -p 原声参考.pth -ap 原声索引 -o voice_en.wav
ffmpeg -i no_vocals.mp4 -i voice_en.wav \
  -filter_complex "[0:a][1:a]amix=inputs=2:duration=first:weights='1 1.2',subtitles=en.srt:force_style='FontName=Arial,FontSize=22'" \
  -c:v libx264 -c:a aac out_en.mp4
```

**RVC 素材准备**：从原视频截取说话人**纯净**段落 ≥30 秒（用①的分离人声，避开 BGM/音效），训练或即用 onnx 一键音色。

---

## 6. 备选方案存档（联网后核验仓库）

> 以下为前 agent 基于已有知识给出，**未经实时核验**，链接/仓库名以实际搜索为准：

| 用途 | 项目 |
|---|---|
| 视频修复 | ProPainter（`ProPainter video inpainting`）、E2FGVI |
| 图像/静态水印 | IOPaint（`Sanster/iopaint`） |
| 一键去字幕工具 | GitHub 搜 `video subtitle remover`（中文社区多个，质量参差） |
| 视频翻译整套 | VideoLingo（`Huanshere/VideoLingo`）、pyvideotrans（`jianchang512/pyvideotrans`）——**注意：用户已选本地自建管线，这两个仅作参考实现，不作主方案** |
| TTS / 克隆 | fish-speech、CosyVoice2、GPT-SoVITS（英文弱）、RVC |
| 超分 | BasicVSR++、RVRT、Real-ESRGAN |
| 转写 / 分离 | whisperX、demucs |

---

## 7. 执行计划（新 agent 从这里开始）

- [ ] **0. 环境探测**：按 §3 逐项验证；shell 不通 → 停下报告用户重启 harness；网络不通 → 报告用户，克隆仓库受阻
- [ ] **1. 确认工作区** `obj114514`，按 §4 盘点视频与 GPU
- [ ] **2. 联网核验** §6 项目（存在性、star、最近提交），选定并记录实际仓库地址
- [ ] **3. 装依赖**：ffmpeg（必需）、ProPainter/whisperX/demucs/TTS/RVC 对应环境（建议 conda 独立 env），**全程不引入付费云服务**
- [ ] **4. 单样例跑通 A**：挑 1 个短视频全链路（mask→修复→融合→超分），抽帧验收马赛克/闪烁
- [ ] **5. 单样例跑通 B**：同 1 个视频全链路（转写→翻译→TTS→RVC→混流），试听音色与时间轴
- [ ] **6. 批量执行**：整文件夹循环 A → B，失败单个记日志不中断
- [ ] **7. 交付**：
  ```
  obj114514/
  ├── 原视频（只读，绝不覆盖）
  └── output/
      ├── clean/    *_clean.mp4              （A 产物，超分后）
      └── en/       *_en.mp4 + *.en.srt      （B 产物）
  ```

**耗时预期**（4080 laptop 12GB）：A 标准链 = 数十分钟级/分钟视频；含扩散重建 = 小时级/分钟视频；B = 十几分钟级/分钟视频。laptop 插电 + 良好散热。

---

## 8. 开放项（仅剩 1 个，开工时向用户确认）

**翻译环节的本地 LLM**：用户选了"全部本地优先"，但英文脚本翻译需要 LLM：
- 先问用户：机器上有没有本地 LLM（ollama / qwen 等）？
- 有 → 全链路零外网零费用；
- 没有 → 先试本地小模型翻译质量，不达标时**这是唯一可能需要开一个翻译 API key 的环节**，届时单独向用户申请，不要擅自接云服务。

---

## 9. 关键注意事项

- **原片只读**：所有产物写入 `output/`，绝不覆盖源文件；单文件失败不中断批处理。
- **API key 永不写入文档/提交**：若 §8 获准用翻译 key，从环境变量或 `.env` 读。
- **当前会话工作区** `D:\Desktop\rak-flowmind-main` 是无关项目（视频本地化 MCP 服务端），仅本文档放在其中；处理任务全部在 `obj114514` 下进行。
- **版权与用途**：用户自有视频的本地化处理；能力仅用于合法用途，责任由用户自负。
- **不要重新选型**：画质档位/音色路线/预算均已敲定（§2），除非用户主动改主意，否则直接执行既定方案。
- **先沟通再动手**：若探测结果与本文档假设不符（如 GPU 不是 4080、文件夹为空、模型装不上），先向用户报告再调整，不要静默降级方案（用户明确要求"先不干活，先沟通"的精神）。

---

## 10. ✅ 执行结果（2026-10-17 已完成交付）

工作区仅 1 个视频（`ad738c4bcfdb9f0d41eeb5ede9a06a2e_raw.mp4`，720×1280/30fps/82.2s，左上「超级工厂」固定水印 + 底部烧录中文字幕），两条流水线均已按「极致无痕档」交付：

| 产物 | 规格 | 说明 |
|---|---|---|
| `output/clean/*_clean.mp4` | **2160×3840 (4K) HEVC** 82.2s | OCR 笔画级 mask → 全分辨率 ProPainter（25 块重叠缝合）→ Real-ESRGAN x4 → 降采样 4K |
| `output/clean/*_clean720.mp4` | 720×1280 | 修复后的 720p 母版 |
| `output/en/*_en.mp4` + `*.en.srt` | 2160×3840 HEVC | **GPT-SoVITS 克隆原片人声**说英文（62.9k star）+ demucs 保留 BGM + 英文字幕烧录 |

**与 §2/§5 既定方案的偏离（均经用户确认或实测决定）**：
1. **音色路线**：用户当场拍板保留 GPT-SoVITS 直接克隆成果（而非 TTS+RVC 两步法）——参考音频 = demucs 分离的原片人声（**限 3~10s** + 对应中文原文作 prompt_text）。
2. **泊松融合**：实测（对比图见 `work/poisson_test/`）ProPainter 直出接缝已不可见，逐帧泊松无增益且有闪烁风险 → 跳过。
3. **翻译环节**：未用 LLM——转写后由 agent 逐句人工校对翻译（修正 ASR 错字：义乌/蜡基单元/净化车间等），零 API 费用，§8 开放项就此关闭。
4. **复杂区扩散重建（AnimateDiff）**：未启用——全分辨率修复+超分后验收已达"无残影无马赛克"，扩散仅在不达标时按 §5 补充。

**关键工程经验（复用必读，详见 `obj114514/work/README-流水线使用说明.md`）**：
- ProPainter 官方脚本跑长视频必 OOM → 已改造为分块+重叠缝合流水线（`work/run_pp_chunked.py`，含 fp16/显存补丁），82s 视频全分辨率约 5.5h。
- 本机提交内存上限 48.8GB 是隐形杀手：全分辨率块进程峰值 ~23GB，跑批时**务必让用户关闭大内存应用**。
- Real-ESRGAN ncnn 默认选 Intel 核显，**必须 `-g 1` 指定 NVIDIA**（核显慢 40 倍）。
- GPT-SoVITS：`torchaudio.load` 依赖 torchcodec → 需要 shared 版 ffmpeg DLL（`work/ffmpeg_shared/`，注入 PATH）；torch 2.11.0+cu126 配套。
- 4K 烧字幕必须用 `hevc_nvenc`（libx264 4K 会慢到不可接受）；libass 字幕样式随分辨率自动等比缩放，无需改字号。
