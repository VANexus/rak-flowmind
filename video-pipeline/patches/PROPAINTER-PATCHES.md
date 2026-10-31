# ProPainter 补丁说明（`propainter.patch`，基于 sczhou/ProPainter HEAD e870e79 应用）

官方脚本直接跑长视频（>10s @720p）必 OOM，本流水线对 `inference_propainter.py` 做了 5 处修改：

| # | 位置 | 修改 | 原因 |
|---|---|---|---|
| 1 | `read_frame_from_videos()` | `torchvision.io.read_video` → OpenCV `cv2.VideoCapture` | torchvision ≥0.27 移除了 read_video |
| 2 | RAFT 短剪辑长度 | `short_clip_len` 强制为 2 | 相关体（corr volume）在 720p 下按原值(4~8)分配 6GB+，必 OOM |
| 3 | 光流存储 | RAFT 输出按半精度（`.half()`）存入列表 | fp32 光流双方向全片驻留 ~2.2GB → fp16 减半 |
| 4 | mask 上设备后 | `flow_masks / masks_dilated` 提前转 fp16 | 与 fp16 推理一致，省 ~1GB 显存 |
| 5 | 重叠帧融合循环 | `a*0.5+b*0.5` 的 float32 临时数组 → `cv2.addWeighted(dst=in-place)` | 全分辨率下每帧 2×10MB float 临时数组加剧内存压力 |
| 6 | 结尾输出 | 去掉 `masked_in.mp4` 调试视频（绿幕叠加预览） | 纯调试产物，省 6.8GB 内存与编码时间 |

配合 `scripts/run_pp_chunked.py`（分块 + 20 帧重叠缝合 + `--save_frames` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`）
即可在 12GB 显存上跑 720p 全分辨率长视频；实测 82s 视频 25 块约 5.5 小时。

应用方式：

```powershell
git clone https://github.com/sczhou/ProPainter.git work\ProPainter
git -C work\ProPainter apply patches\propainter.patch
```

`inference_propainter_patched.py` 是打补丁后的完整文件，仅供对照/比对版本，不需手动复制。
