# Video Localization Engine (VLE) — 架构设计

**目标**: 通用视频本地化模块。输入任意视频, 输出目标语言版本视频。
**当前 Phase**: A (骨架 + 核心数据契约)。

---

## 七层架构

```
┌─────────────────────────────────────────────────────────┐
│ L1 VideoAnalyzer         →  VideoMeta + FrameStream     │
├─────────────────────────────────────────────────────────┤
│ L2 RegionPolicies        →  RegionProposal (per frame)  │
├─────────────────────────────────────────────────────────┤
│ L3 TextDetector          →  TextCandidate (per frame)   │
├─────────────────────────────────────────────────────────┤
│ L4 SubtitleManager       →  SubtitleInstance (跨帧)     │
│    ├─ Tracker  (IoU/centroid/text-similarity 跨帧)     │
│    ├─ Buffer   (new/active/finished lifecycle)         │
│    └─ Classifier (region+time+arrange+font+exclusion)  │
├─────────────────────────────────────────────────────────┤
│ L5 MaskGenerator         →  binary mask (per frame)     │
├─────────────────────────────────────────────────────────┤
│ L6 InpaintingEngine      →  inpainted frame (per frame) │
├─────────────────────────────────────────────────────────┤
│ L7 Localizer (后续)      →  translate+TTS+render+mix    │
└─────────────────────────────────────────────────────────┘
```

---

## 三个核心数据对象 (贯穿所有层)

| 对象 | 定义文件 | 关键属性 |
|---|---|---|
| `VideoMeta`, `FramePacket` | `types/video.py` | 输入流的不可变 + 单帧包 |
| `TextCandidate`, `RegionProposal`, `FrameTextCandidates` | `types/detection.py` | L2/L3 中间产物 |
| `SubtitleInstance`, `InstanceScore`, `SubtitleTrack` | `types/instance.py` | L4 输出, L1-L6 ↔ L7 数据协议 |

**`SubtitleInstance` 是跨帧稳定身份**, 是整个 VLE 设计的核心。
所有"是否字幕"、"什么时候开始/结束"、"几何在哪"都在这个对象上累积。

---

## .vle.json 中间产物

`SubtitleTrack` 序列化结果, 是 L1-L6 与 L7 的解耦点:

- OCR 结果缓存 (Phase C 跑一次, L7 多次复用)
- 多目标语言重生成 (不改 L1-L6, 只换 L7)
- pipeline 断点恢复
- 跨视频比较 / 调试

```json
{
  "version": "0.1.0",
  "video": {
    "source_path": "input.mp4",
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "frame_count": 4500,
    "duration_ms": 150000,
    "orientation": "landscape",
    "has_audio": true,
    "source_locale": "zh"
  },
  "instances": [
    {
      "instance_id": "uuid",
      "status": "active",
      "first_frame": 150,
      "last_frame": 187,
      "frame_count": 38,
      "duration_ms": 1234,
      "representative_text": "你好世界",
      "representative_bbox": [100, 950, 1820, 1010],
      "score": {
        "region_score": 0.95,
        "persistence_score": 1.0,
        "arrangement_score": 0.9,
        "font_score": 0.85,
        "ui_exclusion_score": 0.05
      },
      "detector_id": "paddleocr_ch",
      "locale": "zh"
    }
  ],
  "frame_candidates": [ ... ],
  "region_policies_used": ["bottom_horizontal_landscape"],
  "detector_id": "paddleocr_ch"
}
```

---

## 协议接口 (后续 Phase 填充)

- `L1.VideoAnalyzer`: `meta() -> VideoMeta`, `frames() -> Iterable[FramePacket]`
- `L2.SubtitleRegionPolicy`: `is_applicable(meta) -> bool`, `propose(packet) -> List[RegionProposal]`
- `L3.TextDetectorBackend`: `detect(packet) -> List[TextCandidate]`
- `L4.InstanceTracker`: `update(candidates, frame) -> List[InstanceUpdate]`
- `L4.SubtitleClassifier`: `classify(instance) -> InstanceScore`
- `L5.MaskBackend`: `build(instances, packet) -> np.ndarray`
- `L6.InpaintingBackend`: `inpaint(packet, mask) -> np.ndarray`

每个 backend 通过对应 Registry 注册, 不在调用方硬编码。

---

## 设计原则

1. **不假设字幕位置**: 位置信息来自 L2 policy, 不在代码里写 `if y > 80%`
2. **不假设分辨率**: 全部像素坐标流通, backend 自适配
3. **不假设语种**: detector_id, locale 都是字符串, 不在中文 hard-code
4. **不假设算法**: 每个 backend 一个 Registry, 用户可注册第三方实现
5. **不假设时间**: 字幕 instance 的 first_frame/last_frame 由 tracker 自然产生
6. **每个模块有独立测试**: 不依赖具体测试视频, 用合成数据