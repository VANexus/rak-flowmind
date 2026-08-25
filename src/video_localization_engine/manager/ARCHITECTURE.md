# L4 SubtitleManager — 4 层流水线设计

**目标**: 把 L3 OCR 输出的 `TextCandidate` 转成**跨帧稳定**的 `SubtitleInstance`。
**核心约束**: 不要把 OCR 输出直接当字幕; 必须经过时间和上下文判断。

## 4 层流水线

```
L3 detector 输出
      │
      ▼
[T1] TextLineCandidate   ← 单帧的 TextCandidate + 几何归一化 + 文本归一化
      │                  ← 不跨帧, 同一帧内可能合并多 polygon 为一行
      ▼
[T2] SubtitleCandidate   ← 跨帧聚合: 同一 TextLine 在多帧连续观测
      │                  ← 包含 duration, text_history, 出现频率
      │                  ← 不判定"是不是字幕"
      ▼
[T3] SubtitleInstance    ← 应用 region policy 评分 + 候选状态 (NEW/ACTIVE/ENDING/CLOSED)
      │                  ← 不强制最终判定, 保留所有 candidate, 由 orchestrator 决定
      ▼
output: List[SubtitleInstance] → .vle.json
```

## 与旧 Phase 1.x 的核心区别

| 旧 (Phase 1.x) | 新 (Phase C) |
|---|---|
| 单帧 filter + 硬阈值 | 4 层流水线, 每层独立可测试 |
| 一处 if y > 0.8 → subtitle | 不在 L4 写硬判定, feature 全部保存 |
| `score = position * 0.3 + ...` | `SubtitleInstance` 持有原始 feature dict, 评分交给 classifier |
| Phase 1.6 的固定权重评分 | Phase C 仅保存 feature, Phase D 再做评分 |

## 各层职责

### T1 TextLineCandidate
- **输入**: 单帧的 TextCandidate 列表 + FramePacket
- **输出**: TextLineCandidate 列表 (每个是"一行文字"的候选)
- **做的事**:
  - 同一 TextCandidate 视为一行 (PaddleOCR 默认是 line 级)
  - 可选: 同行多 polygon 合并 (multi-word → multi-char) — 留 extension
- **不做**: 跨帧追踪, 字幕判断

### T2 SubtitleCandidate
- **输入**: 当前帧 TextLineCandidate + 历史活跃 SubtitleCandidate
- **输出**: 更新后的活跃候选列表 + 新候选列表
- **做的事**:
  - IoU / centroid / text similarity 跨帧匹配
  - 累积: duration, frame_count, text_history, polygon_history
  - 没有匹配上的活跃候选 → 进入 ENDING
- **不做**: region 评分, 字幕判定

### T3 SubtitleInstance
- **输入**: SubtitleCandidate (过 ENDING 阈值后) + RegionProposals
- **输出**: SubtitleInstance (active/closed)
- **做的事**:
  - 把 SubtitleCandidate 的 raw features 收集到 SubtitleInstance
  - 不计算最终分数, 仅持有 features
  - 不强制判定 status; status 由 orchestrator 调用 classifier 决定
- **不做**: 评分, 分类 (Phase D/E 才做)

## 生命周期 (4 状态, 不硬编码判定阈值)

```
SubtitleCandidate 状态机:

  NEW          ─→  第一次出现
  ACTIVE       ─→  至少连续观测 N 帧 (N 可配置, 默认 1)
  ENDING       ─→  上一帧未匹配, 但还在 grace period
  CLOSED       ─→  grace period 过期, 提交给 classifier
```

状态转换由 `SubtitleCandidateBuffer` 管理。grace period 帧数可配置, 默认 2。

## 数据流示例

```
frame 10:
  TextCandidate: [poly @ (100,950), text="你好", conf=0.9]
  T1 → TextLineCandidate(poly, text="你好", conf=0.9, ...)
  T2 → 匹配不到任何 active, 创建 SubtitleCandidate(state=NEW, text_history=["你好"])

frame 11:
  TextCandidate: [poly @ (102,952), text="你好", conf=0.91]
  T2 → IoU=0.85, 匹配 frame 10 candidate, 更新: state=ACTIVE, frame_count=2, text_history=["你好"]

frame 12:
  (no text detected)
  T2 → 上一帧 candidate 进入 grace, state=ENDING

frame 13:
  TextCandidate: [poly @ (110,955), text="再见", conf=0.88]
  T2 → IoU=0.75 with ENDING candidate, 但 text 不同 — 创建 NEW candidate
       ENDING candidate 进入 grace2 → CLOSED, 提交给 T3

T3:
  SubtitleInstance(
    instance_id=uuid,
    state=CLOSED,
    text_history=["你好"],
    duration_ms=66,
    frame_count=2,
    polygon_history=[(100,950), (102,952)],
    features={...}  # 原始 feature, 无评分
  )
```

## .vle.json 新增字段

```json
{
  "instances": [
    {
      "instance_id": "uuid",
      "state": "closed",
      "first_frame": 10,
      "last_frame": 11,
      "duration_ms": 66,
      "frame_count": 2,
      "text_history": ["你好"],
      "representative_text": "你好",
      "polygon_history": [[[100, 950], ...], [[102, 952], ...]],
      "features": {
        "avg_confidence": 0.905,
        "text_stability": 1.0,
        "centroid_stability": 0.92,
        "iou_mean": 0.85,
        "char_count": 2,
        "bbox_history": [[100, 900, 300, 950], [102, 902, 302, 952]]
      }
    }
  ]
}
```

## Phase C 边界 — 不做的事

- 不写 region score (需要 classifier, Phase D)
- 不写 ui_exclusion score (需要 UI 特征, Phase D)
- 不决定 SubtitleInstance.status 最终值 (留给 orchestrator)
- 不接 MaskGenerator (Phase D)
- 不接 InpaintingEngine (Phase D)
- 不接 L7 (后续阶段)

## Phase C 测试目标

- TextLineCandidate 正确归一化
- SubtitleCandidate 跨帧正确累积
- grace period 正确转换 NEW→ACTIVE→ENDING→CLOSED
- .vle.json round-trip 完整保留 features
- 3 类合成视频上跑通完整流水线
- **不量化 FP/FN** (这是 Phase D 的事, Phase D 接 classifier 才有意义)
- **不输出最终 mask** (Phase D 边界)