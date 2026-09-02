# 飞书知识库 FAQ 检索技能 (`feishu_kb_search`)

> 飞书 Wiki / 知识库本地化检索：对用户提问做 4 类意图分类 +
> BM25 + TF-IDF 双路召回 + RRF 融合 + 类别加权重排，输出 Top K 命中
> 与四段式因果推理链。

## 适用场景

- 车企 / 客服 / 售前售后 FAQ 智能问答
- 企业知识库问答（任意 4 类意图可配置）
- 任何"问 → 找最相关 N 条 → 给上层 Agent 整合"的场景

## 依赖

- `jieba`（中文分词）
- `rank-bm25`（BM25 排序）
- `scikit-learn`（TF-IDF 向量）
- `numpy`（向量操作）

均已在 `pyproject.toml` 的 `dependencies` 中声明。

## 4 大营销意图

| 营销意图 | 关键词示例 |
|---|---|
| **产品咨询** | 车型 / 配置 / 智驾 / 自动驾驶 / NGP / NOA |
| **故障排查** | 故障 / 报错 / 故障码 / 异响 / 跳枪 / 充不进去电 |
| **充电补能** | 充电 / 快充 / 慢充 / 充电桩 / 续航 / V2L / 外放电 |
| **用车指导** | 怎么用 / 保养 / 胎压 / 质保 / 救援 / 4S 店 |

## 直接调用（非 MCP）

```python
import flowmind.skills  # 触发 @skill 注册
from flowmind.skill import invoke

result = invoke("feishu_kb_search", {
    "query": "普电充的太慢了，车位充电桩装的普通线？",
    "top_k": 3,
})
print(result.ok)            # True
print(result.data.intent_category)  # "充电补能"
print(result.data.intent_confidence)
for hit in result.data.top_k:
    print(f"[{hit.rank}] {hit.faq_id} {hit.final_score:.3f} | {hit.question[:50]}")
print(result.reasoning[0].conclusion)
print(result.reasoning[0].risk_note)
print(result.metrics.latency_ms)  # 框架自动填
print(result.trace.trace_id)      # 框架自动生成
```

## 通过 MCP 调用

启动 MCP server 后即自动暴露为工具 `feishu_kb_search`：

```bash
uv run flowmind-mcp   # stdio 传输
```

调用方传入 `{"query": "用户问的话", "top_k": 3}` 即可。

## 数据准备

技能默认加载 `src/flowmind/skills/feishu_kb_seed.json`（8 条样本）。

生产部署：在用户 `flowmind.config.toml` 中指定 `data_path`：

```toml
[feishu_kb]
data_path = "/path/to/your/faqs.json"
retrieval_top_n = 20
```

JSON 数据格式：

```json
[
  {
    "id": "FAQ-0001",
    "category": "产品咨询",
    "question": "...",
    "answer": "...",
    "source_url": "feishu://kb/FAQ-0001"
  }
]
```

支持任意 category（4 大类只是默认），分类器对未识别类别兜底为「用车指导」。

## 输出结构

```python
SkillResult(
    ok=True,
    skill="feishu_kb_search",
    version="0.1.0",
    trace=TraceContext(trace_id="..."),
    data=FeishuKbReport(
        query="...",
        cleaned_query="...",
        intent_category="充电补能",
        intent_confidence=0.99,
        matched_keywords=["充电"],
        top_k=[FaqItem(rank=1, faq_id="FAQ-...", final_score=0.08, ...), ...],
        agent_reply_hint="用户问题：... 系统分类：...",
    ),
    reasoning=[
        ReasoningChain(
            conclusion="匹配到 3 个候选 FAQ...",
            triggered_rules=[RuleHit(rule_id="KB-INTENT", ...), ...],  # 自动
            evidence=[Evidence(metric="意图类别", ...), ...],          # 自动
            causal_analysis="用户问题归类为「充电补能」...通过 BM25 + TF-IDF 双路召回...",
            risk_note="若 Top 1 final_score < 0.02：建议转人工客服...",
        ),
    ],
    metrics=ReliabilityMetrics(latency_ms=12.3, confidence=0.99, sample_size=8),
)
```

## 四段式因果推理链

| 段 | 来源 |
|---|---|
| 1. `conclusion` | skill 函数组装（命中数 + 置信度） |
| 2. `triggered_rules` | `evaluate_rules()` 自动（KB-INTENT / KB-HAS-HITS / KB-HIGH-CONF） |
| 3. `evidence` | `evaluate_rules()` 自动（意图类别 / Top K 数 / Top 1 分数） |
| 4. `causal_analysis` + `risk_note` | skill 函数组装（解释召回流程 + 风险提示） |

## 阈值与降级

- Top 1 `final_score < 0.02` → `risk_note` 提示转人工
- 数据文件不存在 → `degraded=True` + `degradation_reason="FAQ 数据未配置..."`，`ok=True`（结构化降级，不是崩溃）

## 性能

| 指标 | 值 |
|---|---|
| 8 条种子数据检索延迟 | < 50ms |
| 163 条全量数据检索延迟 | < 200ms |
| 内存占用（8 条） | < 10MB |
| 启动开销 | 无（无模型加载） |

## 验证

通过 `examples/` 冒烟 + 直接 `invoke()` 覆盖：

- 4 大类各覆盖
- 空 query 校验失败
- top_k 参数尊重
- 命中含溯源
- 四段式链 4 要素齐全
- 可靠性指标完整
- trace_id 自动生成
