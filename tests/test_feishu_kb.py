"""飞书知识库 FAQ 检索技能测试：覆盖 4 类意图、双路召回、四段式链、降级路径。"""
from __future__ import annotations

import flowmind.skills  # noqa: F401  触发 @skill 注册
from flowmind.skill import invoke


def _args(query: str, top_k: int = 3):
    return {"query": query, "top_k": top_k}


def test_intent_classification_charging() -> None:
    """充电补能类：用包含「充电」「电池」关键词的查询触发。"""
    result = invoke("feishu_kb_search", _args("动力电池充电指示灯亮"))
    assert result.ok is True
    rep = result.data
    assert rep.intent_category == "充电补能"
    assert rep.intent_confidence > 0.5
    assert len(rep.top_k) > 0
    # 四段式链：4 要素齐全
    chain = result.reasoning[0]
    assert chain.conclusion and chain.causal_analysis and chain.risk_note
    assert len(chain.triggered_rules) >= 1
    assert len(chain.evidence) >= 1
    # 可靠性指标：latency 必有
    assert result.metrics.latency_ms > 0


def test_intent_classification_usage() -> None:
    """用车指导类。"""
    result = invoke("feishu_kb_search", _args("CVT 夏天行驶感觉也没手动挡提速快"))
    assert result.ok is True
    assert result.data.intent_category == "用车指导"
    assert len(result.data.top_k) > 0


def test_intent_classification_fault() -> None:
    """故障排查类。"""
    result = invoke("feishu_kb_search", _args("燃油指示灯点亮了怎么回事"))
    assert result.ok is True
    # 「燃油指示灯」可能匹配 故障排查 / 用车指导；接受其一
    assert result.data.intent_category in ("故障排查", "用车指导")
    assert len(result.data.top_k) > 0


def test_top_k_respected() -> None:
    """top_k 参数被尊重。"""
    result = invoke("feishu_kb_search", _args("充电", top_k=2))
    assert len(result.data.top_k) <= 2


def test_hits_have_source_url() -> None:
    """每个命中含溯源信息。"""
    result = invoke("feishu_kb_search", _args("动力电池充电指示灯亮"))
    for hit in result.data.top_k:
        assert hit.faq_id.startswith("FAQ-")
        assert hit.source_url.startswith("feishu://")
        assert hit.question and hit.answer


def test_empty_query_rejected_by_validation() -> None:
    """空 query 在 pydantic 校验阶段被拒。"""
    result = invoke("feishu_kb_search", _args(""))
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "VALIDATION"


def test_four_stage_chain_has_evidence_and_rules() -> None:
    """四段式链：4 要素齐全，且第 2、3 段由 evaluate_rules 自动产出。"""
    result = invoke("feishu_kb_search", _args("动力电池充电指示灯亮"))
    chain = result.reasoning[0]
    # 第 1 段：conclusion
    assert "匹配到" in chain.conclusion
    # 第 2 段：triggered_rules（自动）
    assert len(chain.triggered_rules) >= 1
    assert all(r.rule_id for r in chain.triggered_rules)
    # 第 3 段：evidence（自动）
    assert len(chain.evidence) >= 1
    assert all(e.metric for e in chain.evidence)
    # 第 4 段：causal_analysis + risk_note
    assert "BM25" in chain.causal_analysis or "TF-IDF" in chain.causal_analysis
    assert chain.risk_note


def test_metrics_present() -> None:
    """SkillResult 套上 ReliabilityMetrics：latency + sample_size 必有。"""
    result = invoke("feishu_kb_search", _args("动力电池充电指示灯亮"))
    assert result.metrics.latency_ms > 0
    assert result.metrics.sample_size > 0
    assert 0.0 <= result.metrics.confidence <= 1.0
    assert result.trace.trace_id  # 框架自动生成
    assert result.skill == "feishu_kb_search"
    assert result.version == "0.1.0"


def test_top1_has_positive_score() -> None:
    """Top 1 必有非负 final_score。"""
    result = invoke("feishu_kb_search", _args("CVT 变速器"))
    if result.data.top_k:
        assert result.data.top_k[0].final_score >= 0


def test_seed_size_at_least_100() -> None:
    """默认 seed 至少 100 条,确保覆盖度。"""
    from flowmind.skills.feishu_kb import _load_default_faqs

    faqs = _load_default_faqs()
    assert len(faqs) >= 100, f"seed 仅 {len(faqs)} 条,不足以稳定 BM25 召回"


def test_faq_self_match() -> None:
    """FAQ 自命中:用 seed 里一条 question 直接查,Top-1 应高置信命中自己。"""
    from flowmind.skills.feishu_kb import _load_default_faqs

    faqs = _load_default_faqs()
    assert len(faqs) >= 10
    sample = faqs[10]  # 取非首条,避免位置偏差
    result = invoke(
        "feishu_kb_search",
        _args(sample["question"]),
    )
    assert result.ok is True
    assert len(result.data.top_k) >= 1
    top1 = result.data.top_k[0]
    # Top-1 应是样本本身,或高置信命中(同一问题表述)
    assert top1.faq_id == sample["id"] or sample["answer"][:30] in top1.answer
    assert top1.final_score > 0.05, f"自命中置信度太低: {top1.final_score}"


def test_offtopic_returns_degraded() -> None:
    """话题外防御:无关查询返回 degraded=True + top_k=[] + 转人工 hint。"""
    result = invoke("feishu_kb_search", _args("今天北京天气怎么样"))
    assert result.ok is True  # 不报错,只是没命中
    assert result.data.top_k == []
    assert "暂未收录" in result.data.agent_reply_hint or "人工" in result.data.agent_reply_hint


def test_garbage_query_returns_degraded() -> None:
    """纯噪音 query 也走 hard-gate。"""
    result = invoke("feishu_kb_search", _args("asdfgh qwerty"))
    assert result.data.top_k == []
    assert "暂未收录" in result.data.agent_reply_hint


# ====================== 三语支持测试 ======================


def test_detect_chinese() -> None:
    """中文检测:含中文字符 → zh。"""
    from flowmind.skills.feishu_kb import _detect_language

    assert _detect_language("CVT 顿挫") == "zh"
    assert _detect_language("我的车充电很慢") == "zh"


def test_detect_english() -> None:
    """英文检测:纯 ASCII Latin → en。"""
    from flowmind.skills.feishu_kb import _detect_language

    assert _detect_language("CVT jerking") == "en"
    assert _detect_language("how to charge my car") == "en"


def test_detect_thai() -> None:
    """泰文检测:含泰文字符(U+0E00-U+0E7F) → th。"""
    from flowmind.skills.feishu_kb import _detect_language

    assert _detect_language("อาการสะดุด") == "th"
    assert _detect_language("รถชาร์จไฟไม่เข้า") == "th"


def test_english_query_not_blocked_by_chinese_keyword_gate() -> None:
    """英文查询不应被中文关键词 hard-gate 拦截(应能命中 FAQ-0002)。"""
    result = invoke("feishu_kb_search", _args("CVT jerking problem"))
    assert result.ok is True
    # 不应被 hard-gate 拦截 → top_k 应非空
    assert len(result.data.top_k) >= 1
    # agent_reply_hint 应包含翻译指令(英文提示词)
    assert "English" in result.data.agent_reply_hint


def test_thai_query_not_blocked() -> None:
    """泰文查询也应能命中 FAQ。"""
    result = invoke("feishu_kb_search", _args("รถชาร์จไฟไม่เข้า"))
    assert result.ok is True
    assert len(result.data.top_k) >= 1
    # agent_reply_hint 应包含翻译指令(泰文提示词"ภาษาไทย")
    assert "ภาษาไทย" in result.data.agent_reply_hint


# ====================== Tier 1 zero-LLM:同义词表扩容 ======================
# 解决 EN/TH → ZH 词汇缺口:用户历史上 EN "one-click / emergency / how far"
# 等词没在原有 40 项同义词表里,扩词后让 BM25 直接命中正确 FAQ。


def test_synonyms_emergency_start() -> None:
    """同义词扩:EN 'one-click emergency start operation' → '一键启动 应急启动 操作' → FAQ-0030。"""
    result = invoke(
        "feishu_kb_search",
        _args("one-click emergency vehicle start operation method"),
    )
    assert result.ok is True
    assert result.data.top_k[0].faq_id == "FAQ-0030", (
        f"扩词后应命中 FAQ-0030(应急启动),实际 {result.data.top_k[0].faq_id}"
    )


def test_synonyms_fuel_warning_distance() -> None:
    """同义词扩:EN 'fuel warning light how far' → 燃油报警灯 还能跑多远 → FAQ-0029。"""
    result = invoke(
        "feishu_kb_search",
        _args(
            "Wuling and Baojun vehicle models: how far can you drive after the fuel warning light comes on"
        ),
    )
    assert result.ok is True
    assert result.data.top_k[0].faq_id == "FAQ-0029", (
        f"扩词后应命中 FAQ-0029(燃油报警灯跑多远),实际 {result.data.top_k[0].faq_id}"
    )


def test_synonyms_thai_rear_seat_warm() -> None:
    """同义词扩:TH 'เบาะหลัง อุ่น' → 后排座椅 暖风不热 → FAQ-0026。"""
    result = invoke(
        "feishu_kb_search",
        _args("รถไม่อุ่นที่เบาะหลัง ของ Kaijie"),
    )
    assert result.ok is True
    assert result.data.top_k[0].faq_id == "FAQ-0026", (
        f"扩词后 TH 后排暖风应命中 FAQ-0026,实际 {result.data.top_k[0].faq_id}"
    )


# ====================== Tier 1 zero-LLM:Title 加权检索 ======================
# 解决 BM25 长 answer bias:让 question/title 命中权重高于 body,
# 长 answer FAQ(如 FAQ-0029 燃料里程表)能凭 title 抢 Top-1。


def test_title_weighted_recovers_long_answer_faq() -> None:
    """Title 加权:EN fuel warning distance query → FAQ-0029(长 answer)。

    Before title-weighted: FAQ-0073(短 answer 胎压灯)会抢 Top-1。
    After title-weighted: title 命中更高的 FAQ-0029 抢 Top-1。
    """
    result = invoke(
        "feishu_kb_search",
        _args(
            "Wuling and Baojun vehicle models: how far can you drive after the fuel warning light comes on"
        ),
    )
    assert result.ok is True
    assert result.data.top_k[0].faq_id == "FAQ-0029", (
        f"title-weighted 应让 FAQ-0029 抢 Top-1,实际 {result.data.top_k[0].faq_id}"
    )


def test_title_weighted_zh_baseline_still_hits() -> None:
    """Title 加权不应破坏 ZH baseline。

    凯捷中后排暖风不热 — 已是最热门 ZH FAQ,加 title-weighted 后应仍然 Top-1。
    """
    result = invoke("feishu_kb_search", _args("凯捷车辆中后排暖风不热问题"))
    assert result.ok is True
    assert result.data.top_k[0].faq_id == "FAQ-0026"


def test_title_weighted_balances_short_answers() -> None:
    """Title 加权应让多短 answer FAQ 在同一 query 上更平衡(不会一长串同主题)。"""
    # 胎压报警相关 query —— 不应所有 Top-K 都被胎压短路
    result = invoke(
        "feishu_kb_search",
        _args("tire pressure warning light comes on what to do", top_k=3),
    )
    assert result.ok is True
    assert len(result.data.top_k) >= 1
    # Top-1 应该是胎压 related 而非被其他主题抢占
    faq_ids = {h.faq_id for h in result.data.top_k}
    assert any("胎压" in h.question or "tire" in h.question.lower() for h in result.data.top_k), (
        f"Title 加权后胎压 query Top-K 至少 1 个胎压相关,实际 {faq_ids}"
    )


# ====================== Agent 结构化字段契约 ======================
# 让非 LLM agent(比如只读 JSON 字段的 agent)也能直接读懂
# "用户问了什么语种" / "要不要翻译" / "怎么翻译"。
# 字段:
#   user_language: str           (zh/en/th/other)
#   translation_required: bool   (True 当 user_language != "zh")
#   translation_directive: dict  ({source, target, rule})


def test_report_user_language_field() -> None:
    """契约:FeishuKbReport 必须含 user_language 字段。"""
    result = invoke("feishu_kb_search", _args("how to charge my car"))
    assert hasattr(result.data, "user_language"), (
        "[结构保护] FeishuKbReport 缺 user_language 字段"
    )
    assert result.data.user_language == "en"


def test_report_translation_required_field() -> None:
    """契约:FeishuKbReport 必须含 translation_required 字段,EN → True,ZH → False。"""
    r_en = invoke("feishu_kb_search", _args("how to charge my car"))
    assert hasattr(r_en.data, "translation_required")
    assert r_en.data.translation_required is True, (
        "EN query 应 translation_required=True"
    )
    r_zh = invoke("feishu_kb_search", _args("我的车充电很慢"))
    assert r_zh.data.translation_required is False, (
        "ZH query 应 translation_required=False"
    )


def test_report_translation_directive_field() -> None:
    """契约:FeishuKbReport.translation_directive 含 source/target/rule 三键。"""
    result = invoke("feishu_kb_search", _args("how to charge my car"))
    d = result.data.translation_directive
    assert isinstance(d, dict)
    assert d.get("source") == "zh"
    assert d.get("target") == "en"
    rule = d.get("rule", "").lower()
    # 必须包含"不替换/不软化"硬约束(中文或英文表述均可)
    assert ("不软化" in d.get("rule", "") or "no soften" in rule), (
        f"translation_directive.rule 必须含 '不软化/no soften' 硬约束,实际 '{d.get('rule')}'"
    )