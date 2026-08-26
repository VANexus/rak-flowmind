"""_content_rules 测试：确定性审计引擎。

覆盖：
- 通用绝对化/医疗功效/数据规则跨平台命中
- 平台专属规则（小红书导流 / 公众号诱导分享 / 抖音金融）
- 平台隔离：某平台规则不误伤其他平台
- severity 排序（error 在前）
- matched_text / rule_id 回填
"""
from __future__ import annotations

from flowmind.skills._content_rules import audit_rules


def test_absolute_word_hit_all_platforms():
    for p in ("xhs", "wechat", "douyin"):
        fs = audit_rules(p, "全网最低价的保温杯", "正文", [])
        assert any(f.category == "absolute" and f.severity == "error" for f in fs), p


def test_medical_claim_hit():
    fs = audit_rules("xhs", "这个水杯能降血压", "正文", [])
    assert any(f.category == "medical" and "降血压" in (f.matched_text or "") for f in fs)


def test_xhs_wechat_guidance_blocked():
    fs = audit_rules("xhs", "需要的加微信领优惠", "正文", [])
    assert any(f.category == "platform" and f.severity == "error" and f.rule_id == "R-XHS-01" for f in fs)


def test_wechat_induce_share_hit():
    fs = audit_rules("wechat", "转发到朋友圈即可领取", "正文", [])
    assert any(f.rule_id == "R-WX-01" for f in fs)


def test_douyin_finance_hit():
    fs = audit_rules("douyin", "这款产品稳赚不赔", "正文", [])
    assert any(f.rule_id == "R-DY-01" for f in fs)


def test_platform_rules_are_isolated():
    """公众号诱导分享规则不命中小红书；小红书导流规则不命中公众号。"""
    share = audit_rules("xhs", "转发到朋友圈即可领取", "正文", [])
    assert not any(f.rule_id == "R-WX-01" for f in share)

    guidance = audit_rules("wechat", "需要的加微信领优惠", "正文", [])
    assert not any(f.rule_id == "R-XHS-01" for f in guidance)


def test_clean_copy_has_no_findings():
    clean = audit_rules("xhs", "316 不锈钢内胆，一键开盖防漏设计", "夏天保冷、冬天保暖，450ml 容量。", ["通勤好物"])
    assert clean == []


def test_severity_sort_error_first():
    fs = audit_rules("xhs", "全网最低价，转发到朋友圈", "加微信领优惠", [])
    sevs = [f.severity for f in fs]
    assert sevs == sorted(sevs)  # error 在 warning 前


def test_tags_are_scanned():
    fs = audit_rules("wechat", "标题", "正文", ["#转发抽奖"])
    assert any(f.rule_id == "R-WX-01" for f in fs)
