"""content_publish_check 技能测试：通过 invoke() 走信封层。

覆盖：干净文案通过 / 标题超限 / 正文超限 / 违禁词 finding / 无效平台。
纯计算类：ok=True + can_publish 判定。
"""
from __future__ import annotations

from flowmind.skill import invoke


def test_check_passes_clean_content():
    r = invoke("content_publish_check", {
        "platform": "xhs",
        "title": "316 不锈钢保温杯",
        "body": "夏天保冷、冬天保暖，450ml 容量刚好通勤。",
        "tags": ["通勤好物"],
        "image_count": 3,
    })
    assert r.ok is True
    d = r.data
    assert d.platform == "xhs"
    assert d.can_publish is True
    assert d.limit_warnings == []


def test_check_detects_title_over_limit():
    # 构造 >20 字的标题
    r = invoke("content_publish_check", {
        "platform": "xhs",
        "title": "这是一篇非常长的标题超过二十个字限制啊啊啊",
        "body": "正文",
        "tags": [],
        "image_count": 0,
    })
    assert r.ok is True
    d = r.data
    assert d.title_length > 20
    assert d.can_publish is False
    assert any("标题" in w for w in d.limit_warnings)


def test_check_detects_body_over_limit():
    r = invoke("content_publish_check", {
        "platform": "xhs",
        "title": "标题",
        "body": "字" * 1200,
        "tags": [],
        "image_count": 0,
    })
    assert r.ok is True
    d = r.data
    assert d.can_publish is False
    assert any("正文" in w for w in d.limit_warnings)


def test_check_detects_image_over_limit():
    r = invoke("content_publish_check", {
        "platform": "xhs",
        "title": "标题",
        "body": "正文",
        "tags": [],
        "image_count": 25,  # xhs 限制 18
    })
    assert r.ok is True
    d = r.data
    assert d.can_publish is False
    assert any("图片" in w for w in d.limit_warnings)


def test_check_catches_absolute_word():
    # "全网最低价" 触发 absolute 规则
    r = invoke("content_publish_check", {
        "platform": "xhs",
        "title": "全网最低价保温杯",
        "body": "正文",
        "tags": [],
        "image_count": 0,
    })
    assert r.ok is True
    d = r.data
    assert d.can_publish is False
    assert any(f.category == "absolute" and f.severity == "error" for f in d.rule_findings)


def test_check_wechat_higher_title_limit():
    # 微信公众号标题限制 64 字
    r = invoke("content_publish_check", {
        "platform": "wechat",
        "title": "这是一个三十个字的公众号标题，测试长度限制是否比小红书宽松",
        "body": "正文",
        "tags": [],
        "image_count": 1,
    })
    assert r.ok is True
    d = r.data
    assert d.title_length == len("这是一个三十个字的公众号标题，测试长度限制是否比小红书宽松")
    assert d.can_publish is True  # 30 < 64


def test_check_invalid_platform():
    r = invoke("content_publish_check", {
        "platform": "pinterest",
        "title": "t",
        "body": "b",
        "tags": [],
        "image_count": 0,
    })
    assert r.ok is False
    assert r.error.code == "VALIDATION"
