"""content_xhs_draft 技能测试：通过 invoke() 走信封层。

覆盖：标签规范化 / 标题裁剪 / 正文裁剪 / 平台规范提醒 / JSON 导出 / 边界空输入。
"""
from __future__ import annotations

import json

from flowmind.skill import invoke


def test_draft_normalizes_tags():
    r = invoke("content_xhs_draft", {
        "title": "夏日通勤好物分享",
        "body": "这款保温杯真的超好用，450ml 容量刚好，316 不锈钢内胆耐腐蚀。",
        "tags": ["通勤", " 好物 ", "##生活", "通勤"],  # 重复 + 空格 + 双 #
        "image_urls": ["https://img.example.com/1.jpg"],
    })
    assert r.ok is True
    d = r.data
    assert d.tags == ["#通勤", "#好物", "#生活"]
    assert d.within_limits is True
    assert d.char_count == len(d.body)


def test_draft_truncates_title_over_20():
    r = invoke("content_xhs_draft", {
        "title": "这是一篇非常非常长的标题绝对超过二十个字符的限制",
        "body": "正文内容",
        "tags": ["标签1"],
        "image_urls": [],
    })
    assert r.ok is True
    d = r.data
    assert len(d.title) <= 20
    assert any("标题已裁剪" in w for w in d.warnings)


def test_draft_truncates_body_over_1000():
    long_body = "字" * 1500
    r = invoke("content_xhs_draft", {
        "title": "正常标题",
        "body": long_body,
        "tags": [],
        "image_urls": [],
    })
    assert r.ok is True
    d = r.data
    assert len(d.body) <= 1000
    assert any("正文已裁剪" in w for w in d.warnings)


def test_draft_truncates_tags_over_10():
    tags = [f"tag{i}" for i in range(15)]
    r = invoke("content_xhs_draft", {
        "title": "标题",
        "body": "正文",
        "tags": tags,
        "image_urls": [],
    })
    assert r.ok is True
    d = r.data
    assert len(d.tags) <= 10
    assert any("标签已裁剪" in w for w in d.warnings)


def test_draft_warns_on_short_title_and_no_images():
    r = invoke("content_xhs_draft", {
        "title": "好物",
        "body": "还不错",
        "tags": [],
        "image_urls": [],
    })
    assert r.ok is True
    d = r.data
    assert any("标题过短" in w for w in d.warnings)
    assert any("正文过短" in w for w in d.warnings)
    assert any("未添加配图" in w for w in d.warnings)
    assert any("未添加标签" in w for w in d.warnings)


def test_draft_exports_valid_json():
    r = invoke("content_xhs_draft", {
        "title": "测试标题",
        "body": "测试正文",
        "tags": ["测试"],
        "image_urls": ["https://img.example.com/a.jpg"],
        "topic": "测试品牌",
    })
    assert r.ok is True
    d = r.data
    payload = json.loads(d.content_json)
    assert payload["platform"] == "xhs"
    assert payload["title"] == d.title
    assert payload["topic"] == "测试品牌"
    assert payload["tags"] == ["#测试"]


def test_draft_truncates_images_over_18():
    urls = [f"https://img.example.com/{i}.jpg" for i in range(25)]
    r = invoke("content_xhs_draft", {
        "title": "标题",
        "body": "正文",
        "tags": [],
        "image_urls": urls,
    })
    assert r.ok is True
    d = r.data
    assert len(d.images) <= 18
    assert any("图片已裁剪" in w for w in d.warnings)


def test_draft_empty_title_rejected():
    r = invoke("content_xhs_draft", {"title": "", "body": "正文", "tags": [], "image_urls": []})
    assert r.ok is False
    assert r.error.code == "VALIDATION"
