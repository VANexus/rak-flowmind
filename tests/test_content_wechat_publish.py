"""content_wechat_publish 技能测试：通过 invoke() 走信封层。

覆盖：无凭证降级 / API 失败降级 / 成功路径（mock）/ 仅草稿模式。
HTTP 依赖类：成功 ok=True，降级 ok=True + degraded=True。
"""
from __future__ import annotations

from unittest.mock import patch

import flowmind.skills.content_wechat_publish as mod
from flowmind.skill import invoke
from flowmind.skills._wechat_client import WechatAPIError


def test_publish_degrades_without_credentials():
    """未设置 WECHAT_APP_ID / WECHAT_APP_SECRET → degraded。"""
    with patch.object(mod, "get_api_key", lambda _env: None):
        r = invoke("content_wechat_publish", {
            "title": "测试文章",
            "content": "<p>正文</p>",
            "thumb_image_url": "https://img.example.com/cover.jpg",
        })
    assert r.ok is True  # HTTP 依赖类
    d = r.data
    assert d.status == "failed"
    assert d.failure_category == "environment"
    assert d.retriable is False
    assert r.metrics.degraded is True


def test_publish_degrades_on_wechat_api_error():
    """微信 API 返回 errcode → degraded。"""
    err = WechatAPIError("invalid credential", category="environment", retriable=False, errcode=40001)

    with patch.object(mod, "get_api_key", lambda _env: "test-app-id" if "ID" in _env else "test-secret"):
        with patch.object(mod, "get_access_token", side_effect=err):
            r = invoke("content_wechat_publish", {
                "title": "测试文章",
                "content": "<p>正文</p>",
                "thumb_image_url": "https://img.example.com/cover.jpg",
            })
    assert r.ok is True
    d = r.data
    assert d.status == "failed"
    assert d.failure_category == "environment"
    assert r.metrics.degraded is True


def test_publish_success_full_flow():
    """完整流程：token → 上传封面 → 草稿 → 发布。"""
    with patch.object(mod, "get_api_key", lambda _env: "test-app-id" if "ID" in _env else "test-secret"):
        with patch.object(mod, "get_access_token", return_value="mock-token"):
            with patch.object(mod, "upload_thumb_image", return_value="thumb-media-123"):
                with patch.object(mod, "add_draft", return_value="draft-media-456"):
                    with patch.object(mod, "free_publish", return_value="pub-789") as mock_pub:
                        r = invoke("content_wechat_publish", {
                            "title": "测试文章",
                            "content": "<p>正文内容</p>",
                            "thumb_image_url": "https://img.example.com/cover.jpg",
                            "publish": True,
                        })
    assert r.ok is True
    d = r.data
    assert d.status == "published"
    assert d.media_id == "draft-media-456"
    assert d.publish_id == "pub-789"
    assert "get_access_token" in d.steps_completed
    assert "upload_thumb" in d.steps_completed
    assert "add_draft" in d.steps_completed
    assert "free_publish" in d.steps_completed
    assert r.metrics.degraded is False
    mock_pub.assert_called_once()


def test_publish_draft_only_mode():
    """publish=False → 只存草稿，不调发布接口。"""
    with patch.object(mod, "get_api_key", lambda _env: "test-app-id" if "ID" in _env else "test-secret"):
        with patch.object(mod, "get_access_token", return_value="mock-token"):
            with patch.object(mod, "upload_thumb_image", return_value="thumb-media-123"):
                with patch.object(mod, "add_draft", return_value="draft-media-456"):
                    with patch.object(mod, "free_publish") as mock_pub:
                        r = invoke("content_wechat_publish", {
                            "title": "测试文章",
                            "content": "<p>正文</p>",
                            "thumb_image_url": "https://img.example.com/cover.jpg",
                            "publish": False,
                        })
    assert r.ok is True
    d = r.data
    assert d.status == "drafted"
    assert d.media_id == "draft-media-456"
    assert d.publish_id is None
    assert "free_publish" not in d.steps_completed
    mock_pub.assert_not_called()


def test_publish_invalid_title_rejected():
    """标题超长（>64 字节）→ VALIDATION。"""
    r = invoke("content_wechat_publish", {
        "title": "标" * 100,
        "content": "<p>正文</p>",
        "thumb_image_url": "https://img.example.com/cover.jpg",
    })
    assert r.ok is False
    assert r.error.code == "VALIDATION"
