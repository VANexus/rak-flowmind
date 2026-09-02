"""微信公众号端到端新增能力测试：排版 / 发布 v2（渠道/定时/正文图）/ 账号测试 / 状态查询。

HTTP 依赖类：成功 ok=True，降级 ok=True + degraded=True；纯本地渲染（typeset）走 raise。
"""
from __future__ import annotations

from unittest.mock import patch

import flowmind.skills.content_wechat_publish as publish_mod
from flowmind.skill import invoke
from flowmind.skills import _wechat_client as client_mod


# ───────────────────────── content_typeset ─────────────────────────

def test_typeset_renders_and_inlines():
    """Markdown → 内联样式 HTML，变量全部解析、主题样式生效。"""
    r = invoke("content_typeset", {
        "markdown": "# 标题\n\n**加粗**正文\n\n## 小节\n\n> 引用",
        "theme": "default",
    })
    assert r.ok is True
    d = r.data
    assert d.html.startswith("<section")
    assert "var(" not in d.html                       # CSS 变量全部解析
    assert "#07C160" in d.html                        # default 主题主色生效
    assert "border-bottom: 2px solid #07C160" in d.html  # h1 样式已内联
    assert d.stats["headings"] == 2
    assert d.stats["paragraphs"] >= 1
    # 元数据（label/css_file）不得混入内联 style（曾回归：section style 出现
    # "label: 经典;css_file: default.css;" 这类无效 CSS 属性）
    assert "label:" not in d.html
    assert "css_file:" not in d.html


def test_typeset_grace_theme():
    """grace 主题主色不同。"""
    r = invoke("content_typeset", {"markdown": "# 标题", "theme": "grace"})
    assert r.ok is True
    assert "#9C6ADE" in r.data.html


def test_typeset_inline_code_class():
    """行内代码带上 .codespan，代码块带上 .code__pre（对齐 doocs 主题约定）。"""
    md = "正文 `code`\n\n```python\nx=1\n```"
    r = invoke("content_typeset", {"markdown": md, "theme": "simple"})
    assert r.ok is True
    assert 'class="codespan"' in r.data.html
    assert 'class="code__pre"' in r.data.html


def test_typeset_unknown_theme_rejected():
    """未知主题 → INTERNAL（纯本地渲染 raise 由 invoke 兜底）。"""
    r = invoke("content_typeset", {"markdown": "# 标题", "theme": "nope"})
    assert r.ok is False
    assert r.error.code == "INTERNAL"


# ───────────────────────── content_wechat_publish v2 ─────────────────────────

def test_publish_mass_channel():
    """channel=mass → 走 mass_send，返回 msg_id 与 mass_sent。"""
    with patch.object(publish_mod, "get_api_key", lambda _env: "app" if "ID" in _env else "secret"):
        with patch.object(publish_mod, "get_access_token", return_value="t"):
            with patch.object(publish_mod, "upload_thumb_image", return_value="thumb"):
                with patch.object(publish_mod, "upload_content_images",
                                  side_effect=lambda **kw: (kw["content"], [])):
                    with patch.object(publish_mod, "add_draft", return_value="draft"):
                        with patch.object(publish_mod, "mass_send", return_value="msg-1") as m:
                            r = invoke("content_wechat_publish", {
                                "title": "群发测试", "content": "<p>x</p>",
                                "thumb_image_url": "https://e.com/c.jpg",
                                "channel": "mass", "publish": True,
                            })
    assert r.ok is True
    d = r.data
    assert d.status == "mass_sent"
    assert d.msg_id == "msg-1"
    assert d.publish_id is None
    assert "mass_send" in d.steps_completed
    m.assert_called_once()


def test_publish_scheduled_time_passed_to_freepublish():
    """publish_time 透传给 free_publish。"""
    with patch.object(publish_mod, "get_api_key", lambda _env: "app" if "ID" in _env else "secret"):
        with patch.object(publish_mod, "get_access_token", return_value="t"):
            with patch.object(publish_mod, "upload_thumb_image", return_value="thumb"):
                with patch.object(publish_mod, "upload_content_images",
                                  side_effect=lambda **kw: (kw["content"], [])):
                    with patch.object(publish_mod, "add_draft", return_value="draft"):
                        with patch.object(publish_mod, "free_publish", return_value="pub-1") as fp:
                            r = invoke("content_wechat_publish", {
                                "title": "定时测试", "content": "<p>x</p>",
                                "thumb_image_url": "https://e.com/c.jpg",
                                "channel": "publish", "publish": True,
                                "publish_time": 1770000000,
                            })
    assert r.ok is True
    assert r.data.status == "published"
    assert r.data.publish_time == 1770000000
    fp.assert_called_once()
    assert fp.call_args.kwargs.get("publish_time") == 1770000000


def test_publish_account_override_used():
    """显式 app_id/app_secret 优先于环境变量。"""
    captured = {}

    def fake_token(**kw):
        captured["app_id"] = kw["app_id"]
        return "t"

    with patch.object(publish_mod, "get_api_key", lambda _env: "env-app"):
        with patch.object(publish_mod, "get_access_token", side_effect=fake_token):
            with patch.object(publish_mod, "upload_thumb_image", return_value="thumb"):
                with patch.object(publish_mod, "upload_content_images",
                                  side_effect=lambda **kw: (kw["content"], [])):
                    with patch.object(publish_mod, "add_draft", return_value="draft"):
                        with patch.object(publish_mod, "free_publish", return_value="pub"):
                            r = invoke("content_wechat_publish", {
                                "title": "账号测试", "content": "<p>x</p>",
                                "thumb_image_url": "https://e.com/c.jpg",
                                "publish": False,
                                "app_id": "override-app", "app_secret": "override-secret",
                            })
    assert r.ok is True
    assert captured["app_id"] == "override-app"


def test_publish_no_credentials_env_and_override():
    """override 为空且环境无凭证 → degraded。"""
    with patch.object(publish_mod, "get_api_key", lambda _env: None):
        r = invoke("content_wechat_publish", {
            "title": "x", "content": "<p>x</p>", "thumb_image_url": "https://e.com/c.jpg",
        })
    assert r.ok is True
    assert r.data.status == "failed"
    assert r.data.failure_category == "environment"
    assert r.metrics.degraded is True


# ───────────────────────── content_wechat_account_test ─────────────────────────

def test_account_test_degrades_without_credentials():
    """无凭证 → degraded environment。"""
    import flowmind.skills.content_wechat_account_test as at_mod
    with patch.object(at_mod, "get_api_key", lambda _env: None):
        r = invoke("content_wechat_account_test", {})
    assert r.ok is True
    assert r.data.ok is False
    assert r.data.failure_category == "environment"
    assert r.metrics.degraded is True


def test_account_test_success_with_mock_token():
    """凭证有效 → ok=True。"""
    import flowmind.skills.content_wechat_account_test as at_mod
    with patch.object(at_mod, "get_api_key", lambda _env: "app" if "ID" in _env else "secret"):
        with patch.object(at_mod, "get_access_token", return_value="t"):
            r = invoke("content_wechat_account_test", {})
    assert r.ok is True
    assert r.data.ok is True
    assert r.metrics.degraded is False
    assert "****" in r.data.app_id_masked


# ───────────────────────── content_wechat_publish_status ─────────────────────────

def test_status_degrades_without_credentials():
    import flowmind.skills.content_wechat_publish_status as st_mod
    with patch.object(st_mod, "get_api_key", lambda _env: None):
        r = invoke("content_wechat_publish_status", {"publish_id": "p1"})
    assert r.ok is True
    assert r.metrics.degraded is True
    assert r.data.failure_category == "environment"


def test_status_publish_success():
    import flowmind.skills.content_wechat_publish_status as st_mod
    with patch.object(st_mod, "get_api_key", lambda _env: "app" if "ID" in _env else "secret"):
        with patch.object(st_mod, "get_access_token", return_value="t"):
            with patch.object(st_mod, "get_publish_status",
                              return_value={"publish_status": 0, "article_detail": {"item": [{"article_url": "https://mp.weixin.qq.com/s/abc"}]}}):
                r = invoke("content_wechat_publish_status", {"publish_id": "p1"})
    assert r.ok is True
    assert r.data.kind == "publish"
    assert r.data.status_text == "发布成功"
    assert r.data.article_url == "https://mp.weixin.qq.com/s/abc"


# ───────────────────────── _wechat_client 新函数 ─────────────────────────

class _FakeResp:
    def __init__(self, status_code=200, json_data=None, content=b"img"):
        self.status_code = status_code
        self._json = json_data or {}
        self.content = content

    def json(self):
        return self._json


class _FakeClient:
    """极简 httpx 客户端桩：按 URL 返回预设响应。"""

    def __init__(self, responses):
        self._responses = responses   # list of (method, url_contains, resp)

    def get(self, url, **kw):
        return self._find("GET", url)

    def post(self, url, **kw):
        return self._find("POST", url)

    def _find(self, method, url):
        for m, sub, resp in self._responses:
            if m == method and sub in url:
                return resp
        raise AssertionError(f"unexpected {method} {url}")


def test_client_upload_content_images_replaces_src():
    """正文图 uploadimg 转存：外链替换为 mmbiz URL，mmbiz 跳过。"""
    client = _FakeClient([
        ("GET", "https://img.e.com/a.png", _FakeResp(200, {}, b"\x89PNG")),
        ("POST", "media/uploadimg", _FakeResp(200, {"url": "https://mmbiz.qpic.cn/new1"})),
    ])
    html = '<p><img src="https://img.e.com/a.png" /></p><p><img src="https://mmbiz.qpic.cn/keep" /></p>'
    out, uploaded = client_mod.upload_content_images(
        access_token="t", content=html, client=client, api_base="https://api.weixin.qq.com/cgi-bin",
    )
    assert "https://mmbiz.qpic.cn/new1" in out
    assert "https://mmbiz.qpic.cn/keep" in out
    assert len(uploaded) == 1 and uploaded[0]["ok"] is True


def test_client_mass_send_body_and_msg_id():
    """mass_send 提交 sendall 并返回 msg_id。"""
    client = _FakeClient([
        ("POST", "message/mass/sendall", _FakeResp(200, {"msg_id": "m-9"})),
    ])
    msg_id = client_mod.mass_send(
        access_token="t", media_id="draft-1", clientmsgid="cid-1",
        client=client, api_base="https://api.weixin.qq.com/cgi-bin",
    )
    assert msg_id == "m-9"


def test_client_get_publish_status_and_article_url():
    client = _FakeClient([
        ("POST", "freepublish/get", _FakeResp(200, {
            "publish_id": "p1", "publish_status": 0,
            "article_detail": {"item": [{"article_url": "https://mp.weixin.qq.com/s/x"}]},
        })),
    ])
    st = client_mod.get_publish_status(access_token="t", publish_id="p1", client=client,
                                       api_base="https://api.weixin.qq.com/cgi-bin")
    assert st["publish_status"] == 0
