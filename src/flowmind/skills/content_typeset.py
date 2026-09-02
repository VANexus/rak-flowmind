"""content_typeset 技能：公众号 Markdown → 内联样式 HTML（排版）。

管线：markdown-it-py 渲染 Markdown → 套用 doocs/md 官方主题 CSS（已 vendored）
→ 把 CSS 变量（--md-primary-color 等）解析为具体值 → css_inline 全部内联。

为什么这么做：微信公众号后台粘贴正文时会剥离 <style> 与外部样式，
只有「每元素内联 style」的 HTML 才能保真。这里只做确定性渲染，
不调用 LLM，是内容创作中心的「排版」环节。

失败契约：纯本地渲染 — 出错 raise，由 invoke() 套 INTERNAL 信封。
"""
from __future__ import annotations

import re
from pathlib import Path

import css_inline
from markdown_it import MarkdownIt
from pydantic import BaseModel, Field

from flowmind.contracts import ReasoningChain, SkillOutput
from flowmind.skill import skill

_VERSION = "0.1.0"

_THEME_DIR = Path(__file__).resolve().parent / "_wechat_themes"

# ── 内置主题 preset：值直接来自 doocs/md 主题 CSS 所消费的 CSS 变量 ──
# --foreground 是 hsl() 三元组（主题 CSS 里写作 hsl(var(--foreground))）。
_DEFAULT_FONT_FAMILY = (
    "-apple-system, BlinkMacSystemFont, 'Helvetica Neue', 'PingFang SC', "
    "'Hiragino Sans GB', 'Microsoft YaHei', sans-serif"
)

THEME_PRESETS: dict[str, dict[str, str]] = {
    "default": {
        "label": "经典",
        "css_file": "default.css",
        "--md-primary-color": "#07C160",
        "--foreground": "222, 12%, 14%",
        "--muted-foreground": "0 0% 45%",
        "--md-link-color": "#576B95",
        "--md-blockquote-background": "rgba(7, 193, 96, 0.08)",
        "--blockquote-background": "rgba(7, 193, 96, 0.08)",
        "--md-font-size": "16px",
        "--md-line-height": "1.8",
        "--md-font-family": _DEFAULT_FONT_FAMILY,
        "--md-block-spacing": "1",
    },
    "grace": {
        "label": "优雅",
        "css_file": "grace.css",
        "--md-primary-color": "#9C6ADE",
        "--foreground": "270, 22%, 16%",
        "--muted-foreground": "0 0% 45%",
        "--md-link-color": "#6A5ACD",
        "--md-blockquote-background": "rgba(156, 106, 222, 0.08)",
        "--blockquote-background": "rgba(156, 106, 222, 0.08)",
        "--md-font-size": "16px",
        "--md-line-height": "1.8",
        "--md-font-family": _DEFAULT_FONT_FAMILY,
        "--md-block-spacing": "1",
    },
    "simple": {
        "label": "简约",
        "css_file": "simple.css",
        "--md-primary-color": "#1F6FEB",
        "--foreground": "215, 22%, 13%",
        "--muted-foreground": "0 0% 45%",
        "--md-link-color": "#0969DA",
        "--md-blockquote-background": "rgba(31, 111, 235, 0.06)",
        "--blockquote-background": "rgba(31, 111, 235, 0.06)",
        "--md-font-size": "16px",
        "--md-line-height": "1.8",
        "--md-font-family": _DEFAULT_FONT_FAMILY,
        "--md-block-spacing": "1",
    },
}

# 匹配 var(--known) 与 var(--known, fallback)；已知变量一律替换为 preset 值。
_KNOWN_VARS = "|".join(re.escape(k) for k in THEME_PRESETS["default"])
_VAR_EXACT_RE = re.compile(rf"var\(({_KNOWN_VARS})\)")
_VAR_FALLBACK_RE = re.compile(rf"var\(({_KNOWN_VARS})\s*,\s*[^)]*\)")

# doocs/md 主题约定：行内代码带 .codespan，代码块带 .code__pre
_INLINE_CODE_RE = re.compile(r"<code(?![^>]*class=)([^>]*)>")
_PRE_RE = re.compile(r"<pre(?![^>]*class=)([^>]*)>")


class TypesetInput(BaseModel):
    """公众号排版入参。"""

    markdown: str = Field(min_length=1, max_length=100000, description="Markdown 正文（结构化 Markdown）")
    theme: str = Field(default="default", description="内置主题 id：default / grace / simple")
    # 可选覆盖（不传则用主题 preset）
    primary_color: str | None = Field(default=None, description="主题主色覆盖（CSS 色值）")
    font_size: str | None = Field(default=None, description="正文字号覆盖（如 16px）")


class TypesetResult(BaseModel):
    """公众号排版业务载荷。"""

    html: str                        # 已内联样式的正文 HTML（可直接粘贴公众号后台 / 用于草稿）
    theme: str
    theme_label: str
    stats: dict = Field(default_factory=dict)  # chars / headings / images / paragraphs


@skill(id="content_typeset", name="公众号排版", version=_VERSION, category="内容创作")
def content_typeset(inp: TypesetInput) -> SkillOutput[TypesetResult]:
    """公众号 Markdown → 内联样式 HTML。

    数据流：markdown-it 渲染 → doocs 主题 CSS 变量解析 → css_inline 内联。
    纯确定性渲染，不调用 LLM。
    """
    preset = THEME_PRESETS.get(inp.theme)
    if preset is None:
        raise ValueError(f"未知排版主题：{inp.theme}（可选：{', '.join(THEME_PRESETS)}）")

    # 1. 渲染 Markdown → HTML
    md = MarkdownIt("gfm-like", {"linkify": True})
    body_html = md.render(inp.markdown)
    body_html = _add_doocs_classes(body_html)

    # 2. 组装 CSS：base + 主题，并解析变量为具体值
    base_css = (_THEME_DIR / "base.css").read_text(encoding="utf-8")
    theme_css = (_THEME_DIR / preset["css_file"]).read_text(encoding="utf-8")
    resolved = dict(preset)
    if inp.primary_color:
        resolved["--md-primary-color"] = inp.primary_color
    if inp.font_size:
        resolved["--md-font-size"] = inp.font_size
    css = _resolve_vars(base_css + "\n" + theme_css, resolved)

    # 3. 包一层 section（携带 CSS 变量），再全部内联。
    # 只内联真正的 CSS 变量（-- 开头），label/css_file 等元数据不得混入 style。
    style_vars = " ".join(f"{k}:{v};" for k, v in resolved.items() if k.startswith("--"))
    wrapped = f'<section class="md-container" style="{style_vars}">{body_html}</section>'
    inlined = css_inline.inline(
        wrapped, extra_css=css, load_remote_stylesheets=False, keep_style_tags=False,
    )
    # css_inline 会把片段包成完整文档（<html><head></head><body>...</body></html>），
    # 公众号只粘贴正文片段，剥掉外壳保留 <section> 内容。
    inlined = _strip_document_wrapper(inlined)

    stats = _collect_stats(inp.markdown, body_html)
    chain = ReasoningChain(
        conclusion=f"公众号排版完成：主题「{preset['label']}」，{stats.get('chars', 0)} 字",
        evidence=[],
        causal_analysis="render_markdown → apply_theme → inline_style",
        risk_note="正文中的外链图片需在发布时转存为公众号图片（content_wechat_publish 自动处理）。",
    )
    return SkillOutput(
        data=TypesetResult(
            html=inlined,
            theme=inp.theme,
            theme_label=preset["label"],
            stats=stats,
        ),
        reasoning=[chain],
        confidence=1.0,
        sample_size=1,
    )


def _strip_document_wrapper(html: str) -> str:
    """去掉 css_inline 包裹的 <html><head></head><body> ... </body></html> 外壳。"""
    s = html
    for prefix in ("<html><head></head><body>", "<html><head></head><body>\n"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.endswith("</body></html>"):
        s = s[: -len("</body></html>")]
    return s


def _add_doocs_classes(html: str) -> str:
    """对齐 doocs/md 主题类名约定：行内代码 .codespan、代码块 .code__pre。"""
    html = _INLINE_CODE_RE.sub(r'<code class="codespan"\1>', html)
    html = _PRE_RE.sub(r'<pre class="code__pre"\1>', html)
    return html


def _resolve_vars(css: str, preset: dict[str, str]) -> str:
    """把主题 CSS 里的已知 CSS 变量替换为具体值（含 var(--x, fallback) 形式）。"""
    css = _VAR_FALLBACK_RE.sub(lambda m: preset[m.group(1)], css)
    css = _VAR_EXACT_RE.sub(lambda m: preset[m.group(1)], css)
    return css


def _collect_stats(markdown: str, body_html: str) -> dict:
    return {
        "chars": len(markdown.replace("\n", "")),
        "headings": len(re.findall(r"<h[1-6][ >]", body_html)),
        "paragraphs": body_html.count("<p"),
        "images": len(re.findall(r"<img\b", body_html)),
    }
