"""内容平台规则库 + 确定性审计引擎（content_audit 技能的规则扫描层）。

规则分两类：
- 通用（平台 `*`）：广告法绝对化用语、医疗功效宣称、无来源数据等——三平台通用。
- 平台专属：小红书导流、公众号诱导分享、抖音金融/口播违禁等。

severity 语义：error=必改（违禁词/高风险），warning=建议复核（疑似风险）。
扫描基于正则（re.IGNORECASE），命中返回 AuditFinding（可解释：命中词/规则 id/建议）。

本模块只做确定性扫描；LLM 复核在 content_audit 技能内调用 _llm_client 补充。
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

Severity = Literal["error", "warning"]
CATEGORY_LABELS = {
    "absolute": "绝对化用语",
    "medical": "医疗功效",
    "advert": "夸大宣传",
    "platform": "平台规范",
    "finance": "金融夸大",
    "data": "数据无来源",
}


class AuditRule(BaseModel):
    """一条规则。platforms 里含 "*" 表示所有平台。"""
    id: str
    platforms: list[str]
    category: str
    severity: Severity
    pattern: str      # 正则表达式
    label: str        # 人类可读的命中描述
    suggestion: str   # 修改建议


class AuditFinding(BaseModel):
    """单条审计结果。"""
    category: str
    severity: Severity
    message: str
    suggestion: str
    matched_text: str | None = None
    rule_id: str | None = None


# ── 规则库 ──
# 通用广告法（`*`）：绝对化用语 / 医疗功效 / 无来源数据
_COMMON_ABSOLUTE = (
    r"全网最低|全国最低|全网最好|全网第一|全球最佳|世界第一|国家级|顶级|绝无仅有|"
    r"唯一|首选|100%|百分百|百分之百|绝对|极致|永久|最强|最佳|最优|最好|最低价|"
    r"第一品牌|排名第一|销量第一|业界领先|遥遥领先"
)
_COMMON_MEDICAL = (
    r"治疗|治愈|根治|药到病除|降血压|降血糖|抗癌|防癌|疗效|康复|处方|"
    r"杀菌率(达|超过)?|抑菌率(达|超过)?|包治百病|立竿见影|医美"
)
_COMMON_DATA = r"获(得|评|颁)?[一二三四五六七八九十\d]+[项届次名]?[奖]|央视(报道|上榜|推荐)"


AUDIT_RULES: list[AuditRule] = [
    # ── 通用绝对化用语（广告法，error）──
    AuditRule(id="R-ABS-01", platforms=["*"], category="absolute", severity="error",
              pattern=_COMMON_ABSOLUTE,
              label="检出绝对化用语", suggestion="删除或改为可证实的具体表述（如'通过 X 项检测'）"),
    # ── 通用医疗功效（error）──
    AuditRule(id="R-MED-01", platforms=["*"], category="medical", severity="error",
              pattern=_COMMON_MEDICAL,
              label="检出医疗功效宣称", suggestion="删除疾病治疗/保健功能表述，必要时标注'普通日用品，非医疗器械'"),
    # ── 通用无来源数据（warning）──
    AuditRule(id="R-DAT-01", platforms=["*"], category="data", severity="warning",
              pattern=_COMMON_DATA,
              label="引用奖项/媒体报道建议补充来源", suggestion="补充权威来源或删除未经证实的头衔"),
    AuditRule(id="R-DAT-02", platforms=["*"], category="data", severity="warning",
              pattern=r"\d+(\.\d+)?%\s*(的|以上|的人|用户)?(表示|认为|选择|首选|信赖)",
              label="数据类表述无来源", suggestion="补充调研/测试报告出处及样本说明"),

    # ── 小红书：导流违禁（error）+ 站外商品引导（warning）──
    AuditRule(id="R-XHS-01", platforms=["xhs"], category="platform", severity="error",
              pattern=r"加\s*微信|微信号|VX|加\s*v|薇信|私信(我|领|获取|免费|报名|扣)|评论区(见|留言|扣)|加群|点击(主页|头像)",
              label="小红书禁止站外导流（微信/私信引导）", suggestion="改为'点我了解'或在平台内互动，删除联系方式"),
    AuditRule(id="R-XHS-02", platforms=["xhs"], category="platform", severity="warning",
              pattern=r"tb\.cn|复制.*打开(淘宝|天猫)|￥[A-Za-z0-9]+￥",
              label="疑似站外商品链接引导", suggestion="删除站外链接；小红书禁止外链导购"),
    AuditRule(id="R-XHS-03", platforms=["xhs"], category="advert", severity="warning",
              pattern=r"刷单|买粉|代购|代发|养号",
              label="检出违规营销词", suggestion="删除与刷单/买粉相关的表述"),

    # ── 公众号：诱导分享（error）+ 诱导关注（warning）──
    AuditRule(id="R-WX-01", platforms=["wechat"], category="platform", severity="error",
              pattern=r"转发(到|至)?(朋友圈|群|好友)|分享(到)?(朋友圈|群)|转发(即|就)?可|转发抽奖|分享抽奖|集齐.*换|助力|砍价",
              label="诱导分享/转发", suggestion="删除强制转发要求；合规的抽奖需说明规则与抽奖资质"),
    AuditRule(id="R-WX-02", platforms=["wechat"], category="platform", severity="warning",
              pattern=r"关注(才|才能|后|即可)|不(关注|点)就|取关",
              label="诱导关注", suggestion="用内容价值吸引关注，避免'关注才可…'胁迫式表述"),

    # ── 抖音：金融夸大（error）+ 口播绝对化（复用通用）+ 导流（error）──
    AuditRule(id="R-DY-01", platforms=["douyin"], category="finance", severity="error",
              pattern=r"稳赚不赔|保本保息|零风险|暴富|躺赚|轻松月入|无风险|高收益|稳赚",
              label="检出金融收益承诺", suggestion="删除收益承诺；金融相关需资质并提示风险"),
    AuditRule(id="R-DY-02", platforms=["douyin"], category="platform", severity="error",
              pattern=r"加\s*微信|VX|加\s*v|私信(我|领|获取)|点(关注)?进入(橱窗|直播间)?抢",
              label="抖音导流/引导话术", suggestion="在平台规则内完成转化（小黄车/链接组件），删除站外导流"),

    # ── 通用导流兜底（warning；小红书/抖音已单独 error）──
    AuditRule(id="R-PLT-01", platforms=["*"], category="platform", severity="warning",
              pattern=r"wechat|vx|v信|加我微信|二维码",
              label="检出联系方式引导", suggestion="确认平台是否允许该联系方式展示"),
]


def audit_rules(platform: str, title: str, body: str, tags: list[str]) -> list[AuditFinding]:
    """确定性扫描 title+body+tags 命中规则，返回按 severity 排序的 findings。"""
    text = "\n".join([title or "", body or "", " ".join(tags or [])])
    findings: list[AuditFinding] = []
    for rule in AUDIT_RULES:
        if "*" not in rule.platforms and platform not in rule.platforms:
            continue
        m = re.search(rule.pattern, text, re.IGNORECASE)
        if m:
            findings.append(AuditFinding(
                category=rule.category,
                severity=rule.severity,
                message=rule.label,
                suggestion=rule.suggestion,
                matched_text=m.group(0)[:80] or None,
                rule_id=rule.id,
            ))
    order = {"error": 0, "warning": 1}
    findings.sort(key=lambda f: order.get(f.severity, 9))
    return findings


def category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, category)
