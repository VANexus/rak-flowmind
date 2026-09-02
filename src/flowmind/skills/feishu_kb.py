"""飞书知识库 FAQ 检索技能：把飞书 Wiki/知识库同步到本地，对用户提问做
4 类意图分类 + BM25+TF-IDF（+本地向量嵌入，GPU 可用时）多路召回 + RRF 融合
+ 重排，输出 Top 3 命中与四段式因果推理链。

设计要点（遵循 FlowMind 约定）：
- 输入用 pydantic 模型校验，输出 SkillOutput[T] 套 SkillResult 信封
- 4 段式链第 2、3 段用 evaluate_rules() 自动产出
- 错误走 degraded=True + SkillError，不抛
- 阈值走 config（用户可覆盖），含通用默认
- trace_id 透传由 invoke() 框架负责，本函数不关心
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jieba
import numpy as np
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from flowmind.config import load_config
from flowmind.contracts import Evidence, ReasoningChain, SkillOutput
from flowmind.rules import Rule, evaluate_rules
from flowmind.skill import skill
from flowmind.skills import _local_embed

_VERSION = "0.1.0"

# 4 大营销意图（与种子数据 SUBCATEGORY_TO_INTENT 对齐）
INTENTS: tuple[str, ...] = (
    "产品咨询",
    "故障排查",
    "充电补能",
    "用车指导",
)

# 4 大类关键词词典（启发式权重，足够 demo；生产应从数据自学习）
INTENT_KEYWORDS: dict[str, dict[str, float]] = {
    "产品咨询": {
        "车型": 1.0, "配置": 1.2, "动力": 1.2, "续航": 1.0, "马力": 1.2,
        "扭矩": 1.3, "轴距": 1.2, "外观": 0.8, "内饰": 0.8, "空间": 0.9,
        "智驾": 1.5, "智能驾驶": 1.6, "辅助驾驶": 1.5, "自动驾驶": 1.5,
        "车道保持": 1.6, "自动泊车": 1.6, "ACC": 1.3, "NGP": 1.4,
        "L2": 1.2, "LCC": 1.2, "NOA": 1.6, "车机": 1.2, "OTA": 1.2,
        "零重力座椅": 1.6, "吸顶屏": 1.5, "迎宾模式": 1.4, "座椅记忆": 1.4,
        "语音助手": 1.3, "按摩": 1.3, "座椅": 1.0, "屏幕": 1.0, "空调": 1.0,
        "氛围灯": 1.3, "方向盘": 1.0, "后视镜": 1.0,
    },
    "故障排查": {
        "故障": 1.4, "报错": 1.5, "故障码": 1.6, "报警": 1.4, "异响": 1.6,
        "抖动": 1.4, "顿挫": 1.6, "失速": 1.8, "无法启动": 1.6,
        "打不着火": 1.6, "黑屏": 1.3, "死机": 1.2, "动力丢失": 1.8,
        "动力中断": 1.8, "跑偏": 1.4, "漏水": 1.5, "漏油": 1.7,
        "不制冷": 1.4, "烧机油": 1.8, "冒烟": 1.5, "跳枪": 1.6,
        "充不进去电": 1.8, "充不上电": 1.8, "充不进电": 1.8,
        "指示灯": 1.5, "灯亮": 1.5, "灯点亮": 1.5, "灯常亮": 1.5,
        "点亮条件": 1.5, "处理措施": 1.5, "报警灯": 1.5,
        "亮灯": 1.4, "灭了": 1.3, "不亮": 1.4, "闪烁": 1.3,
        "无法": 1.3, "不起": 1.3, "不起动": 1.3,
    },
    "充电补能": {
        "充电": 1.2, "快充": 1.5, "慢充": 1.4, "充电桩": 1.6, "充电枪": 1.5,
        "家用充电": 1.5, "公共充电": 1.4, "充电站": 1.4, "充电时间": 1.4,
        "充电功率": 1.5, "充电接口": 1.4, "直流": 1.2, "交流": 1.0,
        "电池": 1.0, "电池保养": 1.5, "电池寿命": 1.4, "电池衰减": 1.5,
        "实际续航": 1.3, "续航里程": 1.2, "冬季续航": 1.5, "夏季续航": 1.4,
        "预约充电": 1.5, "定时充电": 1.5, "V2L": 1.4, "外放电": 1.4,
        "智能保温": 1.5, "保温": 1.4, "充电时长": 1.5, "额定电量": 1.4,
        "动力电池": 1.4, "充电指示灯": 1.6, "电池充电": 1.4,
        "随车充电": 1.5, "LINGCLUB": 1.3, "LING": 1.2,
        "充满电": 1.3, "充满": 1.2, "快充时间": 1.4, "SOC": 1.4,
        "百公里": 1.3, "电耗": 1.3,
    },
    "用车指导": {
        "怎么开": 1.4, "怎么用": 1.3, "如何使用": 1.3, "怎么操作": 1.3,
        "使用方法": 1.3, "保养": 1.4, "保养周期": 1.6, "首保": 1.5,
        "维护": 1.3, "换胎": 1.4, "轮胎": 1.0, "雨刮": 1.2, "玻璃水": 1.2,
        "机油": 1.3, "刹车油": 1.4, "防冻液": 1.4, "儿童锁": 1.4,
        "安全座椅": 1.4, "拖车": 1.3, "搭电": 1.4, "电瓶": 1.3,
        "胎压": 1.4, "质保": 1.5, "三包": 1.5, "救援": 1.4, "4S店": 1.3,
        "CVT": 1.4, "变速器": 1.4, "燃油报警": 1.5, "燃油灯": 1.5,
        "应急启动": 1.5, "应急熄火": 1.5, "应急开门": 1.5,
        "机械钥匙": 1.4, "一键启动": 1.4, "空气净化": 1.3,
        "随车工具": 1.3, "备胎": 1.2, "千斤顶": 1.2,
        "暖风": 1.2, "风道": 1.2, "出风口": 1.2,
        "胎压复位": 1.4, "胎压灯": 1.3,
        "年检": 1.3, "检验": 1.2, "行驶": 1.0,
    },
}


# ====================== Pydantic 模型 ======================


class FeishuKbInput(BaseModel):
    """飞书知识库检索技能入参。"""
    query: str = Field(min_length=1, max_length=2000, description="用户原句（2000 字内）")
    top_k: int = Field(default=3, ge=1, le=20, description="返回条数（默认 3）")


class FaqItem(BaseModel):
    """单条 FAQ 命中。"""
    rank: int
    faq_id: str
    category: str
    question: str
    answer: str
    source_url: str = ""
    final_score: float = 0.0


class FeishuKbReport(BaseModel):
    """技能业务载荷。

    字段分两类:
    - 业务字段:query / cleaned_query / intent / top_k / agent_reply_hint
    - Agent 可直接读取的结构化字段:user_language / translation_required / translation_directive
      让任何 agent(包括非 LLM)直接读懂输出语种 + 翻译规则
    """
    query: str
    cleaned_query: str
    intent_category: str
    intent_confidence: float
    matched_keywords: list[str] = Field(default_factory=list)
    top_k: list[FaqItem] = Field(default_factory=list)
    agent_reply_hint: str = ""  # 给上层 Agent 的回复模板指引

    # ====================== Agent 直接可读的结构化字段 ======================
    user_language: str = "other"                # "zh" / "en" / "th" / "other"
    translation_required: bool = False          # user_language != "zh"
    translation_directive: dict[str, str] = Field(
        default_factory=dict,
        description="结构化翻译指令:{source,target,rule}",
    )


# ====================== 内部辅助 ======================


_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_URL_RE = re.compile(r"https?://[^\s]+|www\.[^\s]+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "☀-⛿"
    "✀-➿"
    "]"
)
_FULL2HALF = {
    "，": ",", "。": ".", "；": ";", "：": ":", "？": "?", "！": "!",
    "（": "(", "）": ")", "【": "[", "】": "]", "～": "~", "、": ",",
}

# 话题外防御：标准转人工文案（Agent 上层直接透传给用户）
_OFFTOPIC_HINT = "暂未收录此类问题，请换个问法或联系人工客服。"

# 翻译不可变性约束:翻译输出语言 = 用户语言。
# 所有 KB 术语输出时禁止:软化 / euphemize / replace / 同义化。
_TRANSLATION_RULE = (
    "verbatim 翻译;不软化 / 不替换 / 不 euphemize / 不 register-shift;"
    "保留 brand / 车型 / 技术术语原文(或附原文括注)"
)

# 三语翻译指令:KB 内容是中文 source of truth,LLM 只做语言转换,不补充信息。
# [Language-MANDATORY] 标记为下游 agent(Claude Code 等)必执行翻译。
_TRANSLATION_HINT = {
    "zh": "",  # 中文无需翻译
    "en": (
        "\n[Language-MANDATORY] User asked in English. "
        "**The user MUST receive the answer in English.** "
        "Translate the Chinese Top-1 answer above into natural English. "
        "Do NOT add information beyond the FAQ content. "
        "DO NOT soften, euphemize, replace, or register-shift any car/technical/brand term."
    ),
    "th": (
        "\n[Language-MANDATORY] ผู้ใช้ถามเป็นภาษาไทย "
        "**ผู้ใช้ต้องได้รับคำตอบเป็นภาษาไทย** "
        "กรุณาแปลคำตอบจากภาษาจีนด้านบนเป็นภาษาไทยที่เป็นธรรมชาติ "
        "ห้ามเพิ่มข้อมูลนอกเหนือจากเนื้อหา FAQ "
        "ห้าม soften/replace/euphemize คำศัพท์รถ/เทคนิค/แบรนด์"
    ),
    "other": (
        "\n[Language] User asked in an unsupported language. "
        "Respond briefly: '暂未收录此类问题,请用中文、英文或泰文提问。'"
    ),
}


def _build_translation_directive(lang: str) -> dict[str, str]:
    """结构化翻译指令:让下游任何 agent(LLM 或 JSON-reader)
    直接读懂用户语种、目标语种、翻译硬约束。
    """
    if lang == "zh" or lang == "other":
        return {}
    return {
        "source": "zh",
        "target": lang,
        "rule": _TRANSLATION_RULE,
    }


def _detect_language(text: str) -> str:
    """检测查询语言: zh / en / th / other。

    实现:基于 Unicode 字符范围(零依赖、轻量、确定性)。
    - 中文字符: CJK Unified Ideographs (U+4E00-U+9FFF)
    - 泰文字符: Thai block (U+0E00-U+0E7F)
    - 英文/拉丁: ASCII / Latin-1 Supplement 及以上

    返回值优先级: 含中文字符 → zh;含泰文字符 → th;仅拉丁 → en;其他 → other。
    """
    if not text:
        return "other"
    has_zh = False
    has_th = False
    has_alpha = False
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:  # CJK
            has_zh = True
        elif 0x0E00 <= cp <= 0x0E7F:  # Thai
            has_th = True
        elif ch.isalpha():
            has_alpha = True
    if has_zh:
        return "zh"
    if has_th:
        return "th"
    if has_alpha:
        return "en"
    return "other"


# 跨语言检索桥接: EN/TH 关键词 → 中文同义词。
# 在非中文 query 上做关键词扩展,把 "charge/battery" 等映射成 "充电/电池",
# 让 BM25+TF-IDF 在中文 FAQ 上能命中。
# 规模: ~150 个 EN 词 + ~80 个 TH 词,覆盖 FAQ 语料里所有核心领域词。
# 不做完整翻译,只做"领域词"映射 —— 通用词由 LLM 翻译层处理。
_CROSS_LANG_SYNONYMS: dict[str, list[str]] = {
    # English → Chinese  ---  充电 / 电池 / 动力 ----------------------
    "charge": ["充电"], "charging": ["充电"], "charger": ["充电桩"], "plug": ["充电枪"],
    "battery": ["电池"], "power": ["动力"], "range": ["续航"], "mileage": ["续航里程"],
    "fuel": ["燃油"], "gasoline": ["燃油"], "petrol": ["燃油"], "diesel": ["柴油"],
    "fuel economy": ["油耗"], "fuel consumption": ["油耗"], "fuel level": ["油量"],
    "fuel pump": ["油泵"], "fuel gauge": ["燃油表"], "fuel light": ["燃油灯"],
    # English → Chinese  ---  轮胎 / 胎压 -----------------------------
    "tire": ["轮胎"], "tyre": ["轮胎"], "pressure": ["胎压"], "flat": ["亏气"],
    "spare tire": ["备胎"], "spare wheel": ["备胎"], "jack": ["千斤顶"], "tools": ["工具"],
    # English → Chinese  ---  发动机 / 启动 ----------------------------
    "engine": ["发动机"], "motor": ["电机"],
    "start": ["启动"], "stall": ["失速"], "stalling": ["失速"],
    "engine start": ["发动机启动"], "key fob": ["遥控钥匙"], "keyless": ["无钥匙"],
    "remote start": ["远程启动"], "cold start": ["冷启动"], "warm up": ["暖车"],
    "alternator": ["发电机"], "starter": ["起动机"],
    # English → Chinese  ---  刹车 / 故障 / 报警 ------------------------
    "brake": ["刹车"], "braking": ["刹车"], "brake light": ["刹车灯"],
    "fault": ["故障"], "error": ["报错"], "warning": ["报警"], "alarm": ["报警"],
    "warning light": ["报警灯"], "warning lamp": ["报警灯"],
    "comes on": ["点亮"], "light up": ["点亮"], "lights up": ["点亮"],
    "illuminate": ["点亮"], "illuminates": ["点亮"], "turns on": ["亮"],
    "indicator": ["指示灯"], "lamp": ["指示灯"], "indicator lamp": ["指示灯"],
    "indicator light on": ["指示灯"], "fuel indicator": ["燃油指示灯"],
    # English → Chinese  ---  异响 / 抖动 / 抖动 ------------------------
    "noise": ["异响"], "noise sound": ["异响"], "vibration": ["抖动"],
    "jerk": ["顿挫"], "jerking": ["顿挫"], "shudder": ["抖动"],
    "loud": ["异响"], "noisy": ["异响"], "rattle": ["异响"],
    # English → Chinese  ---  屏 / 气囊 / 显示 --------------------------
    "screen": ["车机"], "display": ["车机"], "airbag": ["安全气囊"],
    "air bag": ["安全气囊"],
    # English → Chinese  ---  安全系统 ---------------------------------
    "ABS": ["ABS"], "ESC": ["电子稳定"], "ESP": ["电子稳定"],
    "traction": ["牵引力"], "stability": ["稳定"],
    # English → Chinese  ---  空调 / 暖风 -------------------------------
    "AC": ["空调"], "air conditioning": ["空调"], "heater": ["暖风"],
    "heat": ["暖风"], "heating": ["暖风"], "warm": ["暖风"],
    "defrost": ["除霜"], "fog": ["雾"], "fog light": ["雾灯"],
    # English → Chinese  ---  门窗 / 锁 ---------------------------------
    "door": ["车门"], "window": ["车窗"], "lock": ["门锁"], "door handle": ["门把手"],
    "key": ["钥匙"], "remote": ["遥控"], "remote control": ["遥控"],
    "open the door": ["开门"], "unlock": ["开门"], "shut down": ["熄火"],
    "turn off": ["关闭"], "trunk release": ["后备箱开启"], "hood": ["引擎盖"],
    "sunroof": ["天窗"], "moonroof": ["天窗"],
    # English → Chinese  ---  座椅 / 空间 -------------------------------
    "seat": ["座椅"], "seat massage": ["按摩"], "headrest": ["头枕"],
    "rear": ["后排"], "front": ["前排"], "back row": ["后排"],
    "trunk": ["后备箱"], "boot": ["后备箱"],
    # English → Chinese  ---  油液 / 滤芯 -------------------------------
    "transmission fluid": ["变速器油"], "coolant": ["防冻液"],
    "engine oil": ["机油"], "power steering": ["转向助力"],
    "air filter": ["空气滤芯"], "oil filter": ["机油滤芯"],
    "washer fluid": ["玻璃水"], "wiper fluid": ["玻璃水"],
    # English → Chinese  ---  灯光 / 信号 -------------------------------
    "headlight": ["大灯"], "tail light": ["尾灯"], "signal": ["转向灯"],
    "high beam": ["远光"], "low beam": ["近光"], "hazard": ["双闪"],
    "hazard light": ["双闪"], "wiper": ["雨刮"], "wipers": ["雨刮"],
    "horn": ["喇叭"], "windshield": ["前挡风"], "rear window": ["后挡风"],
    # English → Chinese  ---  驾驶辅助 ---------------------------------
    "cruise": ["定速巡航"], "lane": ["车道"], "assist": ["辅助"],
    "departure": ["偏航"], "departure warning": ["车道偏离预警"],
    "lane keep": ["车道保持"], "lane keeping": ["车道保持"],
    "auto park": ["自动泊车"], "autopilot": ["自动驾驶"],
    "adaptive": ["自适应"], "adaptive cruise": ["自适应巡航"],
    "blind spot": ["盲区"], "around view": ["全景"], "camera": ["摄像头"],
    # English → Chinese  ---  启动 / 一键 / 应急 ------------------------
    "one-click": ["一键"], "one touch": ["一键"], "push button": ["一键"],
    "push-button": ["一键"], "one push": ["一键"],
    "emergency": ["应急"], "emergency start": ["应急启动"],
    "emergency shutdown": ["应急熄火"], "emergency stop": ["应急熄火"],
    "emergency open": ["应急开门"],
    # English → Chinese  ---  距离 / 续航 / 操作 ------------------------
    "how far": ["还能跑多远"], "how long": ["续航"], "distance": ["续航", "距离"],
    "operation": ["操作"], "procedure": ["操作"], "method": ["方法"],
    "how to use": ["使用方法"], "instructions": ["操作方法"],
    # English → Chinese  ---  季节 / 温度 -------------------------------
    "winter": ["冬季"], "summer": ["夏季"], "cold": ["低温"], "weather": ["天气"],
    "hot": ["热"],
    # English → Chinese  ---  其他 / 通用 ---------------------------------
    "smoke": ["冒烟"], "leak": ["漏"], "smell": ["气味"], "burning": ["烧"],
    "where": ["在哪"], "where is": ["在哪"], "how": ["怎么"], "why": ["为什么"],
    "what": ["什么"], "which": ["哪个"], "reset": ["复位"], "relearn": ["复位"],
    "CVT": ["CVT"], "transmission": ["变速器"], "gearbox": ["变速器"],
    "kilometers": ["公里"], "km": ["公里"], "mile": ["英里"], "miles": ["英里"],
    # ====================== Thai → Chinese ======================
    "ชาร์จ": ["充电"], "ชาร์จไฟ": ["充电"], "แบตเตอรี่": ["电池"],
    "ไฟฟ้า": ["充电"], "พลังงาน": ["动力"], "ระยะทาง": ["续航"],
    "ยาง": ["轮胎"], "ลมยาง": ["胎压"], "แรงดัน": ["胎压"],
    "เครื่องยนต์": ["发动机"], "มอเตอร์": ["电机"], "เบรก": ["刹车"],
    "ขัดข้อง": ["故障"], "ผิดปกติ": ["报错"], "เตือน": ["报警"],
    "ไฟ": ["灯"], "สัญญาณไฟ": ["指示灯"], "ไฟเตือน": ["指示灯"],
    "สตาร์ท": ["启动"], "ดับ": ["失速"], "เสียงดัง": ["异响"],
    "สั่น": ["抖动"], "กระตุก": ["顿挫"],
    "หน้าจอ": ["车机"], "ถุงลม": ["安全气囊"],
    "แอร์": ["空调"], "เครื่องปรับอากาศ": ["空调"], "ฮีทเตอร์": ["暖风"],
    "ประตู": ["车门"], "กุญแจ": ["钥匙"], "รีโมท": ["遥控"],
    "เบาะ": ["座椅"], "ที่นั่ง": ["座椅"],
    "ฤดูหนาว": ["冬季"], "ฤดูร้อน": ["夏季"], "หนาว": ["低温"],
    # 泰文扩展
    "ฉุกเฉิน": ["应急"], "ฉุกเฉินสตาร์ท": ["应急启动"], "สตาร์ทฉุกเฉิน": ["应急启动"],
    "ไกล": ["多远"], "ไกลแค่ไหน": ["多远"], "ระยะ": ["多远", "距离"],
    "วิธีใช้": ["使用方法"], "วิธีการ": ["操作方法"], "ขั้นตอน": ["操作方法"],
    "ติดสว่าง": ["点亮"], "ไฟติด": ["点亮"], "สว่างขึ้น": ["点亮"],
    "น้ำมัน": ["燃油"], "เชื้อเพลิง": ["燃油"],
    "ยางอะไหล่": ["备胎"], "แม่แรง": ["千斤顶"], "เครื่องมือ": ["工具"],
    "ที่ไหน": ["在哪"], "อยู่ที่ไหน": ["在哪"],
    "อุ่น": ["暖风"], "อุ่นๆ": ["暖风"],
    "เบาะหลัง": ["后排座椅"], "เบาะหน้า": ["前排座椅"],
    "สตาร์ทไม่ติด": ["无法启动"], "เครื่องดับ": ["失速"],
    "เสียง": ["声音", "异响"],
    "ควัน": ["冒烟"], "ควันขาว": ["冒白烟"],
    "น้ำมันรั่ว": ["漏油"], "น้ำรั่ว": ["漏水"],
    "ร้อน": ["热"], "เย็น": ["冷"],
    "เปิดประตู": ["开门"], "ล็อค": ["门锁"], "ล็อก": ["门锁"],
    "เปิด": ["打开"], "ปิด": ["关闭"],
    "สัญญาณไฟเลี้ยว": ["转向灯"], "ไฟสูง": ["远光"], "ไฟต่ำ": ["近光"],
    "ไฟตัดหมอก": ["雾灯"], "ไฟฉุกเฉิน": ["双闪"],
    "กระจก": ["玻璃", "车窗"], "กระจกหน้า": ["前挡风"],
    "หม้อน้ำ": ["水箱"], "น้ำมันเครื่อง": ["机油"], "น้ำมันเบรก": ["刹车油"],
    "กุญแจรีโมท": ["遥控钥匙"], "รีโมทคอนโทรล": ["遥控"],
    "ตั้งค่า": ["复位"], "รีเซ็ต": ["复位"],
    "ฝากระโปรง": ["引擎盖"], "หลังคา": ["天窗"],
    "ขับ": ["驾驶", "行驶"], "ความเร็ว": ["速度"], "เร่ง": ["加速"],
    "รอบ": ["转速"], "เกียร์": ["变速器", "档位"],
}


def _expand_query_for_cross_lang(query: str, lang: str) -> str:
    """对非中文 query 做关键词扩展,把 EN/TH 词翻译成 ZH 词以命中 FAQ。

    仅在 lang != "zh" 时调用;不修改中文 query。
    """
    if lang == "zh" or lang == "other":
        return query
    query_lower = query.lower()
    expansions: list[str] = []
    for term, zh_synonyms in _CROSS_LANG_SYNONYMS.items():
        if term.lower() in query_lower:
            expansions.extend(zh_synonyms)
    if not expansions:
        return query
    # 把 ZH 同义词追加到 query(用空格分隔,tokenize 时会一并处理)
    return f"{query} {' '.join(expansions)}"


def _clean(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = _CTRL_RE.sub(" ", text)
    text = "".join(_FULL2HALF.get(ch, ch) for ch in text)
    return _WS_RE.sub(" ", text).strip()


def _phrase_match_bonus(title: str, query_zh: str) -> float:
    """中文 4 字短语匹配 bonus。

    遍历 title 中所有连续的 4 字中文子串,看是否在 query 出现过。
    每匹配一个 = 0.05 分。多个匹配累加。

    为什么是 4 字:
    - 3 字太宽(单 token 也匹配),噪音多
    - 4 字精确(覆盖 "燃油报警灯" / "后还能跑多远" 这种 FAQ 标题短语)

    防作弊:
    - 只统计 4 个**连续中文**字符
    - query 已剥离 ASCII / 标点 / 空格,纯中文 substring 匹配
    """
    if not title or not query_zh or len(query_zh) < 4:
        return 0.0
    bonus = 0.0
    for i in range(len(title) - 3):
        chunk = title[i : i + 4]
        if all("一" <= ch <= "鿿" for ch in chunk):
            if chunk in query_zh:
                bonus += 0.05
    return bonus


@lru_cache(maxsize=1)
def _jieba_ready() -> bool:
    jieba.initialize()
    return True


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    _jieba_ready()
    return [t for t in jieba.cut(text.lower()) if t.strip()]


def _classify(cleaned: str) -> tuple[str, float, list[str]]:
    """返回 (category, confidence, matched_keywords)。"""
    if not cleaned:
        return "用车指导", 0.0, []
    matched: dict[str, list[str]] = {c: [] for c in INTENTS}
    raw: dict[str, float] = {c: 0.0 for c in INTENTS}
    cleaned_lower = cleaned.lower()
    for cat, kws in INTENT_KEYWORDS.items():
        for kw, w in kws.items():
            if kw.lower() in cleaned_lower:
                matched[cat].append(kw)
                raw[cat] += w
    if max(raw.values()) <= 0:
        return "用车指导", 0.0, []
    sorted_cats = sorted(raw.items(), key=lambda x: x[1], reverse=True)
    top_cat, top_score = sorted_cats[0]
    second = sorted_cats[1][1] if len(sorted_cats) > 1 else 0.0
    margin = (top_score - second) / max(top_score, 1.0)
    confidence = min(0.5 * margin + 0.5 * min(top_score / 3.0, 1.0) + 0.1, 0.99)
    return top_cat, round(confidence, 3), matched[top_cat]


@dataclass
class _Candidate:
    faq_id: str
    category: str
    question: str
    answer: str
    source_url: str
    bm25_score: float
    vector_score: float
    rrf_score: float
    embed_score: float = 0.0   # 本地向量余弦（第三路；无向量路时为 0）


# 语料向量进程内缓存：key=(model, device, docs 哈希)，FAQ 语料不变则不重算
_DOC_EMBED_CACHE: dict[tuple, np.ndarray] = {}


def _make_embed_fn(cfg):
    """按 cfg.embed_backend 决定向量路。返回 encode(texts)->ndarray 或 None。

    - off  = 显式禁用
    - on   = 必须可用，否则显式报错（错误永不静默）
    - auto = sentence-transformers 可导入则启用，否则 None（回落双路）
    """
    chosen = (cfg.embed_backend or "auto").lower()
    if chosen == "off":
        return None
    if chosen == "on":
        if not _local_embed.embed_available():
            raise _local_embed.EmbedError(
                "embed_backend=on 但 sentence-transformers/torch 不可用", category="environment")
        return lambda texts: _local_embed.encode(
            texts, model=cfg.embed_model, device=cfg.embed_device)
    if _local_embed.embed_available():
        return lambda texts: _local_embed.encode(
            texts, model=cfg.embed_model, device=cfg.embed_device)
    return None


def _embed_docs(embed_fn, docs: list[str], model: str, device: str) -> np.ndarray:
    """FAQ 语料嵌入（进程内缓存，FAQ 集不变则只算一次）。"""
    key = (model, device, hashlib.sha1("\x1e".join(docs).encode("utf-8")).hexdigest())
    if key not in _DOC_EMBED_CACHE:
        _DOC_EMBED_CACHE[key] = embed_fn(docs)
    return _DOC_EMBED_CACHE[key]


def _hybrid_search(
    faqs: list[dict], cleaned: str, top_n: int,
    embed_fn=None, *, embed_model: str = "", embed_device: str = "",
) -> list[_Candidate]:
    """多路召回 + RRF 融合：BM25 + TF-IDF（+ 本地向量余弦，embed_fn 提供）。

    长 answer bias 防御（标题加权 + 中文短语匹配 bonus）：
    1) title 在 BM25 语料里复制 2 遍 —— 让 title token 在长文档里占比更高
    2) 给每个候选按 title 命中算中文短语匹配 bonus（见 _phrase_match_bonus）
    """
    if not faqs or not cleaned.strip():
        return []
    # ★ title 复制 2 遍:让 title token 在长 answer 中的占比相对更高
    docs = [
        ((f.get("question", "") + " ") * 2 + f.get("answer", "")).strip()
        for f in faqs
    ]
    corpus_tokens = [_tokenize(d) for d in docs]
    if not any(corpus_tokens):
        return []
    # ★ b 参数回默认 0.75;长 answer bias 由 _rerank 的中文短语匹配 bonus 处理
    bm25 = BM25Okapi(corpus_tokens)
    q_tokens = _tokenize(cleaned)
    if not q_tokens:
        return []
    bm25_scores = np.asarray(bm25.get_scores(q_tokens), dtype="float32")
    bm25_order = np.argsort(-bm25_scores)[:top_n]
    bm25_results: list[tuple[int, float]] = [
        (int(i), float(bm25_scores[i])) for i in bm25_order if bm25_scores[i] > 0
    ]

    tokenized_str = [" ".join(toks) for toks in corpus_tokens]
    try:
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\S+", lowercase=True, min_df=1)
        tfidf_matrix = vectorizer.fit_transform(tokenized_str)
        q_vec = vectorizer.transform([" ".join(q_tokens)])
        sims = linear_kernel(q_vec, tfidf_matrix).flatten()
        vec_order = np.argsort(-sims)[:top_n]
        vec_results: list[tuple[int, float]] = [
            (int(i), float(sims[i])) for i in vec_order if sims[i] > 0
        ]
    except ValueError:
        vec_results = []

    # 第三路：本地向量余弦（bge L2 归一化 → 内积即余弦）。embed_fn=None 时跳过。
    embed_results: list[tuple[int, float]] = []
    if embed_fn is not None:
        doc_vecs = _embed_docs(embed_fn, docs, embed_model, embed_device)
        q_emb = embed_fn([cleaned])[0]
        e_sims = np.asarray(doc_vecs @ q_emb, dtype="float32").flatten()
        e_order = np.argsort(-e_sims)[:top_n]
        embed_results = [(int(i), float(e_sims[i])) for i in e_order if e_sims[i] > 0]

    # RRF 融合
    by_idx: dict[int, _Candidate] = {}
    for rank, (i, raw) in enumerate(bm25_results, start=1):
        c = by_idx.setdefault(i, _Candidate(
            faq_id=faqs[i].get("id", f"FAQ-{i:04d}"),
            category=faqs[i].get("category", "未分类"),
            question=faqs[i].get("question", ""),
            answer=faqs[i].get("answer", ""),
            source_url=faqs[i].get("source_url", ""),
            bm25_score=0.0, vector_score=0.0, rrf_score=0.0,
        ))
        c.bm25_score = raw
        c.rrf_score += 1.0 / (60 + rank)
    for rank, (i, raw) in enumerate(vec_results, start=1):
        c = by_idx.setdefault(i, _Candidate(
            faq_id=faqs[i].get("id", f"FAQ-{i:04d}"),
            category=faqs[i].get("category", "未分类"),
            question=faqs[i].get("question", ""),
            answer=faqs[i].get("answer", ""),
            source_url=faqs[i].get("source_url", ""),
            bm25_score=0.0, vector_score=0.0, rrf_score=0.0,
        ))
        c.vector_score = raw
        c.rrf_score += 1.0 / (60 + rank)
    for rank, (i, raw) in enumerate(embed_results, start=1):
        c = by_idx.setdefault(i, _Candidate(
            faq_id=faqs[i].get("id", f"FAQ-{i:04d}"),
            category=faqs[i].get("category", "未分类"),
            question=faqs[i].get("question", ""),
            answer=faqs[i].get("answer", ""),
            source_url=faqs[i].get("source_url", ""),
            bm25_score=0.0, vector_score=0.0, rrf_score=0.0,
        ))
        c.embed_score = raw
        c.rrf_score += 1.0 / (60 + rank)
    return sorted(by_idx.values(), key=lambda x: x.rrf_score, reverse=True)


def _rerank(
    candidates: list[_Candidate],
    retrieval_query: str,
    intent_category: str,
    top_k: int,
) -> list[FaqItem]:
    """类别命中加权 + 中文短语匹配 bonus + 跨类多样 + 去重 → Top K。

    短语匹配 bonus 解决了 BM25 long-answer bias:
    - BM25 把 query token 摊到整篇 answer 上,长 answer FAQ 反而被稀释
    - 我们额外检查: candidate title 中任一 4 字中文短语是否出现在 query 里,
      出现得越多 = 这条 FAQ 主题越匹配,加 bonus。
    - 这样"燃油报警灯点亮后还能跑多远"这种 4 字短语能从 query "fuel warning how far"
      跨语言命中 FAQ-0029,title 命中率高,抢 Top-1。
    """
    scored: list[tuple[float, _Candidate]] = []
    # 清理 query: 去掉空格 / ASCII(只保留中文)以便做 4 字短语 substring 匹配
    query_zh = re.sub(r"[\s　A-Za-z0-9\.\,\:\?\!，！？]", "", retrieval_query)
    for c in candidates:
        bonus = 0.05 if c.category == intent_category else 0.0
        phrase_bonus = _phrase_match_bonus(c.question, query_zh)
        final = c.rrf_score + bonus + phrase_bonus
        scored.append((final, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    picked: list[tuple[float, _Candidate]] = []
    per_cat_limit = max(1, top_k // 2 + 1)
    cat_count: dict[str, int] = {}
    for s in scored:
        c = s[1]
        q = c.question.strip()
        if q in seen:
            continue
        if cat_count.get(c.category, 0) >= per_cat_limit:
            continue
        seen.add(q)
        picked.append(s)
        cat_count[c.category] = cat_count.get(c.category, 0) + 1
        if len(picked) >= top_k:
            break
    if len(picked) < top_k:
        for s in scored:
            if s in picked:
                continue
            picked.append(s)
            if len(picked) >= top_k:
                break
    out: list[FaqItem] = []
    for rank, (final, c) in enumerate(picked, start=1):
        out.append(FaqItem(
            rank=rank, faq_id=c.faq_id, category=c.category,
            question=c.question, answer=c.answer, source_url=c.source_url,
            final_score=round(final, 4),
        ))
    return out


@lru_cache(maxsize=1)
def _load_default_faqs() -> tuple[dict, ...]:
    """加载内置真实企业 FAQ（与 skill 同目录的 feishu_kb_seed.json，docx 解析产物）。"""
    seed = Path(__file__).parent / "feishu_kb_seed.json"
    if not seed.exists():
        return ()
    return tuple(json.loads(seed.read_text(encoding="utf-8")))


def _load_faqs_from_path(path: str) -> tuple[dict, ...]:
    """从用户配置路径加载 FAQ JSON。

    安全检查：
    - 解析为绝对路径，防止奇怪路径
    - 确认是常规文件（非目录、设备文件等）
    - 大小上限 50MB（防 DoS）
    - 必须是合法 JSON 且为 list[dict]
    """
    if not path:
        return ()
    p = Path(path).resolve()
    if not p.is_file():
        return ()
    if p.stat().st_size > 50 * 1024 * 1024:
        return ()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ()
    if not isinstance(data, list):
        return ()
    return tuple(data)


def _rules(intent_category: str, top1_score: float, has_hits: bool) -> list[Rule]:
    """把命中情况描述为声明式规则，供 evaluate_rules 生成第 2、3 段。"""
    return [
        Rule(
            id="KB-INTENT",
            name="意图分类命中",
            expression=f"intent_category == {intent_category}",
            predicate=lambda m: m.get("intent_category") == intent_category,
            evidence=lambda m: [Evidence(
                metric="意图类别", value=m.get("intent_category", ""),
                threshold="产品咨询/故障排查/充电补能/用车指导", comparison="==",
            )],
        ),
        Rule(
            id="KB-HAS-HITS",
            name="有候选命中",
            expression="len(top_k) > 0",
            predicate=lambda m: m.get("has_hits", False),
            evidence=lambda m: [Evidence(
                metric="Top K 候选数", value=len(m.get("top_k_list", [])),
                threshold=1, comparison=">=",
            )],
        ),
        Rule(
            id="KB-HIGH-CONF",
            name="Top 1 高置信度",
            expression="top1_score >= 0.06",
            predicate=lambda m: m.get("top1_score", 0.0) >= 0.06,
            evidence=lambda m: [Evidence(
                metric="Top 1 final_score", value=round(m.get("top1_score", 0.0), 4),
                threshold=0.06, comparison=">=",
            )],
        ),
    ]


def _build_chain(
    query: str,
    intent_category: str,
    intent_confidence: float,
    matched_keywords: list[str],
    top_k: list[FaqItem],
    retrieval_desc: str = "BM25 + TF-IDF 双路召回",
) -> ReasoningChain:
    """组装四段式因果推理链。"""
    has_hits = len(top_k) > 0
    top1_score = top_k[0].final_score if top_k else 0.0
    metrics = {
        "intent_category": intent_category,
        "has_hits": has_hits,
        "top1_score": top1_score,
        "top_k_list": top_k,
    }
    hits, evidence = evaluate_rules(_rules(intent_category, top1_score, has_hits), metrics)
    if has_hits and top1_score >= 0.06:
        conclusion = f"匹配到 {len(top_k)} 个候选 FAQ，最高 final_score={top1_score:.3f}"
    elif has_hits:
        conclusion = f"匹配到 {len(top_k)} 个候选 FAQ，但 Top 1 置信度偏低（{top1_score:.3f}）"
    else:
        conclusion = "未匹配到任何 FAQ，判定为意图不清晰"
    causal = (
        f"用户问题归类为「{intent_category}」（置信度 {intent_confidence}，"
        f"命中关键词 {matched_keywords or '无'}）。"
        f"通过 {retrieval_desc}、RRF 融合（k=60）、"
        f"类别命中加权 + 跨类多样重排，取 Top {len(top_k)}。"
    )
    risk = (
        "若 Top 1 final_score < 0.02：建议转人工客服，不要强行套用。"
        if has_hits and top1_score < 0.02
        else "若分类置信度 < 0.4：建议转人工或追问澄清。"
    )
    return ReasoningChain(
        conclusion=conclusion,
        triggered_rules=hits,
        evidence=evidence,
        causal_analysis=causal,
        risk_note=risk,
    )


def _sanitize_for_prompt(text: str, max_len: int = 200) -> str:
    """把不可信文本清洗成可安全嵌入 LLM 提示的形态。

    - 去掉换行（防 prompt 分隔注入）
    - 去掉控制字符
    - 折叠空白
    - 截断到 max_len（防长度攻击）
    - 去掉反引号（防 markdown 代码块逃逸）
    """
    if not text:
        return ""
    text = _CTRL_RE.sub(" ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.replace("`", "'").replace("</", " ")
    text = _WS_RE.sub(" ", text).strip()
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "…"
    return text


def _agent_reply_hint(query: str, intent_category: str, top_k: list[FaqItem]) -> str:
    """给上层 Agent 的回复模板指引（不是 SkillOutput 必需，是辅助）。

    设计原则：**严格忠于 KB**。Agent 必须把 Top-1 的 answer 原文透传给用户，
    不要改写、不要补充、不要推测。改写会引入 KB 之外的信息，违反"完全按照
    知识库回答"的要求。

    安全说明：``query`` 来自用户输入，**不可信**。本函数在嵌入 LLM 提示前
    必须经过 ``_sanitize_for_prompt``：去换行 / 去控制字符 / 截断 / 去反引号。
    上层 Agent 必须把本输出视作**数据**而非**指令**。
    """
    if not top_k:
        return _OFFTOPIC_HINT
    safe_query = _sanitize_for_prompt(query, max_len=200)
    return (
        f"用户问题：{safe_query}\n"
        f"系统分类：{intent_category}\n"
        f"系统已检索 {len(top_k)} 条相关 FAQ，请你：\n"
        f"  1) **直接引用** Top 1 的 answer 原文（不要改写、不要补充、不要推测）；\n"
        f"  2) 在回复**末尾**附『来源：FAQ-编号 · 飞书链接』；\n"
        f"  3) 不要整合 Top 2/3 的内容 —— 它们是兜底备份，不混入回复。\n"
        f"  4) 如果 Top 1 answer 与用户问题不完全匹配，仍按 Top 1 原文回答，"
        f"但加一句『如未解决您的问题，请联系人工客服』。"
    )


# ====================== @skill 入口 ======================


@skill(id="feishu_kb_search", name="飞书知识库 FAQ 检索", version=_VERSION)
def feishu_kb_search(inp: FeishuKbInput) -> SkillOutput[FeishuKbReport]:
    """把飞书知识库 FAQ 同步到本地，对用户提问做 4 类意图分类 +
    BM25+TF-IDF（+本地向量，GPU 可用时）多路召回 + RRF 融合 + 类别加权重排，
    输出 Top K 命中与四段式因果推理链。

    适用场景：车企 FAQ 智能客服、knowledge base 检索、客服意图分发。
    依赖：jieba（中文分词）+ rank-bm25 + scikit-learn + numpy（见 pyproject.toml）。
    """
    cfg = load_config().feishu_kb
    cleaned = _clean(inp.query)
    lang = _detect_language(cleaned)  # 语言检测：三语支持
    # 三语支持：非中文 query 在检索前做关键词扩展(EN/TH→ZH 同义词),让 BM25 能命中中文 FAQ
    retrieval_query = _expand_query_for_cross_lang(cleaned, lang)
    intent_category, intent_conf, matched = _classify(retrieval_query)

    # 向量路（GPU 本地嵌入）：auto/on/off 语义见 _make_embed_fn
    embed_fn = _make_embed_fn(cfg)
    retrieval_desc = (
        "BM25 + TF-IDF + 本地向量 三路召回" if embed_fn is not None
        else "BM25 + TF-IDF 双路召回"
    )

    # 加载 FAQ：优先用 cfg.data_path；未配置时用内置真实企业 FAQ（docx 解析产物，非演示数据）
    faqs = list(_load_faqs_from_path(cfg.data_path)) if cfg.data_path else list(_load_default_faqs())
    if not faqs:
        return SkillOutput(
            data=FeishuKbReport(
                query=inp.query, cleaned_query=cleaned,
                intent_category=intent_category, intent_confidence=intent_conf,
                matched_keywords=matched, top_k=[],
                agent_reply_hint=f"未加载到任何 FAQ 数据，请配置 {cfg.data_path}。",
                user_language=lang,
                translation_required=lang != "zh",
                translation_directive=_build_translation_directive(lang),
            ),
            reasoning=[_build_chain(inp.query, intent_category, intent_conf, matched, [],
                                    retrieval_desc)],
            confidence=0.0, sample_size=0, degraded=True,
            degradation_reason="FAQ 数据未配置或文件不存在",
        )

    # 检索 + 重排
    candidates = _hybrid_search(
        faqs, retrieval_query, top_n=cfg.retrieval_top_n,
        embed_fn=embed_fn, embed_model=cfg.embed_model, embed_device=cfg.embed_device,
    )
    top_k = _rerank(candidates, retrieval_query=retrieval_query, intent_category=intent_category, top_k=inp.top_k)

    # ★ Hard-gate：意图分类置信度=0（4 类关键词都没命中）→ 必转人工。
    # 这是"机器人不回复多余话题"的核心防线。原因：FAQ 语料里大量"是啥问题？/怎么样？"
    # 等句式，纯噪音查询也会拿到较高 BM25 分数，单靠分数阈值无法稳定拦截。
    # 意图关键词命中是更可靠的"领域内"信号。
    # 三语支持: 非中文查询(英文/泰文)用 BM25 直接对中文语料检索也能命中(关键词翻译近似),
    # 不走"intent_confidence == 0"的拦截,只走分数阈值,让 LLM 层做语言转换。
    if lang == "zh":
        off_topic = intent_conf == 0.0
    else:
        off_topic = False  # 非中文查询的关键词 gate 不适用
    low_score = bool(top_k) and top_k[0].final_score < cfg.min_top1_score
    if off_topic or low_score:
        reason = []
        if off_topic:
            reason.append("意图分类置信度=0（4 类关键词均未命中）")
        if low_score:
            reason.append(f"Top-1 final_score {top_k[0].final_score:.4f} < 阈值 {cfg.min_top1_score}")
        hint = _OFFTOPIC_HINT + _TRANSLATION_HINT.get(lang, "")
        return SkillOutput(
            data=FeishuKbReport(
                query=inp.query,
                cleaned_query=cleaned,
                intent_category=intent_category,
                intent_confidence=intent_conf,
                matched_keywords=matched,
                top_k=[],
                agent_reply_hint=hint,
                user_language=lang,
                translation_required=lang != "zh",
                translation_directive=_build_translation_directive(lang),
            ),
            reasoning=[_build_chain(inp.query, intent_category, intent_conf, matched, [],
                                    retrieval_desc)],
            confidence=0.0,
            sample_size=len(faqs),
            degraded=True,
            degradation_reason="; ".join(reason),
        )

    hint = _agent_reply_hint(inp.query, intent_category, top_k) + _TRANSLATION_HINT.get(lang, "")
    # 兜底:即使没走 hard-gate,只要 top_k 为空也应标 degraded(无任何命中)
    return SkillOutput(
        data=FeishuKbReport(
            query=inp.query,
            cleaned_query=cleaned,
            intent_category=intent_category,
            intent_confidence=intent_conf,
            matched_keywords=matched,
            top_k=top_k,
            agent_reply_hint=hint,
            user_language=lang,
            translation_required=lang != "zh",
            translation_directive=_build_translation_directive(lang),
        ),
        reasoning=[_build_chain(inp.query, intent_category, intent_conf, matched, top_k,
                                retrieval_desc)],
        confidence=intent_conf,
        sample_size=len(faqs),
        degraded=len(top_k) == 0,
        degradation_reason="无任何 FAQ 命中" if not top_k else None,
    )


__all__ = ["feishu_kb_search", "FeishuKbInput", "FeishuKbReport", "FaqItem"]