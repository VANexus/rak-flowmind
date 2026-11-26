"""packaging_dieline 技能演示 —— 刀板几何复核（Shapely）+ 刀线 SVG / DXF / PDF 出图。

运行：PYTHONPATH=<repo>/src conda run -n flowmind python examples/packaging_dieline_demo.py
（本机没 conda 时：PYTHONPATH=<repo>/src python3 examples/packaging_dieline_demo.py，
  需要 shapely；ezdxf / reportlab 缺席只会少产对应格式，并在 ``missing`` 里如实说明。）

展示：
1. discover() 自动字段发现
2. happy path：两个相贴矩形并集成 40×100 一片料 → 面积/线长/出血全部对得上
3. 数字造假：把报文面积写大 25% → 如实 blocker（技能不给前端背书）
4. 排料重叠：两片料压到一起 → blocker（同版会切串）

报文形状与 cross-dashboard 包装工坊导出的 ``flowmind.packaging.dieline/1`` 一致：
坐标是页面毫米（Y 轴向下），``cut`` / ``bleed`` 是**并集后**的闭合环。
"""

from __future__ import annotations

import flowmind.skills  # noqa: F401  触发 @skill 注册
from flowmind.discover import field_names
from flowmind.skill import invoke


def section(title: str) -> None:
    print(f"\n{'─' * 64}\n  {title}\n{'─' * 64}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def rect(x: float, y: float, w: float, h: float) -> list[dict[str, float]]:
    """矩形闭合环（与前端 DielinePoint[][] 同形：首尾点重复）。"""
    pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return [{"x": px, "y": py} for px, py in pts] + [{"x": x, "y": y}]


def fold(x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
    return {"id": "f", "x1": x1, "y1": y1, "x2": x2, "y2": y2}


# 一片料：盒身 40×80 (10,10) + 插舌 40×20 (10,90)，共享 y=90 那条边
#   并集 = 40×100 矩形 ⇒ area 4000、裁切线长 280、压痕 40
#   出血 3mm（mitre）⇒ 46×106 = 4876
BODY = rect(10, 10, 40, 80)
FLAP = rect(10, 90, 40, 20)
BLEED = rect(7, 7, 46, 106)


def blank(**over) -> dict:
    base = {
        "id": "blank/main",
        "label": "主体",
        "cut": [BODY, FLAP],
        "bleed": [BLEED],
        "folds": [fold(10, 90, 50, 90)],
        "art": [{"id": "front", "face": "front", "x": 10, "y": 10, "w": 40, "h": 80}],
        "areaMm2": 4000.0,
        "bleedAreaMm2": 4876.0,
        "cutLengthMm": 280.0,
        "foldLengthMm": 40.0,
    }
    base.update(over)
    return base


def payload(**over) -> dict:
    base = {
        "schema": "flowmind.packaging.dieline/1",
        "brand": "LUMÉ",
        "productName": "Glass Veil Lip Gloss",
        "style": "straight-tuck",
        "bleedMm": 3.0,
        "page": {"widthMm": 66, "heightMm": 126},
        "box": {"lengthMm": 40, "widthMm": 25, "heightMm": 80},
        "blanks": [blank()],
    }
    base.update(over)
    return base


def report(data) -> None:
    print(f"  片料={data.blankCount} 出血={data.bleedMm}mm "
          f"阻塞={data.blockers} 提醒={data.warnings}")
    print(f"  totals={data.totals}")
    for item in data.artifacts:
        print(f"  产物 {item.kind:<4} {item.fileName} ({item.bytes} B)")
    for miss in data.missing:
        print(f"  缺 {miss}")
    for f in data.findings:
        if f.severity != "info":
            print(f"  [{f.severity}] {f.label}：{f.detail}")


def main() -> None:
    section("0) discover('packaging_dieline') —— Agent 自查字段")
    for path, names in field_names("packaging_dieline").items():
        print(f"  {path}: {names}")

    section("1) Happy path：两环并集 = 40×100，面积/线长/出血逐条对上")
    r = invoke("packaging_dieline", payload())
    report(r.data)
    check(r.ok, "技能应正常返回")
    check(r.data.blockers == 0, f"不该有阻塞项：{[f.id for f in r.data.findings if f.severity == 'blocker']}")
    check(abs(r.data.totals["areaMm2"] - 4000.0) < 0.01, "并集面积应为 4000（不是 4000+800=两环相加）")
    check(abs(r.data.totals["cutLengthMm"] - 280.0) < 0.01, "裁切线长应是并集轮廓 280，不是各环相加 520")
    check(any(a.kind == "svg" for a in r.data.artifacts), "SVG 必出（纯文本，无第三方依赖）")
    print(f"  degraded={r.metrics.degraded} reason={r.metrics.degradation_reason}"
          "（缺 DXF/PDF 时这里会如实为 True）")

    section("2) 数字造假：报文面积写大 25% → 必须 blocker，不给前端背书")
    fake = payload(blanks=[blank(areaMm2=5000.0)])
    r2 = invoke("packaging_dieline", fake)
    report(r2.data)
    check(r2.data.blockers >= 1, "面积对不上必须报阻塞")
    check(r2.metrics.degraded is True, "有阻塞项即 degraded")
    check(r2.metrics.degradation_reason == "geometry-mismatch", "降级原因应写明几何不符")

    section("3) 排料重叠：第二片料压到第一片 → blocker")
    overlapped = payload(
        page={"widthMm": 120, "heightMm": 126},
        blanks=[blank(), blank(id="blank/tray", label="内托",
                               cut=[rect(20, 20, 30, 30)], bleed=[rect(17, 17, 36, 36)],
                               folds=[], art=[], areaMm2=900.0, bleedAreaMm2=1296.0,
                               cutLengthMm=120.0, foldLengthMm=0.0)],
    )
    r3 = invoke("packaging_dieline", overlapped)
    report(r3.data)
    check(any(f.id.startswith("layout/overlap") and f.severity == "blocker"
              for f in r3.data.findings), "两片料压到一起必须报出来")

    section("4) 窄特征：2mm 高的条 → 提醒印厂复核刀模工艺")
    thin = payload(
        blanks=[blank(cut=[rect(10, 10, 40, 2)], bleed=[rect(7, 7, 46, 8)],
                      folds=[], art=[],
                      areaMm2=80.0, bleedAreaMm2=368.0, cutLengthMm=84.0, foldLengthMm=0.0)],
    )
    r4 = invoke("packaging_dieline", thin)
    report(r4.data)
    check(any(f.id.endswith("/thin-feature") and not f.ok for f in r4.data.findings),
          "窄于 minFeatureMm 的特征应提醒")

    print("\nPASS  packaging_dieline：并集复核 / 数字造假拦截 / 排料重叠 / 窄特征提醒")


if __name__ == "__main__":
    main()
