"""packaging_dieline 技能：刀板几何**独立复核** + 印刷级刀线出图（SVG / DXF / PDF）。

为什么要这一步：包装工坊在浏览器里用 ``clipper-lib`` 算并集与 3mm 外扩出血，
「自己算、自己报」天然缺一道外部校验——数字看着齐，不代表模切厂拿去就能开刀。
本技能把交稿报文（``flowmind.packaging.dieline/1``）里每片料的轮廓用 **Shapely**
再算一遍，面积 / 裁切线长 / 压痕线长 / 出血增量逐条对账；对不上就如实进
``findings``，绝不给前端的数字背书。

出图用现成库，不手写文件格式：DXF 走 ``ezdxf``（模切厂真正要的那份），
PDF 走 ``reportlab``（1:1 毫米），SVG 自己拼（带 ``data-layer`` 分层的 mm 矢量）。
DXF / PDF 是**可选层**：对应库没装就少产一件，并在 ``missing`` 里写清原因，
不静默跳过、也不谎称已出（CLAUDE.md「错误永不静默」）。

产物只含**刀线**（裁切 / 压痕 / 出血 / 角线），不含图稿——图稿层由包装工坊按
配色与矢量文本出，本技能出的是可以直接开刀模的那一层。
"""
from __future__ import annotations

import base64
from io import BytesIO, StringIO
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from flowmind.contracts import Evidence, ReasoningChain, RuleHit, SkillOutput
from flowmind.skill import skill

_VERSION = "0.1.0"
DIELINE_SCHEMA = "flowmind.packaging.dieline/1"

# 与前端同参：clipper-lib 侧是 ClipperOffset(miterLimit=2, arcTolerance=2)。
# 尖角的 miter 超出限值会退化成 bevel，两边必须用同一个限值，否则面积天然对不上。
MITRE_LIMIT = 2.0
_EPS = 1e-6


class _Pt(BaseModel):
    x: float
    y: float


class _Fold(BaseModel):
    id: str = ""
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0


class _Art(BaseModel):
    id: str = ""
    face: str = ""
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0


class _Blank(BaseModel):
    id: str = ""
    label: str = ""
    cut: list[list[_Pt]] = Field(default_factory=list)
    bleed: list[list[_Pt]] = Field(default_factory=list)
    folds: list[_Fold] = Field(default_factory=list)
    art: list[_Art] = Field(default_factory=list)
    areaMm2: float = 0.0
    bleedAreaMm2: float = 0.0
    cutLengthMm: float = 0.0
    foldLengthMm: float = 0.0


class _Page(BaseModel):
    widthMm: float = 0.0
    heightMm: float = 0.0


class _Box(BaseModel):
    lengthMm: float = 0.0
    widthMm: float = 0.0
    heightMm: float = 0.0


class PackagingDielineInput(BaseModel):
    """交稿报文 + 出图选项（``schema`` 用别名接前端字段名）。"""

    model_config = ConfigDict(populate_by_name=True)

    schema_name: str = Field(default=DIELINE_SCHEMA, alias="schema")
    brand: str = ""
    productName: str = ""
    style: str = "straight-tuck"
    bleedMm: float = Field(default=3.0, ge=0.0, le=10.0)
    page: _Page = Field(default_factory=_Page)
    box: _Box | None = None
    blanks: list[_Blank] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=lambda: ["svg", "dxf", "pdf"])
    tolerancePct: float = Field(default=1.0, ge=0.05, le=10.0)
    minFeatureMm: float = Field(default=1.5, ge=0.1, le=10.0)


class Check(BaseModel):
    """一条对账结论。``severity``：blocker / warn / info。"""

    id: str
    label: str
    ok: bool
    severity: str = "info"
    expected: float | None = None
    actual: float | None = None
    deltaPct: float | None = None
    detail: str = ""


class BlankReport(BaseModel):
    id: str = ""
    label: str = ""
    rings: int = 0
    areaMm2: float = 0.0
    bleedAreaMm2: float = 0.0
    cutLengthMm: float = 0.0
    foldLengthMm: float = 0.0
    checks: list[Check] = Field(default_factory=list)


class Artifact(BaseModel):
    kind: str
    fileName: str = ""
    text: str = ""
    base64Data: str = ""
    bytes: int = 0


class PackagingDielineData(BaseModel):
    schemaName: str = DIELINE_SCHEMA
    style: str = ""
    blankCount: int = 0
    page: _Page = Field(default_factory=_Page)
    bleedMm: float = 0.0
    totals: dict[str, float] = Field(default_factory=dict)
    blanks: list[BlankReport] = Field(default_factory=list)
    findings: list[Check] = Field(default_factory=list)
    blockers: int = 0
    warnings: int = 0
    artifacts: list[Artifact] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _shapely() -> dict[str, Any]:
    """Shapely 是本技能的硬前置：缺失即显式报错，不出「没校验过」的图。"""
    try:
        from shapely.geometry import LineString, Polygon  # noqa: PLC0415
        from shapely.ops import unary_union  # noqa: PLC0415
        from shapely.validation import make_valid  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "packaging_dieline 需要 shapely>=2.0（几何复核）："
            "conda env update -n flowmind -f environment.yml，或 pip install 'shapely>=2.0'"
        ) from exc
    return {"LineString": LineString, "Polygon": Polygon,
            "unary_union": unary_union, "make_valid": make_valid}


def _coords(ring: list[_Pt]) -> list[tuple[float, float]]:
    pts = [(float(p.x), float(p.y)) for p in ring]
    if len(pts) >= 2 and pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def _union(sh: dict[str, Any], rings: list[list[_Pt]], repairs: list[str], where: str) -> Any:
    """环集 → 并集几何。无效环先 make_valid 再参与并集，并**记账**（不假装它本来就好）。"""
    polys = []
    for i, ring in enumerate(rings):
        pts = _coords(ring)
        if len(pts) < 4:  # 闭合三角形的首点重复算在内，最少 4 个
            repairs.append(f"{where} 第 {i + 1} 环点数不足，已跳过")
            continue
        geom = sh["Polygon"](pts)
        if not geom.is_valid:
            geom = sh["make_valid"](geom)
            repairs.append(f"{where} 第 {i + 1} 环自相交/退化，已 make_valid 修复后参与并集")
        if geom.is_empty:
            repairs.append(f"{where} 第 {i + 1} 环修复后为空，已跳过")
            continue
        polys.append(geom)
    if not polys:
        return None
    merged = sh["unary_union"](polys) if len(polys) > 1 else polys[0]
    return None if merged.is_empty else merged


def _fold_length(sh: dict[str, Any], folds: list[_Fold]) -> float:
    return sum(sh["LineString"]([(f.x1, f.y1), (f.x2, f.y2)]).length for f in folds)


def _pct(expected: float, actual: float) -> float:
    if expected <= _EPS:
        return 0.0 if actual <= _EPS else 100.0
    return abs(actual - expected) / expected * 100.0


def _compare(cid: str, label: str, expected: float, actual: float,
             tol_pct: float, unit: str) -> Check:
    """报文数字 vs Shapely 独立算出的数字。超容差 = blocker（这份稿不能直接交印厂）。"""
    delta = _pct(expected, actual)
    ok = delta <= tol_pct
    return Check(
        id=cid, label=label, ok=ok, severity="info" if ok else "blocker",
        expected=round(expected, 2), actual=round(actual, 2), deltaPct=round(delta, 3),
        detail=(f"报文 {expected:.2f}{unit} vs 复核 {actual:.2f}{unit}"
                f"（偏差 {delta:.3f}%，容差 {tol_pct:g}%）"),
    )


def _check_blank(sh: dict[str, Any], blank: _Blank, bleed_mm: float,
                 tol_pct: float, min_feature_mm: float) -> tuple[BlankReport, list[Check]]:
    where = blank.label or blank.id or "片料"
    repairs: list[str] = []
    findings: list[Check] = []

    cut = _union(sh, blank.cut, repairs, f"{where} 裁切")
    if cut is None:
        findings.append(Check(id=f"{where}/cut-empty", label="裁切轮廓", ok=False,
                              severity="blocker",
                              detail="这片料没有可用环（前端 Clipper 并集为空？）"))
        return BlankReport(id=blank.id, label=where, checks=findings), findings

    bleed = _union(sh, blank.bleed, repairs, f"{where} 出血")
    rings = len(blank.cut)
    area = round(cut.area, 2)
    cut_len = round(cut.length, 2)
    fold_len = round(_fold_length(sh, blank.folds), 2)
    bleed_area = round(bleed.area if bleed is not None else cut.area, 2)

    findings.append(_compare(f"{where}/area", "用料面积", blank.areaMm2, cut.area,
                             tol_pct, "mm²"))
    findings.append(_compare(f"{where}/cut-length", "裁切线长", blank.cutLengthMm, cut.length,
                             tol_pct, "mm"))
    findings.append(_compare(f"{where}/fold-length", "压痕线长", blank.foldLengthMm,
                             _fold_length(sh, blank.folds), tol_pct, "mm"))

    if bleed_mm > 0 and bleed is not None:
        ideal = cut.buffer(bleed_mm, join_style=2, mitre_limit=MITRE_LIMIT)
        findings.append(_compare(f"{where}/bleed-area", f"出血 {bleed_mm:g}mm 后面积",
                                 bleed.area, ideal.area, tol_pct, "mm²"))
        covers = bleed.buffer(0.01, join_style=2).covers(cut)
        findings.append(Check(
            id=f"{where}/bleed-covers-cut", label="出血包住裁切", ok=covers,
            severity="info" if covers else "blocker",
            detail="出血轮廓完整覆盖裁切轮廓" if covers
                   else "出血轮廓没盖住裁切轮廓——印厂会切到图，必须修"))

    survives = cut.buffer(-min_feature_mm, join_style=2)
    findings.append(Check(
        id=f"{where}/thin-feature", label="最小特征尺寸", ok=not survives.is_empty,
        severity="info" if not survives.is_empty else "warn",
        actual=min_feature_mm,
        detail=(f"内缩 {min_feature_mm:g}mm 后仍有料（没有更窄的条/尖角）"
                if not survives.is_empty else
                f"存在窄于 {min_feature_mm:g}mm 的特征：模切排屑与压痕成型有风险，"
                "请与印厂确认刀模工艺（槽舌、扣底卡口最容易踩）")))

    findings.extend(Check(id=f"{where}/repair", label="几何自愈", ok=False, severity="warn",
                          detail=note) for note in repairs)

    return BlankReport(
        id=blank.id, label=where, rings=rings, areaMm2=area, bleedAreaMm2=bleed_area,
        cutLengthMm=cut_len, foldLengthMm=fold_len, checks=findings,
    ), findings


def _layout_checks(drawn: list[Any], page: _Page) -> list[Check]:
    """跨片料：同版排料不得互相压到；出血轮廓不得越出页面。"""
    findings: list[Check] = []
    geoms = [g for g in drawn if g is not None]
    for i in range(len(geoms)):
        for j in range(i + 1, len(geoms)):
            area = float(geoms[i].intersection(geoms[j]).area)
            findings.append(Check(
                id=f"layout/overlap-{i + 1}-{j + 1}", label=f"片料 {i + 1} × {j + 1} 重叠",
                ok=area <= 0.01, severity="info" if area <= 0.01 else "blocker",
                actual=round(area, 3),
                detail="片料互不压到，可以同版排料" if area <= 0.01
                       else f"轮廓重叠 {area:.2f}mm²——同版会切串，请挪位或分开出片"))

    if page.widthMm > 0 and page.heightMm > 0 and geoms:
        boxes = [g.bounds for g in geoms]
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        inside = x0 >= -0.01 and y0 >= -0.01 and x1 <= page.widthMm + 0.01 and y1 <= page.heightMm + 0.01
        findings.append(Check(
            id="layout/page-bounds", label="页面边界", ok=inside,
            severity="info" if inside else "blocker",
            detail=(f"全部落在 {page.widthMm:g} × {page.heightMm:g} mm 页面内" if inside else
                    f"越出页面：轮廓 bbox ({x0:.2f}, {y0:.2f}) → ({x1:.2f}, {y1:.2f})")))
    return findings


def _art_checks(blanks: list[_Blank]) -> list[Check]:
    """图稿位必须落在出血范围内（否则出血区留白，或图稿被裁掉）。"""
    findings: list[Check] = []
    for blank in blanks:
        if not blank.art or not blank.bleed:
            continue
        xs = [p.x for ring in blank.bleed for p in ring]
        ys = [p.y for ring in blank.bleed for p in ring]
        bad = [a.id or a.face for a in blank.art
               if a.x < min(xs) - 0.01 or a.y < min(ys) - 0.01
               or a.x + a.w > max(xs) + 0.01 or a.y + a.h > max(ys) + 0.01]
        where = blank.label or blank.id or "片料"
        findings.append(Check(
            id=f"{where}/art-in-bleed", label="图稿位落在出血内", ok=not bad,
            severity="info" if not bad else "warn",
            detail="所有图稿位都在出血轮廓范围内" if not bad
                   else f"{len(bad)} 个图稿位越出出血框：{'、'.join(bad[:4])}"))
    return findings


# ────────────────────────────── 出图（现成库优先） ──────────────────────────────

def _d(ring: list[_Pt]) -> str:
    return " ".join(f"{'M' if i == 0 else 'L'}{p.x} {p.y}" for i, p in enumerate(ring))


def _slug(inp: PackagingDielineInput) -> str:
    raw = f"{inp.brand} {inp.productName}".lower()
    out = "".join(ch if ch.isalnum() else "-" for ch in raw).strip("-")
    return out[:60] or "dieline"


def _svg(inp: PackagingDielineInput, title_block: str) -> str:
    """刀线 SVG：mm 单位 1:1，按 data-layer 分层（bleed / fold / cut / marks / title）。"""
    w, h = inp.page.widthMm, inp.page.heightMm
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}mm" height="{h}mm" '
        f'viewBox="0 0 {w} {h}">',
        f"<title>{title_block}</title>",
        '<g fill="none" stroke-width="0.15">',
    ]
    for blank in inp.blanks:
        label = (blank.label or blank.id or "blank").replace('"', "'")
        parts.append(f'<g data-layer="bleed" data-blank="{label}" stroke="#8a8f98">')
        parts.extend(f'<path d="{_d(ring)} Z"/>' for ring in blank.bleed)
        parts.append("</g>")
        parts.append(f'<g data-layer="fold" data-blank="{label}" stroke="#2f6fb5" '
                     'stroke-dasharray="2 1.2">')
        parts.extend(f'<line x1="{f.x1}" y1="{f.y1}" x2="{f.x2}" y2="{f.y2}"/>'
                     for f in blank.folds)
        parts.append("</g>")
        parts.append(f'<g data-layer="cut" data-blank="{label}" stroke="#d1434a" '
                     'stroke-width="0.22">')
        parts.extend(f'<path d="{_d(ring)} Z"/>' for ring in blank.cut)
        parts.append("</g>")
    parts.append("</g>")

    m = max(1.0, min(6.0, w / 6, h / 6))
    marks = []
    for cx, cy in ((0, 0), (w, 0), (0, h), (w, h)):
        sx = -1 if cx == 0 else 1
        sy = -1 if cy == 0 else 1
        marks.append(f'<line x1="{cx + sx * 2}" y1="{cy}" x2="{cx + sx * (2 + m)}" y2="{cy}"/>')
        marks.append(f'<line x1="{cx}" y1="{cy + sy * 2}" x2="{cx}" y2="{cy + sy * (2 + m)}"/>')
    parts.append('<g data-layer="marks" stroke="#000000" stroke-width="0.1">'
                 + "".join(marks) + "</g>")
    parts.append(f'<g data-layer="title"><text x="1" y="{h - 0.6}" font-size="2.6" '
                 f'font-family="Helvetica, Arial, sans-serif">{title_block}</text></g>')
    parts.append("</svg>")
    return "\n".join(parts)


def _dxf(inp: PackagingDielineInput) -> bytes | str:
    """DXF（ezdxf）：CUT / FOLD / BLEED / MARKS 四层，$INSUNITS=4（毫米）。"""
    try:
        import ezdxf  # noqa: PLC0415
    except ImportError:
        return "ezdxf-missing"
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 4
    for layer, color in (("CUT", 1), ("FOLD", 5), ("BLEED", 8), ("MARKS", 250)):
        doc.layers.add(layer, color=color)
    msp = doc.modelspace()
    # DXF 用 Y 向上、SVG/报文用 Y 向下：取负对齐，CAD 里量出来的尺寸才是正的。
    for blank in inp.blanks:
        for ring in blank.cut:
            msp.add_lwpolyline([(p.x, -p.y) for p in ring], close=True,
                               dxfattribs={"layer": "CUT"})
        for ring in blank.bleed:
            msp.add_lwpolyline([(p.x, -p.y) for p in ring], close=True,
                               dxfattribs={"layer": "BLEED"})
        for f in blank.folds:
            msp.add_line((f.x1, -f.y1), (f.x2, -f.y2), dxfattribs={"layer": "FOLD"})
    # ezdxf 写的是文本流（DXF 本身是 ASCII），再统一编码成 bytes 回传
    buf = StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")


def _pdf(inp: PackagingDielineInput) -> bytes | str:
    """PDF（reportlab）：页面尺寸 = 展开尺寸，线宽/虚线全部按毫米换算。"""
    try:
        from reportlab.lib.units import mm  # noqa: PLC0415
        from reportlab.pdfgen import canvas  # noqa: PLC0415
    except ImportError:
        return "reportlab-missing"
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(inp.page.widthMm * mm, inp.page.heightMm * mm))
    h = inp.page.heightMm

    def ring_path(ring: list[_Pt]) -> None:
        p = c.beginPath()
        for i, pt in enumerate(ring):
            x, y = pt.x * mm, (h - pt.y) * mm
            p.moveTo(x, y) if i == 0 else p.lineTo(x, y)
        p.close()
        c.drawPath(p, stroke=1, fill=0)

    for blank in inp.blanks:
        c.setDash()
        c.setStrokeColorRGB(0.54, 0.56, 0.60)
        c.setLineWidth(0.15 * mm)
        for ring in blank.bleed:
            ring_path(ring)
        c.setDash(2 * mm, 1.2 * mm)
        c.setStrokeColorRGB(0.18, 0.44, 0.71)
        for f in blank.folds:
            c.line(f.x1 * mm, (h - f.y1) * mm, f.x2 * mm, (h - f.y2) * mm)
        c.setDash()
        c.setStrokeColorRGB(0.82, 0.26, 0.29)
        c.setLineWidth(0.22 * mm)
        for ring in blank.cut:
            ring_path(ring)
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.1 * mm)
    for cx, cy in ((0, 0), (inp.page.widthMm, 0), (0, h), (inp.page.widthMm, h)):
        sx = -1 if cx == 0 else 1
        sy = 1 if cy == 0 else -1
        c.line((cx + sx * 2) * mm, (h - cy) * mm, (cx + sx * 8) * mm, (h - cy) * mm)
        c.line(cx * mm, (h - cy - sy * 2) * mm, cx * mm, (h - cy - sy * 8) * mm)
    c.setFont("Helvetica", 2.6 * mm)
    c.drawString(1 * mm, 1.6 * mm, "flowmind packaging_dieline · 1:1 mm · 刀线层（无图稿）")
    c.showPage()
    c.save()
    return buf.getvalue()


def _artifact(kind: str, file_name: str, text: str = "", data: bytes = b"") -> Artifact:
    raw = text.encode("utf-8") if text else data
    return Artifact(
        kind=kind, fileName=file_name, text=text,
        base64Data=base64.b64encode(data).decode("ascii") if data else "",
        bytes=len(raw),
    )


# ───────────────────────────────── 技能主体 ─────────────────────────────────

@skill(id="packaging_dieline", name="包装刀板复核与印刷级出图", version=_VERSION,
       category="包装刀板", tags=["packaging", "dieline", "print"])
def packaging_dieline(inp: PackagingDielineInput) -> SkillOutput[PackagingDielineData]:
    """Shapely 独立复核刀板几何，并产出可以直接开刀模的 SVG / DXF / PDF。"""
    if not inp.blanks:
        raise ValueError("报文里没有片料（blanks 为空）——无从校验，拒绝出图")

    sh = _shapely()
    notes: list[str] = []
    if inp.schema_name != DIELINE_SCHEMA:
        notes.append(f"报文 schema 为 {inp.schema_name}，本技能按 {DIELINE_SCHEMA} 解读")

    reports: list[BlankReport] = []
    findings: list[Check] = []
    drawn: list[Any] = []
    for blank in inp.blanks:
        report, local = _check_blank(sh, blank, inp.bleedMm, inp.tolerancePct, inp.minFeatureMm)
        reports.append(report)
        findings.extend(local)
        repairs: list[str] = []
        drawn.append(_union(sh, blank.bleed or blank.cut, repairs, "layout"))

    findings.extend(_layout_checks(drawn, inp.page))
    findings.extend(_art_checks(inp.blanks))

    blockers = [f for f in findings if f.severity == "blocker"]
    warns = [f for f in findings if f.severity == "warn"]

    title_block = (f"{inp.brand} {inp.productName} · {inp.style} · {len(inp.blanks)} 片 · "
                   f"出血 {inp.bleedMm:g}mm").replace("<", "&lt;")
    slug = _slug(inp)
    artifacts: list[Artifact] = []
    missing: list[str] = []
    wanted = [o.lower() for o in inp.outputs]
    if "svg" in wanted:
        artifacts.append(_artifact("svg", f"{slug}-dieline.svg", text=_svg(inp, title_block)))
    if "dxf" in wanted:
        dxf = _dxf(inp)
        if isinstance(dxf, str):
            missing.append("dxf：未安装 ezdxf（environment.yml 已声明；conda env update 或 pip install ezdxf）")
        else:
            artifacts.append(_artifact("dxf", f"{slug}-dieline.dxf", data=dxf))
    if "pdf" in wanted:
        pdf = _pdf(inp)
        if isinstance(pdf, str):
            missing.append("pdf：未安装 reportlab（environment.yml 已声明；conda env update 或 pip install reportlab）")
        else:
            artifacts.append(_artifact("pdf", f"{slug}-dieline.pdf", data=pdf))

    totals = {
        "areaMm2": round(sum(r.areaMm2 for r in reports), 2),
        "bleedAreaMm2": round(sum(r.bleedAreaMm2 for r in reports), 2),
        "cutLengthMm": round(sum(r.cutLengthMm for r in reports), 2),
        "foldLengthMm": round(sum(r.foldLengthMm for r in reports), 2),
    }
    data = PackagingDielineData(
        schemaName=inp.schema_name, style=inp.style, blankCount=len(inp.blanks),
        page=inp.page, bleedMm=inp.bleedMm, totals=totals, blanks=reports,
        findings=findings, blockers=len(blockers), warnings=len(warns),
        artifacts=artifacts, missing=missing, notes=notes,
    )

    conclusion = (f"复核 {len(inp.blanks)} 片料 / {len(findings)} 项检查："
                  f"{len(blockers)} 项阻塞、{len(warns)} 项提醒；出图 {len(artifacts)} 件")
    if blockers:
        conclusion += "（几何与报文对不上，先别交印厂）"
    elif missing:
        conclusion += "（刀线可用，缺 " + "、".join(m.split("：")[0] for m in missing) + "）"

    return SkillOutput(
        data=data,
        reasoning=[ReasoningChain(
            conclusion=conclusion,
            triggered_rules=[
                RuleHit(rule_id="metrics-match", name="报文数字与复核一致",
                        expression=f"abs(报文-复核)/报文 <= {inp.tolerancePct:g}%",
                        hit=any(f.severity == "blocker" and f.deltaPct is not None
                                for f in findings)),
                RuleHit(rule_id="bleed-covers-cut", name="出血包住裁切",
                        expression="bleed.buffer(0.01).covers(cut) == True",
                        hit=any(f.id.endswith("/bleed-covers-cut") and not f.ok
                                for f in findings)),
                RuleHit(rule_id="thin-feature", name="最小特征尺寸",
                        expression=f"cut.buffer(-{inp.minFeatureMm:g}mm) 非空",
                        hit=any(f.id.endswith("/thin-feature") and not f.ok
                                for f in findings)),
            ],
            evidence=[
                Evidence(metric="复核用料面积(mm²)", value=totals["areaMm2"], comparison="=="),
                Evidence(metric="复核裁切线长(mm)", value=totals["cutLengthMm"], comparison="=="),
                Evidence(metric="复核压痕线长(mm)", value=totals["foldLengthMm"], comparison="=="),
                Evidence(metric="阻塞项", value=len(blockers), threshold=0, comparison="<="),
            ],
            causal_analysis="前端 clipper-lib 与后端 Shapely 是两套独立实现，数字一致才说明"
                            "轮廓并集与出血外扩都没写错。本技能只读坐标（毫米）、不改几何："
                            "有偏差就报出来，改刀还是改口径由设计侧决定。",
            risk_note=("存在阻塞项，按 findings 修正后重跑。" if blockers else
                       ("可选格式缺失（见 missing），刀线本身可用。" if missing else
                        "对账通过。槽舌 / 扣底卡口等尺寸仍属示意级，须印厂按刀模工艺复核；"
                        "本技能不替代打样。")),
        )],
        confidence=0.35 if blockers else 1.0,
        sample_size=len(inp.blanks),
        degraded=bool(blockers) or bool(missing),
        degradation_reason=("geometry-mismatch" if blockers
                            else "artifact-skipped" if missing else None),
    )
