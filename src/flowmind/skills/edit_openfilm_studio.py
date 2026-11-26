"""edit_openfilm_studio：按 project_id 打开 FlowMind 管理的 OpenFilm Studio。

调用方只传项目 id；工作区、CLI 参数、浏览器行为全部由
``flowmind.editing.openfilm_backend`` 决定。Studio 在外部浏览器窗口
打开：本技能启动/复用 Studio 并返回本机地址（默认不擅自拉起浏览器，
``FLOWMIND_OPENFILM_NO_BROWSER=0`` 可交还给 OpenFilm CLI 的默认行为）。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from flowmind.contracts import Evidence, ReasoningChain, SkillOutput
from flowmind.editing.openfilm_backend import (
    OpenFilmError,
    open_studio,
)
from flowmind.skill import skill

_VERSION = "0.1.0"


class StudioInput(BaseModel):
    """打开 Studio 入参：只有 FlowMind 项目 id，不接受路径/命令。"""

    project_id: str = Field(
        min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$",
        description="FlowMind 剪辑项目 id（edit_project 返回）")


class StudioReport(BaseModel):
    project_id: str
    opened: bool = False
    state: str = ""
    studio_url: str | None = None
    project_path: str = ""
    message: str = ""


@skill(id="edit_openfilm_studio", name="OpenFilm Studio 打开",
       version=_VERSION, category="智能剪辑",
       tags=["edit", "openfilm", "studio"])
def edit_openfilm_studio(
        inp: StudioInput) -> SkillOutput[StudioReport]:
    """启动/复用 OpenFilm Studio 并打开项目，返回外部浏览器地址。"""
    try:
        result = open_studio(inp.project_id)
    except OpenFilmError as exc:
        return SkillOutput(
            data=StudioReport(
                project_id=inp.project_id, opened=False,
                message=str(exc)[:1000]),
            reasoning=[ReasoningChain(
                conclusion="OpenFilm Studio 未打开",
                causal_analysis=str(exc)[:500],
                risk_note=(
                    "OpenFilm 是可选能力：安装 OpenFilm/Node 并配置 "
                    "FLOWMIND_OPENFILM_CLI 后重试；不影响 CyberCut/旧流水线"
                    if exc.category == "environment" else
                    "确认项目由 OpenFilm 后端创建且工作区含 film.html"),
                confidence=1.0)],
            confidence=1.0, degraded=True,
            degradation_reason=exc.category)
    return SkillOutput(
        data=StudioReport(
            project_id=inp.project_id, opened=True,
            state=result.get("state", ""),
            studio_url=result.get("studio_url"),
            project_path=result.get("workspace", ""),
            message=("Studio 已启动并打开项目；在外部浏览器打开 studio_url。"
                     "Studio 与 FlowMind 共享同一项目工作区，人工改动会落盘")),
        reasoning=[ReasoningChain(
            conclusion="OpenFilm Studio 已就绪",
            evidence=[Evidence(metric="cli_duration_s",
                               value=str(result.get("cli_duration_s", "")),
                               comparison=">=", window="本次调用")],
            causal_analysis=(
                "edit_openfilm_studio(project_id) → 校验项目/工作区 → "
                "运行 openfilm open <workspace>（参数数组、shell=False）→ "
                "校验返回项目路径是本工作区、Studio 地址是本机 → 记录地址"),
            risk_note=("Studio 是 OpenFilm 的本机单实例服务；同一台机器上"
                       "多个 FlowMind 项目会复用它并在页面间切换"),
            confidence=1.0)],
        confidence=1.0, sample_size=1,
    )
