"""MCP 暴露层：遍历注册表，把每个技能登记为一个 MCP 工具。

skills 融合 mcp——技能只需 @skill 定义，无需改动本文件即被暴露。
"""
from __future__ import annotations

import anyio
from mcp.server.fastmcp import FastMCP

import flowmind.skills  # noqa: F401  触发技能注册
from flowmind.contracts import SkillResult
from flowmind.skill import invoke, registry

mcp = FastMCP("FlowMind Skills")


def _make_tool(spec):
    """为一个技能构造 MCP 工具函数（输入=其 pydantic 模型，输出=SkillResult）。

    动态工具注册依赖 FastMCP v1 的内部协议：通过覆写 ``tool.__annotations__``
    让 FastMCP 从注解推断入参 schema，再以 ``server.add_tool`` 显式登记。
    升级到 FastMCP v2 前需重新审视这些反射点；当前由 pyproject 中
    ``mcp>=1.27,<2`` 锁版本保护。

    技能体统一调度到工作线程执行：
      - HTTP 传输下事件循环不可阻塞——登录（5 分钟等待）、浏览器抓取等长任务
        若在 loop 线程执行会卡死全部并发请求；
      - Playwright Sync API / httpx 等同步库在无 loop 的工作线程中可正常运行。
    """
    input_model = spec.input_model
    skill_id = spec.id

    async def tool(inp) -> SkillResult:
        raw = inp.model_dump() if hasattr(inp, "model_dump") else dict(inp)
        return await anyio.to_thread.run_sync(lambda: invoke(skill_id, raw))

    # 让 FastMCP 从注解推断输入 schema 与返回类型
    tool.__name__ = skill_id
    tool.__doc__ = spec.name
    tool.__annotations__ = {"inp": input_model, "return": SkillResult}
    return tool


def register_all(server: FastMCP) -> None:
    """把注册表中所有技能登记为 MCP 工具。

    FastMCP v1 的 ``add_tool`` 接受可调用对象并按其签名/注解构建工具 schema，
    本流程假设该签名在 v1.x 范围内稳定（见 pyproject ``mcp>=1.27,<2``）。
    """
    for spec in registry().values():
        server.add_tool(_make_tool(spec), name=spec.id, description=spec.name)


register_all(mcp)


def main() -> None:
    """flowmind-mcp 入口：以 stdio 传输启动 MCP 服务器。"""
    mcp.run()


if __name__ == "__main__":
    main()
