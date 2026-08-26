"""A2A Task 存储（内存实现）。

按 task_id 索引任务，支持 save / get / cancel。
简单 dict + asyncio.Lock 保证并发安全。未来可替换为持久化后端
（Redis / DB），只需保持 TaskStore 接口不变。
"""
from __future__ import annotations

import asyncio


class TaskStore:
    """任务存储（内存实现，并发安全）。"""

    def __init__(self) -> None:
        self._tasks: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def save(self, task: dict) -> None:
        """保存任务（按 task_id 索引）。"""
        async with self._lock:
            self._tasks[task["id"]] = task

    async def get(self, task_id: str) -> dict | None:
        """获取任务。不存在返回 None。"""
        async with self._lock:
            return self._tasks.get(task_id)

    async def cancel(self, task_id: str) -> dict | None:
        """取消任务（将状态置为 canceled）。

        Returns:
            更新后的任务字典；不存在返回 None。
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task["status"]["state"] = "canceled"
            return task

    async def clear(self) -> None:
        """清空存储（测试用）。"""
        async with self._lock:
            self._tasks.clear()


# 模块级单例（服务器运行时共享）
_store = TaskStore()


async def save_task(task: dict) -> None:
    """保存任务到全局存储。"""
    await _store.save(task)


async def get_task(task_id: str) -> dict | None:
    """从全局存储获取任务。"""
    return await _store.get(task_id)


async def cancel_task(task_id: str) -> dict | None:
    """在全局存储中取消任务。"""
    return await _store.cancel(task_id)


async def clear_store() -> None:
    """清空全局存储（测试用）。"""
    await _store.clear()
