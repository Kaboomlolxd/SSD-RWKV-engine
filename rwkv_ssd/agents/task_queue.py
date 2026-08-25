"""File-based task queue for the autonomous development loop."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    id: str
    goal_id: str
    title: str
    description: str
    agent_type: str  # "explorer" | "librarian" | "fixer" | "oracle" | "designer"
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: str | None = None
    error: str | None = None
    depends_on: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class TaskQueue:
    tasks: list[Task] = field(default_factory=list)
    path: str | None = None

    def add(self, task: Task) -> None:
        self.tasks.append(task)
        self._save()

    def get(self, task_id: str) -> Task | None:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    def update(self, task_id: str, **kwargs: Any) -> Task | None:
        t = self.get(task_id)
        if t is None:
            return None
        for k, v in kwargs.items():
            if k == "status":
                v = TaskStatus(v) if isinstance(v, str) else v
            setattr(t, k, v)
        t.updated_at = time.time()
        self._save()
        return t

    def pending(self) -> list[Task]:
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def ready(self) -> list[Task]:
        completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        return [
            t
            for t in self.tasks
            if t.status == TaskStatus.PENDING
            and all(dep in completed_ids for dep in t.depends_on)
        ]

    def in_progress(self) -> list[Task]:
        return [t for t in self.tasks if t.status == TaskStatus.IN_PROGRESS]

    def failed(self) -> list[Task]:
        return [t for t in self.tasks if t.status == TaskStatus.FAILED]

    def summary(self) -> dict[str, int]:
        return {
            "total": len(self.tasks),
            "pending": len(self.pending()),
            "in_progress": len(self.in_progress()),
            "completed": len(
                [t for t in self.tasks if t.status == TaskStatus.COMPLETED]
            ),
            "failed": len(self.failed()),
            "cancelled": len(
                [t for t in self.tasks if t.status == TaskStatus.CANCELLED]
            ),
        }

    def _save(self) -> None:
        if self.path:
            data = {
                "tasks": [t.to_dict() for t in self.tasks],
                "summary": self.summary(),
            }
            Path(self.path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_queue(path: str | Path) -> TaskQueue:
    p = Path(path)
    if not p.is_file():
        return TaskQueue(path=str(p))
    data = json.loads(p.read_text(encoding="utf-8"))
    tasks = []
    for td in data.get("tasks", []):
        td["status"] = TaskStatus(td["status"])
        tasks.append(Task(**td))
    q = TaskQueue(tasks=tasks, path=str(p))
    q._save()
    return q


def save_queue(queue: TaskQueue, path: str | Path) -> None:
    queue.path = str(path)
    queue._save()
