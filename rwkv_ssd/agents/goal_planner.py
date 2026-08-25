"""Break down a high-level goal into a task queue for the agent loop."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rwkv_ssd.agents.task_queue import Task, TaskQueue, TaskStatus

logger = logging.getLogger(__name__)

_AGENT_TYPES = {
    "research": "explorer",
    "read": "explorer",
    "find": "explorer",
    "search": "explorer",
    "implement": "fixer",
    "write": "fixer",
    "refactor": "fixer",
    "fix": "fixer",
    "test": "oracle",
    "verify": "oracle",
    "design": "designer",
    "review": "oracle",
}


def _infer_agent_type(description: str) -> str:
    desc_lower = description.lower()
    for keyword, agent_type in _AGENT_TYPES.items():
        if keyword in desc_lower:
            return agent_type
    if any(
        w in desc_lower
        for w in ["how", "what", "why", "compare", "research", "investigate"]
    ):
        return "explorer"
    return "fixer"


def _parse_goal_steps(goal: str) -> list[dict[str, str]]:
    """Parse a goal string into steps. Supports simple markdown lists."""
    steps = []
    for line in goal.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith(">"):
            continue
        for prefix in ["- ", "* ", "1. ", "2. ", "3. ", "4. ", "5. "]:
            if line.startswith(prefix):
                text = line[len(prefix) :].strip()
                if ":" in text and len(text.split(":", 1)[0]) < 20:
                    label, desc = text.split(":", 1)
                    steps.append({"label": label.strip(), "description": desc.strip()})
                else:
                    steps.append({"label": text[:50], "description": text})
                break
    if not steps:
        sentences = [
            s.strip() + "."
            for s in goal.replace("?", ".").split(".")
            if len(s.strip()) > 20
        ]
        for s in sentences[:10]:
            steps.append({"label": s[:50], "description": s})
    return steps


def plan_from_goal(goal: str, goal_id: str | None = None) -> TaskQueue:
    if goal_id is None:
        import hashlib

        goal_id = hashlib.md5(goal.encode()).hexdigest()[:8]

    steps = _parse_goal_steps(goal)
    if not steps:
        steps = [{"label": goal[:50], "description": goal}]

    queue = TaskQueue()
    prev_task_id: str | None = None

    for i, step in enumerate(steps):
        task_id = f"{goal_id}-{i:03d}"
        title = step.get("label", f"Step {i}")
        desc = step.get("description", title)
        agent_type = _infer_agent_type(desc)
        depends = [prev_task_id] if prev_task_id and agent_type != "explorer" else []

        metadata: dict[str, Any] = {}
        if i == 0:
            metadata["phase"] = "research"
        elif agent_type == "explorer":
            metadata["phase"] = "research"
        elif agent_type == "fixer":
            metadata["phase"] = "implementation"
        else:
            metadata["phase"] = "verification"

        task = Task(
            id=task_id,
            goal_id=goal_id,
            title=title,
            description=desc,
            agent_type=agent_type,
            depends_on=depends,
            metadata=metadata,
        )
        queue.add(task)
        prev_task_id = task_id

    return queue


def plan_from_file(path: str | Path) -> tuple[str, TaskQueue]:
    content = Path(path).read_text(encoding="utf-8")
    goal = content.strip()
    goal_id = Path(path).stem
    return goal_id, plan_from_goal(goal, goal_id=goal_id)


def save_goal(goal: str, path: str | Path) -> None:
    Path(path).write_text(goal.strip() + "\n", encoding="utf-8")


def load_goal(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()
