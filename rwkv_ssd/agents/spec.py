"""Agent task specifications — the contract between the orchestrator and sub-agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentSpec:
    """A task specification that gets passed to a sub-agent via the Task tool."""

    task_id: str
    goal_id: str
    agent_type: str
    title: str
    prompt: str
    context: dict[str, Any] = field(default_factory=dict)
    expected_output: str = "implementation"
    verify_command: str | None = None

    def for_task_tool(self) -> dict[str, Any]:
        return {
            "description": self.title[:50],
            "prompt": self._build_prompt(),
            "subagent_type": self.agent_type,
        }

    def _build_prompt(self) -> str:
        lines = [
            f"## Task: {self.title}",
            f"Task ID: {self.task_id}",
            f"Goal: {self.goal_id}",
            "",
            self.prompt,
        ]
        if self.expected_output == "implementation":
            lines.extend(
                [
                    "",
                    "### Expected output:",
                    "- Working code that passes existing tests",
                    "- Follow existing code conventions (typing, imports, error handling)",
                    "- Do NOT add comments unless they explain non-obvious design decisions",
                    "- Run `python -m pytest tests/ -x -q` to verify before returning",
                ]
            )
        if self.verify_command:
            lines.extend(["", "### Verification:", self.verify_command])
        if self.context:
            c = self.context
            if "files" in c:
                lines.extend(
                    ["", "### Context files:", *[f"- {f}" for f in c["files"]]]
                )
        return "\n".join(lines)


@dataclass
class ResearchSpec(AgentSpec):
    agent_type: str = "explorer"

    def __post_init__(self) -> None:
        self.expected_output = "research"


@dataclass
class ImplementSpec(AgentSpec):
    agent_type: str = "fixer"

    def __post_init__(self) -> None:
        self.expected_output = "implementation"


@dataclass
class TestSpec(AgentSpec):
    agent_type: str = "oracle"

    def __post_init__(self) -> None:
        self.expected_output = "verification"
