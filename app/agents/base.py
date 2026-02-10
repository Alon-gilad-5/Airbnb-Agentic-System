"""Base agent interfaces and shared result types."""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas import StepLog


@dataclass
class AgentResult:
    """Standard output returned by all domain agents."""

    response: str
    steps: list[StepLog]


class Agent:
    """Minimal interface that future agents should implement."""

    name: str

    def run(self, prompt: str, context: dict[str, object] | None = None) -> AgentResult:
        """Execute an agent with prompt and optional structured context."""

        raise NotImplementedError
