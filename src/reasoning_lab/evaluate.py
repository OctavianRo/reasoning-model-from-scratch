"""Evaluation helpers for reasoning experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .dataset import Problem
from .models import Response


class Solver(Protocol):
    def solve(self, problem: Problem) -> Response:
        ...


@dataclass(frozen=True)
class EvalResult:
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


def reward(problem: Problem, response: Response) -> int:
    """Return a verifiable reward for final-answer correctness."""

    return int(response.final_answer == problem.answer)


def evaluate(solver: Solver, problems: list[Problem]) -> EvalResult:
    correct = sum(reward(problem, solver.solve(problem)) for problem in problems)
    return EvalResult(total=len(problems), correct=correct)

