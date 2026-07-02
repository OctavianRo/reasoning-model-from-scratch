"""Evaluation helpers for reasoning experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .dataset import Problem
from .models import Response
from .verifiers import VerificationReport, verify_response


class Solver(Protocol):
    def solve(self, problem: Problem) -> Response:
        ...


@dataclass(frozen=True)
class EvalResult:
    total: int
    correct: int
    trace_consistent: int = 0
    extractable: int = 0
    average_reward: float = 0.0

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    @property
    def trace_consistency_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.trace_consistent / self.total

    @property
    def extractable_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.extractable / self.total


def reward(problem: Problem, response: Response) -> int:
    """Return a verifiable reward for final-answer correctness."""

    return int(response.final_answer == problem.answer)


def evaluate_response(problem: Problem, response: Response) -> VerificationReport:
    """Return the full verification breakdown for one response."""

    return verify_response(problem, response)


def evaluate(solver: Solver, problems: list[Problem]) -> EvalResult:
    reports = [evaluate_response(problem, solver.solve(problem)) for problem in problems]
    total_reward = sum(report.total_reward for report in reports)
    return EvalResult(
        total=len(problems),
        correct=sum(report.answer_correct for report in reports),
        trace_consistent=sum(report.trace_consistent for report in reports),
        extractable=sum(report.final_answer_extractable for report in reports),
        average_reward=total_reward / len(reports) if reports else 0.0,
    )
