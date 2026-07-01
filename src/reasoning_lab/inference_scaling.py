"""Inference-time scaling with self-consistency voting."""

from __future__ import annotations

from collections import Counter
from typing import Protocol

from .dataset import Problem
from .models import Response


class Solver(Protocol):
    def solve(self, problem: Problem) -> Response:
        ...


def self_consistency(solver: Solver, problem: Problem, samples: int = 7) -> Response:
    """Sample multiple solutions and vote on the final answer."""

    responses = [solver.solve(problem) for _ in range(samples)]
    vote_counts = Counter(response.final_answer for response in responses)
    winning_answer, _ = vote_counts.most_common(1)[0]

    for response in responses:
        if response.final_answer == winning_answer:
            return Response(
                trace=response.trace + f" Vote selected from {samples} samples.",
                final_answer=winning_answer,
            )

    return responses[0]

