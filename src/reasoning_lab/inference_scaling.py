"""Inference-time scaling with self-consistency voting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from .dataset import Problem
from .models import Response
from .verifiers import composite_reward


class Solver(Protocol):
    def solve(self, problem: Problem) -> Response:
        ...


@dataclass(frozen=True)
class CandidateSet:
    """The sampled responses considered during inference-time scaling."""

    responses: list[Response]
    selected: Response
    selection_method: str

    @property
    def answer_votes(self) -> dict[int, int]:
        return dict(Counter(response.final_answer for response in self.responses))


def self_consistency(solver: Solver, problem: Problem, samples: int = 7) -> Response:
    """Sample multiple solutions and vote on the final answer."""

    return sample_self_consistency(solver, problem, samples).selected


def sample_self_consistency(
    solver: Solver,
    problem: Problem,
    samples: int = 7,
) -> CandidateSet:
    """Return the full candidate set behind self-consistency voting."""

    responses = [solver.solve(problem) for _ in range(samples)]
    vote_counts = Counter(response.final_answer for response in responses)
    winning_answer, _ = vote_counts.most_common(1)[0]

    for response in responses:
        if response.final_answer == winning_answer:
            selected = Response(
                trace=response.trace + f" Vote selected from {samples} samples.",
                final_answer=winning_answer,
                confidence=response.confidence,
            )
            return CandidateSet(
                responses=responses,
                selected=selected,
                selection_method="majority_vote",
            )

    return CandidateSet(
        responses=responses,
        selected=responses[0],
        selection_method="fallback_first",
    )


def best_of_n(solver: Solver, problem: Problem, samples: int = 7) -> CandidateSet:
    """Select the candidate with the highest verifier-style reward."""

    responses = [solver.solve(problem) for _ in range(samples)]
    selected = max(
        responses,
        key=lambda response: (
            composite_reward(problem, response),
            response.confidence,
            -len(response.render().split()),
        ),
    )
    return CandidateSet(
        responses=responses,
        selected=Response(
            trace=selected.trace + f" Best-of-{samples} selected by verifier score.",
            final_answer=selected.final_answer,
            confidence=selected.confidence,
        ),
        selection_method="best_of_n",
    )
