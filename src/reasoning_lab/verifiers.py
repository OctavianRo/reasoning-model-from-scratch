"""Verifiers and reward functions for reasoning-model refinement."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .dataset import Problem
from .models import Response


FINAL_ANSWER_PATTERN = re.compile(r"Final:\s*(-?\d+)")


@dataclass(frozen=True)
class VerificationReport:
    """Breakdown of the checks used to score a reasoning response."""

    answer_correct: bool
    final_answer_extractable: bool
    trace_consistent: bool
    has_reasoning_trace: bool
    response_length: int

    @property
    def total_reward(self) -> float:
        """Combine correctness, format, and trace quality into one reward."""

        reward = 0.0
        reward += 1.0 if self.answer_correct else 0.0
        reward += 0.15 if self.final_answer_extractable else -0.15
        reward += 0.20 if self.trace_consistent else -0.05
        reward += 0.10 if self.has_reasoning_trace else 0.0
        return reward


def extract_final_answer(rendered_response: str) -> int | None:
    """Extract the final answer from a rendered model response."""

    match = FINAL_ANSWER_PATTERN.search(rendered_response)
    if not match:
        return None
    return int(match.group(1))


def trace_matches_arithmetic(problem: Problem, response: Response) -> bool:
    """Check whether the visible trace follows the known arithmetic steps."""

    if not response.trace:
        return False

    after_add = problem.start + problem.add
    required_fragments = [
        f"Start with {problem.start}",
        f"Add {problem.add} to get {after_add}",
        f"Subtract {problem.subtract} to get {response.final_answer}",
    ]
    return all(fragment in response.trace for fragment in required_fragments)


def verify_response(problem: Problem, response: Response) -> VerificationReport:
    """Verify a response with the same kinds of signals used in RLVR."""

    extracted = extract_final_answer(response.render())
    return VerificationReport(
        answer_correct=response.final_answer == problem.answer,
        final_answer_extractable=extracted is not None,
        trace_consistent=trace_matches_arithmetic(problem, response),
        has_reasoning_trace=bool(response.trace),
        response_length=len(response.render().split()),
    )


def composite_reward(problem: Problem, response: Response) -> float:
    """Reward used by the richer refinement demos."""

    return verify_response(problem, response).total_reward

