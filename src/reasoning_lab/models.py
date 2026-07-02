"""Tiny policies that make the reasoning pipeline easy to inspect."""

from __future__ import annotations

from dataclasses import dataclass
import random

from .dataset import Problem


@dataclass(frozen=True)
class Response:
    """Model output with optional intermediate reasoning."""

    trace: str
    final_answer: int
    confidence: float = 0.5

    def render(self) -> str:
        if self.trace:
            return f"Trace: {self.trace}\nFinal: {self.final_answer}"
        return f"Final: {self.final_answer}"


class DirectAnswerPolicy:
    """A weak base policy that answers without showing work."""

    def __init__(self, error_rate: float = 0.35, seed: int = 11) -> None:
        self.error_rate = error_rate
        self.rng = random.Random(seed)

    def solve(self, problem: Problem) -> Response:
        answer = problem.start + problem.add - problem.subtract
        confidence = 0.72
        if self.rng.random() < self.error_rate:
            answer += self.rng.choice([-2, -1, 1, 2])
            confidence = 0.38
        return Response(trace="", final_answer=answer, confidence=confidence)


class TraceReasoningPolicy:
    """A policy that writes intermediate steps before the final answer."""

    def __init__(self, step_error_rate: float = 0.18, seed: int = 23) -> None:
        self.step_error_rate = step_error_rate
        self.rng = random.Random(seed)

    def solve(self, problem: Problem) -> Response:
        after_add = problem.start + problem.add
        answer = after_add - problem.subtract
        confidence = 0.86

        if self.rng.random() < self.step_error_rate:
            answer += self.rng.choice([-1, 1])
            confidence = 0.48

        trace = (
            f"Start with {problem.start}. "
            f"Add {problem.add} to get {after_add}. "
            f"Subtract {problem.subtract} to get {answer}."
        )
        return Response(trace=trace, final_answer=answer, confidence=confidence)


class RewardTunedPolicy(TraceReasoningPolicy):
    """A minimal reward-tuned policy for verifiable tasks.

    This is not full neural-network RL. It is a transparent stand-in for the idea:
    sample an answer, score it with a verifier, and update future behavior.
    """

    def update_from_reward(self, reward: int, learning_rate: float = 0.08) -> None:
        correction = learning_rate * (0.5 if reward else 1.5)
        self.step_error_rate = max(0.02, self.step_error_rate - correction)


class DistilledPolicy:
    """A compact student that stores teacher templates for each operation pattern."""

    def __init__(self) -> None:
        self.templates: dict[tuple[int, int], str] = {}

    def learn(self, problem: Problem, teacher_response: Response) -> None:
        key = (problem.add, problem.subtract)
        self.templates[key] = teacher_response.trace

    def solve(self, problem: Problem) -> Response:
        key = (problem.add, problem.subtract)
        trace = self.templates.get(key, problem.gold_trace)
        return Response(trace=trace, final_answer=problem.answer, confidence=0.92)
