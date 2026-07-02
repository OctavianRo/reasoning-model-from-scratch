"""Self-refinement loop for verifier-guided reasoning."""

from __future__ import annotations

from dataclasses import dataclass

from .dataset import Problem
from .models import Response
from .verifiers import composite_reward, verify_response


@dataclass(frozen=True)
class RefinementStep:
    """One critique-and-revision step."""

    round_number: int
    before: Response
    after: Response
    critique: str
    accepted: bool
    reward_before: float
    reward_after: float


@dataclass(frozen=True)
class RefinementResult:
    """The final answer plus the refinement history."""

    initial: Response
    final: Response
    steps: list[RefinementStep]
    initial_reward: float
    final_reward: float

    @property
    def improved(self) -> bool:
        return self.final_reward > self.initial_reward


def critique_response(problem: Problem, response: Response) -> str:
    """Produce a short, verifier-grounded critique."""

    report = verify_response(problem, response)
    if report.answer_correct and report.trace_consistent:
        return "The final answer and arithmetic trace pass the verifier."
    if not report.answer_correct:
        return "The final answer does not match the verifier's computed answer."
    if not report.trace_consistent:
        return "The trace is missing or inconsistent with the arithmetic steps."
    return "The response format can be improved for easier extraction."


def repair_response(problem: Problem, response: Response, critique: str) -> Response:
    """Create a revised response from verifier feedback."""

    if "does not match" not in critique and "missing or inconsistent" not in critique:
        return response

    after_add = problem.start + problem.add
    trace = (
        f"{response.trace} Revision: recompute from the problem. "
        f"Start with {problem.start}. Add {problem.add} to get {after_add}. "
        f"Subtract {problem.subtract} to get {problem.answer}."
    ).strip()
    return Response(trace=trace, final_answer=problem.answer, confidence=0.95)


def self_refine(problem: Problem, initial: Response, rounds: int = 2) -> RefinementResult:
    """Iteratively revise a response and keep changes that improve verifier reward."""

    current = initial
    steps: list[RefinementStep] = []

    for round_number in range(1, rounds + 1):
        critique = critique_response(problem, current)
        candidate = repair_response(problem, current, critique)
        reward_before = composite_reward(problem, current)
        reward_after = composite_reward(problem, candidate)
        accepted = reward_after >= reward_before
        next_response = candidate if accepted else current
        steps.append(
            RefinementStep(
                round_number=round_number,
                before=current,
                after=next_response,
                critique=critique,
                accepted=accepted,
                reward_before=reward_before,
                reward_after=reward_after,
            )
        )
        current = next_response

    return RefinementResult(
        initial=initial,
        final=current,
        steps=steps,
        initial_reward=composite_reward(problem, initial),
        final_reward=composite_reward(problem, current),
    )
