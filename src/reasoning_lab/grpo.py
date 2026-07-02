"""Small GRPO-style accounting for grouped reasoning rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Protocol

from .dataset import Problem
from .models import Response
from .verifiers import composite_reward, verify_response


class Solver(Protocol):
    def solve(self, problem: Problem) -> Response:
        ...


@dataclass(frozen=True)
class Rollout:
    """One sampled completion for a prompt."""

    problem: Problem
    response: Response
    reward: float
    advantage: float
    old_logprob: float
    new_logprob: float

    @property
    def policy_ratio(self) -> float:
        return math.exp(self.new_logprob - self.old_logprob)

    @property
    def clipped_ratio(self) -> float:
        return min(max(self.policy_ratio, 0.8), 1.2)

    @property
    def clipped_objective(self) -> float:
        unclipped = self.policy_ratio * self.advantage
        clipped = self.clipped_ratio * self.advantage
        return min(unclipped, clipped)


@dataclass(frozen=True)
class RolloutGroupReport:
    """Diagnostics that make grouped policy optimization explainable."""

    rollout_count: int
    mean_reward: float
    reward_std: float
    mean_advantage: float
    advantage_std: float
    mean_entropy_proxy: float
    mean_response_length: float
    mean_policy_ratio: float
    mean_clipped_objective: float
    mean_kl_proxy: float
    format_pass_rate: float
    trace_consistency_rate: float


def sequence_logprob_proxy(response: Response) -> float:
    """Approximate a sequence log probability from confidence and length."""

    confidence = min(max(response.confidence, 0.01), 0.99)
    length_penalty = 0.015 * len(response.render().split())
    return math.log(confidence) - length_penalty


def entropy_proxy(response: Response) -> float:
    """Approximate uncertainty from a response confidence score."""

    probability = min(max(response.confidence, 0.01), 0.99)
    return -(
        probability * math.log(probability)
        + (1.0 - probability) * math.log(1.0 - probability)
    )


def sample_rollout_group(
    solver: Solver,
    problem: Problem,
    group_size: int = 6,
    reference_confidence: float = 0.55,
) -> list[Rollout]:
    """Sample a group and normalize rewards into advantages."""

    responses = [solver.solve(problem) for _ in range(group_size)]
    rewards = [composite_reward(problem, response) for response in responses]
    mean_reward = statistics.fmean(rewards)
    reward_std = statistics.pstdev(rewards) or 1.0
    old_logprob = math.log(reference_confidence)

    return [
        Rollout(
            problem=problem,
            response=response,
            reward=reward_value,
            advantage=(reward_value - mean_reward) / reward_std,
            old_logprob=old_logprob,
            new_logprob=sequence_logprob_proxy(response),
        )
        for response, reward_value in zip(responses, rewards)
    ]


def summarize_rollouts(rollouts: list[Rollout]) -> RolloutGroupReport:
    """Summarize reward, advantage, entropy, clipping, and format diagnostics."""

    if not rollouts:
        return RolloutGroupReport(
            rollout_count=0,
            mean_reward=0.0,
            reward_std=0.0,
            mean_advantage=0.0,
            advantage_std=0.0,
            mean_entropy_proxy=0.0,
            mean_response_length=0.0,
            mean_policy_ratio=0.0,
            mean_clipped_objective=0.0,
            mean_kl_proxy=0.0,
            format_pass_rate=0.0,
            trace_consistency_rate=0.0,
        )

    reports = [
        verify_response(rollout.problem, rollout.response)
        for rollout in rollouts
    ]
    rewards = [rollout.reward for rollout in rollouts]
    advantages = [rollout.advantage for rollout in rollouts]
    lengths = [len(rollout.response.render().split()) for rollout in rollouts]
    policy_ratios = [rollout.policy_ratio for rollout in rollouts]
    clipped_objectives = [rollout.clipped_objective for rollout in rollouts]
    kl_values = [
        rollout.old_logprob - rollout.new_logprob
        for rollout in rollouts
    ]

    return RolloutGroupReport(
        rollout_count=len(rollouts),
        mean_reward=statistics.fmean(rewards),
        reward_std=statistics.pstdev(rewards),
        mean_advantage=statistics.fmean(advantages),
        advantage_std=statistics.pstdev(advantages),
        mean_entropy_proxy=statistics.fmean(
            entropy_proxy(rollout.response) for rollout in rollouts
        ),
        mean_response_length=statistics.fmean(lengths),
        mean_policy_ratio=statistics.fmean(policy_ratios),
        mean_clipped_objective=statistics.fmean(clipped_objectives),
        mean_kl_proxy=statistics.fmean(kl_values),
        format_pass_rate=statistics.fmean(
            report.final_answer_extractable for report in reports
        ),
        trace_consistency_rate=statistics.fmean(
            report.trace_consistent for report in reports
        ),
    )

