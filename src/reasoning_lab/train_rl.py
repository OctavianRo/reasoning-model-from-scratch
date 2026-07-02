"""Reward-tuning demo for verifiable reasoning tasks."""

from __future__ import annotations

from dataclasses import dataclass

from .dataset import Problem
from .evaluate import reward
from .grpo import RolloutGroupReport, sample_rollout_group, summarize_rollouts
from .models import RewardTunedPolicy


@dataclass(frozen=True)
class TrainingEpochReport:
    """Metrics to inspect while refining a reasoning policy."""

    epoch: int
    error_rate: float
    rollout_report: RolloutGroupReport


def tune_with_rewards(
    policy: RewardTunedPolicy,
    problems: list[Problem],
    epochs: int = 5,
    learning_rate: float = 0.08,
) -> list[float]:
    """Tune a policy from binary rewards and return error-rate history."""

    history = [policy.step_error_rate]
    for _ in range(epochs):
        for problem in problems:
            response = policy.solve(problem)
            policy.update_from_reward(
                reward=reward(problem, response),
                learning_rate=learning_rate,
            )
        history.append(policy.step_error_rate)
    return history


def tune_with_grpo_diagnostics(
    policy: RewardTunedPolicy,
    problems: list[Problem],
    epochs: int = 5,
    learning_rate: float = 0.08,
    group_size: int = 6,
) -> list[TrainingEpochReport]:
    """Tune with binary rewards while reporting GRPO-style diagnostics."""

    reports: list[TrainingEpochReport] = []
    for epoch in range(1, epochs + 1):
        epoch_rollouts = []
        for problem in problems:
            epoch_rollouts.extend(
                sample_rollout_group(policy, problem, group_size=group_size)
            )
            response = policy.solve(problem)
            policy.update_from_reward(
                reward=reward(problem, response),
                learning_rate=learning_rate,
            )

        reports.append(
            TrainingEpochReport(
                epoch=epoch,
                error_rate=policy.step_error_rate,
                rollout_report=summarize_rollouts(epoch_rollouts),
            )
        )

    return reports
