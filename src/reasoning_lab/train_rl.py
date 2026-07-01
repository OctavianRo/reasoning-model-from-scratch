"""Reward-tuning demo for verifiable reasoning tasks."""

from __future__ import annotations

from .dataset import Problem
from .evaluate import reward
from .models import RewardTunedPolicy


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

