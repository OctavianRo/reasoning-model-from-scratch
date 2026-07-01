"""Teacher-to-student distillation demo."""

from __future__ import annotations

from .dataset import Problem
from .models import DistilledPolicy, TraceReasoningPolicy


def distill_teacher(
    teacher: TraceReasoningPolicy,
    problems: list[Problem],
) -> DistilledPolicy:
    """Create a compact student from teacher traces."""

    student = DistilledPolicy()
    for problem in problems:
        student.learn(problem, teacher.solve(problem))
    return student

