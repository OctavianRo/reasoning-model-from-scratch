"""Teacher-to-student distillation demo."""

from __future__ import annotations

from dataclasses import dataclass

from .dataset import Problem
from .evaluate import evaluate
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


@dataclass(frozen=True)
class DistillationReport:
    """Numerical metrics that make distillation easy to explain."""

    baseline_accuracy: float
    teacher_accuracy: float
    student_accuracy: float
    teacher_calls_before: int
    teacher_calls_after: int

    @property
    def accuracy_gain(self) -> float:
        return self.student_accuracy - self.baseline_accuracy

    @property
    def teacher_retention(self) -> float:
        if self.teacher_accuracy == 0:
            return 0.0
        return self.student_accuracy / self.teacher_accuracy

    @property
    def teacher_call_reduction(self) -> float:
        if self.teacher_calls_before == 0:
            return 0.0
        saved = self.teacher_calls_before - self.teacher_calls_after
        return saved / self.teacher_calls_before


def measure_distillation(
    baseline,
    teacher: TraceReasoningPolicy,
    student: DistilledPolicy,
    eval_problems: list[Problem],
) -> DistillationReport:
    """Compare the distilled student against a baseline and its teacher."""

    return DistillationReport(
        baseline_accuracy=evaluate(baseline, eval_problems).accuracy,
        teacher_accuracy=evaluate(teacher, eval_problems).accuracy,
        student_accuracy=evaluate(student, eval_problems).accuracy,
        teacher_calls_before=len(eval_problems),
        teacher_calls_after=0,
    )
