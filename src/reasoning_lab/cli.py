"""Command line walkthrough for the reasoning lab."""

from __future__ import annotations

import argparse

from .dataset import make_dataset
from .distill import distill_teacher, measure_distillation
from .evaluate import evaluate
from .inference_scaling import best_of_n, self_consistency
from .models import DirectAnswerPolicy, RewardTunedPolicy, TraceReasoningPolicy
from .refine import self_refine
from .train_rl import tune_with_grpo_diagnostics, tune_with_rewards


class SelfConsistencySolver:
    def __init__(self, solver: TraceReasoningPolicy, samples: int) -> None:
        self.solver = solver
        self.samples = samples

    def solve(self, problem):
        return self_consistency(self.solver, problem, samples=self.samples)


class BestOfNSolver:
    def __init__(self, solver: TraceReasoningPolicy, samples: int) -> None:
        self.solver = solver
        self.samples = samples

    def solve(self, problem):
        return best_of_n(self.solver, problem, samples=self.samples).selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=int, default=50)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    problems = make_dataset(args.problems, seed=args.seed)
    base = DirectAnswerPolicy()
    trace = TraceReasoningPolicy()
    scaled = SelfConsistencySolver(TraceReasoningPolicy(seed=29), samples=args.samples)
    best = BestOfNSolver(TraceReasoningPolicy(seed=37), samples=args.samples)
    tuned = RewardTunedPolicy(step_error_rate=0.45, seed=31)

    before = evaluate(tuned, problems)
    history = tune_with_rewards(tuned, problems, epochs=5)
    after = evaluate(tuned, problems)
    diagnostic_policy = RewardTunedPolicy(step_error_rate=0.45, seed=33)
    diagnostic_reports = tune_with_grpo_diagnostics(
        diagnostic_policy,
        problems[: max(1, min(10, len(problems)))],
        epochs=3,
        group_size=4,
    )

    teacher = TraceReasoningPolicy(step_error_rate=0.0, seed=41)
    student = distill_teacher(teacher, problems[: max(1, args.problems // 2)])
    distillation = measure_distillation(
        baseline=DirectAnswerPolicy(seed=11),
        teacher=TraceReasoningPolicy(step_error_rate=0.0, seed=41),
        student=student,
        eval_problems=problems,
    )

    print("Reasoning Model Lab")
    print("===================")
    print(f"Problems: {len(problems)}")
    print(f"Direct-answer accuracy:       {evaluate(base, problems).accuracy:.1%}")
    trace_eval = evaluate(trace, problems)
    scaled_eval = evaluate(scaled, problems)
    best_eval = evaluate(best, problems)
    print(f"Trace reasoning accuracy:     {trace_eval.accuracy:.1%}")
    print(f"Trace consistency rate:       {trace_eval.trace_consistency_rate:.1%}")
    print(f"Self-consistency accuracy:    {scaled_eval.accuracy:.1%}")
    print(f"Best-of-N accuracy:           {best_eval.accuracy:.1%}")
    print(f"Best-of-N avg reward:         {best_eval.average_reward:.2f}")
    print(f"Reward-tuned before:          {before.accuracy:.1%}")
    print(f"Reward-tuned after:           {after.accuracy:.1%}")
    print(f"Reward error-rate history:    {[round(x, 3) for x in history]}")
    print("\nGRPO-Style Diagnostics")
    print("----------------------")
    for report in diagnostic_reports:
        rollout = report.rollout_report
        print(
            f"Epoch {report.epoch}: reward={rollout.mean_reward:.2f}, "
            f"adv_std={rollout.advantage_std:.2f}, "
            f"entropy={rollout.mean_entropy_proxy:.2f}, "
            f"ratio={rollout.mean_policy_ratio:.2f}, "
            f"trace={rollout.trace_consistency_rate:.1%}, "
            f"error_rate={report.error_rate:.3f}"
        )
    print("\nDistillation Metrics")
    print("--------------------")
    print(f"Baseline accuracy:            {distillation.baseline_accuracy:.1%}")
    print(f"Teacher accuracy:             {distillation.teacher_accuracy:.1%}")
    print(f"Distilled student accuracy:   {distillation.student_accuracy:.1%}")
    print(f"Accuracy gain vs baseline:    {distillation.accuracy_gain:+.1%}")
    print(f"Teacher accuracy retained:    {distillation.teacher_retention:.1%}")
    print(f"Teacher calls before:         {distillation.teacher_calls_before}")
    print(f"Teacher calls after:          {distillation.teacher_calls_after}")
    print(f"Teacher call reduction:       {distillation.teacher_call_reduction:.1%}")

    example = problems[0]
    first_draft = TraceReasoningPolicy(step_error_rate=1.0, seed=9).solve(example)
    refined = self_refine(example, first_draft, rounds=2)
    print("\nExample")
    print("-------")
    print("Question:", example.question)
    print(teacher.solve(example).render())
    print("\nSelf-Refinement Example")
    print("-----------------------")
    print("Initial:", first_draft.render().replace("\n", " | "))
    print("Final:  ", refined.final.render().replace("\n", " | "))


if __name__ == "__main__":
    main()
