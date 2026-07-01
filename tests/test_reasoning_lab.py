import unittest

from reasoning_lab.dataset import make_dataset
from reasoning_lab.distill import distill_teacher, measure_distillation
from reasoning_lab.evaluate import evaluate
from reasoning_lab.inference_scaling import self_consistency
from reasoning_lab.models import DirectAnswerPolicy, TraceReasoningPolicy


class ReasoningLabTests(unittest.TestCase):
    def test_dataset_answers_match_trace(self):
        problem = make_dataset(1, seed=1)[0]
        self.assertEqual(problem.answer, problem.start + problem.add - problem.subtract)
        self.assertIn(f"Subtract {problem.subtract}", problem.gold_trace)

    def test_trace_policy_can_solve_when_error_rate_is_zero(self):
        problems = make_dataset(20, seed=2)
        policy = TraceReasoningPolicy(step_error_rate=0.0)
        result = evaluate(policy, problems)
        self.assertEqual(result.accuracy, 1.0)

    def test_self_consistency_returns_response(self):
        problem = make_dataset(1, seed=3)[0]
        policy = TraceReasoningPolicy(step_error_rate=0.1)
        response = self_consistency(policy, problem, samples=5)
        self.assertIsInstance(response.final_answer, int)
        self.assertIn("Vote selected", response.trace)

    def test_distilled_student_matches_teacher_task_answers(self):
        problems = make_dataset(10, seed=4)
        teacher = TraceReasoningPolicy(step_error_rate=0.0)
        student = distill_teacher(teacher, problems)
        self.assertEqual(evaluate(student, problems).accuracy, 1.0)

    def test_distillation_report_has_clear_metrics(self):
        problems = make_dataset(20, seed=4)
        baseline = DirectAnswerPolicy(error_rate=0.5, seed=6)
        teacher = TraceReasoningPolicy(step_error_rate=0.0)
        student = distill_teacher(teacher, problems)
        report = measure_distillation(baseline, teacher, student, problems)
        self.assertGreater(report.accuracy_gain, 0.0)
        self.assertEqual(report.teacher_retention, 1.0)
        self.assertEqual(report.teacher_call_reduction, 1.0)

    def test_direct_answer_policy_is_not_perfect_with_errors(self):
        problems = make_dataset(100, seed=5)
        policy = DirectAnswerPolicy(error_rate=0.5, seed=6)
        result = evaluate(policy, problems)
        self.assertLess(result.accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
