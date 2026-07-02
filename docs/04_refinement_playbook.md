# Reasoning Model Refinement Playbook

This is the checklist I want this project to make concrete. The toy arithmetic policies are deliberately simple, but each stage maps to something required when refining a real reasoning model.

## 1. Start With A Measurable Task

A reasoning model needs a task where improvement can be measured. In this lab, every problem has a known numeric answer. In a real setup, the verifier might be a symbolic math checker, a unit-test runner, a formal proof checker, or a domain-specific rules engine.

Useful signals:

- final-answer accuracy,
- answer extractability,
- trace consistency,
- response length,
- confidence or log-probability proxy,
- reward by problem type.

## 2. Separate Reasoning From Grading

The model can produce fluent traces that are still wrong. The verifier should grade the answer independently of whether the reasoning sounds convincing.

In this project:

- `Response.render()` creates the model-style output,
- `extract_final_answer()` checks whether the final answer is parseable,
- `trace_matches_arithmetic()` checks whether the visible steps agree with the known operations,
- `verify_response()` returns the full grading breakdown.

## 3. Improve At Inference Time Before Training

Inference-time scaling is the cheapest refinement experiment because it does not change model weights.

This project includes two forms:

- self-consistency: sample several answers and vote on the final answer,
- best-of-N: sample several answers and select by verifier-style reward.

A real LLM version would add temperature, top-p sampling, stopping rules, tie-breaking by log probability, and a budget policy that decides when more samples are worth the latency.

## 4. Add Self-Refinement

Self-refinement uses the model or a verifier to critique an answer and produce a revised answer. The important part is not simply asking for a revision; it is deciding whether the revision should be accepted.

In this lab:

- `critique_response()` turns verifier failures into feedback,
- `repair_response()` creates a revised candidate,
- `self_refine()` accepts the revision only when reward improves or stays stable.

For a real model, this becomes a loop of draft, critique, revise, score, and accept. The loop needs stopping criteria so it does not spend unlimited compute.

## 5. Convert Rewards Into Learning Signals

Reward-based training needs more than a binary pass/fail score. A group of sampled rollouts lets me compare several answers to the same prompt and ask which ones were better than the group average.

The toy GRPO-style flow is:

1. sample multiple rollouts for the same prompt,
2. score each rollout with a verifier,
3. compute the group mean and standard deviation,
4. normalize each reward into an advantage,
5. inspect whether advantages are meaningful enough to train on.

In real GRPO, those advantages weight policy updates for generated token sequences.

## 6. Watch For Reward Hacking And Instability

Refinement can make a model worse if the reward is incomplete or the update is too aggressive.

Metrics to watch:

- reward trend,
- evaluation accuracy,
- response length,
- format pass rate,
- advantage standard deviation,
- entropy,
- policy ratio,
- clipped objective,
- KL drift from a reference model.

The point of `grpo.py` is to make these words concrete before moving to neural training code.

## 7. Distill For Efficient Deployment

After building a stronger teacher through prompting, inference-time scaling, self-refinement, or RLVR, the next question is cost. Distillation trains a smaller student to imitate useful teacher traces and answers.

In this lab, `DistilledPolicy` stores compact templates. In a real LLM project, this would become supervised fine-tuning on teacher-generated reasoning data, followed by evaluation against both the teacher and a direct-answer baseline.

## 8. Combine The Methods

The deeper lesson is that these techniques are not isolated:

- evaluate first,
- improve with inference-time scaling,
- refine weak answers,
- train with verifiable rewards,
- stabilize with diagnostics,
- distill into a cheaper student,
- keep evaluating after every change.

That loop is what turns this from a toy script into a miniature reasoning-model refinement pipeline.

