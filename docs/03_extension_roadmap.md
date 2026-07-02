# My Extension Roadmap

I can use this roadmap to turn the project into a deeper GitHub portfolio piece as my understanding improves.

## Milestone 1: Better Tasks

- Add multiplication, division, and two-hop word problems.
- Add JSONL export for generated datasets.
- Track accuracy by problem type.

## Milestone 2: Real Token Generation

- Add a tokenizer walkthrough.
- Replace the toy policy with a small open-source decoder model.
- Implement greedy decoding, sampling, and stopping criteria.

## Milestone 3: Stronger Evaluation

- Add pass-at-k and majority-vote metrics.
- Save wrong examples to `runs/errors.jsonl`.
- Add trace-quality checks separate from final-answer checks.
- Add verifier-specific dashboards for format failures, arithmetic failures, and trace failures.

## Milestone 4: Inference-Time Scaling

- Compare greedy, sampling, self-consistency, and best-of-n.
- Plot accuracy versus number of samples.
- Measure latency versus accuracy.
- Add a budget controller that decides when to stop sampling.
- Add self-refinement comparisons across 0, 1, 2, and 3 revision rounds.

## Milestone 5: Reward-Based Training

- Replace the transparent reward demo with policy-gradient or GRPO-style training.
- Use verifiable rewards from arithmetic, code tests, or symbolic solvers.
- Track reward curves across training.
- Save rollout groups with rewards, advantages, sequence log probabilities, entropy, KL, and response length.
- Compare reward-only training against format-plus-correctness rewards.
- Add guardrails for reward hacking, such as penalizing invalid formats and unnecessarily long traces.

## Milestone 6: Distillation

- Generate teacher traces from a stronger model.
- Fine-tune a smaller student on the traces.
- Compare teacher cost, student cost, and accuracy.
- Evaluate teacher-retention by problem type.
- Compare direct distillation against distill-then-RLVR.

## Milestone 7: Interview Polish

- Add diagrams to the README.
- Add a short demo video or GIF.
- Add a table of experiment results.
- Add a "what I learned" section after each milestone.
