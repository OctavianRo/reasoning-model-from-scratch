# My Interview Guide

## 60-Second Pitch

I built this small reasoning-model lab for my own understanding of how modern reasoning LLMs are developed. I started with a weak direct-answer baseline, added intermediate reasoning traces, evaluated final-answer accuracy, improved results with self-consistency, then demonstrated reward tuning and distillation on verifiable tasks.

I later enriched it so it reflects more of a real refinement workflow: answer extraction, trace checking, best-of-N selection, verifier-guided self-refinement, GRPO-style rollout diagnostics, and teacher-to-student efficiency metrics.

## Architecture Talking Points

- The dataset creates arithmetic word problems with known answers.
- The base policy answers directly, which represents a conventional generator.
- The reasoning policy emits a trace plus a final answer.
- The evaluator gives objective feedback by checking the final answer, answer format, and trace consistency.
- Self-consistency samples multiple traces and votes.
- Best-of-N selection chooses the strongest candidate by verifier reward.
- Self-refinement critiques and revises weak answers, then accepts only non-worse revisions.
- Reward tuning updates behavior using verifier feedback.
- GRPO-style diagnostics show rewards, normalized advantages, entropy, clipping, KL drift, and format pass rate.
- Distillation compresses teacher behavior into a smaller student.

## Questions I Can Answer

**Why not just prompt the model to think step by step?**

Prompting helps, but I still need evaluation. A model can produce a plausible trace and still land on the wrong answer.

**What does inference-time scaling buy you?**

It trades latency and compute for accuracy. Sampling multiple reasoning paths can recover from occasional bad paths. I implemented both majority voting and best-of-N scoring to show two common ways to spend extra inference budget.

**What is self-refinement doing here?**

It turns verifier feedback into a small revise-and-accept loop. The important part is the acceptance rule: a revised answer is only kept if the reward improves or stays stable.

**Where does reinforcement learning fit?**

RL is useful when a reward can be computed. In math, code, and logic tasks, correctness can often be checked automatically.

**Why track GRPO-style diagnostics in a toy project?**

Because the hard part is not just computing a reward. A real reasoning run needs to know whether rewards are improving, whether advantages are meaningful, whether entropy is collapsing, whether outputs are getting too long, and whether policy updates are drifting too far.

**Why distill a reasoning model?**

The strongest reasoning process may be too slow or expensive. Distillation trains a smaller model to imitate useful reasoning patterns. In my demo, the distilled student improves from a 76.0% direct-answer baseline to 100.0% accuracy, which is a +24.0 percentage-point gain, while retaining 100.0% of the teacher's accuracy and reducing teacher calls during evaluation from 25 to 0.

**How would you scale this from toy code to a real LLM?**

I would replace the toy policy with a small decoder LLM, keep the evaluator, generate supervised reasoning traces, add self-consistency decoding, then fine-tune with supervised learning or reward-based optimization.

## Demo Script

1. Run `python -m reasoning_lab.cli --problems 50 --samples 7`.
2. Show the accuracy gap between direct answers and reasoning traces.
3. Explain why self-consistency improves robustness.
4. Point to best-of-N and self-refinement as extra inference-time refinement strategies.
5. Show reward-tuning before and after.
6. Show the GRPO-style diagnostics.
7. Open `src/reasoning_lab/verifiers.py` to emphasize verifiable rewards.
