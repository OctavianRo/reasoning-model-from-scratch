# My Interview Guide

## 60-Second Pitch

I built this small reasoning-model lab for my own understanding of how modern reasoning LLMs are developed. I started with a weak direct-answer baseline, added intermediate reasoning traces, evaluated final-answer accuracy, improved results with self-consistency, then demonstrated reward tuning and distillation on verifiable tasks.

## Architecture Talking Points

- The dataset creates arithmetic word problems with known answers.
- The base policy answers directly, which represents a conventional generator.
- The reasoning policy emits a trace plus a final answer.
- The evaluator gives objective feedback by checking the final answer.
- Self-consistency samples multiple traces and votes.
- Reward tuning updates behavior using verifier feedback.
- Distillation compresses teacher behavior into a smaller student.

## Questions I Can Answer

**Why not just prompt the model to think step by step?**

Prompting helps, but I still need evaluation. A model can produce a plausible trace and still land on the wrong answer.

**What does inference-time scaling buy you?**

It trades latency and compute for accuracy. Sampling multiple reasoning paths can recover from occasional bad paths.

**Where does reinforcement learning fit?**

RL is useful when a reward can be computed. In math, code, and logic tasks, correctness can often be checked automatically.

**Why distill a reasoning model?**

The strongest reasoning process may be too slow or expensive. Distillation trains a smaller model to imitate useful reasoning patterns.

**How would you scale this from toy code to a real LLM?**

I would replace the toy policy with a small decoder LLM, keep the evaluator, generate supervised reasoning traces, add self-consistency decoding, then fine-tune with supervised learning or reward-based optimization.

## Demo Script

1. Run `python -m reasoning_lab.cli --problems 50 --samples 7`.
2. Show the accuracy gap between direct answers and reasoning traces.
3. Explain why self-consistency improves robustness.
4. Show reward-tuning before and after.
5. Open `src/reasoning_lab/evaluate.py` to emphasize verifiable rewards.
