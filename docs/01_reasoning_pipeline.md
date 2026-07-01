# My Reasoning Pipeline Notes

I created this project to make the reasoning-model pipeline concrete. The PDF gave me the learning path, and this repository gives me a small version I can inspect, run, and explain.

## 1. Base Generation

A conventional LLM is usually trained to predict the next token. In my project, `DirectAnswerPolicy` stands in for that base behavior: it produces a final answer directly and does not expose any intermediate reasoning.

The point I want to be able to explain: next-token prediction can produce fluent answers, but fluency is not the same as reliable multi-step problem solving.

## 2. Reasoning Traces

`TraceReasoningPolicy` adds intermediate steps before the final answer. In a real LLM, those steps are generated as tokens. Here, they are generated as explicit arithmetic steps so I can inspect correctness.

The point I want to be able to explain: reasoning is useful when intermediate steps help decompose the task and make evaluation easier.

## 3. Evaluation

`evaluate.py` checks final answers against ground truth. This mirrors verifiable reasoning benchmarks where answers can be graded automatically.

The point I want to be able to explain: reasoning models need objective evaluation. A convincing trace is not enough if the final answer is wrong.

## 4. Inference-Time Scaling

`inference_scaling.py` samples several candidate traces and votes on the final answer. This is a tiny version of self-consistency.

The point I want to be able to explain: I can improve reasoning without changing weights by spending more compute at inference time.

## 5. Reward Tuning

`train_rl.py` uses binary rewards from the evaluator to reduce future error. This is intentionally transparent rather than a full neural RL implementation.

The point I want to be able to explain: reinforcement learning becomes practical when the task has a verifier, such as exact answers in math, code tests, or formal constraints.

## 6. Distillation

`distill.py` lets a compact student learn from a stronger teacher's traces.

The point I want to be able to explain: distillation trades expensive teacher-time reasoning for cheaper student-time inference.
