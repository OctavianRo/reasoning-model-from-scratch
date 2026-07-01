# Reasoning Pipeline

This project follows the learning path from the PDF while keeping the implementation small enough to inspect.

## 1. Base Generation

A conventional LLM is usually trained to predict the next token. In this repository, `DirectAnswerPolicy` stands in for that base behavior: it produces a final answer directly and does not expose any intermediate reasoning.

The important interview point: next-token prediction can produce fluent answers, but fluency is not the same as reliable multi-step problem solving.

## 2. Reasoning Traces

`TraceReasoningPolicy` adds intermediate steps before the final answer. In a real LLM, those steps are generated as tokens. Here, they are generated as explicit arithmetic steps so you can inspect correctness.

The important interview point: reasoning is useful when intermediate steps help decompose the task and make evaluation easier.

## 3. Evaluation

`evaluate.py` checks final answers against ground truth. This mirrors verifiable reasoning benchmarks where answers can be graded automatically.

The important interview point: reasoning models need objective evaluation. A convincing trace is not enough if the final answer is wrong.

## 4. Inference-Time Scaling

`inference_scaling.py` samples several candidate traces and votes on the final answer. This is a tiny version of self-consistency.

The important interview point: you can improve reasoning without changing weights by spending more compute at inference time.

## 5. Reward Tuning

`train_rl.py` uses binary rewards from the evaluator to reduce future error. This is intentionally transparent rather than a full neural RL implementation.

The important interview point: reinforcement learning becomes practical when the task has a verifier, such as exact answers in math, code tests, or formal constraints.

## 6. Distillation

`distill.py` lets a compact student learn from a stronger teacher's traces.

The important interview point: distillation trades expensive teacher-time reasoning for cheaper student-time inference.

