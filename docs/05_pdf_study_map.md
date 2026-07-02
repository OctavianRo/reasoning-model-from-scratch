# PDF Study Map

Source studied: `Build_a_Reasoning_Model_(From_Scratch).pdf`, Sebastian Raschka, Manning, 440 pages.

This document is my project-oriented map of the PDF. It is not a replacement for the book and does not copy the book text. It records the concepts I used to enrich this repository.

## Chapter-To-Project Mapping

| Book area | What I took from it | Project location |
| --- | --- | --- |
| Understanding reasoning models | Reasoning is treated as a pipeline, not a magic property: generate, evaluate, improve, train, distill. | `README.md`, `docs/01_reasoning_pipeline.md` |
| Text generation with a pretrained LLM | A real system needs tokenizer handling, autoregressive decoding, sampling controls, KV caching, and efficient inference. | Represented by toy policies in `models.py`; future work in `docs/03_extension_roadmap.md` |
| Evaluating reasoning models | Final answers must be extracted and checked with a verifier. Accuracy alone is useful but incomplete. | `evaluate.py`, `verifiers.py` |
| Inference-time scaling | Spend more inference compute through self-consistency, top-p/temperature-style diversity, and vote or score selection. | `inference_scaling.py` |
| Self-refinement | Draft, critique, revise, score, and accept only useful revisions. | `refine.py` |
| Reinforcement learning with verifiable rewards | Use verifier rewards from rollouts and turn group-relative reward differences into training signals. | `train_rl.py`, `grpo.py` |
| Improving GRPO | Track reward, advantage statistics, entropy, response length, clipping, KL drift, and format rewards to avoid unstable training. | `grpo.py`, CLI diagnostics |
| Distillation | Use a stronger teacher to create reasoning data, then train a cheaper student and measure retained accuracy. | `distill.py` |

## Full Refinement Loop

The folder now represents this loop:

1. Generate a candidate answer.
2. Extract the final answer.
3. Verify final-answer correctness.
4. Check whether the reasoning trace is consistent.
5. Reward useful answer format and trace structure.
6. Sample multiple rollouts for the same prompt.
7. Normalize rewards into advantages.
8. Watch stability diagnostics.
9. Refine or train the policy.
10. Distill the stronger behavior into a cheaper student.

## Concepts Added To The Project

- final-answer extraction,
- format reward,
- trace consistency reward,
- composite verifier reward,
- candidate sets for sampled inference,
- best-of-N selection,
- self-refinement with reward-based acceptance,
- grouped rollouts,
- normalized advantages,
- entropy proxy,
- response-length diagnostics,
- policy-ratio clipping,
- KL drift proxy,
- teacher-retention and teacher-call-reduction metrics.

## What Is Still Toy-Sized

This project still uses arithmetic policies instead of neural language models. That is intentional. The goal is to make the reasoning refinement pipeline inspectable before replacing the policy with a small decoder LLM.

To make it more realistic, the next implementation steps would be:

- add JSONL datasets with train/validation/test splits,
- add problem types beyond add/subtract word problems,
- introduce tokenizer and decoding code,
- replace the toy policies with a small open-source decoder model,
- compute real sequence log probabilities,
- save rollout traces and failed examples,
- run supervised fine-tuning for distillation,
- implement real policy-gradient or GRPO updates.

