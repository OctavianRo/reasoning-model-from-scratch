# Build A Reasoning Model From Scratch - Student Study Notes

Source: Sebastian Raschka, *Build a Reasoning Model (From Scratch)*, Manning, 2026.

These notes are a student-friendly guide to the book's main ideas. They are paraphrased study notes, not a replacement for the book. The goal is to explain what you need to know from each chapter, define important AI terms, and give small examples that make the concepts easier to remember.

## How To Read These Notes

The book follows one big storyline:

1. Start with a normal pretrained language model.
2. Learn how it generates text.
3. Build ways to evaluate whether its reasoning answers are correct.
4. Improve answers at inference time without changing the model weights.
5. Add self-refinement so the model can critique and revise answers.
6. Use reinforcement learning with verifiable rewards to train the model.
7. Stabilize and monitor that reinforcement learning process.
8. Distill the stronger reasoning behavior into a cheaper student model.

The core loop is:

```text
base model -> generate answer -> verify answer -> improve at inference time
           -> train with rewards -> monitor stability -> distill into student
```

## Chapter 1 - Understanding Reasoning Models

### The Big Idea

A reasoning model is not a completely different species of model. It is usually a conventional language model that has been prompted, trained, or post-trained to produce more useful intermediate reasoning before giving an answer.

The chapter sets up the book's roadmap:

- load a pretrained LLM,
- build a text generation pipeline,
- evaluate reasoning behavior,
- improve reasoning without training,
- improve reasoning with training,
- distill stronger reasoning into a smaller model.

### What You Need To Know

**Conventional LLM training has stages.**

A typical LLM pipeline often includes:

- **Pretraining**: the model learns broad language patterns from huge text datasets.
- **Instruction fine-tuning**: the model learns to follow user instructions.
- **Preference tuning**: the model is adjusted to produce responses people prefer.
- **Reasoning post-training**: the model is further improved for multistep tasks such as math, code, logic, or planning.

**Reasoning in LLMs usually means useful intermediate steps.**

In this book, reasoning is practical rather than philosophical. A model is treated as a better reasoning model if it can produce intermediate steps that improve final-answer correctness.

Example:

```text
Question: A notebook costs $4. A pen costs $2. You buy 3 notebooks and 5 pens. What is the total?

Weak answer:
Final: $18

Reasoned answer:
3 notebooks cost 3 * 4 = 12.
5 pens cost 5 * 2 = 10.
Total = 12 + 10 = 22.
Final: $22
```

The second answer is not valuable merely because it is longer. It is valuable because the intermediate steps help the model reach the correct answer and give us something to inspect.

**Pattern matching and logical reasoning are different.**

LLMs are trained statistically. They learn patterns in data and generate likely continuations. That can simulate reasoning very well, but it is not the same as a hand-written logic engine that applies explicit rules.

Simple contrast:

```text
Rule-based system:
If A implies B, and A is true, then B is true.

LLM:
Given many examples like "A implies B, A is true, therefore B is true,"
it learns to continue similar text in a way that often follows the pattern.
```

The practical question is not "does the model reason like a human?" The practical question is "can we make it solve multistep tasks more reliably?"

### Important Terms

**LLM, or large language model**  
A neural network trained to predict and generate text. It usually predicts the next token given previous tokens.

**Reasoning model**  
An LLM improved to solve tasks that benefit from multistep thinking, verification, or tool use.

**Chain of thought**  
Intermediate reasoning steps generated before a final answer.

**Inference-time scaling**  
Using more compute during generation, such as sampling multiple answers, without changing model weights.

**Post-training**  
Training that happens after pretraining, often to make a model follow instructions, align with preferences, or reason better.

**Distillation**  
Training a smaller or cheaper student model to imitate a stronger teacher model.

### Illustrative Example

Imagine a student solving arithmetic.

```text
Direct answer:
17 + 28 - 9 = 31

Step-by-step:
17 + 28 = 45.
45 - 9 = 36.
Final: 36
```

The direct answer can be wrong with no explanation. The step-by-step answer gives an evaluator a chance to check where the calculation went right or wrong.

### What To Remember

- Reasoning models are usually built on top of ordinary LLMs.
- Reasoning improvement can happen at inference time or training time.
- Evaluation is central: you need to measure whether reasoning actually improved correctness.
- The book is a practical pipeline, not only theory.

### Self-Check

- What is the difference between pretraining and reasoning post-training?
- Why is a long chain of thought not automatically a good chain of thought?
- Why should reasoning improvements be measured with an evaluator?

## Chapter 2 - Generating Text With A Pretrained LLM

### The Big Idea

Before you can build a reasoning model, you need to understand how a normal LLM generates text. The model takes tokens as input, predicts scores for possible next tokens, selects one token, appends it, and repeats.

Reasoning is built on this same generation machinery.

### What You Need To Know

**Text must become tokens before the model can process it.**

Models do not read words exactly the way humans do. A tokenizer converts text into token IDs.

Example:

```text
Text:
"Solve 2 + 3"

Possible token pieces:
["Solve", " 2", " +", " 3"]

Possible token IDs:
[41820, 220, 489, 220, 18]
```

The exact tokens depend on the tokenizer.

**Generation is autoregressive.**

Autoregressive means the model generates one token at a time, using previous tokens as context.

Example:

```text
Prompt: "The answer is"
Step 1: model predicts " 4"
Prompt becomes: "The answer is 4"
Step 2: model predicts "."
Prompt becomes: "The answer is 4."
```

For reasoning models, this same process generates reasoning steps:

```text
"First, compute 7 + 5 = 12. Then subtract 3..."
```

**The model outputs logits.**

Before probabilities, the model produces raw scores called logits. Higher logits mean the model currently prefers those tokens more.

Example:

```text
Candidate next tokens:
"yes" -> logit 3.0
"no"  -> logit 1.0
"maybe" -> logit 0.5
```

Softmax turns these raw scores into probabilities.

**KV caching speeds up generation.**

During generation, earlier tokens are reused again and again. KV caching stores intermediate attention information so the model does not recompute everything from scratch each time it generates the next token.

Without caching:

```text
Generate token 1: process prompt
Generate token 2: process prompt + token 1 again
Generate token 3: process prompt + token 1 + token 2 again
```

With caching:

```text
Generate token 1: process prompt
Generate token 2: reuse cached prompt information
Generate token 3: reuse cached earlier information
```

This matters because reasoning models often generate many more tokens than direct-answer models.

**Model compilation can improve runtime.**

Compilation tools can optimize model execution so repeated operations run faster. In PyTorch, `torch.compile` can sometimes speed up inference by optimizing the computation graph.

### Important Terms

**Token**  
A chunk of text represented by an integer ID. Tokens can be words, word pieces, spaces, punctuation, or other symbols.

**Tokenizer**  
The component that converts text to token IDs and token IDs back to text.

**Autoregressive generation**  
Generating one token at a time, where each new token depends on previous tokens.

**Logits**  
Raw model scores for possible next tokens.

**Softmax**  
A function that converts logits into probabilities that sum to 1.

**KV cache**  
Stored key and value tensors used by the attention mechanism to speed up generation.

**Inference**  
Using a trained model to produce outputs.

### Illustrative Example

Suppose the model has these logits for the next token:

```text
"4" -> 4.0
"5" -> 1.0
"blue" -> -2.0
```

After softmax, `"4"` gets most of the probability. If you use greedy decoding, the model picks `"4"`.

For reasoning, the model might instead generate:

```text
2 + 2 = 4, so the final answer is 4.
```

The important point is that the model is still generating tokens one by one.

### What To Remember

- LLMs generate text by predicting one token at a time.
- Tokenization is the bridge between text and model input.
- Logits become probabilities through softmax.
- KV caching and compilation matter because reasoning outputs can be long and expensive.

### Self-Check

- What is a token?
- Why does a reasoning model usually cost more to run than a direct-answer model?
- What problem does KV caching solve?

## Chapter 3 - Evaluating Reasoning Models

### The Big Idea

You cannot improve what you cannot measure. For reasoning models, evaluation means more than asking whether the output sounds intelligent. The book focuses on verifier-based evaluation, especially for math problems where answers can be checked automatically.

### What You Need To Know

**There are several broad evaluation styles.**

Common ways to evaluate LLMs include:

- multiple-choice benchmarks,
- verifier-based grading,
- leaderboard tasks,
- LLM-as-judge evaluation.

For reasoning, verifier-based evaluation is especially useful because it can grade free-form answers.

**A verifier checks whether the answer is correct.**

The model may produce a long response, but the evaluator usually needs to extract the final answer first.

Example model output:

```text
We need to solve x + 3 = 8.
Subtract 3 from both sides: x = 5.
Final: 5
```

The verifier should extract `5`, then compare it with the reference answer.

**Extraction is a real problem.**

Models do not always put answers in the same format.

Examples:

```text
Final: 5
The answer is 5.
\boxed{5}
x = 5
```

A robust evaluator needs rules for extracting answers from varied output formats.

**Normalization makes equivalent answers comparable.**

Two answers can look different but mean the same thing.

Examples:

```text
0.5
1/2
\frac{1}{2}
```

A math verifier should normalize expressions before comparing them.

**Symbolic equivalence is stronger than string matching.**

String matching would say these are different:

```text
x^2 - 1
(x - 1)(x + 1)
```

But mathematically they are equivalent. A symbolic tool such as SymPy can check that.

**Evaluation must be repeatable.**

If a model uses sampling, the same prompt can produce different answers. Good evaluation should control seeds, record prompts, save outputs, and compare models under similar conditions.

### Important Terms

**Verifier**  
A program or rule that checks whether a model output is correct.

**Answer extraction**  
Finding the final answer inside a longer response.

**Normalization**  
Converting different answer formats into a common representation.

**Symbolic equivalence**  
Checking whether two expressions are mathematically the same, even if written differently.

**Benchmark**  
A dataset and scoring protocol used to compare models.

**LLM judge**  
Another model used to grade outputs, often useful for open-ended tasks but less deterministic than a verifier.

### Illustrative Example

Question:

```text
What is 2/4 simplified?
```

Reference:

```text
1/2
```

Model outputs:

```text
Output A: Final: 0.5
Output B: Final: \frac{1}{2}
Output C: Final: 2/4
```

A string matcher may mark these as different. A math-aware verifier should mark them as equivalent.

### What To Remember

- Evaluation is the foundation of reasoning-model improvement.
- A good verifier extracts, normalizes, and checks answers.
- Correct-looking reasoning is not enough; the final answer must be right.
- For math, symbolic equivalence is often better than raw string matching.

### Self-Check

- Why is answer extraction necessary?
- What is the difference between string equality and mathematical equivalence?
- Why might an LLM judge be less reliable than a verifier for math?

## Chapter 4 - Improving Reasoning With Inference-Time Scaling

### The Big Idea

Inference-time scaling improves answer quality by spending more compute during generation, without updating model weights.

The chapter covers chain-of-thought prompting, temperature, top-p sampling, and self-consistency.

### What You Need To Know

**Prompting can change the model's behavior.**

A prompt like "explain step by step" encourages the model to generate intermediate reasoning.

Example:

```text
Prompt A:
What is 18 * 7?

Prompt B:
What is 18 * 7? Explain step by step and give the final answer.
```

Prompt B is more likely to produce a calculation:

```text
18 * 7 = 18 * 5 + 18 * 2 = 90 + 36 = 126.
Final: 126
```

**Temperature controls randomness.**

Temperature rescales logits before sampling.

- Low temperature: more deterministic, safer, less diverse.
- High temperature: more diverse, more creative, more error-prone.

Example:

```text
Low temperature:
The model almost always chooses the highest-probability token.

High temperature:
The model gives lower-probability tokens more chance.
```

For reasoning, some diversity is useful because different sampled solutions may explore different paths. Too much diversity can produce nonsense.

**Top-p sampling keeps only the most likely probability mass.**

Top-p sampling, also called nucleus sampling, sorts candidate tokens by probability and keeps the smallest set whose cumulative probability reaches `p`.

Example:

```text
Token probabilities:
A: 0.50
B: 0.25
C: 0.15
D: 0.07
E: 0.03

top_p = 0.80 keeps A, B, and C because 0.50 + 0.25 + 0.15 = 0.90.
```

The goal is to allow variety while filtering out very unlikely tokens.

**Self-consistency samples multiple reasoning paths.**

Instead of trusting one answer, generate several answers and vote on the final result.

Example:

```text
Sample 1 final answer: 42
Sample 2 final answer: 41
Sample 3 final answer: 42
Sample 4 final answer: 42
Sample 5 final answer: 43

Majority vote: 42
```

This can improve accuracy if correct reasoning paths are more common than incorrect ones.

**The tradeoff is accuracy versus compute.**

Sampling 10 answers can be more accurate than sampling 1 answer, but it can also cost about 10 times more generation work.

### Important Terms

**Inference-time scaling**  
Using more generation-time compute to improve output quality.

**Chain-of-thought prompting**  
Prompting the model to write intermediate reasoning before the final answer.

**Temperature**  
A parameter controlling how sharp or flat the next-token probability distribution is.

**Top-p sampling**  
A sampling method that keeps only the smallest high-probability token set reaching cumulative probability `p`.

**Self-consistency**  
Generating multiple reasoning paths and selecting the most common final answer.

**Majority vote**  
Choosing the answer that appears most often among sampled candidates.

### Illustrative Example

Question:

```text
A ticket costs $12. You buy 4 tickets and pay a $6 fee. What is the total?
```

One sampled answer might make a mistake:

```text
4 * 12 = 48. Add 6 gives 56. Final: 56
```

Other samples may be correct:

```text
4 * 12 = 48. Add 6 gives 54. Final: 54
```

If most samples say `54`, self-consistency can recover from occasional errors.

### What To Remember

- You can improve reasoning without training by spending more inference compute.
- Temperature and top-p control output diversity.
- Chain-of-thought prompting can improve multistep accuracy.
- Self-consistency works by sampling multiple answers and voting.
- More inference compute means higher cost and latency.

### Self-Check

- Why can self-consistency improve accuracy?
- What happens when temperature is too high?
- What is the practical cost of generating many samples?

## Chapter 5 - Inference-Time Scaling Via Self-Refinement

### The Big Idea

Self-refinement improves a single answer through a loop:

```text
draft -> critique -> revise -> score -> accept or reject
```

This differs from self-consistency. Self-consistency samples many independent answers. Self-refinement starts with one answer and tries to improve it.

### What You Need To Know

**A scoring function is needed.**

If the model revises an answer, we need a way to decide whether the revision is better. The book discusses simple rule-based scores and probability-based scores.

Example rule-based scoring:

```text
+1.0 if the final answer is extractable
+1.0 if the answer is correct
-0.1 for overly long output
```

This is not the only possible scoring scheme, but it shows the idea.

**Token probabilities can estimate confidence.**

The model produces probabilities for each generated token. If the model assigns high probability to the tokens it generated, that can be treated as a rough confidence signal.

Example:

```text
Generated answer tokens:
["Final", ":", " 42"]

Probabilities:
0.80, 0.90, 0.70
```

The raw product is:

```text
0.80 * 0.90 * 0.70 = 0.504
```

For long sequences, products become tiny, so we use log probabilities.

**Log probabilities make sequence scoring stable.**

Multiplying many probabilities can underflow. Adding log probabilities is numerically easier.

Example:

```text
log(0.80) + log(0.90) + log(0.70)
```

Higher average log probability usually means the model was more confident, though confidence is not the same as correctness.

**Self-refinement needs an acceptance rule.**

If every revision is accepted blindly, the model can make things worse.

Example:

```text
Initial answer: correct but short.
Revision: longer, more confident-sounding, but wrong.
```

A good refinement loop should score the revision and reject it if it lowers the score.

**Refinement can use feedback.**

The critique stage can say what seems wrong:

```text
Critique:
The final answer is not in the required format.

Revision:
Keep the reasoning, but add "Final: 42" at the end.
```

For math, feedback can come from a verifier. For open-ended writing, feedback may come from a judge model or checklist.

### Important Terms

**Self-refinement**  
An inference-time process where a model critiques and revises its own answer.

**Rule-based score**  
A score computed from hand-designed rules, such as answer format or length.

**Token probability**  
The probability assigned to a generated token.

**Log probability**  
The logarithm of a probability, useful for stable sequence scoring.

**Sequence likelihood**  
A score estimating how likely the whole generated sequence was under the model.

**Acceptance rule**  
A decision rule for whether to keep a revised answer.

### Illustrative Example

Question:

```text
Solve: 9 + 8 - 5
```

Initial draft:

```text
9 + 8 = 17. 17 - 5 = 13.
Final: 13
```

Verifier says:

```text
Expected 12, got 13.
```

Critique:

```text
The subtraction step is wrong.
```

Revision:

```text
9 + 8 = 17. 17 - 5 = 12.
Final: 12
```

If the verifier reward improves, accept the revision.

### What To Remember

- Self-refinement is draft, critique, revise, and score.
- Revision is useful only if there is a way to judge improvement.
- Log probabilities help score long generated sequences.
- Confidence is useful but not identical to correctness.

### Self-Check

- How is self-refinement different from self-consistency?
- Why are log probabilities preferred over raw probability products?
- Why does self-refinement need an acceptance rule?

## Chapter 6 - Training Reasoning Models With Reinforcement Learning

### The Big Idea

Inference-time methods improve outputs without changing the model. Reinforcement learning changes the model so it becomes more likely to produce high-reward reasoning outputs in the future.

The chapter introduces reinforcement learning with verifiable rewards and GRPO.

### What You Need To Know

**RL can be used after pretraining.**

In LLMs, reinforcement learning is usually a post-training method. The model already knows language. RL adjusts behavior toward outputs that receive higher reward.

**RLHF uses human preference signals.**

RLHF means reinforcement learning from human feedback. A common setup is:

1. Humans compare outputs.
2. A reward model learns to predict human preferences.
3. The LLM is optimized to produce outputs with higher predicted reward.

**RLVR uses deterministic verifiers.**

RLVR means reinforcement learning with verifiable rewards. Instead of asking humans to rank outputs, a program checks correctness.

Example:

```text
Math answer correct and extractable: reward = 1
Wrong or unparseable answer: reward = 0
```

RLVR is attractive for domains like:

- math,
- code with unit tests,
- formal logic,
- structured data tasks,
- some tool-use tasks.

**A rollout is a full generated answer.**

For one prompt, the model may generate several complete answers. Each complete generated answer is a rollout.

Example:

```text
Prompt: "What is 15 - 6?"

Rollout 1: "15 - 6 = 9. Final: 9" -> reward 1
Rollout 2: "15 - 6 = 8. Final: 8" -> reward 0
Rollout 3: "Final: 9" -> reward 1
```

**Advantages compare a rollout with its group.**

Raw rewards are useful, but GRPO uses group-relative advantages.

Example rewards:

```text
[0, 1, 1, 0]
```

Mean reward:

```text
0.5
```

Standard deviation:

```text
0.5
```

Advantages:

```text
(0 - 0.5) / 0.5 = -1
(1 - 0.5) / 0.5 =  1
(1 - 0.5) / 0.5 =  1
(0 - 0.5) / 0.5 = -1
```

So the model is pushed toward the better-than-average rollouts and away from the worse-than-average rollouts.

**GRPO avoids a separate value model.**

Some RL algorithms use a value model to estimate expected future reward. GRPO is more lightweight because it compares rollouts within a group and normalizes rewards into advantages.

**Sequence log probabilities connect rewards to model updates.**

To train the model, you need to know how likely the model made each rollout. If a rollout had a positive advantage, you increase its likelihood. If it had a negative advantage, you decrease its likelihood.

### Important Terms

**Reinforcement learning, or RL**  
Training by rewarding desired behavior and discouraging undesired behavior.

**RLHF**  
Reinforcement learning from human feedback.

**RLVR**  
Reinforcement learning with verifiable rewards.

**Reward**  
A numerical score for a generated output.

**Rollout**  
A full generated completion for a prompt.

**Advantage**  
A normalized signal showing whether a rollout was better or worse than comparable rollouts.

**GRPO**  
Group Relative Policy Optimization, an RL method that uses groups of sampled outputs and relative rewards.

**Policy**  
In RL language, the model being trained. It maps prompts to output probabilities.

### Illustrative Example

Suppose the model answers one problem four times:

```text
Answer A: Final 9, reward 1
Answer B: Final 8, reward 0
Answer C: Final 9, reward 1
Answer D: no final answer, reward 0
```

The group contains both good and bad rollouts. GRPO turns this into:

```text
Increase probability of A and C.
Decrease probability of B and D.
```

Over many prompts, this can teach the model to produce correct, properly formatted reasoning more often.

### What To Remember

- RL changes model behavior, not just decoding behavior.
- RLVR is powerful when correctness can be checked automatically.
- GRPO compares rollouts within a group.
- Advantages are normalized reward signals.
- Sequence log probabilities are needed to update the model.

### Self-Check

- What is the difference between RLHF and RLVR?
- What is a rollout?
- Why does GRPO normalize rewards within a group?

## Chapter 7 - Improving GRPO For Reinforcement Learning

### The Big Idea

Getting GRPO to work once is not enough. Training can become unstable. Chapter 7 is about monitoring, diagnosing, and stabilizing reward-based reasoning training.

### What You Need To Know

**Loss alone is not enough.**

In supervised learning, loss curves are often central. In GRPO, loss can be less intuitive. You need to track several metrics at once.

Useful metrics include:

- average reward,
- evaluation accuracy,
- response length,
- advantage mean and standard deviation,
- entropy,
- policy ratios,
- KL divergence,
- format pass rate.

**Average reward can hide problems.**

Reward can increase while the model becomes worse in another way.

Example:

```text
Reward function:
+1 if output contains "Final:"

Bad model behavior:
The model prints "Final:" everywhere, even when the answer is wrong.
```

This is reward hacking: the model exploits the reward without solving the real task.

**Response length matters.**

Reasoning models can learn to produce very long answers. Sometimes this helps, but it can also waste compute or hide errors.

Example:

```text
Short correct answer:
Final: 12

Useful reasoning:
9 + 8 = 17. 17 - 5 = 12. Final: 12

Unnecessarily long reasoning:
Several paragraphs of unrelated calculations, then Final: 12
```

Longer is not automatically better.

**Entropy measures uncertainty.**

Entropy tells you how spread out the model's token distribution is.

- Very low entropy: the model may be collapsing into repetitive or rigid outputs.
- Very high entropy: the model may be too random.

**Policy ratios measure how much the model changed.**

During RL, we compare the probability of an output under the new policy with its probability under the old policy.

```text
policy_ratio = new_probability / old_probability
```

If the ratio is very large, the model changed aggressively. That can destabilize training.

**Clipping limits aggressive updates.**

Policy-ratio clipping caps how much an update can push the model.

Example:

```text
Allowed ratio range: 0.8 to 1.2

Actual ratio: 1.8
Clipped ratio: 1.2
```

This prevents the model from moving too far based on one batch of rewards.

**KL divergence measures drift from a reference model.**

KL divergence is a measure of how different two probability distributions are. In RL for LLMs, it can be used to keep the trained model from drifting too far away from the original model.

Example intuition:

```text
Reference model: balanced, fluent, general-purpose.
RL model after too much reward pressure: narrow, repetitive, reward-hacking.
KL penalty: discourages drifting too far.
```

**Format rewards can help.**

A verifier may require answers in a specific format. A format reward encourages the model to use that structure.

Example:

```text
Good format:
<think>
Compute 6 + 7 = 13.
</think>
Final: 13

Bad format:
thirteen probably
```

Format rewards are helpful, but they should not replace correctness rewards.

### Important Terms

**Training instability**  
When training metrics improve briefly but then collapse, oscillate, or degrade.

**Reward hacking**  
When the model exploits the reward function without doing the intended task.

**Entropy**  
A measure of uncertainty in the model's output distribution.

**Policy ratio**  
The new policy probability divided by the old policy probability for an output.

**Clipping**  
Limiting policy updates to prevent overly large changes.

**KL divergence**  
A measure of difference between two probability distributions.

**Format reward**  
A reward for producing outputs in a required structure.

### Illustrative Example

Suppose a model is trained with reward:

```text
+1 if the response contains a number
```

The model may learn:

```text
Final: 999
```

for everything. It gets the format reward but fails the real task.

A better reward might combine:

```text
+1.0 for correct answer
+0.2 for extractable final format
+0.1 for reasonable length
-0.2 for missing final answer
```

The broader reward is harder to exploit.

### What To Remember

- GRPO needs monitoring beyond loss.
- Reward can improve while behavior gets worse.
- Track reward, accuracy, length, entropy, advantages, ratios, KL, and format.
- Clipping and KL are stability tools.
- Format rewards help structure outputs but must not replace correctness.

### Self-Check

- Why can average reward be misleading?
- What does entropy tell you during generation?
- Why might a KL penalty be useful?
- What is reward hacking?

## Chapter 8 - Distilling Reasoning Models For Efficient Reasoning

### The Big Idea

Strong reasoning can be expensive. Distillation trains a smaller or cheaper model to imitate the useful behavior of a stronger teacher model.

The book frames distillation as a practical way to turn expensive reasoning traces into a deployable student model.

### What You Need To Know

**Teacher models generate training data.**

The teacher is a stronger model or a stronger inference process. It may use self-consistency, self-refinement, RLVR, or larger model capacity.

Example:

```text
Teacher input:
Solve: 14 + 9 - 5

Teacher output:
14 + 9 = 23. 23 - 5 = 18. Final: 18
```

This becomes a training example for the student.

**Hard distillation uses teacher outputs as targets.**

The student learns to produce the teacher's answer text.

Example training pair:

```text
Input: "Solve 14 + 9 - 5"
Target: "14 + 9 = 23. 23 - 5 = 18. Final: 18"
```

This is like supervised fine-tuning on teacher-generated data.

**Soft distillation can use probability distributions.**

Instead of only copying final tokens, soft distillation can train the student to match the teacher's probability distribution. This gives richer information, but it usually requires access to teacher logits.

**Dataset quality matters.**

If the teacher produces wrong or messy reasoning, the student can learn those mistakes.

A distillation dataset should be:

- correct,
- consistently formatted,
- representative of the target task distribution,
- split into train and validation sets,
- evaluated after training.

**Loss measures how well the student matches the training targets.**

For text generation, cross-entropy loss is commonly used. It measures how well the model predicts the target tokens.

Example intuition:

```text
Target next token: "18"
Student assigns high probability to "18": low loss
Student assigns low probability to "18": high loss
```

**Distillation is about cost and deployment.**

A teacher might be more accurate but too slow or expensive. A student may be slightly less capable but much cheaper to run.

### Important Terms

**Teacher model**  
The stronger model or process used to generate training data.

**Student model**  
The smaller or cheaper model trained to imitate the teacher.

**Hard distillation**  
Training on the teacher's generated outputs as fixed targets.

**Soft distillation**  
Training on the teacher's probability distributions or logits.

**Cross-entropy loss**  
A loss function that measures how well predicted probabilities match target tokens.

**Validation set**  
Held-out data used to check whether the student generalizes beyond the training examples.

### Illustrative Example

Suppose a teacher uses 16 sampled reasoning paths and majority voting to answer each problem. That is expensive.

Instead of doing that every time, we can generate many teacher answers offline:

```text
Prompt 1 -> teacher trace and answer
Prompt 2 -> teacher trace and answer
Prompt 3 -> teacher trace and answer
```

Then train a smaller student:

```text
Student learns to produce similar traces directly in one pass.
```

At deployment time:

```text
Before distillation:
16 teacher generations per problem

After distillation:
1 student generation per problem
```

The student may not match the teacher perfectly, but it can provide a better cost-performance tradeoff.

### What To Remember

- Distillation compresses expensive reasoning behavior.
- Teacher data quality is crucial.
- Hard distillation trains on teacher outputs.
- Soft distillation trains on richer teacher probability information.
- The student should be evaluated against both the teacher and a baseline.

### Self-Check

- Why distill a reasoning model?
- What is the difference between hard and soft distillation?
- Why can bad teacher traces harm the student?

## Appendix Concepts Worth Knowing

The appendices support the main chapters. You do not need to memorize every detail at first, but these topics matter when turning the learning project into a serious implementation.

### Qwen3 Source Code

The book includes source-code-level material for the Qwen3 model architecture. The key learning point is that a real LLM is built from repeated transformer blocks, attention, feed-forward layers, embeddings, normalization, and output projection.

Terms:

**Transformer**  
The neural architecture used by most modern LLMs.

**Attention**  
The mechanism that lets tokens use information from other tokens in the context.

**Embedding**  
A learned vector representation of a token.

**Feed-forward network**  
A neural network block inside each transformer layer.

### Using Larger LLMs

Larger models usually need more memory, more compute, and more careful execution. The main tradeoff is capability versus cost.

Example:

```text
Small model:
cheap, fast, less capable

Large model:
more capable, slower, more expensive
```

### Batching And Throughput

Batching means running multiple prompts together to improve hardware utilization.

Example:

```text
Without batching:
Run prompt 1, then prompt 2, then prompt 3.

With batching:
Run prompts 1, 2, and 3 together.
```

Batching can improve throughput, but it can complicate variable-length generation.

### Common Evaluation Approaches

Different tasks need different evaluation methods:

- exact match for simple answers,
- symbolic verification for math,
- unit tests for code,
- multiple choice for benchmark exams,
- LLM judges for open-ended responses,
- human review for subjective quality.

The best evaluator is the one that matches the task's real success condition.

### Building A Chat Interface

A chat interface wraps the model in a user-facing loop:

```text
user message -> prompt formatting -> model generation -> response display
```

For reasoning models, the interface also needs to decide:

- whether to show reasoning traces,
- how much reasoning budget to spend,
- when to use tools or verifiers,
- how to handle failed or uncertain answers.

## Master Glossary Of AI Terms

### Advantage

A normalized reward signal showing whether one rollout was better or worse than comparable rollouts.

### Autoregressive Model

A model that generates one token at a time, each token conditioned on previous tokens.

### Benchmark

A dataset and scoring method used to measure model performance.

### Chain Of Thought

Intermediate reasoning text produced before a final answer.

### Cross-Entropy Loss

A training loss that penalizes the model when it assigns low probability to the correct target token.

### Distillation

Training a student model to imitate a stronger teacher model.

### Entropy

A measure of uncertainty or spread in a probability distribution.

### GRPO

Group Relative Policy Optimization, an RL method that compares multiple sampled rollouts for the same prompt.

### Inference

Running a trained model to generate outputs.

### Inference-Time Scaling

Improving output quality by spending more compute during generation instead of changing model weights.

### KL Divergence

A measure of how different one probability distribution is from another.

### KV Cache

Stored attention information that speeds up autoregressive generation.

### Logit

A raw score produced by the model before conversion to probabilities.

### Log Probability

The logarithm of a probability, used for stable sequence scoring.

### Policy

In RL, the model that chooses actions. For LLMs, the policy chooses output tokens.

### Preference Tuning

Post-training that adjusts a model toward outputs humans prefer.

### Pretraining

The large-scale initial training stage where a model learns general language patterns.

### Reinforcement Learning

Training a model by rewarding desirable behavior.

### Reward

A numerical score assigned to a model output.

### Reward Hacking

When a model exploits the reward function without truly solving the task.

### RLHF

Reinforcement learning from human feedback.

### RLVR

Reinforcement learning with verifiable rewards.

### Rollout

A complete generated answer for a prompt.

### Self-Consistency

Generating multiple answers and selecting the most common final answer.

### Self-Refinement

Generating an answer, critiquing it, revising it, and accepting the revision only if it improves.

### Softmax

A function that turns logits into probabilities.

### Temperature

A decoding parameter that controls randomness by reshaping next-token probabilities.

### Token

A unit of text represented by a model as an integer ID.

### Tokenizer

The tool that converts text to token IDs and token IDs back to text.

### Top-p Sampling

A decoding method that samples only from the smallest set of likely tokens whose cumulative probability reaches `p`.

### Verifier

A programmatic checker that determines whether an answer is correct.

## The Whole Book In One Mental Model

If you remember only one thing, remember this:

```text
Reasoning models are built by combining generation, evaluation, inference-time search,
reward-based training, stability monitoring, and distillation.
```

More concretely:

1. The base model knows how to generate text.
2. A prompt can encourage reasoning traces.
3. A verifier checks whether answers are correct.
4. Sampling multiple answers can improve reliability.
5. Self-refinement can revise weak answers.
6. RLVR can train the model using automatic rewards.
7. GRPO turns group rewards into learning signals.
8. Diagnostics prevent unstable or reward-hacked training.
9. Distillation makes strong reasoning cheaper to deploy.

That is the book's practical recipe for building and understanding reasoning models.

