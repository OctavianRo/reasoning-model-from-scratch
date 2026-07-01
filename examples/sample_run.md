# Sample Run

Command:

```bash
python -m reasoning_lab.cli --problems 25 --samples 7
```

Expected shape of the output:

```text
Reasoning Model Lab
===================
Problems: 25
Direct-answer accuracy:       ...
Trace reasoning accuracy:     ...
Self-consistency accuracy:    ...
Reward-tuned before:          ...
Reward-tuned after:           ...
Reward error-rate history:    [...]
Distilled student accuracy:   ...

Example
-------
Question: ...
Trace: Start with ...
Final: ...
```

The exact percentages can change because several policies intentionally sample mistakes.

