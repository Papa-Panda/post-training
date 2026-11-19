# SFT vs GRPO on the Same Tasks

> Reading edition: https://papa-panda.github.io/post-training/sft-vs-grpo/

A research synthesis comparing supervised fine-tuning on trajectories with GRPO reinforcement learning, trained on the same tasks.

Headline conclusion: there is no universal winner. It is a division of labour. SFT installs format, knowledge, and reasoning patterns, with the largest in-distribution gains, but it tends to memorise trajectories and degrade out-of-distribution behaviour. GRPO sharpens what the model can already sometimes do, raising Pass@1 and preserving out-of-distribution behaviour better, but only when Pass@k is already above zero, and only with a verifiable reward and many on-policy rollouts. Sequential SFT followed by GRPO is the dominant production recipe, provided SFT is not over-trained first.

The page covers:

- Head-to-head empirical comparisons on math, coding, and agentic tasks
- The Pass@1 versus Pass@k debate: amplification versus new capability
- Mechanisms: objective difference, zero-advantage groups, entropy collapse, distribution shift, and SFT over-training destroying RL plasticity
- Ordering effects: SFT then RL, RL then SFT, synchronised, and interleaved variants
- Agent behaviour after training: trajectory length, tool calls and turns, redundancy, abstention, error recovery, interaction collapse, and entropy
- A scenario selector for when to use SFT only, GRPO only, or SFT followed by GRPO
- Negative and null results, and how behaviour is measured

Sources are primary papers with arXiv links, listed on the page. Figures flagged there as secondary summaries are marked as such on the page and should not be quoted as primary measurements.

## Files

```text
index.html    self-contained reading page (English)
README.md     this file
```
