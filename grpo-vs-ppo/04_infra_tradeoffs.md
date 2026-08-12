# 04 — Infra / vLLM Rollout Tradeoffs

## Infra 架构

RLHF/GRPO loop:

```
policy_gen (vLLM) → reward scoring → advantage → training (FSDP/megatron)
```

PPO 需额外 `critic forward` 每 step；GRPO 省这一步。

## vLLM Rollout

- GRPO 天生的 **group rollout**: $q$ → $G$ samples。vLLM `n=G` 并行，KV cache复用 prefix。
- PPO 单条采样多，吞吐要求高：often 1 sample per prompt * many prompts。
- Token accounting: GRPO 优势显存 → 可把 batch 换成更大 rollout token 数。经验：7B, L=2048, G=16, B=128 queries → rollout tokens 4.1M/token-step，vLLM 150-300 tok/s/GPU。

Config tip (verl):

```yaml
rollout:
  engine: vllm
  n: 16               # G
  temperature: 0.9
  top_p: 0.95
  max_tokens: 2048
  tensor_parallel: 2
```

## Critic elimination

- FSDP shard: PPO: actor-shard + critic-shard 双份 all-gather。GRPO: 单份。
- CPU offload: PPO critic optimizer state 占 2x P (Adam m/v)。GRPO 省。
- Case 70B on 8x H100: PPO 4-model needs 2 nodes (16 GPUs) 才不 OOM; GRPO 4-model minus critic fits 8 GPUs with ZeRO-3 + offload.

## Multi-step Agentic RL

Tool RL loop 长：

- PPO: $V(s_t)$ 给中间 step credit，但 $V$ 预训练弱时崩。需 process reward.
- GRPO: 最简可用 outcome reward 0/1 驱动整条 trajectory，配合 DAPO token-level loss masking (tool output 不训)。

Practice:

- Mask tool observations in loss: loss only on `assistant` tokens.
- Length bias: GRPO 原版除以 $|o_i|$ 导致偏短；Dr.GRPO 去掉此项更好解释长度 hack.

## Memory-saving recipe (no employer IDs)

- LoRA RL: Freeze base, train LoRA for actor only, ref stays full (KL exact).
- Recompute + CPU offload + sequence parallel for rollout->train bridging.
- Async rollout: vLLM server separate from trainer, queue queries.

## Eval

- track `group_std`, `all_correct_rate`, `all_wrong_rate` to know when advantage zero.
- PPO track `value_loss`, `explained_variance` to debug critic.
- rollout time / train time ratio target 1:1 to 2:1.

> ZH：省 critic 的显存拿去堆 G 和 vLLM 吞吐，是 GRPO 上大模型 infra 主因。  
> EN: Save critic memory to scale G and vLLM throughput.
