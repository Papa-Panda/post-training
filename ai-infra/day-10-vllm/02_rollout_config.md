# 02 — vLLM Rollout Config for RL vs Inference

RL rollouts ≠ prod inference. Different arrival, different failure tolerance.

## Key args

| Arg | RL rollout effect | Inference effect | H100 7B/13B rule |
|-----|-------------------|------------------|------------------|
| `tensor_parallel_size` | splits model, lowers TP memory, adds NCCL all-reduce in decode (TPOT +5-10%) | lowers latency at low QPS | 7B:1, 13B:2, 30B:4, 70B:8 |
| `max_num_seqs` | caps concurrency L in Little's law. Too low → under-util, queue grows upstream | same | 7B: 64-128, 13B 64, 70B 32. Must be >= λ*W |
| `max_num_batched_tokens` | caps prefill burst, avoids long prompt starving decode | TTFT control | 4096-8192 for mixed prompt, 16k if 32k prompts |
| `gpu_memory_utilization` | 0.90 Reserve for KV cache. RL CoT 5000 needs 0.90→OOM faster than 0.85 (more evict, more recompute) | 0.90 ok for short | 0.90 for 7B short, 0.85 for 13B long, 0.80 for 70B long |
| `max_model_len` | must ≥ prompt+CoT. Too large wastes KV blocks (>0.5GB) | truncate | set to max(prompt)+max(CoT)+256 |
| `enable_chunked_prefill` | on → long prompt (32k) chunked, protects TPOT P95 | on recommended | on for prompt mix >4k |
| `enforce_eager` | false→CUDA graph for decode (TPOT -10%) but fails on dynamic FIM | off for perf | false, but toggle if deadlock |
| `swap_space` | CPU swap for KV — hides OOM as slowdown (TPOT +30%) | no swap | 0 for stress test to expose OOM, 4GB for prod |

## RL specific

- Temperature 0.7-0.9, top_p 0.95 — higher entropy → length variance big → TPOT P95 >> P50, must sweep length mix.
- `best_of_n` / `n` sampling for Agentic RL: `n=4`  → 4× KV cache, L must /4 or OOM. Prefer sequential rollout groups (see sweep.sh).
- Separated mp: train job ≠ vLLM job — Day-07 notes rollout 80% wall. Using same H100 for both → GPU util <50% due to train preemption. Split: 8×H100 train, 4-8×H100 vLLM pool.

## Example presets

```bash
# 7B short CoT 500, high QPS
vllm serve meta-llama/Llama-3-8B \
 --tensor-parallel-size 1 \
 --gpu-memory-utilization 0.90 \
 --max-num-seqs 128 --max-num-batched-tokens 8192 \
 --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser llama3_json

# 13B long CoT 5000, lower QPS, more swap headroom
vllm serve meta-llama/Llama-3-13B \
 --tensor-parallel-size 2 \
 --gpu-memory-utilization 0.85 \
 --max-num-seqs 64 --max-num-batched-tokens 4096 \
 --max-model-len 9216 --enable-chunked-prefill

# 70B (needs 8× TP)
vllm serve meta-llama/Llama-3-70B \
 --tensor-parallel-size 8 \
 --gpu-memory-utilization 0.80 \
 --max-num-seqs 32 --enable-chunked-prefill --enforce-eager False
```

## RL rollout tip
Pass `sampling_params.seed` per rollout ID for reproducibility; log `request_id → fail_type` for taxonomy (04).
