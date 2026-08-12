# 01 — Serving Metrics Math

Serving ≠ training. 3 core user-facing numbers + 2 system numbers.

## TTFT (Time To First Token)
```
TTFT = queue + prefill
prefill ≈ (prompt_len^2 * d) / FLOPS for attention, linear in prompt_len for KV write
```
- prompt_len 4k vs 32k: TTFT grows ~8× due to quadratic prefill + KV contiguity.
- vLLM controls: `max_num_batched_tokens` caps prefill burst, `enable_chunked_prefill` splits long prompt.

## TPOT (Time Per Output Token) / decode latency
```
TPOT = max( matmul_time, HBM_read(KV) + HBM_read(weights) )
decode is memory-bound: TPOT ≈ (KV_size + weight_size) / HBM_BW
KV_size per token = 2 * L * d * bytes  (K+V)
For 32L 4096h bf16: ~0.5 MB / token / layer? actually 2*4096*2B=16KB per layer per token → 512KB/token total
```
- CoT 500→5000: TPOT stays flat until KV evict, then P95 spikes due to recompute.
- Expect 7B TPOT ~12-18ms, 13B ~22-30ms H100.

## Throughput
```
throughput_tokens_per_sec = concurrent_requests / avg_TPOT  (decode bound)
throughput_reqs_per_sec = 1 / (TTFT + cot_len*TPOT)  (Little's law inverted)
```
- vLLM 7B H100 single: 80-120 req/s @ cot_len 500 → 40k-60k tok/s
- same GPU long 5000: 8-15 req/s → 8-15k tok/s (KV pressure, not compute)

## GPU Util
```
GPU_util = (active_kernels_time) / wall_time
MVP: <70% means queue-starved or blocked on KV mem, not compute bottleneck
```
- Engine step timer: if `num_batched_tokens` << `max_num_batched_tokens` consistently, increase concurrency.

## Little's Law for Serving
```
L = λ * W
L = mean concurrent in system (queue+running)
λ = arrival rate (req/s)
W = mean latency = TTFT + cot_len*TPOT

=> required concurrency to saturate: L_sat = λ_target * W
If λ_target=100 req/s, W=0.6s (500tok*12ms) => L=60 running
vLLM needs max_num_seqs >= L + buffer (1.3×) => 80
```

Fail condition: `W` grows with queue → λ_fixed causes L blowup → KV OOM.

## $/useful rollout
Reuse from day-07:
```
cost_per_useful = gpu_hour_price * p95_wall_hours / useful_rollouts
useful = total * (1 - fail_rate)
fail_rate = fails / total   (5-way, see 04)
```
This is new PUE — was infra PUE, now RL infra.
