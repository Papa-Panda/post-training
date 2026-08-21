# 04 — Failure Taxonomy (Rollout side, mirrors FSDP taxonomy)

Mapping from FSDP failure modes (NCCL timeout, OOM, deadlock) to vLLM rollout failures.

## 5-way + 2 infra (same as day-07 sim, extended)

| # | Type | Symptom | Root cause | Detect | Mitigate | $ bleed |
|---|------|---------|------------|--------|----------|---------|
| 1 | `timeout` | wall > SLO 30s P95 | long CoT 5000 decode bound + queue, λ > μ (Little's law) | P95 wall >3×P50 | ↓λ, ↑max_num_seqs, chunked prefill | 40% of fails, dominant long |
| 2 | `tool_call` | `json parse error`, sandbox returns 500 | code exec timeout, unparsable tool | log `tool_parser` error | retry 3× + cool 10min, sandboxed timeout 15s | 30% |
| 3 | `vcj_verifier` | reward 0 but answer right | verifier prompt brittle, unit test timeout | mismatch manual audit | DAPO decoupled clip — don't discard, down-weight | 15% |
| 4 | `oom_kv` | `KVCacheFull`, `EngineDead` | concurrency L >> cache blocks, max_model_len too big, `gpu_mem_util=0.90` no headroom | `free_blocks==0`, frag <0.3 | swap_space 0 for detection, prod 4GB; lower L or max_num_seqs | 10% |
| 5 | `contiguity` / `fragment` | free>0 but alloc fails | long prompt 16k splits block table non-contig (P17) | `frag>0.4` && OOM | enable_chunked_prefill, defrag on preempt restart | part of 4 |
| 6 | `nccl_preempt` | NCCL `Watchdog` 15m, engine stalls | TP=2/4/8 all-reduce deadlocks when one rank OOM, H100 NVLink | `NCCL WARN`, `engine dead` | `NCCL_ASYNC_ERROR_HANDLING=1`, TP restart group | 5% |
| 7 | `engine_deadlock` | no output, metrics frozen 30s, `async_engine` stuck | producer-consumer `step()` blocking on `seq_group_metadata` lock (vLLM #5678) | `metrics endpoint` not changing, `queue depth` rising | `max_num_seqs` too high, enforce eager off, vLLM bump |

## How to classify (reuse FSDP script logic)

Same classifier as `post-training/rl-infra/day-02-fsdp/…` but rollout side:

```python
def classify(log_line, metrics):
    if "TimeoutError" in log_line or metrics.wall > SLO: return "timeout"
    if "tool" in log_line.lower() or "JSON" in log_line: return "tool_call"
    if "verifier" in log_line or "VCJ" in log_line: return "vcj_verifier"
    if "KV cache" in log_line or metrics.free_blocks==0: return "oom_kv"
    if metrics.frag>0.4: return "contiguity"
    if "NCCL" in log_line: return "nccl_preempt"
    if metrics.frozen>30: return "engine_deadlock"
    return "ok"
```

Log to `failure_log.json`: `{idx, fail_type, wall, prompt_len, cot_len, free_blocks, frag, tp_rank_error}` — mirrors day-07 `/tmp/vllm_rollout_fail.json`.

## Rollout vs Training separation lesson

- Day-07 expected 80% wall-clock rollout (short) → 90% long. Real bottleneck is failure retries, not pure toks/sec.
- Reusable infra idea: translate datacenter PUE / burst prediction (autoscaling) → predict `queue_depth` burst, async eval, hysteresis retry (Day-06 note).
- Failure budget: keep ≤15% long CoT, otherwise $/useful rollout doubles.

## Checklist before claiming "H100 ready"

- [ ] CPU sim passes (fail_rate 7% short / 14% long) — `pytest` in `code/`
- [ ] H100 single run 50 samples logs TTFT/TPOT/toks/s P50/P95
- [ ] sweep.csv shows knee (max goodput before OOM)
- [ ] taxonomy CSV 5-way adds to 100% of failures, not ok
- [ ] NCCL env set: `NCCL_ASYNC_ERROR_HANDLING=1`, `TORCH_NCCL_HEARTBEAT_TIMEOUT=900`
- [ ] Metrics endpoint scraped 1Hz, no 30s freeze
