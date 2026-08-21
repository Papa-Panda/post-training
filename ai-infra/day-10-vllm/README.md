# vllm-rollout — vLLM Rollout Stress Test for H100 beyond 7B

> Goal: turn `day-07-h100-beyond-7b` theory into reproducible vLLM rollout infra. Train needs FSDP (>7B => 13B/30B/70B), Rollout needs vLLM separation — 80% wall-clock → 90% when CoT 500→5000. This repo is the rollout half.

**Why this repo exists**
- RL rollouts dominate wall-clock, not training.
- 7B is minimal viable (2×80GB), but Staff scope = prove scaling to 13B/30B/70B + quantify $/useful rollout.
- `vllm_rollout_stress_test.py` in `post-training/rl-infra/day-07` was prototype; here we productize it with sweep + failure taxonomy aligned to FSDP failure taxonomy.

## Structure
```
.
├── README.md
├── 01_metrics.md              # TTFT/TPOT/throughput math, Little's law
├── 02_rollout_config.md       # tensor_parallel_size, max_num_seqs, gpu_memory_utilization, max_model_len
├── 03_stress_test.md          # concurrency sweep, prompt mix, OOM/contiguity detection
├── 04_failure_taxonomy.md     # KV pressure, engine deadlock, NCCL timeouts (mirror FSDP)
└── code/
    ├── vllm_rollout_stress_test.py
    ├── sweep.sh
    └── metrics_profile.py
```

## Quick link to theory source
Reuses `post-training/rl-infra/day-07-h100-beyond-7b/{README, vllm_rollout_stress_test.py, fsdp_h100_profiler_beyond7b.py}` — same $/useful rollout formula, same 5-way failure split.

## Scope
- H100-only for perf numbers; CPU fallback provides logic CI via Poisson arrival sim.
- No employer names; formulas > anecdotes.
- 7B/13B/30B expectations, not just 7B toy.
