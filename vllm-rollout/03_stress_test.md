# 03 — Stress-test design and configuration

## 1. A benchmark is a workload contract

Record this manifest before running:

```yaml
engine:
  version_or_commit: REQUIRED
  engine_mode: v1_or_v0
  model_revision: REQUIRED
  tokenizer_revision: REQUIRED
  dtype: REQUIRED
  quantization: none_or_named
  parallelism: {tensor: 1, pipeline: 1, data: 1}
  scheduler: {max_num_seqs: null, max_num_batched_tokens: null}
  memory: {gpu_memory_utilization: null, kv_cache_dtype: null}
workload:
  arrival_process: poisson_or_trace_or_closed_loop
  prompt_length_distribution: REQUIRED
  output_length_distribution: REQUIRED
  shared_prefix_fraction: REQUIRED
  candidates_per_prompt: REQUIRED
  stop_conditions: REQUIRED
  timeout_s: REQUIRED
policy:
  snapshot_id: REQUIRED
  rollout_logprob_source: REQUIRED
  max_accepted_version_lag: REQUIRED
```

A closed-loop client with fixed concurrency answers “how fast when clients always wait for completion?” An open-loop trace answers “what happens at a given offered rate?” Do not compare them as if they measured the same queue.

## 2. Staged matrix

### A. Correctness and low-load baseline

- one request, then small concurrency;
- exact prompt/output token counts;
- finish reasons and stop-token behavior;
- deterministic sampling case where supported;
- behavior-policy snapshot and behavior logprobs captured as required by the learner;
- no timeout, preemption, or rejected response hidden by the client.

### B. Length matrix

Test at least short/medium/long prompt crossed with short/medium/long output. Use empirical quantiles if a trace exists. Long prompts stress prefill and initial KV; long outputs stress resident KV and decode duration.

### C. Offered-load sweep

For open-loop arrival rate $\lambda$, increase load from clearly idle through saturation. Hold the request distribution fixed. At every point record:

- offered and completed requests/s;
- prompt and generation tokens/s;
- accepted-token goodput;
- queue depth and oldest age;
- TTFT, ITL/TPOT, and E2E percentiles;
- KV usage and preemption deltas;
- timeout, transport, engine, application, verifier, and freshness-drop counts;
- policy-version lag.

The saturation knee is where additional offered load no longer improves useful goodput or violates a constraint. It is not simply the highest generated tokens/s point.

### D. Scheduler and memory sweep

Sweep one dimension at a time around the baseline:

1. `max_num_seqs`;
2. `max_num_batched_tokens`;
3. engine memory utilization or explicit cache sizing;
4. prefix caching on/off for a controlled shared-prefix workload;
5. parallelism/replica layout.

Warm up before measurement, repeat runs, and report dispersion. A single run cannot separate scheduler variance from a real effect.

### E. Failure injection

- client disconnect and deadline expiry;
- invalid request and impossible context length;
- worker termination/restart;
- slow or failing tool/environment;
- scorer/verifier timeout;
- learner update during a long rollout batch;
- reduced KV capacity to force preemption.

Ensure retries preserve a stable logical rollout ID while each attempt receives a unique attempt ID.

## 3. CPU rehearsal

The simulator exercises admission, decode-priority scheduling, chunked prefill, finite KV tokens, recompute-shaped preemption, timeout, acceptance filtering, and policy-version age.

```bash
python3 code/rollout_lab.py simulate --config configs/stable.json --out /tmp/stable.json
python3 code/rollout_lab.py simulate --config configs/overload.json --out /tmp/overload.json
bash code/sweep.sh configs/stable.json /tmp/sweep.jsonl
```

Expected qualitative checks, not hardware claims:

- overload grows peak queue and timeout count;
- a prompt larger than total KV-token capacity is rejected as a capacity failure;
- accepted tokens never exceed completed tokens, which never exceed attempted generation;
- tighter KV capacity can increase recomputation/preemption and wasted work;
- longer queue residence can increase completion-time policy-version lag.

## 4. Real vLLM benchmark

Use the benchmark command shipped with the **installed** vLLM version; the official benchmark docs describe the current `vllm bench` suite. Save `--help` output with the run because flags evolve. Prefer a request dataset/trace that preserves token lengths and shared-prefix structure. Never label CPU-lab output as a vLLM measurement.

Recommended artifact layout:

```text
runs/<timestamp>/
  manifest.yaml
  client-summary.json
  requests.jsonl
  metrics.csv
  engine.log
  versions.txt
  environment.txt
  analysis.md
```

## 5. Acceptance gates

Choose numeric gates before the sweep. Example categories:

- request success and accepted-trajectory rate;
- P95/P99 TTFT and ITL;
- maximum queue age;
- preemption rate per completed request;
- useful-token goodput;
- maximum accepted policy-version lag;
- zero unexplained token-accounting discrepancy;
- zero silent fallback from real execution to simulation.

Values are workload-specific. The repository intentionally provides categories and equations rather than unsupported thresholds.
