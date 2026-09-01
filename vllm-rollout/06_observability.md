# 06 — Observability and evidence

## 1. Four correlated layers

### Client/workload

- offered request rate and concurrency;
- prompt/output length histograms;
- candidate groups and completion skew;
- client queue, timeout, cancellation, retry, and duplicate counts.

### Engine/scheduler

- running and waiting requests;
- KV-cache usage;
- preemption and recomputation counters where exposed;
- prompt and generation token counters;
- TTFT, ITL/TPOT, E2E, and queue latency histograms;
- finish reasons and engine restarts.

### Hardware/runtime

- device memory used/free and out-of-memory events;
- utilization and power sampled with timestamps;
- host CPU/RAM, network, and storage where relevant;
- collective errors and worker health for distributed layouts.

### RL data plane

- behavior and learner policy versions;
- wall age and version lag;
- tool/scorer/verifier outcomes;
- accepted, dropped, duplicated, and retried tokens;
- group closure latency.

All layers need a shared run ID and timestamps from synchronized clocks.

## 2. Metric names are an API surface

vLLM exposes Prometheus metrics from the API server's `/metrics` endpoint, but names have changed across releases. For example, versioned docs show both `vllm:gpu_cache_usage_perc` in older releases and `vllm:kv_cache_usage_perc` in newer documentation. Pin the release and save a raw scrape before writing dashboards.

The helper preserves labels in long-form CSV and recognizes both cache-name variants:

```bash
python3 code/metrics_profile.py \
  --endpoint http://127.0.0.1:8000/metrics \
  --interval 1 --duration 120 --out /tmp/vllm-metrics.csv
```

It exits nonzero and deletes partial output if scraping fails. It never substitutes synthetic values for a missing server. Use `--all` during discovery, then explicitly select metrics in analysis.

## 3. Dashboard panels

A minimal dashboard should align:

1. offered versus completed request rate;
2. requested, processed, cached, recomputed, completed, and accepted token rates;
3. running/waiting requests and oldest queue age;
4. TTFT, ITL/TPOT, and E2E percentiles;
5. KV usage, preemption rate, and engine restarts;
6. timeout/failure counts by taxonomy;
7. useful goodput and retry amplification;
8. policy lag/age and freshness drops.

Correlations matter more than isolated thresholds. Rising waiting requests + flat completions indicates overload. High KV use + rising preemptions + rising recomputed prompt tokens indicates cache pressure. Stable engine latency + rising scorer backlog indicates the bottleneck is downstream.

## 4. SLO decomposition

Define separate service indicators:

- **availability:** request attempts completed without infrastructure failure;
- **latency:** percentile by request class and length bucket;
- **correctness:** protocol-valid output and token-accounting reconciliation;
- **data utility:** accepted trajectories or tokens per unit time/cost;
- **freshness:** accepted trajectories within the declared lag policy.

Do not make one “failure rate” serve all five.

## 5. Run evidence checklist

- [ ] engine and benchmark versions/commits;
- [ ] model/tokenizer revisions and dtype/quantization;
- [ ] exact server arguments and environment manifest;
- [ ] workload trace or generator seed;
- [ ] warm-up and measurement boundaries;
- [ ] raw client records with rollout and attempt IDs;
- [ ] raw Prometheus scrape and engine logs;
- [ ] hardware topology and telemetry source;
- [ ] policy snapshot lineage and acceptance rule;
- [ ] token-ledger reconciliation;
- [ ] repeated runs with dispersion;
- [ ] no synthetic output labeled as hardware measurement.

## 6. Alert examples

Prefer multi-signal alerts:

- **overload:** waiting requests and oldest queue age rise across several intervals while completion rate is flat;
- **KV thrash:** preemption/recompute rate rises with high KV usage and worsening TTFT/E2E;
- **replica failure:** request errors plus missing scrape/health signal, not a single absent metric name;
- **stale-data waste:** freshness drops rise while raw generation remains high;
- **downstream bottleneck:** engine queue stable but scorer/tool queue and group-closure latency rise.

Every alert should link to the run manifest and preserve the evidence needed for [failure classification](04_failure_taxonomy.md).
