# 07 — Observability and Correctness

## 1. Four planes of telemetry

A single “tokens/s” dashboard cannot explain an RL system. Organize telemetry into four planes.

| Plane | Main question | Examples |
|---|---|---|
| data | What samples entered and survived? | task mix, lengths, groups, duplicates, invalid/reject reasons |
| estimator | Is the update mathematically plausible? | advantages, ratios, KL, clip fraction, entropy, ESS, gradient norms |
| systems | Where is capacity/time spent? | queue age, prefill/decode, reward latency, train step, reshard, barriers |
| outcome | Did capability improve safely? | held-out success, regressions, calibration, reward/evaluator disagreement |

Every metric should carry run, policy, dataset, reward, environment, and code/config identities where applicable.

## 2. Throughput funnel

Measure successive rates:

```text
scheduled prompts
  -> completed trajectories
  -> schema-valid trajectories
  -> scored trajectories
  -> estimator-eligible trajectories
  -> consumed trajectories
  -> accepted action tokens
```

For each edge, record count, token count, and reason-coded loss. Define useful throughput as

$$X_{\rm useful}=\frac{N_{\rm accepted\ action\ tokens}}{T_{\rm wall}}.$$

A rollout optimization that raises generated token rate but increases timeout, stale rejection, or invalid log-probability rate may reduce useful throughput.

## 3. Latency and utilization

Track distributions, not only averages:

- prompt queue wait;
- prefill and decode time;
- environment/tool wait;
- reward/verifier latency;
- buffer residence time;
- learner batch assembly;
- forward/backward/optimizer time;
- checkpoint, reshard, publication, and worker load acknowledgement;
- end-to-end sample-to-update age.

Report median, p90/p95/p99, maximum, and task/length buckets. Long-tail rollout creates synchronous barrier bubbles even when mean latency looks healthy.

Useful utilization proxies include:

$$U_{\rm rollout}=\frac{\text{worker busy time}}{\text{worker available time}},\qquad U_{\rm learner}=\frac{\text{compute-active time}}{\text{allocated time}}.$$

High utilization can still be wasteful if work is rejected or duplicated, so pair it with $X_{\rm useful}$ .

## 4. Freshness dashboard

At learner consumption, log:

- per-segment version lag $k-b(t)$ and per-trajectory maximum lag;
- wall-clock age;
- mean/max absolute token log-ratio;
- ratio percentiles and non-finite count;
- clip fraction;
- approximate KL by task/length;
- importance-weight ESS;
- stale/ratio rejection counts and wasted tokens.

Plot held-out improvement and gradient statistics by lag bucket. A version bound is justified only when these diagnostics and outcome tests support it.

## 5. Estimator health

### Ratio and clipping

TRL's official PPO docs expose metrics such as objective KL, policy clip fraction, entropy, and value/ratio diagnostics. Their exact names vary by implementation; preserve definitions in the run schema.

Warning patterns:

- ratio median drifts from one: behavior/current mismatch or denominator bug;
- extreme tails increase with lag/length: stale or prefix mismatch;
- clip fraction near one: most samples are outside the useful trust region;
- KL jumps only on one backend: weight/tokenizer/mask mismatch;
- non-finite ratios: missing log-probs, underflow/overflow, or impossible tokens.

### Advantage and reward

Track reward and advantage by prompt group, task, length, and verifier version. Warning patterns:

- all-zero GRPO groups: no within-group learning signal;
- near-zero advantage variance everywhere: normalization or reward collapse;
- reward rises while held-out success falls: reward exploitation, leakage, or narrowing;
- high reward/evaluator disagreement: unstable proxy or environment bug;
- success plateaus within an epoch: repeated data may be exhausted or clipping may dominate.

### Entropy

Token entropy under current policy is

$$H_t=-\sum_a\pi_k(a\mid s_t)\log\pi_k(a\mid s_t).$$

A rapid entropy collapse can indicate excessive updates, reward overoptimization, narrow data, or masking errors. The desirable level is task/model-specific; use trends and controlled baselines rather than a universal threshold.

## 6. Cross-backend correctness

For a frozen calibration corpus, compare rollout and learner views:

1. exact token IDs and model-visible prefixes;
2. action masks and position IDs;
3. policy/reference artifact digests;
4. per-token log-probability deltas;
5. MoE route identity where required;
6. stop/truncation classification;
7. normalized loss numerators and denominators.

Fail the run before full training if mismatches exceed the empirically validated tolerance.

## 7. Reproducibility manifest

A run manifest should contain:

- source commit and dirty-state marker;
- framework and backend commits/releases;
- resolved config, environment/container identity, hardware/topology;
- model, tokenizer, adapter, and checkpoint digests;
- dataset/curriculum/filter versions;
- reward/verifier/environment/harness versions;
- random seeds and deterministic-mode limits;
- batching units and remainder/group policy;
- publication/admission/staleness settings;
- evaluation suites and contamination checks.

Bitwise determinism across distributed GPU kernels may be unavailable. Reproducibility then means the exact nondeterminism is declared and replicated runs show controlled variance.

## 8. Debugging ladder

### Level 0 — schema

Can every sampled action be joined to token, mask, behavior version, log-probability, group, reward evidence, and termination?

### Level 1 — frozen equivalence

With no optimizer step, do rollout and learner recomputation agree on tokens, masks, and log-probabilities?

### Level 2 — single batch

Can one fixed batch produce finite, manually checked advantages, ratios, losses, gradients, and optimizer changes?

### Level 3 — single worker loop

Does generate → score → update → publish change the next rollout's recorded version and behavior?

### Level 4 — distributed synchronous

Do sharding, reduction, packing, and publication preserve the single-worker result within tolerance? Where is barrier idle?

### Level 5 — overlap/async

Only now introduce queues, mixed versions, interruption, and independent scaling. Compare against the synchronous quality/control baseline at matched tokens and wall-clock.

Skipping directly to Level 5 makes nearly every failure look like “RL instability.”

## 9. Failure symptom matrix

| Symptom | First hypotheses | Disambiguating evidence |
|---|---|---|
| reward rises, eval falls | reward hacking, leakage, task-mix shift | raw verifier evidence, frozen held-out suite, distribution dashboard |
| learner idle | rollout/reward too slow, batch constraints | queue depth by stage, group completeness, p95 latency |
| queue grows | producer faster than learner | arrival/service rates, age/lag histogram |
| high stale rejection | publication too fast or learner too slow | version/time lag, queue residence, ratio tails |
| low GPU use with full queue | packing/micro-batch/kernel/communication | tokens per rank, profiler, collective and padding time |
| log-ratio tails explode | wrong denominator, prefix/mask/version mismatch | frozen cross-backend calibration |
| many all-zero groups | task too hard/easy, verifier bug, homogeneous sampling | reward evidence and task/difficulty buckets |
| throughput collapses after publish | reshard/load/cache invalidation | publication phases, bytes, acknowledgements, cache hit by version |
| duplicate updates | lease/retry bug | trajectory/update manifests and idempotency keys |

## 10. Promotion gate

A faster system configuration is promotable only when:

1. schema and frozen-equivalence tests pass;
2. no unexplained shift appears in ratio/KL/advantage diagnostics;
3. accepted valid token throughput improves;
4. held-out quality is non-inferior under a declared statistical rule;
5. resource cost and failure/recovery behavior are acceptable;
6. the run is reproducible from a pinned manifest.

System speed and learning quality are coupled outputs of one experiment, not independent checkboxes.
