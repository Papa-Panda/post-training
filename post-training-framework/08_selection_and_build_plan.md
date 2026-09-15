# 08 — Selection and Build Plan

## 1. Start from the research question

Framework selection should follow the first falsifiable experiment.

Write down:

- policy objective and exact estimator;
- prompt, trajectory, group, reward, and mask semantics;
- model size/type, dense or MoE, full fine-tuning or adapter;
- environment and tool latency distribution;
- expected rollout/train/reward bottleneck;
- hardware/topology and operational constraints;
- held-out success and regression criteria.

If these are unknown, a large distributed framework increases search space before it solves a demonstrated bottleneck.

## 2. Stage 0 — Research notebook

Goal: prove reward, data, and estimator logic without distributed execution.

Deliverables:

- 10–100 hand-inspected trajectories;
- deterministic reward/verifier tests with raw evidence;
- manually checked token masks and group advantages;
- contamination and reward-hacking checklist;
- fixed held-out prompts never used for proposal tuning;
- trajectory schema that can survive migration.

Exit criterion: one fixed batch produces the expected finite loss and update direction.

## 3. Stage 1 — Single-process or single-node baseline

A TRL-based baseline is often appropriate when its trainer supports the algorithm, with Unsloth considered when its supported efficiency path matches the model/hardware.

Measure:

- model memory by phase;
- rollout and train token rates;
- reward latency;
- log-probability recomputation agreement;
- all-zero group, entropy, KL, clip fraction, and held-out success;
- exact batch identities from [Chapter 04](04_batch_semantics.md).

Exit criterion: the learning signal and evaluation trend are reproducible across seeds or repeated runs within declared variance.

## 4. Stage 2 — Distributed synchronous

Move to veRL or a suitable slime configuration when model size, rollout throughput, or backend needs justify it. Preserve estimator semantics first; do not add async at the same time.

Add:

- explicit resource pools and role placement;
- versioned buffer/envelopes even if batches are synchronous;
- training-to-generation reshard/publication protocol;
- backend calibration for tokens, masks, log-probabilities, and MoE routes;
- barrier and long-tail metrics;
- restart tests for rollout, reward, learner, and publisher.

Exit criterion: distributed synchronous results match the single-node baseline within declared numerical/statistical tolerance, and the bottleneck is measured.

## 5. Stage 3 — Overlap and disaggregation

Before full asynchrony, overlap one boundary at a time:

1. reward with rollout;
2. evaluation with the next iteration;
3. rollout prefill with late training work;
4. dedicated rollout and training pools with explicit publication barriers.

For every change, compare useful accepted-token throughput and held-out quality. This stage often captures much of the utilization benefit while retaining bounded freshness.

Exit criterion: each overlap has a measurable benefit and no unexplained estimator shift.

## 6. Stage 4 — Fully asynchronous agent RL

Use an AReaL-style async architecture, or an explicitly supported async mode in another stack, only when heavy-tail rollout/environment latency dominates and the team can own mixed-version semantics.

Required controls:

- immutable behavior version/log-probability per token or segment;
- queue age and version-lag distributions;
- ratio/KL/ESS/clip diagnostics by lag bucket;
- max-lag and hard-ratio admission gates;
- high/low watermark backpressure;
- interruption/resume policy and group disposition;
- atomic weight publication and worker acknowledgement;
- matched-token and matched-wall-clock quality experiments.

Exit criterion: async improves useful throughput or time-to-quality over the synchronous control while passing quality and recovery gates.

## 7. Migration contract

Keep these portable across frameworks:

| Contract | Portable content |
|---|---|
| trajectory | IDs, tokens/actions, masks, behavior log-probs, versions, termination |
| reward | raw evidence, transform/version, invalid reason |
| grouping | prompt/group IDs, expected size, partial-group policy |
| update manifest | sample IDs, weights, normalization, learner/old/reference versions |
| evaluation | frozen datasets, harness, metrics, confidence rule |
| observability | stage timestamps, queue/lag/ratio fields, failure taxonomy |
| publication | checkpoint digest, source learner step, target layout, acknowledgement |

If migration changes these contracts, it is a new experiment, not only a system port.

## 8. Migration costs

### Algorithm cost

Trainer callbacks and objective code may assume framework-specific tensor shapes, reduction units, or old-policy semantics. Revalidate the math.

### Data cost

Buffers, rollout outputs, and reward schemas need adapters. Preserve immutable IDs so paired comparisons remain possible.

### Backend cost

Model formats, optimizer states, sharding layouts, tokenizers, kernels, and generation engines may differ. A loadable checkpoint is not proof of log-probability equivalence.

### Operations cost

Cluster scheduling, images, logging, fault recovery, storage, and permissions become part of the experiment. The team must be able to debug them.

### Evaluation cost

A speed-focused migration can change decoding, task mix, truncation, or reward concurrency. Run the same frozen evaluator and inspect trajectory-level differences.

## 9. Build checklist

### Mathematical

- [ ] behavior, old, learner, and reference policies are distinct fields;
- [ ] loss normalization unit is declared;
- [ ] group completion/partial policy is declared;
- [ ] stale/off-policy correction and rejection rules are explicit;
- [ ] truncation/bootstrap semantics are explicit.

### Data

- [ ] trajectory schema has one writer per field;
- [ ] tokenizer, decoding, harness, environment, reward, and policy are versioned;
- [ ] duplicate, contamination, reward-hacking, and disagreement handling exists;
- [ ] update manifests make consumption auditable.

### Systems

- [ ] placement and peak memory are measured by phase;
- [ ] rollout/train/reward service rates and queue ages are known;
- [ ] weight synchronization is atomic and acknowledged;
- [ ] KV caches are namespaced/invalidate by policy identity;
- [ ] partial rollout and side-effect recovery are tested.

### Evaluation

- [ ] frozen equivalence batch passes across rollout and learner backends;
- [ ] single-node and distributed results are compared;
- [ ] held-out capability and regressions gate promotion;
- [ ] useful accepted-token throughput is reported;
- [ ] async is compared at matched tokens and matched wall-clock.

## 10. Example decision records

### Case A — early GRPO experiment on one model

Choose the shortest trainer path, keep groups intact, verify rewards and masks, and establish held-out improvement. Do not deploy a distributed queue before rollout is measured as the bottleneck.

### Case B — large model, synchronous RL, shared GPU fleet

Evaluate veRL's placement/resharding abstractions or slime when Megatron/SGLang is the committed stack. Start with synchronized policy versions and measure phase bubbles.

### Case C — web/tool agents with minute-scale heavy tails

Build durable environment state and trajectory envelopes first. Compare a bounded-overlap baseline to a fully async AReaL-style design with explicit lag/ratio controls.

### Case D — consumer GPU or limited memory

Evaluate whether Unsloth supports the exact model and RL path, using TRL-style trainer semantics as the comparison baseline. Treat speed/memory claims as configuration-specific and verify log-probability/mask consistency.

## 11. The final selection rule

Choose the smallest stack that preserves the estimator and removes the measured bottleneck:

$$F^*=\arg\min_{F\in\mathcal F}\mathrm{Complexity}(F)\quad\text{subject to correctness, capacity, and time-to-quality constraints}.$$

“Most features” is not the objective. A framework is successful when it makes the experiment easier to falsify, the update easier to audit, and failures easier to isolate.
