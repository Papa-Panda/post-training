# 03 — Control, Placement, and Resharding

## 1. Two control scales

HybridFlow's central idea is easier to understand by separating control scales.

### Inter-model dataflow

A single logical controller expresses coarse dependencies among policy generation, reward, reference, critic, and actor update:

```text
prompts
  -> actor.generate
  -> reward.compute + reference.logprob + critic.value
  -> advantage
  -> actor.update + critic.update
```

This view lets an algorithm researcher change the RL dataflow without manually programming every rank-to-rank message.

### Intra-model execution

One logical model operation is collectively executed by many workers under data/tensor/pipeline/expert parallelism. A multi-controller worker abstraction handles these SPMD-like operations close to the distributed backend.

The hierarchy is therefore:

```text
single controller: dependencies between models and stages
multi-controller workers: collective execution inside one model role
```

It is not simply “Ray versus MPI.” The useful abstraction boundary is global RL logic versus internal model parallelism.

## 2. Orchestration substrate versus RL semantics

Ray-style actors and tasks are useful for launching role processes, reserving resources, handling futures, and coordinating failure/restart. That is the **orchestration substrate**. It does not by itself define:

- which policy/log-probability generated a token;
- when a GRPO group is complete;
- whether stale data is eligible;
- how training shards become inference shards;
- which reward or environment version is valid.

Keep these framework-level contracts above the scheduler. The same logical dataflow may use Ray, a batch scheduler plus RPC services, or another runtime; replacing orchestration must not silently change the estimator.

## 3. Role placement is a schedule

Let role $j$ require memory $M_j(t)$ and compute time $C_j(t)$ during phase $t$ . If two roles are colocated on one device, feasibility requires

$$\max_t\sum_j M_j(t)\le M_{\rm device},$$

unless inactive role state is offloaded, sharded, or transformed. Placement also determines communication and idle time.

Three common patterns are:

### Fully colocated

Rollout and training reuse the same devices in alternating phases.

- advantage: high aggregate device utilization when one phase would otherwise leave dedicated hardware idle;
- cost: switching layouts, freeing caches, and reshaping weights; no true rollout/train overlap on the same capacity.

### Disaggregated

Dedicated inference workers and training workers run concurrently.

- advantage: independent scaling and overlap;
- cost: repeated weight transfer, duplicated model memory, network dependency, and freshness lag.

### Hybrid

Some roles share devices while reward/environment services or overflow rollout remain separate. This is often the realistic choice when model roles have different memory and latency profiles.

## 4. A phase-time model

For alternating colocated execution,

$$T_{\rm iter}^{\rm coloc}=T_{\rm rollout}+T_{\rm switch}^{g\rightarrow t}+T_{\rm train}+T_{\rm switch}^{t\rightarrow g}.$$

For an idealized disaggregated pipeline at steady state,

$$T_{\rm iter}^{\rm disagg}\gtrsim\max(T_{\rm rollout},T_{\rm train},T_{\rm reward})+T_{\rm sync},$$

but the pipeline fill/drain, queueing, and stale-sample loss are hidden in that lower bound. A fast nominal pipeline can have worse useful throughput if $T_{\rm sync}$ or rejection is large.

## 5. Training layout versus generation layout

Training and generation optimize different communication patterns:

- training shards parameters, gradients, and optimizer state to fit memory and scale matrix operations;
- generation may use tensor/pipeline parallelism optimized for token latency, KV-cache capacity, and many concurrent sequences.

If a parameter tensor $W$ is held as training shards $S_i^{\rm train}(W)$ and inference requires shards $S_j^{\rm gen}(W)$ , publication performs a layout transform:

$$\{S_i^{\rm train}(W)\}\rightarrow\{S_j^{\rm gen}(W)\}.$$

This may require all-gather, all-to-all, point-to-point redistribution, host staging, or a distributed checkpoint format. “Copy weights” hides the main systems problem.

## 6. Resharding cost model

Let $P$ be total published parameter bytes. A rough lower bound is

$$T_{\rm reshard}\ge \max\left(\frac{P_{\rm moved}}{B_{\rm effective}},\,T_{\rm serialization},\,T_{\rm synchronization}\right).$$

`P_moved` may be less than $P$ for delta synchronization, but effective bandwidth includes topology, contention, serialization, and receiving-side installation. Publication frequency $f$ consumes network rate approximately

$$R_{\rm publish}\approx fP_{\rm moved}.$$

Reducing $P_{\rm moved}$ with deltas saves bandwidth only if all workers share a verified base version and updates are applied atomically.

## 7. 3D-HybridEngine as an architectural lesson

HybridFlow/veRL's 3D-HybridEngine addresses switching an actor between training and generation while using different parallel layouts. The durable lesson is not one API name:

1. model identity must survive layout changes;
2. resharding must be part of the schedule and cost model;
3. generation caches and training activations have distinct lifecycles;
4. transition barriers must prevent mixed versions;
5. placement should be chosen jointly with the algorithm's dataflow.

See the paper and project in [sources.md](sources.md) for implementation-specific claims.

## 8. Resource-pool abstraction

Represent devices as named resource pools with explicit membership and topology:

```text
pool_train: nodes 0..3, GPUs 0..7
pool_rollout: nodes 4..5, GPUs 0..7
pool_reward: nodes 6..7, GPUs 0..3
```

A role-to-pool mapping should declare:

- capacity and exclusivity;
- allowed parallel backends;
- colocated roles and phase schedule;
- network path to producers/consumers;
- failure/restart domain;
- weight and data format at boundaries.

A flexible placement API is valuable because the best mapping changes with model sizes, sequence lengths, reward cost, and cluster topology.

## 9. Communication surfaces

List every cross-role transfer, not only gradients:

| Transfer | Typical payload | Critical property |
|---|---|---|
| sampler → rollout | tokenized prompts, seeds, group IDs | deterministic identity |
| rollout → buffer | trajectories, log-probs, traces | append/idempotency |
| buffer → learner | packed token batches and manifests | eligibility and lease |
| actor → rollout | full/delta weights and metadata | atomic version |
| actor → reference/critic | optional initialization/checkpoints | lifecycle independence |
| environment → verifier | execution evidence | replayability |
| evaluator → gate | metrics and confidence intervals | isolation from training |

For long-context agent RL, trajectory and KV-related data can rival model-sync traffic. Measure bytes by surface rather than assuming parameter movement always dominates.

## 10. Failure domains

- A single-controller crash should be recoverable from durable stage/manifests rather than rerunning external side effects.
- A worker-group failure should invalidate its collective operation without corrupting unrelated roles.
- A publisher failure should leave the previous complete version active.
- An inference worker that misses an update should advertise its loaded behavior version rather than accepting jobs under the new label.
- A reward-service backlog should trigger backpressure or prioritization, not silently train on unscored placeholders.

## 11. Placement experiment checklist

For each candidate mapping, record:

1. peak memory by phase and role;
2. rollout, reward, train, and switch durations;
3. bytes and topology for weight/data transfers;
4. barrier idle and queue age;
5. policy lag and accepted-token rate;
6. restart blast radius;
7. algorithm code changes required by the placement.

A placement wins only if it improves the end-to-end objective under controlled quality, not because one kernel or one role reports higher utilization.
