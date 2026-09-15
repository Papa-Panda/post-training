# Post-Training Frameworks

> A framework is not an algorithm name. It is the executable contract that turns prompts, policy versions, rollout engines, environments, rewards, buffers, and learners into a reproducible update.

This topic synthesizes the **Agent RL engineering** section of the shared reading notebook. The notebook is an index of questions and articles, not a specification for a new framework. Here, its questions are reorganized around one system problem:

$$\text{prompt}\rightarrow\text{rollout}\rightarrow\text{score}\rightarrow\text{advantage}\rightarrow\text{update}\rightarrow\text{publish}\rightarrow\text{evaluate}.$$

The main architectural reference is **HybridFlow/veRL**. **AReaL**, **slime**, **Hugging Face TRL**, and **Unsloth** are compared at their actual abstraction levels rather than forced into a single leaderboard.

## What this topic answers

A useful post-training framework must make five things explicit:

1. **Data semantics:** Which policy produced every token, which log-probability was stored, which reward/version/mask belongs to it, and whether a sample may be consumed.
2. **Control semantics:** Which component owns the global RL dataflow, and which processes collectively execute one model operation.
3. **Placement semantics:** Which models share devices, when memory changes role, and how weights move from training layout to generation layout.
4. **Freshness semantics:** How behavior-policy lag is measured, bounded, corrected, rejected, and monitored.
5. **Evaluation semantics:** Whether throughput gains preserve the intended estimator, trajectory validity, and held-out capability.

## Reading route: mathematics → system → runnable model

| Step | Chapter | Question |
|---|---|---|
| 0 | [Problem and unified loop](00_problem_and_unified_loop.md) | What mathematical object flows through the system, and what must never be ambiguous? |
| 1 | [Roles and data contracts](01_roles_data_contracts.md) | Which component owns each field, transition, and failure? |
| 2 | [Synchronization and staleness](02_sync_async_staleness.md) | What changes when rollout and learning overlap? |
| 3 | [Control and placement](03_control_and_placement.md) | How do HybridFlow-style controllers, colocation, and resharding fit together? |
| 4 | [Batch semantics](04_batch_semantics.md) | How do train, mini-, and micro-batches map to optimizer steps and GRPO groups? |
| 5 | [Rollout and weight synchronization](05_rollout_and_weight_sync.md) | When are weights, caches, tokenization, and partial trajectories valid? |
| 6 | [Framework comparison](06_framework_comparison.md) | Which stack fits which research and scale regime? |
| 7 | [Observability and correctness](07_observability_correctness.md) | Which metrics distinguish speed from estimator corruption? |
| 8 | [Selection and build plan](08_selection_and_build_plan.md) | How should a project migrate without paying cluster complexity too early? |
| 9 | [Evidence ledger](sources.md) | Which claims come from papers, official project docs, or secondary reading leads? |

## Unified architecture

```text
prompt sampler / curriculum
          |
          v
rollout policy(version=b) <---- published weights ---- learner(version=k)
          |                                              ^
          v                                              |
environment / tools / compiler -> trajectory envelope -> buffer
                                      |                   |
                                      v                   |
                            reward / verifier / critic ---+
                                      |
                                      v
                           advantage + update batch
                                      |
                                      v
                           held-out evaluation + gates
```

The **trajectory envelope**, not a loose list of strings, is the unit of exchange. It carries prompt/group identity, token IDs, action masks, behavior-policy version and log-probabilities, reward/verifier versions, termination reason, and environment provenance.

## Framework-selection decision tree

```text
Need the shortest path from a model and dataset to an RL experiment?
  ├─ Yes, standard trainer API is enough
  │    ├─ Need a memory-efficient local/single-node path? -> TRL + Unsloth option
  │    └─ Prefer a transparent general trainer baseline?  -> TRL
  └─ No, need distributed rollout/training orchestration
       ├─ Want synchronous/on-policy-first dataflow and flexible placement? -> veRL
       ├─ Fully asynchronous agent rollouts are the primary bottleneck?     -> AReaL
       └─ Want Megatron training + SGLang rollout with a hackable buffer?   -> slime
```

This is a starting heuristic, not a benchmark result. Model support, licenses, hardware compatibility, operational maturity, and the exact project version must be checked at decision time.

## Boundaries with neighboring topics

| Existing topic | Canonical responsibility | This topic only covers |
|---|---|---|
| [`grpo-vs-ppo/`](../grpo-vs-ppo/README.md) | objective derivations and PPO/GRPO comparison | how estimators constrain batching, freshness, and observability |
| [`vllm-rollout/`](../vllm-rollout/README.md) | rollout-serving metrics, stress tests, failure taxonomy | how rollout engines connect to the learner and version contract |
| [`ai-infra/`](../ai-infra/README.md) | GPU/distributed compute, communication, memory, collectives | role placement, resharding, and framework control plane |
| [`harness-engineering/`](../harness-engineering/README.md) | agent state, tools, workflow, permissions, harness promotion | how an environment/harness produces replayable RL trajectories |
| [`model-aware-data-curation/`](../model-aware-data-curation/README.md) | model-aware sample value, coverage, generation, retention | buffer hygiene and curriculum interfaces only |

## Runnable CPU model

The standard-library simulator is deliberately small, but its queue tests are semantic: policy publication, long-tail barriers, queueing, learner-admission staleness gates, and strict GRPO grouped packing are real state transitions rather than printed claims. Separate unit tests cover importance-ratio arithmetic and clipping. The simulator reports makespan, useful trajectory throughput, maximum queue depth, mean/p95/max version lag, dropped trajectories, learner/rollout utilization proxies, and effective sample fraction. It intentionally admits by version lag only, represents one behavior version per trajectory, and does not model evolving target log-probabilities, ratio-based admission, ESS, or mixed-version segments; production interrupted/resumed rollouts require the segment-level schema in Chapters 01, 02, and 05.

```bash
python3 post-training-framework/code/demo.py
python3 -m unittest discover -s post-training-framework/tests -v
```

It is **not** a GPU performance predictor. It isolates correctness relationships that should also hold in a real distributed implementation.

## Non-negotiable invariants

- A token has one behavior policy, even if the learner later evaluates it under many policies.
- `policy_version` is metadata; the measured log-ratio is evidence. Equal version IDs do not guarantee equal distributions if decoding, tokenizer, masks, or kernels differ.
- A stale sample is not made on-policy by renaming its version. Any reuse must be justified by the estimator and measured ratios.
- GRPO groups remain atomic until group-relative statistics have been computed.
- KV state is valid only for the exact model/cache identity declared by the serving engine.
- Throughput is not an optimization target by itself; the useful unit is accepted, valid training tokens per wall-clock second at controlled quality.
- Reward improvements without held-out task improvement are a debugging signal, not success.

## Source policy

Primary papers and official project documentation support technical claims. The Chinese articles from the notebook remain in [sources.md](sources.md) as **secondary navigation leads**. Their unsourced staleness thresholds and unrelated source-leak/runtime claims are not promoted into facts here.
