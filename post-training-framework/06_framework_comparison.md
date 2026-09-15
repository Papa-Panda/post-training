# 06 — Framework Comparison

## 1. Compare layers before features

These projects overlap, but they are not interchangeable products at one abstraction level.

- **veRL/HybridFlow:** distributed post-training dataflow and placement framework with multiple training and rollout backends.
- **AReaL:** distributed system designed around fully asynchronous agentic RL and mixed-version training.
- **slime:** a hackable distributed RL stack coupling Megatron training, SGLang rollout, and a data-buffer interface.
- **Hugging Face TRL:** trainer library and reference path for post-training algorithms in the Transformers ecosystem.
- **Unsloth:** local/small-scale efficiency and workflow layer that exposes RL/fine-tuning paths, commonly around trainer APIs.

The comparison below uses the official sources in [sources.md](sources.md), checked for this topic on 2026-09-03. Project capabilities are version-sensitive.

## 2. Decision matrix

| Dimension | veRL | AReaL | slime | TRL | Unsloth |
|---|---|---|---|---|---|
| Primary abstraction | distributed RL dataflow + workers/placement | fully async rollout/train system | distributed training-rollout-buffer stack | algorithm trainer API | efficient local training/runtime workflow |
| Natural starting regime | synchronous/on-policy-first, flexible placement | asynchronous agent rollouts | disaggregated or colocated cluster stack | single process/node baseline, then distributed integrations | local/consumer or memory-constrained iteration |
| Training backend posture | official repo documents FSDP/FSDP2 and Megatron-LM integrations | project-specific distributed trainer/system | Megatron-LM-centered | Transformers/Accelerate ecosystem | optimized kernels/memory path around supported trainers/models |
| Rollout posture | official repo documents vLLM, SGLang, and Transformers integrations | async generation with interruptible rollout and reward service | SGLang engine/router | trainer-side generation/integration | local generation and supported RL workflow |
| Control focus | hierarchical RL dataflow and flexible placement | remove rollout/train barriers while controlling staleness | customize generation/reward/environment through buffer boundary | concise experiment API | reduce memory/time-to-first experiment |
| Main complexity paid | distributed role configuration and backend contracts | mixed-version estimator, queues, publication, admission | Megatron/SGLang operations and interface debugging | scaling/agent-environment orchestration grows outside core trainer | feature/backend support matrix and less cluster-level control |

This table is architectural, not a throughput ranking.

## 3. veRL / HybridFlow

### Best fit

- algorithm work needs a clear global RL dataflow;
- actor, critic, reference, reward, and rollout roles require flexible placement;
- model execution needs established distributed training/inference backends;
- synchronous correctness is the initial priority, with selective overlap or newer async features evaluated explicitly.

### Architectural value

HybridFlow presents a hierarchical hybrid programming model: a single controller expresses inter-model dataflow, while distributed worker groups execute intra-model operations. Its 3D-HybridEngine addresses actor transitions between training and generation layouts.

### Tradeoffs

- integration breadth creates a large compatibility surface;
- the best placement depends on model size, topology, sequence distribution, and reward cost;
- “supports backend X” does not prove every combination/version is equally mature;
- project documentation must be pinned to a commit/release for reproducible builds.

The HybridFlow paper reports 1.53x–20.57x throughput improvement over evaluated baselines/configurations. That range is evidence for those experiments, not a portable speedup promise.

## 4. AReaL

### Best fit

- long-horizon agent/environment rollout is heavy-tailed;
- rollout and training should scale and progress independently;
- the team is ready to model mixed behavior-policy versions explicitly;
- staleness-aware admission/update and strong queue observability are acceptable complexity.

### Architectural value

The AReaL paper makes fully asynchronous execution the primary design: generation, reward, and training overlap; rollouts can be interrupted; the learner can receive mixed-version batches; and policy staleness is part of the algorithm/system contract. Its interruptible workers can discard old KV state, load new weights, recompute the prefix, and continue, so one trajectory may contain segments from different policy versions; segment-level provenance and the paper's estimator are therefore essential.

### Tradeoffs

- async speedups are inseparable from estimator and admission choices;
- debugging and reproducibility require three-clock/version observability;
- fast generation can still waste work if the learner rejects stale trajectories;
- environment side effects complicate interruption/replay.

The current v5 abstract reports up to 2.77x speedup in its stated experiments; the paper body also contains a 2.57x comparison for a particular setup. Preserve those contexts instead of merging them into one universal number.

## 5. slime

### Best fit

- Megatron-LM is the intended training backend;
- SGLang is the intended rollout engine/router;
- researchers want a direct, customizable data-buffer boundary for generation, reward, verifier, and environment logic;
- independent debugging of training and rollout components is valuable.

### Architectural value

The official repository describes a three-part shape: Megatron training, SGLang rollout, and a data buffer connecting them. It documents customization paths and current examples for weight synchronization and asynchronous execution.

### Tradeoffs

- a focused stack can be easier to modify but narrows backend choices;
- users inherit operational knowledge of Megatron, SGLang, routing, and weight transfer;
- repository examples are moving project state, not peer-reviewed guarantees;
- custom buffer logic can accidentally redefine estimator semantics unless schemas and tests are strict.

## 6. Hugging Face TRL

### Best fit

- establish an auditable algorithm baseline quickly;
- compare reward/loss/metric behavior before cluster orchestration;
- work within the Transformers dataset/model/trainer ecosystem;
- the environment loop is simple enough to express through available trainer hooks.

### Architectural value

TRL's official PPO and GRPO trainer documentation exposes algorithm configuration and interpretable metrics. It is useful as a readable baseline and for small-to-medium experiments where the trainer abstraction remains the center.

### Tradeoffs

- complex distributed agent environments, independent rollout fleets, and custom cluster placement may require substantial integration outside the trainer;
- a short API does not remove behavior-logprob, mask, group, or version contracts;
- external acceleration layers must be validated against the trainer's semantics.

## 7. Unsloth

### Best fit

- local or constrained-GPU iteration is the bottleneck;
- the supported model/algorithm path matches the experiment;
- memory reduction and setup speed matter more than custom cluster orchestration;
- a researcher wants a practical route from notebook to fine-tuning/RL.

### Architectural value

The official repository and RL guide document RL workflows, including GRPO/DPO-family paths, and emphasize memory/speed optimizations. It can be a strong efficiency layer for the experiment regimes it supports.

### Tradeoffs

- it should not be compared as though it exposes the same cluster control plane as veRL, AReaL, or slime;
- optimized kernels/patches and model support are version-sensitive;
- headline speed or memory claims need their exact model, hardware, sequence, batch, and baseline before reuse;
- integrating a custom distributed agent runtime may outgrow the intended workflow.

## 8. A staged choice is better than a permanent choice

A project can evolve:

```text
TRL baseline
   -> Unsloth-enabled local efficiency if supported
   -> veRL or slime for explicit distributed rollout/training
   -> AReaL-style async path when heavy-tail rollout dominates
```

This is not a mandatory migration order. A team already committed to Megatron/SGLang may start with slime; a team designing fully async agent RL may start with AReaL. The important practice is to keep trajectory schemas, evaluation sets, and metric definitions portable so framework migration does not also redefine the experiment.

## 9. Questions to ask before choosing

1. Are policy updates required to be strictly on-policy, bounded-lag, or explicitly off-policy corrected?
2. Is the dominant cost training FLOPs, rollout decoding, environment latency, reward scoring, or weight synchronization?
3. Which training and inference backends are already operationally trusted?
4. Does the model use dense or MoE routing, full fine-tuning or adapters, and which precision?
5. Do trajectories contain tools/side effects that require durable event state and replay controls?
6. How variable are response lengths and reward latency?
7. Can the team debug distributed queues, version skew, and backend log-prob mismatches?
8. Which exact project commits/releases support the model and hardware?
9. What is the smallest baseline that can falsify the research idea?
10. Can the same held-out evaluation be run before and after migration?

## 10. Non-conclusions

The evidence does **not** establish that one framework is universally fastest, most stable, or most flexible. Published speedups use different models, clusters, baselines, rollout distributions, and definitions. Choose by bottleneck and required semantics, then benchmark the exact pinned stack.
