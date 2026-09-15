# 01 — Roles and Data Contracts

## 1. Components are roles, not necessarily processes

A small experiment may run all roles in one Python process. A cluster may spread one role across hundreds of processes. Correctness comes from the contract between roles, not from process count.

| Role | Owns | Must not silently own |
|---|---|---|
| prompt sampler | prompt IDs, curriculum, group assignment, seeds | reward normalization or learner versions |
| rollout actor | sampling actions under published policy | retroactively editing policy version |
| environment | state transitions, observations, termination evidence | policy gradients |
| context manager / harness | prompt construction, tool protocol, truncation | claiming reward correctness |
| verifier / reward service | raw evidence, score, reward version | changing trajectory bytes |
| reference model | reference log-probabilities | rollout behavior provenance |
| critic | value estimates and value targets | policy publication |
| advantage builder | return/baseline/group statistics | splitting an unfinished group |
| buffer | durable envelopes, indexes, leases, admission result | inventing missing fields |
| learner | update manifest, gradients, optimizer state, checkpoints | marking a checkpoint published before validation |
| publisher | atomic policy artifact and worker acknowledgement | changing optimizer state |
| evaluator | held-out metrics under frozen contracts | feeding hidden answers into training |

The same model can serve multiple roles, but role identity still matters. For example, a policy and reference model may share initialization yet have different update lifecycles.

## 2. Minimum trajectory envelope

A practical schema is grouped by provenance rather than by convenience:

```json
{
  "trajectory_id": "content-addressed-id",
  "prompt": {
    "prompt_id": "p-123",
    "group_id": "g-45",
    "dataset_version": "d7",
    "curriculum_version": "c3",
    "seed": 91
  },
  "policy": {
    "segments": [
      {
        "action_start": 0,
        "action_end": 2,
        "behavior_version": 12,
        "artifact_digest": "...",
        "behavior_logprobs": [-0.7, -1.1]
      }
    ],
    "tokenizer_digest": "...",
    "decoding_digest": "..."
  },
  "sequence": {
    "input_ids": [101, 102],
    "output_ids": [201, 202],
    "action_mask": [0, 0, 1, 1],
    "termination": "environment_success"
  },
  "environment": {
    "name": "compiler-task",
    "version": "e5",
    "trace_digest": "..."
  },
  "score": {
    "raw_evidence": {"tests_passed": 8, "tests_total": 8},
    "reward": 1.0,
    "reward_version": "r4",
    "valid": true
  }
}
```

The exact serialization can differ. The important property is that each field has one writer and an auditable version. A single-version rollout has one segment. If an interrupted rollout resumes under newly published weights, append a segment with its own action interval, behavior version, artifact digest, and token-aligned behavior log-probabilities; never collapse the trajectory to one top-level version.

## 3. Why IDs and digests are separate

Human-readable versions help operations; content digests detect accidental mismatch.

- `behavior_version=12` supports ordering and lag $k-b$ .
- `artifact_digest` proves which concrete weight manifest was loaded.
- `tokenizer_digest` prevents silent token-ID reinterpretation.
- `decoding_digest` captures temperature, top- $p$ , stop rules, max length, and generation implementation choices that alter the behavior distribution.
- `environment.trace_digest` lets the system verify replay evidence without placing a large trace in every learner batch.

A version counter alone cannot detect a bad deployment in which one inference shard loaded different bytes under the same label.

## 4. Action masks are semantic data

A chat/agent transcript can contain system prompts, user tokens, tool outputs, inserted observations, and model actions. Only selected model actions should receive policy loss. If $M(\tau)=\{t:m_t=1\}$ , then a masked loss is normalized over valid actions:

$$L_{\rm NLL}(\tau)=-\frac{\sum_t m_t\ell_t}{\max(1,\sum_t m_t)}.$$

Two common corruptions are:

1. training on environment or tool-output tokens as if the policy chose them;
2. comparing behavior and learner log-probabilities under different prefix construction or truncation.

Therefore the mask contract and context-construction version belong in provenance.

## 5. Grouped estimators require grouped records

For a prompt $x_i$ , GRPO-like methods sample $G$ completions with rewards $R_{i,1},\ldots,R_{i,G}$ . A simple standardized group advantage is

$$A_{i,j}=\frac{R_{i,j}-\bar R_i}{s_i+\varepsilon},\qquad \bar R_i=\frac1G\sum_{j=1}^G R_{i,j}.$$

The framework must know when a group is complete, partially failed, or timed out. If the buffer releases arbitrary samples before group statistics are finalized, the estimator changes. Valid choices include:

- wait for all $G$ members;
- apply a declared partial-group policy with a minimum size;
- replace failed members using the same prompt/version contract;
- reject the group.

Silently computing a group baseline over whichever completions arrived first is not an implementation detail.

## 6. Reward evidence and reward value

Store both the evidence and its transformation:

$$R=F_{r}(E(\tau),\text{task metadata}).$$

`E` might contain compiler exit status, unit-test names, simulator state, rubric votes, or a reward-model score. `F_r` may normalize, clip, combine process/outcome rewards, or apply penalties. Versioning only the final float prevents later audits from distinguishing an environment bug from a normalization change.

Invalid execution should be typed:

- policy produced an invalid action;
- tool service failed;
- environment timed out;
- verifier failed;
- trace was truncated;
- schema mismatch.

Only the first is automatically evidence about policy quality. Infrastructure failures need separate treatment.

## 7. Buffer semantics

A data buffer connecting generation and training needs more than `put/get`:

- idempotent append keyed by trajectory ID;
- indexes by policy, task, group, length, score state, and creation time;
- eligibility/admission result with reason;
- leases and acknowledgements;
- high/low watermarks for backpressure;
- retention and replay policy;
- dead-letter storage for malformed or failed records;
- a manifest for every learner batch.

For queue size $Q(t)$ , mean arrival rate $\lambda$ and service rate $\mu$ , sustained $\lambda>\mu$ implies unbounded growth in the simplest model:

$$\mathbb E[Q(t+\Delta)-Q(t)]\approx(\lambda-\mu)\Delta.$$

Backpressure can slow generation, reduce concurrency, lower sampling length, or reject low-priority work. Dropping arbitrary old samples changes the training distribution and must be observable.

## 8. Ownership and failure isolation

A safe failure boundary follows three rules:

1. **Rollout failure does not mutate learner state.** It creates a typed failed envelope.
2. **Learner failure does not republish partial weights.** The last committed policy remains active.
3. **Reward failure does not erase the trajectory.** Scoring can be retried against immutable content if the reward contract permits it.

For agent environments, external side effects also need idempotency keys or sandbox reset. Replaying a tool call is not equivalent to replaying a pure token transition.

## 9. Data hygiene before admission

The buffer is also the clean interface for:

- exact and semantic deduplication;
- prompt/trajectory contamination checks;
- hard-case and curriculum tags;
- reward-hacking filters based on raw evidence;
- distribution rebalance by task, difficulty, length, language, or source;
- disagreement routing when annotators, verifiers, or reward models conflict.

Filtering is a versioned transformation. Otherwise a later experiment cannot reconstruct why two runs with the same prompt dataset trained on different effective distributions.

## 10. Contract tests worth automating

- the number of action-mask ones equals the number of stored action log-probabilities;
- token IDs decode under the declared tokenizer digest;
- behavior log-probabilities can be recomputed within a declared tolerance;
- every group is complete or has an explicit partial-group disposition;
- a consumed trajectory appears in exactly one update manifest per allowed reuse epoch;
- policy publication manifest and loaded shard digests agree;
- reward evidence satisfies the declared verifier schema;
- hidden evaluation identifiers never occur in the training buffer.
