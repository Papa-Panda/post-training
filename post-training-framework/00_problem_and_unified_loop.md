# 00 — Problem and Unified Loop

## 1. The framework's real optimization object

Let $x\sim\mathcal D$ be a prompt plus environment initial state. A behavior policy with immutable publication version $b$ generates a trajectory

$$\tau=(x,o_0,a_0,o_1,a_1,\ldots,o_T),\qquad \tau\sim\pi_b(\tau\mid x,h_b,e_b).$$

Here $h_b$ identifies the harness/context contract and $e_b$ the environment contract. A text-only completion is the special case in which observations are token prefixes and actions are generated tokens. An agent trajectory may additionally contain tool calls, compiler output, web observations, retries, and explicit termination events.

The intended objective is an expected return under the current policy:

$$J(\theta)=\mathbb E_{x\sim\mathcal D,\,\tau\sim\pi_\theta}[R(\tau,x)].$$

The system rarely evaluates that expectation directly. It samples with a behavior policy $\pi_b$ , scores the trajectory, constructs advantages, and updates learner version $k$ . Therefore the actual framework problem is to preserve the estimator's data contract while scheduling expensive, stateful components.

## 2. Token-level record

For a generated action token $y_t$ , define:

| Field | Meaning |
|---|---|
| $m_t\in\{0,1\}$ | action mask; \$1\$ only where policy loss is allowed |
| $\ell_t^b=\log\pi_b(y_t\mid s_t)$ | behavior log-probability captured for the sampled token |
| $\ell_t^k=\log\pi_k(y_t\mid s_t)$ | learner-policy log-probability recomputed at update time |
| $\ell_t^{\rm ref}$ | frozen reference-policy log-probability, if a KL term uses one |
| $A_t$ | token- or trajectory-level advantage assigned to this token |
| $v_t$ | optional critic estimate |

Let $\mathcal A(\tau)$ contain every stochastic policy-generated action in the trajectory. The loss mask selects $M(\tau)=\{t\in\mathcal A(\tau):m_t=1\}$ ; therefore $M(\tau)$ may be a strict subset of the actions needed for an exact trajectory density ratio.

A token-level current-to-behavior importance ratio is

$$r_t(k,b)=\exp(\ell_t^k-\ell_t^b).$$

In the synchronous PPO case where $\pi_b=\pi_{\rm old}$ , the proximal ratio equals this behavior ratio and one common policy term is

$$L_{\rm clip}=\mathbb E_t\left[m_t\min\left(r_tA_t,\,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)\right].$$

When $\pi_b\ne\pi_{\rm old}$ , these roles must not be conflated: behavior correction uses $\pi_{\rm old}/\pi_b$ , while proximal clipping is centered on $\pi_k/\pi_{\rm old}$ . How those factors are truncated or combined is algorithm-specific; simply clipping $\pi_k/\pi_b$ around one is not standard PPO.

This chapter does not rederive PPO or GRPO. Those derivations live in [`grpo-vs-ppo/`](../grpo-vs-ppo/README.md). The systems point is that every denominator must name the concrete distribution it represents. A field named `old_logprob` is not sufficient unless its provenance is clear.

## 3. Four policies that are often conflated

| Name | Role | Lifecycle |
|---|---|---|
| behavior policy $\pi_b$ | sampled the recorded action | immutable provenance per action segment; one trajectory may have several segments |
| learner/current policy $\pi_k$ | receives gradients now | changes after optimizer updates |
| old/update policy $\pi_{\rm old}$ | denominator for a chosen update epoch | often a snapshot, not necessarily the rollout policy in async systems |
| reference policy $\pi_{\rm ref}$ | anchors KL or preference objective | usually frozen or updated on a slower schedule |

In a strictly synchronous one-update loop, $\pi_b=\pi_{\rm old}$ at batch construction, so code can hide the distinction. In asynchronous execution, $b<k$ is normal, and $\pi_b$ , $\pi_{\rm old}$ , and $\pi_k$ can all differ. A robust schema names each explicitly.

## 4. Unified six-stage transition

### Stage A — sample prompts

The sampler selects prompts, group IDs, seeds, difficulty bands, and environment versions. A curriculum is part of the data distribution and must be versioned.

### Stage B — generate trajectories

Rollout workers load published policy $b$ , then generate tokens/actions. Every trajectory is sealed with tokenizer, decoding, harness, and environment identities plus one or more behavior-policy segments. A segment records its action interval, concrete weight identity, and token-aligned behavior log-probabilities.

### Stage C — score and validate

A rule verifier, reward model, process reward model, environment, or combination produces raw evidence and normalized rewards. Invalid executions are not silently converted to low reward; they retain a failure reason.

### Stage D — estimate advantages

The framework computes returns, critic targets, group-relative baselines, or leave-one-out baselines. This step defines which records must remain together.

### Stage E — update

The learner creates mini-batches and micro-batches, recomputes current-policy log-probabilities, applies the chosen correction/clipping, accumulates gradients, and steps the optimizer.

### Stage F — publish and evaluate

A successfully committed learner checkpoint becomes publication version $k+1$ . Rollout workers switch only after loading and acknowledging that version. Held-out evaluation observes both capability and system correctness.

## 5. State-machine view

A trajectory should move monotonically through explicit states:

```text
reserved -> generating -> sealed -> scored -> eligible -> leased
         -> consumed
         -> rejected(reason)
         -> expired(reason)
```

`sealed` means generation can no longer append tokens. `eligible` means all required fields, versions, and masks passed validation. `leased` prevents duplicate concurrent consumption. A lease timeout permits recovery after trainer failure without pretending the first attempt never happened.

## 6. Core invariants

### I1 — provenance completeness

For every sampled action token:

$$m_t=1\Longrightarrow(b,\ell_t^b,\text{tokenizer},\text{decoding},\text{mask contract})\text{ are known}.$$

### I2 — trajectory immutability

After sealing, content-addressed trajectory bytes and provenance do not change. Derived fields can be attached as separately versioned records.

### I3 — estimator eligibility

Before learner version $k$ consumes a sample, it evaluates

$$\mathrm{eligible}(\tau,k)=\mathrm{schema\_ok}\land\mathrm{reward\_ok}\land\mathrm{freshness\_ok}\land\mathrm{ratio\_ok}.$$

Version lag is only one input to `freshness_ok`; measured ratios and task-specific validation remain necessary.

### I4 — exact-once accounting

A trajectory may be retried operationally, but an update manifest records whether it contributed zero or one weighted contribution to a particular optimizer step.

### I5 — publication atomicity

A policy version is visible to rollout only when all required shards and metadata belong to the same committed checkpoint. Mixed-generation shards are invalid.

## 7. Three clocks

Framework bugs often come from assuming one global step. Keep three clocks:

1. **generation clock:** rollout reservations and completions;
2. **learner clock:** optimizer updates $k$ ;
3. **publication clock:** versions acknowledged by inference workers.

The learner can be at update $k=17$ while the latest complete publication is $p=16$ and a slow rollout still uses $b=14$ . That is a normal async state, not necessarily a bug. It becomes a bug when the data contract or configured admission policy cannot describe it.

## 8. Evaluation is part of the loop

Training reward is not the final objective. At minimum record:

- held-out task success under a frozen evaluator contract;
- reward/verifier versions and disagreement rates;
- policy KL, entropy, clip fraction, and importance-ratio tails;
- invalid/timeout/tool-error rates by environment version;
- accepted valid tokens per second, not just generated tokens per second;
- duplicate, contamination, and reward-hacking filters.

A rising training reward with falling held-out success can come from reward exploitation, distribution narrowing, stale corrections, masking errors, or evaluation leakage. A framework must preserve enough evidence to distinguish them.
