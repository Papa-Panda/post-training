# 04 — Batch Semantics

## 1. Four nested quantities

Names vary by framework, so define meanings before copying a configuration.

- $P$ : prompts sampled for one outer rollout/update iteration.
- $G$ : completions sampled per prompt.
- $B=PG$ : trajectories consumed by the outer update dataset.
- $M$ : **mini-batch size**, trajectories represented by one optimizer step.
- $\mu$ : **micro-batch size per GPU**, trajectories processed by one forward/backward invocation on each data-parallel rank.
- $D$ : number of data-parallel ranks.
- $E$ : number of epochs over the same train batch.

A common veRL-style configuration mapping is:

| Configuration concept | Symbol here | Unit to verify |
|---|---:|---|
| `train_batch_size` | commonly $P$ | often prompts before rollout expansion; verify whether the pinned version already counts trajectories |
| rollout response count (often `rollout.n`) | $G$ | completions per prompt |
| derived update trajectories | $B=PG$ | completed sequences before filtering/remainder handling |
| `ppo_mini_batch_size` | $M$ | global trajectories/sequences represented by one optimizer step |
| `ppo_micro_batch_size_per_gpu` | $\mu$ | trajectories/sequences per data-parallel rank per forward/backward wave |
| data-parallel world size | $D$ | replicas after accounting for tensor/pipeline/expert parallelism |
| `ppo_epochs` | $E$ | passes over the generated train batch |

Do not infer units from the option names: inspect the pinned framework version and log resolved prompt, trajectory, and action-token counts.

Assuming divisibility and equal trajectory weighting, the number of mini-batches per epoch is

$$N_{\rm mini}=\frac{B}{M}.$$

The global micro-batch capacity per forward/backward wave is

$$G_{\rm micro}=D\mu.$$

The number of gradient-accumulation waves per optimizer step is

$$A=\frac{M}{D\mu}.$$

The total optimizer steps and forward/backward waves for the train batch are

$$N_{\rm step}=E\frac{B}{M},\qquad N_{\rm fb}=E\frac{B}{D\mu}.$$

These identities require $B\bmod M=0$ and $M\bmod(D\mu)=0$ unless the framework has an explicit remainder policy.

## 2. Concrete example

Let `train_batch_size` count $P=256$ prompts and rollout sample $G=4$ completions per prompt. Then

$$B=PG=1024,\qquad M=256,\qquad \mu=4,\qquad D=8,\qquad E=2.$$

Then:

- the outer update contains $256\times4=1024$ trajectories;
- global micro-batch capacity is $D\mu=32$ trajectories;
- gradient accumulation is $A=256/32=8$ waves per optimizer step;
- mini-batches per epoch are $1024/256=4$ ;
- optimizer steps are $2\times4=8$ ;
- forward/backward waves are $2\times1024/32=64$ .

Changing `micro_batch_size_per_gpu` from 4 to 2 doubles accumulation waves and usually lowers activation memory, but it does **not** change $M$ , $B$ , or the number of optimizer steps.

## 3. “Batch” may count prompts, trajectories, or tokens

With $P$ prompts and $G$ completions per prompt, trajectory count is

$$B_{\rm traj}=PG.$$

If response lengths are $L_{i,j}$ , action-token count is

$$B_{\rm tok}=\sum_{i=1}^{P}\sum_{j=1}^{G}L_{i,j}.$$

A configuration saying `train_batch_size=256` is ambiguous unless it declares its unit. Prompt-count batching preserves group accounting; token-count batching improves device balance. Real systems need both logical grouping and physical packing.

## 4. GRPO group integrity

For group size $G\ge2$ , all completions for one prompt must be present when computing group-relative statistics; $G=1$ has no within-prompt comparison signal. A mini-batch boundary is valid when either:

1. $M$ is a multiple of $G$ and records are grouped contiguously; or
2. the framework computes group statistics before shuffling/packing and attaches immutable advantages.

The simple structural condition is

$$M\bmod G=0.$$

That condition is not sufficient by itself: a careless shuffle can still place partial groups into a mini-batch. Validate group IDs explicitly.

### All-zero or zero-variance groups

If all rewards are equal, then $R_{i,j}-\bar R_i=0$ . With standardized advantages, the group contributes no preference direction. Log:

- fraction of groups with zero variance;
- fraction with all-zero reward;
- group size after failures/timeouts;
- reward diversity by task and curriculum bucket.

An increasing all-zero fraction can mean tasks are too hard, verifier failure, mode collapse, or reward discretization—not just “bad data.”

## 5. Token-balanced packing

Trajectory counts poorly predict memory and compute under variable length. Let estimated cost of trajectory $i$ be

$$c_i=\alpha L_i^{\rm prompt}+\beta L_i^{\rm response}+\gamma(L_i^{\rm total})^2,$$

where the quadratic term approximates attention work when applicable. A packer should balance $\sum c_i$ across data-parallel ranks while preserving semantic group constraints.

Useful outputs are:

- action tokens per rank;
- padded tokens per rank;
- maximum sequence length per rank;
- group fragments per rank;
- estimated and measured step time.

Sequence parallelism, packing, and backend kernels change the true cost model, so the estimator must be calibrated rather than treated as law.

## 6. Loss normalization changes the estimator

Two common reductions are trajectory-mean and token-mean:

$$L_{\rm traj}=\frac1B\sum_{i=1}^{B}\frac{1}{|M_i|}\sum_{t\in M_i}\ell_{i,t},$$

$$L_{\rm token}=\frac{\sum_i\sum_{t\in M_i}\ell_{i,t}}{\sum_i|M_i|}.$$

`L_token` gives long responses more total weight; `L_traj` gives each trajectory equal weight. Under distributed training, local means followed by rank averaging are incorrect when valid-token counts differ. Aggregate numerator and denominator globally.

## 7. Reuse epochs and policy semantics

If $E>1$ , later epochs reuse trajectories after the learner has changed. Even a synchronously generated batch becomes off-policy relative to later mini-batches. The update-policy snapshot, ratio denominator, shuffling, and early stopping must be explicit.

A framework should distinguish:

- outer rollout/update iteration;
- update epoch over stored data;
- optimizer step;
- gradient-accumulation micro-step;
- policy publication version.

Calling all five `step` makes logs impossible to interpret.

## 8. Critic and actor batch sizes may differ

The actor, critic, reference model, and reward model need not use the same micro-batch because their memory and compute differ. But they must agree on trajectory/token identity. Join by immutable IDs and masks, not by incidental array order.

A critic may train for more epochs or use bootstrapped targets. Its version should be logged separately from the policy version.

## 9. Remainder policy

Production data is rarely perfectly divisible. Choose explicitly:

- drop incomplete mini-batch/group;
- pad with zero-loss records;
- carry remainder to the next update while preserving behavior version;
- form a smaller final optimizer step with corrected normalization;
- wait for replacement trajectories.

Each changes efficiency or sampling. A hidden `drop_last=True` can systematically discard long or slow trajectories if ordering correlates with completion time.

## 10. Configuration validator

Before launch, check:

```text
B > 0, M > 0, micro > 0, D > 0, epochs > 0
B % M == 0                         unless remainder policy is declared
M % (D * micro) == 0               unless uneven accumulation is supported
logical group size divides group-statistics batch
no physical pack splits an unfinished group contract
normalization unit is declared: prompt / trajectory / action token
every step counter has a precise meaning
```

The runnable validator in [`code/`](code/) intentionally implements the stricter first mode: optimizer batches retain complete groups. It does not model the alternative in which group statistics are precomputed, sealed, and then packed across group boundaries.
