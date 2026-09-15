# 02 — Synchronization, Asynchrony, and Staleness

## 1. Three execution regimes

### Synchronous alternating

```text
publish k -> generate complete batch with k -> stop rollout
          -> score/update k to k+1 -> publish -> repeat
```

The strongest simple invariant is $b=k$ at batch admission. Its cost is a global barrier: fast workers wait for the slowest rollout, and rollout devices may wait while training runs.

### Overlapped pipeline

Generation for a future batch overlaps scoring or learning for the current batch, but the system preserves bounded stages and explicit version windows. This can reduce bubbles without accepting arbitrary lag.

### Fully asynchronous

Rollout, reward, buffering, learning, and publication advance independently. A learner at $k$ may consume trajectories from several behavior versions $b$ . It removes global barriers but turns freshness into an estimator and queue-control problem.

## 2. Version lag is necessary but not sufficient

Let $b(t)$ be the behavior-policy version for action token $t$ . Define per-action lag and a conservative trajectory summary

$$\Delta_v(t,k)=k-b(t),\qquad \Delta_v^{\max}(\tau,k)=\max_{t\in \mathcal A(\tau)}\Delta_v(t,k).$$

For a single-version trajectory, this reduces to $k-b(\tau)$ . Also record wall-clock age

$$\Delta_t(\tau)=t_{\rm consume}-t_{\rm sample}.$$

Neither directly measures distribution shift. Many small updates may move the policy more than one large-looking version interval, and a version can change decoding or tokenizer metadata without an optimizer step. Distributional diagnostics include:

$$d_t=\ell_t^k-\ell_t^{b(t)},\qquad r_t=e^{d_t},$$

and aggregate KL estimates, ratio quantiles, effective sample size, and clip fraction.

For nonnegative sequence weights $w_i$ , one importance-weight diagnostic is

$$\mathrm{ESS}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.$$

A high version lag with ratios near one may be usable; low lag with extreme ratios can be dangerous. Admission should therefore combine metadata and measurements.

## 3. Change of measure

For a one-step expectation under current policy $\pi_k$ , data from behavior policy $\pi_b$ can be reweighted:

$$\mathbb E_{a\sim\pi_k}[f(a)]=\mathbb E_{a\sim\pi_b}\left[\frac{\pi_k(a\mid s)}{\pi_b(a\mid s)}f(a)\right].$$

Holding the initial-state distribution and environment/harness transition kernel fixed, a full trajectory with a possibly piecewise behavior policy has the following exact action-likelihood ratio:

$$\rho(\tau;k)=\prod_{t\in \mathcal A(\tau)}\frac{\pi_k(a_t\mid s_t)}{\pi_{b(t)}(a_t\mid s_t)}=\exp\left(\sum_{t\in \mathcal A(\tau)}d_t\right).$$

Here $\mathcal A(\tau)$ includes every stochastic policy action, even if the policy-loss mask later excludes some actions from optimization. Products become high variance for long trajectories. PPO-style token clipping controls each local contribution, but does not make arbitrarily stale trajectory data equivalent to fresh on-policy data. Clipping changes the estimator by trading variance for bias; advantage estimates and state visitation may also be stale.

## 4. Why clipping is not a freshness proof

In the synchronous case $\pi_b=\pi_{\rm old}$ , suppose $r_t=\pi_k/\pi_{\rm old}$ is outside $[1-\epsilon,1+\epsilon]$ . The clipped surrogate limits one gradient term, but:

- the state/prefix was visited under $\pi_b$ , not $\pi_k$ ;
- a critic target may have been built under an older value function;
- group membership and rewards reflect the old rollout distribution;
- long sequences accumulate many small mismatches;
- clipped samples may contribute little useful gradient while consuming training budget.

Therefore monitor lag, ratio tails, clip fraction, and accepted-token efficiency together.

## 5. Admission policy

A transparent async learner can use a staged gate:

```text
schema valid?
  no -> reject(schema)
group/reward complete?
  no -> wait or reject(incomplete)
version lag <= configured hard bound?
  no -> reject(stale_version)
recomputed ratios finite and within hard safety limits?
  no -> reject(ratio)
otherwise -> accept with the estimator's declared weighting/clipping
```

The hard version bound is an engineering control, not a universal algorithmic constant. It should be tuned against policy-change rate, ratio distributions, task horizon, and held-out quality.

## 6. The unsupported threshold trap

The source notebook includes rules of thumb such as particular stale-version ranges being safe for PPO or GRPO. Those thresholds are **not treated as verified facts here**. A version is not a standardized unit across frameworks or experiments.

Replace folklore with an experiment:

1. record per-segment/per-token $\Delta_v$ , trajectory $\Delta_v^{\max}$ , $\Delta_t$ , token log-ratios, KL, clip fraction, and ESS;
2. bucket samples by lag and task/length;
3. compare gradient norm/direction and held-out improvement by bucket;
4. sweep admission bounds under a fixed token and wall-clock budget;
5. promote a bound only if quality and stability remain controlled.

## 7. Throughput model

Let rollout task times in a synchronous wave be $T_1,\ldots,T_n$ . The wave duration is

$$T_{\rm wave}=\max_iT_i.$$

Rollout-worker idle time caused by the barrier is

$$I_{\rm barrier}=n\max_iT_i-\sum_iT_i.$$

Long-tail agent trajectories make this term large. An async queue can immediately give a free worker another prompt, reducing the barrier loss. But if trajectory arrival rate $\lambda$ exceeds learner service rate $\mu$ , the queue grows and freshness degrades.

A more meaningful throughput is

$$X_{\rm useful}=\frac{\text{accepted valid action tokens}}{\text{wall-clock time}}.$$

Generated tokens later rejected for staleness or schema failure consume capacity but do not increase $X_{\rm useful}$ .

## 8. Queue and control loop

A practical controller observes:

- queue depth and age histogram;
- learner consumption and rollout production rates;
- lag/ratio distributions;
- drop and retry reasons;
- publication/load latency;
- worker utilization and long-tail quantiles.

Possible controls include generation concurrency, prompt priority, maximum active rollout length, publication cadence, admission lag, learner batch size, and interruption policy. Each control changes both systems performance and the sampled training distribution.

## 9. Partial and interrupted rollout

An asynchronous system may interrupt a long trajectory after a newer policy is published or when queue pressure rises. Three semantics are possible:

1. **discard:** preserve complete-trajectory estimator assumptions at wasted compute cost;
2. **truncate:** treat the prefix as terminal with an explicit truncation reason and bootstrap rule;
3. **resume:** continue from a serialized environment/context state, usually under the original behavior policy.

Resuming the same prefix under a different policy makes the final trajectory piecewise-behavior. It is valid only if every segment carries its own behavior version/log-probabilities and the estimator explicitly supports that contract.

## 10. Sync versus async decision

Prefer synchronous execution when:

- estimator simplicity and reproducibility matter more than peak device utilization;
- rollout lengths are relatively homogeneous;
- policy changes quickly, making stale data expensive;
- the team lacks strong versioning and queue observability.

Consider asynchronous execution when:

- environment/tool latency dominates and is heavy-tailed;
- rollout and training resources can be scaled independently;
- the estimator and schema explicitly support mixed behavior policies;
- publication, admission, and held-out validation are mature.

AReaL is a primary reference for fully asynchronous execution. HybridFlow/veRL is a primary reference for flexible dataflow and placement with synchronous/on-policy-first designs. The framework comparison in [Chapter 06](06_framework_comparison.md) keeps these goals separate.
