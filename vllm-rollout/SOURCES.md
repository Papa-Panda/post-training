# Sources and claim ledger

Accessed 2026-09-01. Versioned links are intentional because CLI flags, defaults, and metric names change.

## vLLM primary/official material

1. **Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention,” SOSP 2023.** [arXiv abstract](http://arxiv.org/abs/2309.06180v1)
   Used for: block-based non-contiguous KV storage, dynamic request memory, sharing, and the motivation for PagedAttention. The paper's benchmark speedups are not reused as universal expectations here.

2. **vLLM, Optimization and Tuning (latest docs).** [Official documentation](https://docs.vllm.ai/en/latest/configuration/optimization/)
   Used for: V1 recompute preemption, tuning actions under frequent preemption, V1 chunked-prefill scheduling, and the qualitative `max_num_batched_tokens` trade-off. “Latest” is mutable; save a versioned copy with real experiment artifacts.

3. **vLLM, Automatic Prefix Caching design.** [Official documentation](https://docs.vllm.ai/en/stable/design/prefix_caching/)
   Used for: full-block prefix caching, hash identity, allocation, references, and eviction behavior. “Stable” is mutable; match it to the installed release.

4. **vLLM v0.16.0, Production Metrics.** [Official versioned documentation](https://docs.vllm.ai/en/v0.16.0/usage/metrics/)
   Used for: `/metrics`, current metric examples including request state, KV usage, token counters, prefix-cache counters, and preemption/recomputation counters.

5. **vLLM, Benchmark Suites.** [Official documentation](https://docs.vllm.ai/en/latest/benchmarking/)
   Used for: existence and scope of the current `vllm bench` tools. The guide intentionally tells operators to save local `--help` output rather than freeze mutable command flags here.

6. **vLLM v0.7.0, Disaggregated Prefilling.** [Official versioned documentation](https://docs.vllm.ai/en/v0.7.0/features/disagg_prefill.html)
   Used only for the bounded claim that disaggregated prefill can separate TTFT/ITL tuning and that those docs explicitly state it does not improve throughput. The feature was marked experimental in that release.

## RL primary material

7. **Schulman et al., “Proximal Policy Optimization Algorithms,” 2017.** [arXiv](https://arxiv.org/abs/1707.06347v2)
   Used for: PPO's behavior/current-policy probability ratio, clipped surrogate, and multiple epochs over collected samples.

8. **Espeholt et al., “IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures,” ICML 2018.** [PMLR](https://proceedings.mlr.press/v80/espeholt18a.html)
   Used for: actor–learner policy lag as an off-policy problem and V-trace as an explicit correction in that architecture.

## Deliberately unsupported claims removed

The previous notes contained hardware throughput/latency ranges, fixed model-to-device mappings, failure percentages, GPU prices, wall-clock shares, an unverified issue reference, and a contiguous-KV-fragmentation detector. They were removed because this directory had no reproducible run artifacts or primary evidence for them.
