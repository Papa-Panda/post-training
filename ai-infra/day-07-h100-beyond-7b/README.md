# Large-model FSDP Capacity Planning

This historical lab previously mixed model-state estimates, synthetic rollout failures, and unverified H100 throughput ranges. It is now narrowed to one defensible task: **make FSDP model-state byte accounting explicit before attempting a hardware run**.

It is an analytical planning lab, not an H100 benchmark.

## 1. State assumptions

For $P$ parameters, the default example assigns:

| State | Bytes per parameter | Sharded by full-shard FSDP? |
|---|---:|---|
| bf16 parameter | 2 | yes |
| bf16 gradient | 2 | yes |
| fp32 Adam first/second moments | 8 | yes |
| fp32 master parameter copy | 4 | yes, when the chosen stack maintains it |

The default therefore uses 16 bytes per parameter. Some stacks omit or represent the master copy differently; the command exposes every byte assumption instead of hiding it in a label such as “bf16 mix.”

For $G$ ranks, the evenly sharded resident model-state estimate is:

$$M_{resident}=\frac{P(b_p+b_g+b_o+b_m)}{G}.$$

If the largest FSDP wrapped unit has $L$ parameters and its parameter shards are materialized for compute, the additional parameter bytes above the resident local shard are:

$$M_{gather}=L b_p\frac{G-1}{G}.$$

The planner reports $M_{resident}+M_{gather}$ as a **model-state lower bound**, not total peak memory.

## 2. What the bound omits

A fit decision must additionally measure or bound:

- saved activations and activation checkpointing policy;
- attention/MLP temporary buffers;
- communication workspaces and prefetch overlap;
- allocator fragmentation;
- embeddings or modules outside the wrapped-unit policy;
- CUDA context, kernels, and framework runtime;
- sequence length, microbatch, dtype, optimizer implementation, and compilation behavior.

Therefore a number below device capacity does not prove that the workload fits. A number above capacity is useful as an immediate rejection under the stated assumptions.

## 3. Run the planner

```bash
python3 ai-infra/day-07-h100-beyond-7b/fsdp_h100_profiler_beyond7b.py \
  --params-b 7 --ranks 8 --largest-layer-m 220

python3 -m unittest discover \
  -s ai-infra/day-07-h100-beyond-7b -p 'test_*.py' -v
```

To model a stack without a separate fp32 master copy, add `--master-parameter-bytes 0`. Do not choose that flag merely to make the estimate fit; confirm the optimizer/framework state representation first.

## 4. Hardware validation contract

A measured follow-up should record:

1. exact model/configuration and parameter count;
2. FSDP version and wrapping/sharding/mixed-precision policies;
3. optimizer and state representation;
4. sequence length, microbatch, accumulation, and checkpointing;
5. warm-up and measured iterations;
6. synchronized step time and `torch.cuda.max_memory_allocated()`;
7. profiler evidence for collectives and exposed communication;
8. hardware, topology, CUDA, NCCL, driver, and framework versions.

Until those fields exist, this directory makes no claims about H100 throughput, failure rate, communication percentage, or model fit.

## 5. Rollout boundary

The duplicate rollout script formerly stored here was removed. Rollout metrics, failure taxonomy, and stress testing are maintained in [`day-10-vllm`](../day-10-vllm/) and the canonical [`vllm-rollout/`](../../vllm-rollout/README.md) track.

## Sources

- PyTorch FSDP API and memory/limiter semantics: https://docs.pytorch.org/docs/stable/fsdp.html
- PyTorch profiler: https://docs.pytorch.org/docs/stable/profiler.html
- PyTorch distributed checkpoint: https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
