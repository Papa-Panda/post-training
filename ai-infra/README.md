# AI Infrastructure Labs

A systems-first path through the compute, communication, and memory costs of training and serving large models. The organizing question is always:

> What extra work or complexity does a technique introduce, what resource does it save, and where does the trade stop paying off?

This directory contains two generations of material. The `r2-day-*` sequence is the current, cleaner learning path. The older `day-*` directories are retained as experiments and historical notes; their CPU simulations are not hardware benchmarks.

## Current path

```text
model computation
  -> framework execution
  -> topology and collectives
  -> replicated data parallelism
  -> declarative sharding
  -> GPU memory wall
  -> CUDA execution and access patterns
  -> FSDP / tensor-pipeline-sequence parallelism
  -> checkpointing
  -> inference and rollout serving
  -> evaluation and reliability gates
```

| Lesson | Focus | Executable evidence | Status |
|---|---|---|---|
| [`r2-day-01-transformer`](r2-day-01-transformer/README.md) | Decoder dimensions and parameter accounting | Python dimension/parameter model | CPU model |
| [`r2-day-02-pytorch-loop`](r2-day-02-pytorch-loop/README.md) | Training-loop state, optimizer, checkpoint | Minimal loop with explicit fallback | CPU path; accelerator profiling pending |
| [`r2-day-03-topo-nccl`](r2-day-03-topo-nccl/README.md) | Topology labels and ring collective cost | Unit-aware $\alpha$–$\beta$ model and semantic tests | CPU model; NCCL measurement pending |
| [`r2-day-04-ddp`](r2-day-04-ddp/README.md) | DDP ownership, data sharding, gradient synchronization | `torchrun` demo plus dependency-free invariants | CPU/Gloo when PyTorch is available |
| [`r2-day-05-jax-mesh`](r2-day-05-jax-mesh/README.md) | Mesh and declarative partitioning | JAX/fallback shape model | Multi-device execution pending |
| [`r2-day-06-gpu-architecture`](r2-day-06-gpu-architecture/README.md) | Roofline, HBM traffic, shared-memory capacity | Analytical model and six tests | CPU model; CUDA measurement pending |
| [`r2-day-07-cuda-programming-model`](r2-day-07-cuda-programming-model/README.md) | Grid/block/warp, coalescing, bank conflicts | Address model, eight tests, CUDA source | CPU model; CUDA run pending |

The full intended sequence is in [`ROADMAP_45D.md`](ROADMAP_45D.md). It is a curriculum map, not a claim that every planned lesson has been implemented.

## Core models

### Communication

For a ring over $p$ ranks and a payload of $S$ bytes per rank, the idealized per-rank transfer volumes are:

$$V_{\mathrm{RS}}=V_{\mathrm{AG}}=\frac{p-1}{p}S,\qquad V_{\mathrm{AR}}=2\frac{p-1}{p}S.$$

A simple latency-bandwidth estimate is:

$$T_{\mathrm{ring}}\approx n_{\mathrm{steps}}\alpha+\frac{V}{B_{\mathrm{effective}}}.$$

This is a lower-order mechanism model. Real NCCL behavior also depends on topology, channel count, protocol, chunking, contention, and its algorithm selection. Product “total bandwidth” and measured collective bandwidth are not interchangeable.

### Memory

For arithmetic intensity $I=F/Q$ with $F$ FLOPs and $Q$ bytes transferred from the bottleneck memory level, the Roofline bound is:

$$P\le\min(P_{\mathrm{peak}},B I).$$

A memory-saving method is incomplete until its additional communication, recomputation, fragmentation, and temporary buffers are accounted for.

### Correctness before speed

Every distributed experiment should separate three questions:

1. **Semantics:** Are samples partitioned as intended, gradients synchronized, and replicas equal after the step?
2. **Accounting:** Are bytes, bandwidth units, and collective phases defined consistently?
3. **Measurement:** Was the real backend synchronized and measured with the workload/configuration recorded?

A CPU fallback can answer the first two in limited cases. It cannot establish GPU latency, bandwidth, MFU, or scaling efficiency.

## Older experiments

The original `day-*` labs remain useful as focused prototypes:

- distributed basics: [`day-01-ddp-basics`](day-01-ddp-basics/), [`day-02-fsdp`](day-02-fsdp/), [`day-03-fsdp-perblock`](day-03-fsdp-perblock/), [`day-07-checkpoint-recovery`](day-07-checkpoint-recovery/), [`day-15-megatron-3d`](day-15-megatron-3d/);
- post-training links: [`day-04-rlhf-vs-agentic-rl`](day-04-rlhf-vs-agentic-rl/), [`day-08-eval-infra`](day-08-eval-infra/), [`day-10-vllm`](day-10-vllm/), [`day-12-reward-model`](day-12-reward-model/), [`day-13-reliability-slo`](day-13-reliability-slo/);
- capacity/profiling prototypes: [`day-07-h100-beyond-7b`](day-07-h100-beyond-7b/), [`day-12b-profile-tool-legacy`](day-12b-profile-tool-legacy/);
- **non-core side tracks:** [`day-06-paper1-rl-infra`](day-06-paper1-rl-infra/), [`day-11-paper2-mech-load`](day-11-paper2-mech-load/), and [`day-14-pue-cost`](day-14-pue-cost/) concern workload forecasting or facility/thermal models rather than AI-infrastructure mechanisms.

Numbers in those directories marked simulation, proxy, estimate, or pending hardware validation must remain labeled that way.

## Boundaries with other tracks

- [`gpu-architecture/`](../gpu-architecture/README.md) owns the deeper hardware/kernel treatment. These labs use that cost model in training and serving systems.
- [`vllm-rollout/`](../vllm-rollout/README.md) is the canonical rollout-serving stress-test track; `day-10-vllm` is retained as an earlier snapshot.
- [`grpo-vs-ppo/`](../grpo-vs-ppo/README.md) owns policy-objective derivations. This directory discusses their systems consequences only.
- [`model-aware-data-curation/`](../model-aware-data-curation/README.md) owns model-aware data selection; this directory owns execution and resource costs.
- [`harness-engineering/`](../harness-engineering/README.md) owns agent runtime/control-plane design, not GPU or collective implementation.

## Run the current checks

```bash
python3 -m unittest discover -s ai-infra/r2-day-03-topo-nccl -p 'test_*.py' -v
python3 -m unittest discover -s ai-infra/r2-day-04-ddp -p 'test_*.py' -v
python3 -m unittest discover -s ai-infra/r2-day-06-gpu-architecture -p 'test_*.py' -v
python3 -m unittest discover -s ai-infra/r2-day-07-cuda-programming-model -p 'test_*.py' -v
```

## Primary references

- PyTorch DistributedDataParallel design note: https://docs.pytorch.org/docs/main/notes/ddp
- PyTorch DistributedDataParallel API: https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
- NCCL collective semantics: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
- CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- CUDA C++ Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
