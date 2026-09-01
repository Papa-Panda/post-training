# r2-Day04 — Distributed Data Parallel Semantics

## Connection to Prev

r2-Day03 separated logical collective volume from physical bandwidth. DDP supplies the training semantics around those collectives: replicated parameters, partitioned inputs, averaged gradients, and identical optimizer updates.

## 1. What DDP guarantees—and what it does not

PyTorch DDP synchronizes module state when the wrapper is constructed and registers autograd hooks that communicate gradient buckets during backward. It does **not** partition input data; the application normally supplies a `DistributedSampler`.

For equally sized local minibatches on $G$ ranks, DDP's averaged gradient is:

$$g=\frac{1}{G}\sum_{r=0}^{G-1}g_r.$$

That equals the gradient of the concatenated global minibatch under the usual per-example mean loss. Unequal local batch sizes require weighting; an unweighted mean of rank means is then not the global example mean.

The default correctness invariants are:

1. every rank starts from the same logical dataset and a synchronized model state;
2. sampler ownership has the expected coverage, padding, or dropped-tail behavior;
3. corresponding parameters receive the same averaged gradient;
4. replicas remain equal after each optimizer step.

Rank-local minibatch losses need **not** be equal because each rank processes different examples.

## 2. Data ownership

`DistributedSampler` pads by repeating indices when the dataset is not divisible by world size and `drop_last=False`. Therefore “each sample appears exactly once” is only true for divisible sizes (or an explicitly dropped tail). With shuffling, call `sampler.set_epoch(epoch)` so every rank uses the same deterministic permutation before taking its strided shard.

Every rank in the demo constructs the same deterministic logical dataset. The earlier version seeded dataset generation with `42 + rank`, which created different rank-local datasets before sampling and invalidated the intended partitioning model.

## 3. Communication and overlap

For a ring all-reduce over a gradient payload of $S$ bytes, idealized per-rank sent volume is:

$$V_{AR}=2\frac{G-1}{G}S.$$

DDP buckets parameters so a bucket can become ready and start reducing while autograd continues computing earlier layers. Bucket size is a tunable trade-off between startup overhead, overlap opportunity, and memory. Do not treat a historical default value as a universal optimum.

## 4. Run

Dependency-free semantic tests:

```bash
python3 -m unittest discover -s ai-infra/r2-day-04-ddp -p 'test_*.py' -v
```

Optional PyTorch integration:

```bash
torchrun --standalone --nproc-per-node=2 \
  ai-infra/r2-day-04-ddp/ddp_demo.py
```

The integration demo reads `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` from `torchrun`; uses Gloo on CPU and NCCL when CUDA is available; checks parameter checksums after DDP initialization and every optimizer step; and writes one replicated-state checkpoint on rank 0 only.

## Status

- Verified without PyTorch: sampler coverage/padding, gradient averaging, update equality, and ring volume.
- Not verified in this environment: PyTorch two-process execution (PyTorch is not installed), NCCL, GPU performance, bucket overlap, or MFU.

## Primary sources

- DDP design note: https://docs.pytorch.org/docs/main/notes/ddp
- DDP API: https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
- DistributedSampler API: https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
