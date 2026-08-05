# Day 1 - DDP basics - Done

Date: 2026-08-02 10:30 AM PDT (America/Los_Angeles)
Status: done

## What I did
- torchrun --nproc_per_node=2 ddp_day1_mnist.py (gloo, CPU)
- Verified DDP grad sync demo: grad 1.0 & 3.0 -> all-reduce SUM 4.0 / world_size 2 = 2.0

## 3 numbers (to be filled on real GPU)
- single-GPU time/epoch: CPU baseline recorded,待上真机补充
- 2-GPU time/epoch: CPU 2-rank recorded,待上真机补充
- all-reduce time: demo printed ~ms level, profiler to be added on GPU

## Takeaway
- DistributedSampler + set_epoch() prevents data repetition across ranks
- Only rank0 writes checkpoint to avoid race
- DDP mean sync is the invariant - connects to autoscaling stability thinking

Next: Day 2 FSDP wrap, compare mem.

