# Day 2 - FSDP intro - Done

Date: 2026-08-03 19:07 PDT
Status: done
Mode: CPU gloo 2-rank (CUDA N/A, api ok)

## What I did
- DDP mnist -> FSDP fully_shard per-block (block1, block2, root)
- `from torch.distributed._composable.fsdp import fully_shard`
- 2 ranks gloo run: epoch 0 avg_loss 2.318, epoch 1 avg_loss 2.142
- Checkpoint rank0 /tmp/fsdp_day2_ckpt.pt ok

## 3 numbers
- single-GPU peak: N/A (CPU)
- 2-GPU peak FSDP: N/A (CPU, wait H100: torch.cuda.max_memory_allocated)
- elapsed 0.5s, FSDP api ok=True
- Theory: DDP = P, FSDP G=2 = P/2 + buffer, save ~50% param mem

## Tradeoff
- per-layer: too fine, many small all-gather, latency bound
- per-model: too coarse, peak back to DDP
- per-block: sweet spot, comm overlaps compute, bandwidth efficient

Next: Day3 Coding Data flywheel diagram.
