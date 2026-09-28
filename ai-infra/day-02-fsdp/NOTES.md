# Day 2 - FSDP intro - Done

> 📖 阅读版：https://papa-panda.github.io/post-training/ai-infra/day-02-fsdp/

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

<!-- viz:vs: DDP | 显存常驻 P; 每个 rank 存全量参数 || FSDP G=2 | 显存常驻 P/2 + buffer; 省约 50% 参数显存 -->
<!-- viz:flow: per-layer 太细延迟受限 → per-block 甜点通信计算重叠 → per-model 太粗峰值回 DDP -->
