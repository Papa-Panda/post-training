# Day 7 NOTES - 2026-08-08 14:30 PDT (补)

Date: 2026-08-08 14:30 PDT (补 08-11)
Status: done (CPU gloo 验证逻辑，待 H100 NCCL 验证 peak mem + DCP throughput)

## 核心

- FSDP checkpoint 两种：
  - Full: `model.state_dict()` (composable FSDP 会自动 gather 成 full)，rank0写 `/tmp/fsdp_day7_full.pt`
  - Sharded DCP: `torch.distributed.checkpoint.save()` 每卡写自己分片，并行快
- Recovery：启动时找最新 ckpt，`load_state_dict` + optimizer + epoch + `sampler.set_epoch(epoch)`
- 失败模拟：epoch 1 中间手动 `raise` 或杀进程，再重启脚本自动从 epoch 1 恢复

## 3 numbers (CPU gloo，待 H100)

- epoch 0 avg_loss 2.31 (CPU gloo 2-rank)
- epoch 1 avg_loss 2.14 (从 ckpt 恢复后继续，未清零)
- checkpoint 写耗时 rank0 12ms (full, CPU, 0.1M param)，真机 7B 预期 2-4s 并行写 NVMe

## Code

- `fsdp_day7_checkpoint.py` 支持：
  - `torchrun --nproc_per_node=2 fsdp_day7_checkpoint.py --ckpt /tmp/fsdp_day7_full.pt`
  - 第二次跑自动检测 ckpt 存在 → load → 从 epoch 1 开始
  - 模拟 crash：`--crash-epoch 1` 第二个 epoch 开头抛异常，下次重启验证 recovery

Recipe:

```
# 正常跑 2 epoch
torchrun --nproc_per_node=2 fsdp_day7_checkpoint.py

# 模拟中途崩
torchrun --nproc_per_node=2 fsdp_day7_checkpoint.py --crash-epoch 1
# 失败后重跑（自动恢复）
torchrun --nproc_per_node=2 fsdp_day7_checkpoint.py
```

待真机：补 `torch.cuda.max_memory_allocated` 对比，DCP 并行写测 `tokens/sec` 无影响
