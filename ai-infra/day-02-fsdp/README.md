# Day 2 - FSDP intro

- Learning Goal: 知道 FSDP 为什么比 DDP 省显存
- Task: 把昨天的 mnist 改成 FSDP，对比显存/速度
- Work Connection: 跟你做机械负载建模一样：分片是为了让系统不超限

## Key idea
FSDP = ZeRO-3 变体，参数/梯度/optimizer 都分片。
前向 all-gather 聚成全参算，反向 reduce-scatter 再散回去。
按 transformer block 包是甜点：避免 per-layer 太碎的启动开销，也避免 per-model 太大的峰值。

## How to measure (lightest)
```python
torch.cuda.reset_peak_memory_stats()
# train step
print(torch.cuda.max_memory_allocated()/1024**2)
```
DDP 跑一次，FSDP 跑一次，对比。

CPU gloo 也能跑通，显存记 N/A 待 H100。

## Tradeoff mental model
- DDP 显存: P (全参)
- FSDP (G=2): ~ P/2 + buffer，省 ~50%
- 通信量一样都是 P，但分成 num_blocks 次，latency 微增
