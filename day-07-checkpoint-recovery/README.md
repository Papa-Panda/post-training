# Day 7 - 2026-08-08 - Checkpoint & Recovery

**Learning Goal**: 理解大规模训练为什么总要断点续训

**Small Task**: 给 FSDP 加上 checkpoint，模拟一次失败恢复

## 为什么要存盘？

- 训练 70B 跑 3 天，机器烧了/网络抖了/被抢占，丢了几小时 = 几千刀
- Checkpoint = 存档点，Recovery = 读档重来
- FSDP 难点：参数被切成碎片，怎么拼回可用的存盘？

## 两类 checkpoint

| 类型 | 玩法 | 适用 |
|------|------|------|
| Full checkpoint | rank0 gather 全量再存，简单但大、慢 | 调试、小模型 |
| Sharded checkpoint (DCP) | 每卡只存自己的分片，能并行存，7B+必用 | 生产、大模型 |

## 你组里类比

- 你 SLO 压测：预测瓶颈 + 重试策略，checkpoint 就是“预测失败 + 自动读档”
- autoscaling：机器挂了能回滚到上一版扩容方案

## 流程图

```
Train Step N -> 每 K 步 checkpoint -> 机器崩了 -> 启动新作业 -> 找最新 ckpt -> load -> 从 Step N 继续
```

关键：存什么？
- model sharded state
- optimizer sharded state (Adam m/v 占 2*P，别丢)
- epoch / step / sampler epoch / rng
- dataloader 进度（否则重跑一 epoch）

## 真机 vs CPU

CPU gloo：逻辑跑通，演示 rank0 存 / rank1 等 barrier / crash 后 load
GPU NCCL：`torch.cuda.max_memory_allocated` 看峰值，DCP 并行写 NVMe 最快
