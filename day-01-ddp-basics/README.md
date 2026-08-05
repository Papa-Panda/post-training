# Day 1 - DDP basics

- Learning Goal: 理解 DDP 梯度同步原理
- Task: 用 3 行代码启动 DDP 跑通 mnist，记录通信开销
- Work Connection: 类比 autoscaling 资源波动预测

## What to record
- 单卡 time/epoch
- 双卡 time/epoch
- all-reduce 占比

Demo: 2 GPUs grad 1.0 & 3.0 -> after all-reduce SUM=4.0 / 2 = 2.0 mean, both ranks see same grad.
