# Day 17 - 混合精度 / 梯度累积 / 重计算

> Connection to Prev: Day16 3D并行 TP/PP/SP → Day17 混合精度: TP把矩阵切小了但单卡仍需BF16省一半显存才能塞下Adam；Day15 FSDP/ZeRO用通信换显存的坑在今天用计算换显存（重计算）+ 精度换显存（BF16）双路解决。
> 草帽路飞分层：第二层分布式优化，问“牺牲什么换取什么”。

## 核心

- BF16 vs FP16 都是16位，为何大模型更偏爱BF16：BF16指数8位 vs 5位，动态范围接近FP32，不易overflow/underflow，大多可不做Loss Scaling，像草稿低精度算关键步骤高精度交卷
- 梯度累积：有限显存模拟更大有效Batch
- Activation Checkpointing：用计算换显存，只保存部分激活，需要时重算，像考试不留草稿只记关键中间结果需时重推

## 小任务（30-60min）

写3条对比：
1. ZeRO-2 vs ZeRO-3 一句区分
2. BF16 vs FP16 指数位差异
3. 重计算 vs ZeRO 都是省显存，代价分别是什么

## 检验

- 能口算7B FP16 14GB + Adam 56GB 单80GB能否放下，是否必须ZeRO
- 回答BF16 8位指数为何稳

## 真数

CPU占位，待H100 NCCL补 `torch.cuda.max_memory_allocated` + BF16 vs FP16 loss曲线对比
