# r2-Day04 — DDP：问答归档

用户在当天 ai 课程聊天中提出的问题与总结。问题原文保留，答案按聊天记录整理；
术语与中文解释保留英文术语（DDP、TP、PP、GPipe）。

## 2026-08-28（周五）

### Q1：gpipe讲的是啥？ pp？

**A：** 对，GPipe 讲的是 Pipeline Parallelism（PP）。模型按连续的层切成多个 stage：

```text
GPU0: Layer 0–7
GPU1: Layer 8–15
GPU2: Layer 16–23
GPU3: Layer 24–31
```

一个大 batch 再切成多个 micro-batch，像流水线一样流过各 stage：

```text
micro-batch 1: GPU0 → GPU1 → GPU2 → GPU3
micro-batch 2:       GPU0 → GPU1 → GPU2 → GPU3
```

核心机制：

1. **模型按层切**，解决单卡放不下的问题；
2. stage 之间只传 activation；backward 时反向传 activation gradient；
3. 一个 batch 内先跑完所有 micro-batch 的 forward，再统一 backward，
   叫 **GPipe schedule / all-forward-all-backward**；
4. 开头和结尾部分 GPU 会闲着，叫 **pipeline bubble**；
   micro-batch 越多，bubble 占比越小；
5. 用 activation recomputation：forward 不保存所有中间激活，
   backward 时重算，省显存换计算。

和当天课程的并行方式对比：

- **DP/DDP**：每卡完整模型，切数据，梯度 AllReduce。
- **TP**：每层矩阵横着切（Column/RowParallel），几乎每层都通信。
- **PP/GPipe**：模型按连续层竖着切，只在 stage 边界传 activation，
  但会有流水线 bubble。
