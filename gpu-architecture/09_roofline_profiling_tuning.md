# 09 — Roofline、Profiling 与性能调优

## 1. Roofline 是假设生成器

算术强度：

$$I=\frac{F}{Q}\quad[\mathrm{FLOP/byte}].$$

单层 Roofline：

$$P_{\mathrm{attainable}}\le\min(P_{\mathrm{peak}},B I).$$

ridge point：

$$I^*=\frac{P_{\mathrm{peak}}}{B}.$$

若 $I<I^*$，模型预测 bandwidth-bound；若 $I>I^*$，预测 compute-bound。这里 $Q$ 必须说明是哪一级流量（HBM、L2、shared 或 link），$P/B$ 必须使用同一设备/精度/口径。

## 2. Hierarchical Roofline

一个 kernel 可能对 HBM 有高复用，却受 L2/shared bandwidth 限制。可分别写：

$$P\le\min(P_{\mathrm{peak}},B_{\mathrm{HBM}}I_{\mathrm{HBM}},B_{\mathrm{L2}}I_{\mathrm{L2}},B_{\mathrm{SMEM}}I_{\mathrm{SMEM}}).$$

这解释了为什么“HBM throughput 不满”不能直接判定 compute-bound：瓶颈可能在 latency、L2、shared bank conflicts、instructions 或并发不足。

## 3. 一个正确的调优闭环

```text
定义 workload/正确性
  -> baseline + timeline
  -> 提出瓶颈假设
  -> 选择能证伪的 counters
  -> 一次只改一个机制
  -> 重新测 correctness + latency distribution
  -> 检查跨 shape/device 的回归
```

不要先打开数百 counters 再找故事。Profiler 本身会增加 overhead；先用 timeline 找区间，再对单 kernel 做深分析。

## 4. 症状—指标—动作

| 假设 | 观测 | 候选动作 |
|---|---|---|
| HBM bandwidth-bound | 高 DRAM throughput、低 $I$、memory stalls | 合并访问、复用、fusion、压缩 dtype |
| compute/Tensor Core-bound | 高 pipe utilization、足够 $I$ | 更好指令/shape、减少多余 FLOPs、负载平衡 |
| latency-bound | throughput 均不高、dependency stalls、grid 小 | 增加并发/ILP、合并小 kernel、Graph |
| occupancy/resource-bound | register/shared ceiling、少 active warps | 调 tile/block/stages；防 spill |
| branch/irregular | active lanes 低、divergence | 数据分组、warp specialization、改算法 |
| launch-bound | timeline 多短 kernel/gaps | fusion、Graph、batching、persistent kernel |
| collective-bound | NCCL 占关键路径、链路利用/step 多 | bucket、算法/拓扑/rank placement、overlap |
| page/migration-bound | faults/migrations | prefetch、placement、working-set 分块 |

动作只是实验候选，不是看到一个 counter 就自动套用。

## 5. Effective bandwidth 与 MFU

有效带宽：

$$B_{\mathrm{eff}}=\frac{Q_{\mathrm{useful}}}{T}.$$

若 $Q_{\mathrm{useful}}$ 不含过取/重试，它与物理 DRAM bytes 不是同一量。Memory Load/Store Efficiency 描述 requested vs transferred；DRAM throughput 描述实际 memory subsystem activity。

Model FLOPs Utilization 常写为：

$$\mathrm{MFU}=\frac{F_{\mathrm{model}}/T}{P_{\mathrm{peak}}}.$$

必须声明 model FLOPs convention、精度峰值、dense/sparse 口径和是否含 recompute；否则不同报告不可比。

## 6. Occupancy 与利用率分开

- occupancy：可驻留 warps 比例；
- issue/pipe utilization：执行资源实际繁忙程度；
- achieved occupancy：运行时平均 active warps；
- eligible warps：当前可发射而非等待的 warps。

高 occupancy + 低 eligible warps 可能说明 memory dependency/barrier；低 occupancy + 高 compute utilization 可能完全健康。

## 7. Benchmark 纪律

- 固定 shape/dtype/layout/seed；
- warmup，避免首次 JIT/library plan/clock state；
- 使用 device events 测 kernel，timeline 测 end-to-end；
- 报 median 与 tail，不只报最好一次；
- 校验输出和 numerical tolerance；
- 记录 GPU 型号、功耗模式、driver/runtime/library；
- 标注 theoretical estimate、simulated 和 measured，绝不混写。

## 8. 可运行 Roofline 模型

`roofline()` 输入 FLOPs、bytes、peak compute、bandwidth，返回 $I$、ridge、bound 与理想时间。它有意不包含 overlap/cache/launch；作用是验证单位与数量级，并指导下一次测量。

## 9. 从 Roofline 到 LLM

- 大 GEMM 通过 tiling 提高 $I$，常接近 compute roof；
- elementwise/norm/embedding 更常被 bytes 限制；
- decode 中 batch/token 维度小，权重与 KV traffic 摊薄困难；
- multi-GPU 还要加 network roof：$P\le B_{\mathrm{link}}I_{\mathrm{comm}}$。

下一章将这些信号映射到 Transformer。

## 导航

- 上一篇：[08 VM/IOMMU/UVM](08_virtual_memory_iommu_smmu_uvm.md)
- 下一篇：[10 LLM 训练/推理映射](10_llm_training_inference_mapping.md)
- 可运行模型：[code/gpu_models.py](code/gpu_models.py)
