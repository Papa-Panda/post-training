# 04 — Tensor Core、GEMM Tiling 与数据流

## 1. Tensor Core 加速的是 tile-level MMA

GEMM 为：

$$C_{M\times N}=A_{M\times K}B_{K\times N}+C_{M\times N}.$$

一次标量 FMA 贡献 2 FLOPs；总工作量近似：

$$F=2MNK.$$

Tensor Core 让一个 warp/warpgroup 对矩阵 fragment 执行 MMA，而不是让单线程拥有一个“超快乘法器”。其有效吞吐取决于 shape、dtype、layout、对齐、数据供给和 accumulator 管理。

## 2. Mixed precision 是数值协议

常见模式是低精度输入、较高精度累加：

$$C_{ij}^{(t+1)}=\mathrm{accum}_{\mathrm{high}}\left(C_{ij}^{(t)}+\mathrm{cast}(A_{ik})\mathrm{cast}(B_{kj})\right).$$

Volta 引入 Tensor Core；Turing/Ampere 扩展整数和矩阵指令，Ampere 加入 TF32/BF16 路径；Hopper Transformer Engine 面向 FP8/FP16 动态选择。硬件支持不等于模型精度自动安全，仍需 scaling、amax/history、loss scaling 或量化校准。

不要把不同格式的峰值相除后直接宣称端到端加速，因为：

- 非 GEMM 算子占比不变；
- 转换/scale 和额外 reduction 有成本；
- shape 可能无法充分铺满 tile；
- HBM/collective/launch 仍可能主导。

## 3. 三层 tiling

高性能 GEMM 通常至少有：

1. **CTA/block tile**：决定一个 block 负责的 $C$ 区域与 shared-memory footprint；
2. **warp/warpgroup tile**：映射到 MMA 指令协作组；
3. **instruction tile**：硬件支持的最小 MMA shape。

循环沿 $K$ 维推进：

```text
for k_tile:
    global A/B -> shared stage
    shared fragments -> registers
    MMA accumulates register C fragment
fused epilogue -> global C
```

理想 tile arithmetic intensity 随复用提高，但 accumulator/register 与 shared memory 同时增长。

## 4. Inner、Outer 与 Systolic 的统一解释

### Inner-product 数据流

每个 output 元素按 $K$ 归约：

$$C_{ij}=\sum_{k=1}^{K}A_{ik}B_{kj}.$$

优点是 output ownership 清晰；难点是同时向大量 output 喂 A/B，并保留 partial sums。

### Outer-product 数据流

每个 $k$ 贡献一个 rank-1 update：

$$C=\sum_{k=1}^{K}A_{:,k}B_{k,:}.$$

A column 和 B row 可广播给一个 output tile；代价是大量 C partial sums 必须在近端保存或归并。

### Systolic 数据流

processing elements 让 A/B 沿阵列传播，partial sum 在阵列内累积。其本质是通过时空调度提高 operand reuse，减少远端存储访问。不能把所有 NVIDIA Tensor Core 都简单等同于某一种公开的二维 systolic array 实现；软件可依赖的是指令/fragment/layout contract，而非未经官方公开的微架构细节。

三者共同优化目标可写成：

$$\min Q_{\mathrm{far}}\quad\text{s.t. compute units stay fed and partial sums stay local}.$$

## 5. Epilogue fusion

GEMM 后常接 bias、activation、scale、residual。若分成多个 kernel，中间矩阵至少经历写回和重读：

$$Q_{\mathrm{extra}}\ge2MN e.$$

将 epilogue 融入 GEMM 可减少 HBM traffic 和 launch，但会增加 registers、代码复杂度和 kernel specialization。是否获益仍看 shape 与瓶颈。

## 6. Blackwell 与“寄存器自救”应如何理解

低精度吞吐增长使数据供应和 accumulator 容量更紧张。架构和库会通过更大协作组、专用搬运、fragment 重排与 staged pipeline 缓解 register/shared pressure。可靠结论应停留在官方暴露的编程模型和规格；不要从反汇编图示推断未公开的物理阵列或把单个 microbenchmark 外推到所有 GEMM。

## 7. 库和 DSL 的角色

- cuBLAS/cuBLASLt：成熟 GEMM 算法选择与 epilogue；
- CUTLASS：显式分层 tile 和 pipeline 的模板化实现；
- Triton/tile DSL：以 program/tile 表达数据并行，编译器负责更低层 mapping；
- 手写 CUDA/PTX：需要控制特殊指令或不规则融合时使用，维护成本最高。

先证明库做不到，再下探抽象层。

## 8. 工程检查表

- M/N/K、leading dimensions 是否适合目标 dtype/tile？
- 实际是否命中 Tensor Core 指令？
- HBM→shared 与 shared→register 是否重叠？
- accumulator 是否引起 register spill？
- tail/padding 代价是否吞掉小 shape 收益？
- epilogue fusion 减少多少 bytes，增加多少资源？

## 导航

- 上一篇：[03 内存层次](03_memory_hierarchy_coalescing.md)
- 下一篇：[05 CUDA 编程模型](05_cuda_programming_model.md)
- 相关：[06 库与 DSL](06_libraries_and_kernel_dsl.md) · [10 LLM 映射](10_llm_training_inference_mapping.md)
