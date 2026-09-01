# 06 — CUDA Libraries、CUTLASS 与 Tile DSL

## 1. 抽象层级不是能力排名

```text
framework graph/compiler
  └─ cuBLAS/cuDNN/NCCL or generated kernels
       └─ CUTLASS / Triton / tile DSL
            └─ CUDA C++
                 └─ PTX / architecture-specific instructions
```

越低层控制越强，但验证、调优、可移植和维护成本也越高。工程默认应从成熟 library 开始，以 profiler 证明 gap，再决定是否下探。

## 2. 库负责的不只是“调用硬件”

以 GEMM 为例，library 需要根据：

$$\mathcal A=f(M,N,K,\mathrm{dtype},\mathrm{layout},\mathrm{alignment},\mathrm{epilogue},\mathrm{workspace},\mathrm{architecture})$$

选择算法、tile、stages、split-K 和 epilogue。固定一个 kernel 很难覆盖 tall-skinny、small-batch、large square、grouped GEMM 等形状。

- cuBLAS/cuBLASLt：GEMM 和可配置 epilogue；
- cuDNN：卷积、attention 等深度学习算子图；
- NCCL：拓扑感知 collective；
- CUDA 标准库生态还包括 CUB、Thrust、cuSPARSE、cuFFT 等。

## 3. CUTLASS：把 GEMM 层次显式化

CUTLASS 将 device/CTA/warp/instruction tile、copy atom、layout、pipeline 和 epilogue编码进模板。它适合：

- 学习/控制 GEMM 数据流；
- 构建定制 fused epilogue；
- 作为更高层 kernel generator 的 building blocks。

代价是模板组合和架构特定细节复杂，编译时间与 binary size 也要管理。

## 4. Triton 与 Tile-level DSL

Triton 程序通常让一个 program instance 处理一个 tile，通过 program IDs、向量化 load/store 和 mask 表达数据流。优势是 Python 级表达与自动 lower；边界是：

- 编译器仍需找到合理 layout/schedule；
- 特殊硬件特性未必即时暴露；
- irregular synchronization/communication 可能更适合 CUDA；
- 自动调参结果依赖 shape/dtype/device，不能只在一个尺寸上选 winner。

TileLang、cuTile 等材料可用于理解“把优化单位从 thread 提升到 tile”的趋势；专题只使用用户给出的 Bilibili 页面可见标题/主题作为线索，不把视频未取得的具体观点当证据。

## 5. Fusion 的收益方程

算子 A 产生中间张量 $X$，算子 B 消费它。若融合消除一次写回和一次读取，理想减少：

$$\Delta Q\approx2|X|e.$$

但 fusion 可能增加：

- live ranges 和 registers；
- shared memory；
- 代码分支和编译 specialization；
- 复用成熟库 kernel 的难度。

应比较：

$$T_{\mathrm{saved\ memory+launch}}>T_{\mathrm{occupancy\ loss+extra\ compute}}.$$

## 6. Autotuning 必须防过拟合

调参空间可包括 block shape、num warps、stages、split-K、layout。正确流程：

1. 固定 correctness oracle 和 tolerance；
2. 分离 tuning shapes 与 held-out shapes；
3. warmup 后测 latency distribution；
4. 限定 workspace/compile-time/binary-size；
5. 按 device/dtype/version 缓存结果；
6. library/driver 升级后重新验证。

单个 benchmark shape 的最优配置不等于生产 workload 最优。

## 7. 如何选择实现层

| 条件 | 优先选择 |
|---|---|
| 标准大 GEMM/conv/attention | 成熟库/框架 fused op |
| 简单 elementwise/reduction fusion | compiler 或 Triton |
| 定制 GEMM epilogue/layout | cuBLASLt、CUTLASS、Triton 对比 |
| 新架构专用数据搬运/指令 | CUTLASS/CUDA，明确 target 与 fallback |
| 通信 collective | NCCL；不要先手写 peer-copy ring |
| 极小动态 kernel | 同时考虑 fusion、Graph、persistent kernel |

## 8. Bilibili 参考的可访问边界

用户提供的第二个视频页面可读到标题“Tilelang/Triton/cuTile 的异同与选择”和相关主题，但没有取得完整转录；因此本章没有归因视频中的具体技术判断。第一个视频页面只显示 CUDA 软硬件课程标题/相关推荐，同样不作为规格证据。详见 [`sources.md`](sources.md)。

## 导航

- 上一篇：[05 CUDA 编程模型](05_cuda_programming_model.md)
- 下一篇：[07 多 GPU 互连与 NCCL](07_multi_gpu_interconnect_collectives.md)
- 相关：[04 GEMM](04_tensor_core_gemm_dataflows.md) · [09 调优](09_roofline_profiling_tuning.md)
