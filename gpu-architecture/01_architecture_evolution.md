# 01 — GPU 架构演进：从吞吐处理器到 AI 数据通路

## 1. 不按产品表背架构

GPU 的核心矛盾长期没有变：晶体管预算有限，怎样让更多能量花在**可并行的有效运算**，而不是复杂控制、等待数据和搬运数据。现代 GPU 因而同时包含：

- 大量吞吐型 SM；
- SIMT 调度与层次化线程组织；
- register file、shared memory/L1、L2、HBM；
- Tensor Core 等专用矩阵数据通路；
- copy/async engines、互连和地址翻译单元。

不能把一代 GPU 概括成“核心更多”。性能来自计算格式、数据路径、缓存、调度、互连和软件栈的共同变化。

## 2. 一条有边界的演进主线

| 阶段 | 主要变化 | 对程序员的含义 |
|---|---|---|
| 早期可编程 GPU | shader 逐步通用化 | 数据并行适合吞吐执行，但控制流代价高 |
| CUDA/G80 以后 | 通用 thread/block/grid + SM | kernel 成为可移植并行程序，硬件仍按 warp 执行 |
| Volta | 第一代 Tensor Core；Independent Thread Scheduling | 矩阵乘加成为显式专用路径；旧式隐式 warp-synchronous 假设需谨慎 |
| Turing/Ampere | Tensor Core 支持更多精度；Ampere 增加 TF32/BF16 与异步 global→shared copy 路径 | 数值格式与 pipeline 成为性能设计的一部分 |
| Hopper | Transformer Engine、Tensor Memory Accelerator、thread-block cluster、异步 transaction barrier | 更大粒度的数据搬运/协作被硬件化，但具有架构特定约束 |
| Blackwell | 继续扩展低精度矩阵计算和架构特定 Tensor Core 特性 | 编译目标、数据格式和库版本更加重要；不能假设特殊指令前向兼容 |

这里只列官方文档可支持的方向性变化，不把不同 SKU 的峰值数字混成“代际倍数”。例如 H100 SXM、H100 PCIe、H200 的 HBM 容量/带宽不同；比较前必须固定产品、精度、是否使用 sparsity 以及功耗形态。

## 3. SM 是资源池，不是“CUDA Core 的盒子”

一个 kernel block 被放到 SM 后，会同时消耗：

- threads/warps slots；
- registers；
- shared memory；
- block slots；
- 指令发射和各类 execution pipe 的容量。

若 block 使用资源向量 $r_b=(T_b,W_b,R_b,S_b,1)$，SM 上可驻留 block 数的教学上界为：

$$N_{\mathrm{block}}=\min\left(\left\lfloor\frac{T_{\max}}{T_b}\right\rfloor,\left\lfloor\frac{W_{\max}}{W_b}\right\rfloor,\left\lfloor\frac{R_{\max}}{R_b}\right\rfloor,\left\lfloor\frac{S_{\max}}{S_b}\right\rfloor,N_{\max}\right).$$

真实硬件还有 register/shared allocation granularity 等约束，应交给 CUDA occupancy API 或 Nsight Compute 核对。

## 4. 为什么专用化仍然需要通用 SIMT

Tensor Core 擅长规则 tile 上的矩阵乘加：

$$D=A B+C.$$

但 LLM kernel 还包含地址计算、归约、mask、normalization、采样、稀疏路由和控制逻辑。通用 CUDA cores、load/store pipeline 与专用矩阵 pipe 是协作关系，而不是“Tensor Core 替代 CUDA Core”。

真正的问题是：

1. 输入 tile 能否及时到达 register/shared memory；
2. shape/layout/dtype 是否命中高效指令和库 kernel；
3. epilogue 能否融合，避免把中间结果写回 HBM；
4. 并行度是否足够隐藏 pipeline 和 memory latency。

## 5. 代际比较的四个陷阱

1. **峰值口径混淆**：dense/sparse、FP32/TF32/FP16/FP8、boost/base clock 不可横比。
2. **产品形态混淆**：PCIe 与 SXM、不同 HBM 容量版本不是同一规格。
3. **把峰值当实测**：$P_{\mathrm{achieved}}\le P_{\mathrm{peak}}$，layout、shape、occupancy 和搬运会拉开差距。
4. **把硬件特性当透明加速**：架构特定特性常需要新指令、编译 target 和库支持。

## 6. 对后文的接口

- 下一章把 SM 资源池展开成 warp 调度；
- 第 3 章解释为何数据供应决定执行 pipe 利用率；
- 第 4 章解释 Tensor Core 上的 tile 数据流；
- 第 9 章用 Roofline/profiler 把“架构故事”变成可证伪判断。

## 官方锚点

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Ampere Tuning Guide](https://docs.nvidia.com/cuda/archive/12.0.1/ampere-tuning-guide/index.html)
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [NVIDIA Blackwell and CUDA architecture-specific features](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/)

## 导航

- 上一篇：[README](README.md)
- 下一篇：[02 SIMT 与 warp 调度](02_simt_warp_scheduling.md)
- 总索引：[路线图](README.md#路线图) · [证据账本](sources.md)
