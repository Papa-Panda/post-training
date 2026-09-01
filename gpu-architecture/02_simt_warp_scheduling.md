# 02 — SIMT、Warp 调度与 Occupancy

## 1. 软件层次与硬件执行层次

CUDA 暴露 `grid → block → thread`；当前 NVIDIA GPU 的 SM 以 32 threads 的 warp 为基本执行组。thread 拥有自己的逻辑 program counter/register state，但同一 warp 的 active lanes 共享指令发射机会。

$$W_b=\left\lceil\frac{T_b}{32}\right\rceil.$$

一个 100-thread block 会占 4 个 warps 的资源，最后一个 warp 只有 4 个有效 lanes。边界线程仍需 guard。

## 2. SIMT 不等于 SIMD

SIMD 通常是一条显式向量指令操作向量元素；SIMT 让程序员写标量 thread，硬件把多个 thread 聚成 warp 执行。两者都利用 lane 并行，但状态模型和软件接口不同。

若 warp 在条件分支上分裂，硬件必须执行各 active path，并对不属于该路径的 lane 关闭执行：

$$\eta_{\mathrm{branch}}\approx\frac{\sum_p n_p I_p}{32\sum_p I_p},$$

其中 $n_p$ 是路径 $p$ 的 active lanes，$I_p$ 是该路径指令数。这个比值只是 lane-utilization 教学量；predication、编译器控制流和 Independent Thread Scheduling 会改变实际时间。

## 3. AMD 对照：概念可迁移，常数不可照搬

ROCm 使用 work-group/work-item、Compute Unit 与 wavefront 等术语。wavefront 大小、CU 资源和矩阵指令路径取决于 AMD 架构；不能把本章 NVIDIA 的 32-thread warp、SM limits 或 CUDA intrinsics 直接移植。可迁移的是资源方程、发散、延迟隐藏和访存局部性这些方法。AMD 官方 [GPU architecture documentation](https://rocm.docs.amd.com/en/docs-6.3.0/conceptual/gpu-arch.html) 应作为目标设备的常数来源。

## 4. 延迟隐藏的 Little’s-law 直觉

假设某 pipeline 发起操作后平均等待 $L$ cycles，每个 warp 每隔 $s$ cycles 可发出一个独立操作。维持流水线所需在途并发量约为：

$$N_{\mathrm{inflight}}\gtrsim\frac{L}{s}.$$

GPU 通过切换 ready warps 隐藏等待，但前提是：

- 有足够 resident warps；
- warp 之间有独立工作；
- 没有同时卡在同一 barrier、memory dependency 或 pipeline saturation。

因此 occupancy 是**潜在延迟隐藏容量**，不是性能目标。

## 5. Occupancy 的资源方程

设 active warps 为 $W_{\mathrm{active}}$，硬件上限为 $W_{\max}$：

$$\mathrm{occupancy}=\frac{W_{\mathrm{active}}}{W_{\max}}.$$

resident blocks 同时受 threads、warps、registers、shared memory 和 blocks/SM 限制，见第 1 章方程。常见取舍：

- 增大 tile → 更高复用，但 shared memory/register 增加；
- 每线程更多 accumulator → 指令级并行提高，但 occupancy 可能降低；
- 限制 registers → 可能增加 occupancy，也可能触发 local-memory spill；
- block 太大 → 每 SM resident blocks 变少，barrier 尾部效应更重。

一个计算密集 GEMM 在 25%–50% occupancy 下也可能已充分利用 Tensor Core；一个 memory-latency kernel 可能需要更多 active warps。

## 6. Warp primitives 与正确性

`__shfl_sync`、`__ballot_sync`、`__reduce_*_sync` 允许 warp 内交换/归约而不经过 shared memory。要点是显式 mask：

```cuda
unsigned mask = __activemask();
float x = __shfl_down_sync(mask, value, offset);
```

不要把“同 warp”误当成任意条件下的隐式同步。Volta 以后的 Independent Thread Scheduling 使依赖旧 lockstep 假设的代码更危险；使用 `_sync` intrinsics、`__syncwarp(mask)` 或 Cooperative Groups 表达参与者。

## 7. Block、Cluster 与全局同步边界

- `__syncthreads()` 只同步一个 block，且所有参与线程必须一致到达；
- Cooperative Groups 可以显式表达 block/tile/grid 等 group，但 grid-wide synchronization 需要 cooperative launch 等前提；
- Hopper thread-block cluster 提供比 block 更大的硬件协作域，但不是所有 GPU 的通用保证；
- 跨 kernel launch 是最清晰的全局 phase boundary，代价是 launch/调度开销和潜在中间流量。

## 8. 工程诊断

| 症状 | 可能原因 | 先验证什么 |
|---|---|---|
| occupancy 低 | register/shared/block ceiling | launch statistics 与 resource usage |
| occupancy 高但慢 | memory dependency、分支、pipeline 饱和 | stall reasons、memory throughput、issue utilization |
| 少数 block 拖尾 | grid 太小或 block 工作不均 | waves per SM、block duration distribution |
| warp 发散 | data-dependent control flow | branch efficiency/active threads，不先盲改 block size |
| spill | register 压力过大 | local load/store 与 compiler register report |

## 9. 可运行模型

[`code/gpu_models.py`](code/gpu_models.py) 的 `occupancy()` 计算简化 resource ceilings。它故意忽略架构特定分配粒度，用来检查“谁先成为上限”，不能代替 occupancy API。

## 导航

- 上一篇：[01 架构演进](01_architecture_evolution.md)
- 下一篇：[03 内存层次与合并访问](03_memory_hierarchy_coalescing.md)
- 总索引：[路线图](README.md#路线图) · [证据账本](sources.md)
