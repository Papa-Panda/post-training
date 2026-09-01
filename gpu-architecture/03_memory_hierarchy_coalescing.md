# 03 — 内存层次、Coalescing 与数据复用

## 1. 性能问题首先是“数据从哪里来”

大致从近到远：register → shared memory/L1 → L2 → HBM → host/peer memory。容量、带宽、延迟和可见范围不同：

- register：每线程私有的最快状态；过多会限制 occupancy，spill 到 local memory 后实际走 device memory/cache；
- shared memory：block 内显式管理的 on-chip scratchpad；适合 tile、重排和协作复用；
- L1/L2：硬件管理 cache；命中率依赖访问模式与工作集；
- HBM：容量大、峰值带宽高，但远低于 Tensor Core 消费数据的潜在速率；
- host/peer：还受 PCIe/NVLink、地址翻译和同步控制。

## 2. Global-memory coalescing

对 CC 6.0+ 常用教学模型，一个 warp 的 global access 被拆成覆盖其地址的若干 32-byte segments。若 32 lanes 从 32-byte 对齐地址连续读取 FP32：

$$Q_{\mathrm{request}}=32\times4=128\ \mathrm{B},\qquad N_{\mathrm{seg}}=4.$$

搬运效率：

$$\eta_{\mathrm{coal}}=\frac{Q_{\mathrm{request}}}{N_{\mathrm{seg}}\times32\ \mathrm{B}}.$$

- 对齐连续：4 segments，100%；
- 错位 4 bytes：覆盖 5 segments，教学效率 80%；
- stride 2：覆盖 8 segments，教学效率 50%。

cache 可复用相邻 warp 多取的数据，所以这个模型不是 DRAM 实测。关键原则是：让相邻 lanes 访问相邻、自然对齐的数据，并用 shared memory 做必要重排。

## 3. Shared-memory bank conflicts

将 32-bit word index $i$ 简化映射为：

$$\mathrm{bank}(i)=i\bmod32.$$

warp 按列访问 `tile[32][32]` 时，第 $\ell$ lane 的 word index 为 $32\ell+c$，全部落到 bank $c$；padding 为 `tile[32][33]` 后：

$$\mathrm{bank}(33\ell+c)=(\ell+c)\bmod32,$$

分散到 32 banks。多个 lanes 读取**同一地址**可广播，不应误判为 32-way conflict。实际 bank width/模式和特殊指令仍应按目标架构文档与 profiler 判断。

## 4. Tiling 的数学收益

朴素 GEMM 中每个 $C_{ij}$ 读取 $K$ 个 A/B 元素，忽略 cache 的输入流量约：

$$Q_{\mathrm{naive}}\approx2MNK e,$$

其中 $e$ 是元素 bytes。若每个 $T_M\times T_N$ output tile 协作加载 A/B，则理想化输入流量：

$$Q_{\mathrm{tile}}\approx e\left(\left\lceil\frac{N}{T_N}\right\rceil MK+\left\lceil\frac{M}{T_M}\right\rceil KN\right).$$

更大 tile 提高复用，但会增加 shared memory、register accumulators、barrier 和尾块浪费。最优 tile 是多资源平衡，不是越大越好。

## 5. 异步搬运与流水化

高性能 kernel 把搬运和计算组织成 pipeline：

```text
HBM/L2 -> shared-memory stage k+1
              overlap
Tensor Core consumes stage k -> registers -> epilogue
```

Ampere 的 async copy、Hopper 的 TMA/transaction barrier 扩展了可表达的数据搬运，但共同目标是：

$$T_{\mathrm{tile}}\approx\max(T_{\mathrm{load}},T_{\mathrm{compute}})$$

而不是串行的 $T_{\mathrm{load}}+T_{\mathrm{compute}}$。要实现重叠，需要足够 stages、正确 barrier、无 buffer hazard，并避免 stages 过多挤占 shared memory。

## 6. AoS/SoA 与 layout

若 warp 只读取结构体中的一个字段，Array-of-Structures 可能让 lane 地址带 stride；Structure-of-Arrays 往往更易合并。矩阵 layout 还决定：

- 哪个维度 contiguous；
- load 是否 vectorized/aligned；
- Tensor Core tile 的 mapping；
- transpose 是否可在 shared memory 中完成而不增加 HBM pass。

“逻辑 shape 相同”不代表物理数据流相同。

## 7. 典型误区

- `cudaMalloc` 基地址对齐不代表每个 row/block 起点都对齐；pitch/leading dimension 仍重要。
- shared memory 比 HBM 快，不代表搬进去一定赚；一次使用的数据可能只增加 copy/barrier。
- cache hit 高不自动代表快；也可能是反复加载无效数据或 pipeline 受别处限制。
- bytes 下降不自动等于 time 等比例下降；kernel 可能跨过 ridge point 变成 compute-bound。

## 8. 可运行模型

- `coalescing_report()`：枚举 touched 32-byte segments；
- `shared_bank_conflict_degree()`：区分 bank conflict 与 broadcast；
- `gemm_traffic()`：对比朴素和 tiled GEMM 的理想化 HBM traffic。

## 导航

- 上一篇：[02 SIMT 与 warp](02_simt_warp_scheduling.md)
- 下一篇：[04 Tensor Core 与 GEMM 数据流](04_tensor_core_gemm_dataflows.md)
- 相关：[09 Roofline](09_roofline_profiling_tuning.md) · [10 LLM 映射](10_llm_training_inference_mapping.md)
