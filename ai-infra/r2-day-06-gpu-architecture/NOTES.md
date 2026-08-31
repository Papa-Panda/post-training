# NOTES — r2-Day06 GPU 架构与 HBM Memory Wall

## 准确术语

- **SM (Streaming Multiprocessor)**：warp 调度与执行的基本硬件单元。
- **CUDA Core**：执行通用标量/向量算术的执行通路；本课 FP32 vector add 用它建立 roofline 直觉。
- **Tensor Core**：面向矩阵乘加的专用执行单元；高峰值算力会把某些算子的 ridge point 推得更高，因此喂饱 Tensor Core 更依赖数据复用。
- **HBM bandwidth**：GPU device memory 的理论峰值带宽。H100 SXM 官方值 3.35 TB/s；不能和 H100 NVL、A100 PCIe/SXM 混写。
- **Arithmetic intensity**：每从 HBM 搬 1 byte 做多少 FLOPs，单位 FLOP/byte。
- **Memory-bound / compute-bound**：Roofline 模型中分别先撞 $B_{HBM}I$ 或 $P_{peak}$ 上界；是模型判断，不是 profiler 结论。
- **Shared-memory capacity**：H100 为 228 KB/SM，但单 block 最大为 227 KB；两者不是同一个限制。

## 可手算小例子：为什么 vector add 喂不饱 H100 算力

设 $N=2^{20}=1{,}048{,}576$，计算 FP32 `C[i] = A[i] + B[i]`。

### 1. FLOPs

每个元素 1 次加法：

$$\text{FLOPs}=N=1{,}048{,}576$$

### 2. HBM 流量

理想化地假设每个元素只读/写一次，FP32 每元素 4 bytes：

- 读 A：$4N$ bytes
- 读 B：$4N$ bytes
- 写 C：$4N$ bytes

所以：

$$\text{bytes}=12N=12{,}582{,}912\ \text{bytes}=12\ \text{MiB}$$

### 3. Arithmetic intensity

$$I=\frac{N}{12N}=\frac{1}{12}\approx0.0833\ \text{FLOP/byte}$$

### 4. H100 SXM FP32 ridge point

使用官方理论峰值 FP32 67 TFLOP/s 与 HBM 3.35 TB/s；两者都按十进制前缀：

$$I^*=\frac{67}{3.35}=20\ \text{FLOP/byte}$$

因为 $0.0833\ll20$，这个算子在理想 Roofline 下是 memory-bound。

### 5. 理论带宽上界与时间下限

$$P_{BW}=3.35\times\frac{1}{12}=0.2792\ \text{TFLOP/s}$$

$$t_{min}=\frac{12{,}582{,}912}{3.35\times10^{12}}=3.756\ \mu s$$

这只是**理论估计**。它未计入 kernel launch、cache、transaction granularity、ECC/协议、时钟、竞争与访存合并等影响，不是 benchmark。

## 可手算小例子：tile 是否放得进 shared memory

为了 GEMM 复用，假设一个 block 同时缓存 A、B 两个 $128\times128$ BF16 tile：

$$2\ \text{tiles}\times128\times128\times2\ \text{bytes}=65{,}536\ \text{bytes}=64\ \text{KiB}$$

64 KiB 小于 H100 的 227 KiB/block opt-in 上限，容量上可放下。若改成 $256\times256$：

$$2\times256\times256\times2=262{,}144\ \text{bytes}=256\ \text{KiB}$$

超过 227 KiB/block，单 block 放不下。即使容量上能放下，也不代表性能最好：还要考虑 register 用量、活跃 blocks/SM、bank conflict 与同步。

## A100 / H100 / H200：只比较已核对字段

| 产品 | 显存 | 理论 HBM 带宽 |
|---|---:|---:|
| A100 80GB PCIe | 80 GB HBM2e | 1,935 GB/s |
| A100 80GB SXM | 80 GB HBM2e | 2,039 GB/s |
| H100 SXM | 80 GB | 3.35 TB/s |
| H200 | 141 GB HBM3e | 4.8 TB/s |

结论不是“H200 一定比 H100 快 1.43 倍”，而是 memory-bound 且能有效利用带宽的部分有更高上限；compute-bound、launch-bound、同步受限或访问不规则的 workload 不会自动按带宽比例加速。

## 今日一句话

**GPU 优化首先不是“多算”，而是判断数据从哪一级来：若每 byte 做的 FLOPs 太少，算力再高也会等 HBM。**

## 讨论难点

同一个 GEMM 为什么既可能 compute-bound，也可能 memory-bound？请从矩阵尺寸、batch、tile 复用、dtype、Tensor Core 路径和 occupancy 说明：提高 arithmetic intensity 能移动 $B\cdot I$ 上界，但 tile 变大同时可能因 shared memory/register 压力降低 occupancy，所以“减少 HBM 流量”不自动等于“更快”。

## 验证状态

- `gpu_memory_wall.py` 与 6 个 CPU 单元测试已执行。
- 所有数字输出均标为 theoretical estimate。
- **execution not validated on H100 / 待H100验证**：CUDA kernel、Nsight、有效带宽、cache hit rate、occupancy、Tensor Core 吞吐均未验证。

## 原始 / 官方来源

1. NVIDIA H100 product specifications: https://www.nvidia.com/en-us/data-center/h100/?trk=article-ssr-frontend-pulse_little-text-block
2. NVIDIA Hopper Tuning Guide: https://docs.nvidia.com/cuda/archive/12.2.2/hopper-tuning-guide/index.html
3. NVIDIA A100 product specifications: https://www.nvidia.com/en-us/data-center/a100/?xd_co_f=3f62097c-89dd-4617-b76f-814ed24253f2
4. NVIDIA H200 technical post: https://developer.nvidia.com/blog/nvidia-h200-tensor-core-gpus-and-nvidia-tensorrt-llm-set-mlperf-llm-inference-records/
5. CUDA C++ Best Practices Guide: https://docs.nvidia.com/cuda/archive/12.1.0/cuda-c-best-practices-guide/index.html
6. NVIDIA H100 architecture whitepaper: https://nvdam.widen.net/content/tdwwiwotwr/original/gtc22-whitepaper-hopper.pdf
