# r2-Day06 — GPU 架构与 HBM Memory Wall

## Connection to Prev

r2-Day05 用 JAX `Mesh` / `PartitionSpec` 声明“张量放在哪些设备”；r2-Day06 再向下一层，判断**一张卡内的数据放在哪一级存储、算子最终受算力还是 HBM 带宽约束**。

- **牺牲什么**：tiling、fusion、数据复用会增加实现/调参复杂度，并占用 register/shared memory；tile 太大还会降低 occupancy，甚至 spill 回 local/global memory。
- **换取什么**：减少 HBM 往返，把热点数据留在 register/shared memory/L2，让 Tensor Core / CUDA Core 少等数据。
- **何时不赚**：算子已是 compute-bound；没有可复用数据；工作集放不进片上存储；或为复用付出的同步、重算、低 occupancy 成本超过省下的 HBM 流量。

## 今天要形成的硬件心智模型

```text
Host CPU
   |
GPU device
   +-- SM (Streaming Multiprocessor): warp 调度与执行的基本单元
       +-- CUDA Cores: 通用标量/向量算术
       +-- Tensor Cores: 矩阵乘加路径
       +-- registers: 每线程私有，最快但容量最紧
       +-- shared memory / L1: 每个 SM，程序可显式复用 tile
   +-- L2 cache: 全 GPU 共享
   +-- HBM: 大容量、高带宽，但比片上存储更远
```

“register > shared/L1 > L2 > HBM”表达的是**优先复用、减少远端搬运**，不是一组可跨工作负载直接套用的固定延迟数字。最终要用 profiler 验证。

## 先钉死 SKU，避免混用规格

| GPU 产品 | 显存 | 官方峰值显存带宽 | 说明 |
|---|---:|---:|---|
| A100 80GB PCIe | 80 GB HBM2e | 1,935 GB/s | PCIe 产品 |
| A100 80GB SXM | 80 GB HBM2e | 2,039 GB/s | SXM 产品 |
| H100 SXM | 80 GB | 3.35 TB/s | 本课算例所用 SKU |
| H200 | 141 GB HBM3e | 4.8 TB/s | 容量与带宽升级，不代表所有算子等比例加速 |

Hopper 调优指南还给出 H100 的 **50 MB L2、228 KB shared memory/SM、227 KB 最大 shared memory/block**。注意 228 KB/SM 不等于单个 block 全部可用；超过默认静态 shared-memory 限额时还需要显式 opt-in。

## Memory Wall：用 Roofline 做第一步判断

定义 arithmetic intensity：

$$I = \frac{\text{FLOPs}}{\text{bytes transferred from/to HBM}}$$

理论 Roofline 上界：

$$P \le \min(P_{peak},\; B_{HBM} I)$$

H100 SXM 的 FP32 峰值为 67 TFLOP/s、HBM 峰值带宽为 3.35 TB/s，因此用同一十进制单位计算的 ridge point 为：

$$I^* = \frac{67\times 10^{12}}{3.35\times 10^{12}} = 20\ \text{FLOP/byte}$$

- $I < 20$：理论上先撞带宽屋顶，倾向 memory-bound。
- $I > 20$：才可能先撞 FP32 算力屋顶；“可能”是因为实际瓶颈还可能来自指令、依赖、occupancy 或访存模式。

这里的 67 TFLOP/s 与 3.35 TB/s 都是**产品理论峰值**，不是本仓库测得的 benchmark。

## 可执行小任务

`gpu_memory_wall.py` 会真正执行四步，而不是只打印流程：

1. 载入 H100 SXM 规格；
2. 计算 FP32 vector add 的 FLOPs 与读写字节数；
3. 计算 arithmetic intensity、ridge point、带宽 roof 与理论最短 HBM 时间，并分类 memory/compute bound；
4. 计算两个 BF16 方形 tile 的工作集，检查能否放进 227 KiB/block 的 opt-in shared memory 上限。

```bash
python3 ai-infra/r2-day-06-gpu-architecture/gpu_memory_wall.py
python3 -m unittest discover -s ai-infra/r2-day-06-gpu-architecture -p 'test_*.py' -v
```

默认算例输出的关键结论：

- `C=A+B`，$N=2^{20}$ 个 FP32 元素：1,048,576 FLOPs；读 A、读 B、写 C，共 12,582,912 bytes。
- $I=1/12\approx0.0833$ FLOP/byte，远低于 20，故理论分类为 memory-bound。
- 带宽 roof：$3.35\times(1/12)=0.2792$ TFLOP/s。
- 仅按峰值 HBM 带宽估计的下限：$12{,}582{,}912/(3.35\times10^{12})\approx3.756\ \mu s$。
- 两个 $128\times128$ BF16 tile 占 65,536 bytes，低于 227 KiB/block；两个 $256\times256$ tile 占 262,144 bytes，超出上限。

## 状态

- 已验证：纯 Python 公式、单位、边界条件与 6 个 CPU 单元测试。
- **execution not validated on H100 / 待H100验证**：没有运行 CUDA kernel、Nsight、实测带宽、kernel latency、occupancy、Tensor Core 吞吐或缓存命中率。
- 因缺少 H100/CUDA，本课状态保持 `blocked`，不能把 theoretical estimate 当 benchmark。

## 原始 / 官方来源

- NVIDIA H100 product specifications（80 GB、3.35 TB/s、FP32 67 TFLOP/s）：https://www.nvidia.com/en-us/data-center/h100/?trk=article-ssr-frontend-pulse_little-text-block
- NVIDIA Hopper Tuning Guide（50 MB L2、228 KB/SM、227 KB/block）：https://docs.nvidia.com/cuda/archive/12.2.2/hopper-tuning-guide/index.html
- NVIDIA A100 product specifications（A100 80GB PCIe/SXM 带宽）：https://www.nvidia.com/en-us/data-center/a100/?xd_co_f=3f62097c-89dd-4617-b76f-814ed24253f2
- NVIDIA H200 technical post（141 GB HBM3e、4.8 TB/s）：https://developer.nvidia.com/blog/nvidia-h200-tensor-core-gpus-and-nvidia-tensorrt-llm-set-mlperf-llm-inference-records/
- CUDA C++ Best Practices Guide（effective bandwidth 的 read+write 字节公式）：https://docs.nvidia.com/cuda/archive/12.1.0/cuda-c-best-practices-guide/index.html
- NVIDIA H100 Tensor Core GPU Architecture whitepaper：https://nvdam.widen.net/content/tdwwiwotwr/original/gtc22-whitepaper-hopper.pdf
