# GPU Architecture & Programming

> 从晶体管和数据通路一路推到 LLM kernel：**硬件资源 → SIMT 执行 → 内存层次 → Tensor Core/GEMM → CUDA runtime → 多卡通信 → 地址翻译 → Roofline/profiler → 训练与推理系统**。

本专题不是芯片型号百科，也不是 CUDA API 抄表。目标是建立一套可计算的性能心智模型：给定一个 kernel 或分布式算子，能先判断它在搬什么数据、在哪一级存储复用、由什么资源限制并发、何时被带宽/延迟/算力卡住，以及应该用什么 profiler 证伪。

## 统一性能模型

把一次 GPU 工作写成资源时间上界：

$$T\gtrsim\max\left(\frac{F}{P_{\mathrm{peak}}},\frac{Q_{\mathrm{HBM}}}{B_{\mathrm{HBM}}},\frac{Q_{\mathrm{link}}}{B_{\mathrm{link}}}\right)+T_{\mathrm{launch}}+T_{\mathrm{sync}}+T_{\mathrm{translation}}.$$

- $F$：执行的 FLOPs；
- $Q_{\mathrm{HBM}}$：HBM 与芯片之间移动的 bytes；
- $Q_{\mathrm{link}}$：GPU 间或 GPU–host 间移动的 bytes；
- $P,B$：可达到的计算/带宽上限，而非默认等于宣传峰值；
- launch、同步、page fault/TLB miss 等延迟在小 kernel、decode 和不规则访问中不可忽略。

这条式子不是可直接相加的精确模拟器，而是贯穿专题的诊断骨架。

## 路线图

### Phase A — 一张 GPU 内部（01–04）

1. [`01_architecture_evolution.md`](01_architecture_evolution.md)：为什么 GPU 从图形流水线演化为通用 SIMT + 专用矩阵数据通路；不要把“代际”误写成单一倍数。
2. [`02_simt_warp_scheduling.md`](02_simt_warp_scheduling.md)：thread/block/warp/SM、分支发散、延迟隐藏、occupancy 的资源方程。
3. [`03_memory_hierarchy_coalescing.md`](03_memory_hierarchy_coalescing.md)：register/shared/L1/L2/HBM、coalescing、bank conflict、异步搬运与数据复用。
4. [`04_tensor_core_gemm_dataflows.md`](04_tensor_core_gemm_dataflows.md)：Tensor Core、mixed precision、GEMM tiling，以及 inner/outer/systolic 数据流的共同本质。

### Phase B — 编程与测量（05、06、09）

5. [`05_cuda_programming_model.md`](05_cuda_programming_model.md)：warp primitives、Cooperative Groups、streams、events、graph 与正确同步。
6. [`06_libraries_and_kernel_dsl.md`](06_libraries_and_kernel_dsl.md)：cuBLAS/cuDNN/CUTLASS 与 CUDA/Triton/tile DSL 的抽象层级和选择边界。
9. [`09_roofline_profiling_tuning.md`](09_roofline_profiling_tuning.md)：Roofline、occupancy、Nsight 指标、从症状到实验的调优流程。

### Phase C — 从单卡到系统（07、08、10）

7. [`07_multi_gpu_interconnect_collectives.md`](07_multi_gpu_interconnect_collectives.md)：PCIe/NVLink/NVSwitch/NIC、NCCL、ring/tree、拓扑与计算通信重叠。
8. [`08_virtual_memory_iommu_smmu_uvm.md`](08_virtual_memory_iommu_smmu_uvm.md)：GPU virtual memory、IOMMU/SMMU、SVA/PASID、UVM、page migration 与隔离。
10. [`10_llm_training_inference_mapping.md`](10_llm_training_inference_mapping.md)：把前九章映射到 GEMM、attention、KV cache、parallelism 与 collective。

证据和材料可访问性见 [`sources.md`](sources.md)；CPU-only 模型见 [`code/`](code/)，测试见 [`tests/`](tests/)。

```text
01 架构演进
   ↓
02 SIMT / warp ──► 03 内存层次 ──► 04 Tensor Core / GEMM
   │                    │                    │
   └──────────────► 05 CUDA ◄───────────────┘
                         ↓
                  06 库与 kernel DSL
                         ↓
07 多卡互连/collective ◄─┼─► 08 VM/IOMMU/UVM
                         ↓
                  09 Roofline / profiling
                         ↓
                  10 LLM kernel / 系统映射
```

## 可运行模型

```bash
python3 gpu-architecture/code/demo.py
python3 -m unittest discover -s gpu-architecture/tests -v
```

模型覆盖：

- 32-byte segment coalescing；
- registers/shared-memory/thread ceilings 下的 occupancy；
- 单层 Roofline；
- ring/tree collective 的 $\alpha$–$\beta$ 教学模型；
- tiled GEMM 的理想化 HBM traffic；
- shared-memory bank conflict。

它们明确是**机制模型，不是真机 benchmark**。架构特定 allocation granularity、cache、协议、拓扑争用和 kernel 实现会使实测偏离。

## 与仓库其他专题的边界

| 专题 | 回答的问题 | 本专题只提供的接口 |
|---|---|---|
| [`ai-infra/`](../ai-infra/) | 如何搭 DDP/FSDP/checkpoint 等训练系统 | 硬件与 kernel 的底层成本模型；复用已有 Day06/07，不复制每日练习 |
| [`vllm-rollout/`](../vllm-rollout/) | rollout 的 TTFT/TPOT/KV 压力和压测 | decode memory wall、KV cache 与 TP collective 的硬件解释 |
| [`grpo-vs-ppo/`](../grpo-vs-ppo/) | RL objective 与训练算法 | 不讨论 PPO/GRPO 数学，只解释其 kernel/通信代价 |
| [`harness-engineering/`](../harness-engineering/) | Agent runtime、context、workflow 与可回归自改进 | 不讨论 agent harness；只讨论其推理 kernel 和 GPU runtime 资源 |

## 阅读后的能力检查

完成后应能回答：

1. 为什么“更高 occupancy”“更少 FLOPs”“更少 kernel”都不自动等于更快？
2. 一个 warp 的地址模式如何变成 memory transactions，shared-memory padding 为什么有效？
3. Tensor Core 的收益为什么依赖 tile、layout、精度、数据搬运和 epilogue？
4. ring all-reduce 为什么带宽效率高但 step 数随 GPU 数线性增长？
5. 为什么 LLM prefill 常比 decode 更 compute-heavy，而 decode 更容易被权重/KV 流量和 launch latency 限制？
6. profiler 的哪个观测能证伪自己的瓶颈假设？
