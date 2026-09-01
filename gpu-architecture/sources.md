# Sources & Evidence Ledger

> 原则：官方 programming/tuning guides 与原论文支撑技术结论；中文文章/视频只作为选题线索。未取得正文时不推断内容。访问状态记录于 2026-09-01。

## 1. Primary / official sources

| 主题 | 来源 | 本专题使用范围 |
|---|---|---|
| CUDA execution/memory/streams/VM | [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) | grid/block/thread、warp、memory model、streams、Managed Memory/VMM |
| Coalescing/shared memory/measurement | [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | CC 6.0+ 的 32-byte segment 教学规则、shared-memory tiling、benchmark 纪律 |
| Ampere | [Ampere Tuning Guide](https://docs.nvidia.com/cuda/archive/12.0.1/ampere-tuning-guide/index.html) | async global→shared copy、Tensor Core precision/instruction evolution |
| Hopper | [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) | thread-block cluster、TMA、异步 transaction、架构限制 |
| H100 architecture | [NVIDIA H100 Tensor Core GPU Architecture Whitepaper](https://nvdam.widen.net/content/tdwwiwotwr/original/gtc22-whitepaper-hopper.pdf) | Hopper SM/Tensor Core/Transformer Engine/NVLink 的官方架构背景 |
| Blackwell compilation | [Blackwell and CUDA architecture-specific features](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/) | `sm_*a` 特性非前向兼容的边界 |
| Profiling | [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html) | occupancy、sections、Roofline/profiling 工作流 |
| Cooperative Groups | [CUDA Programming Guide: Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups) | 显式 group 与同步 scope |
| GEMM library | [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/) | GEMM/cuBLASLt 接口与 algorithm selection |
| GEMM templates | [CUTLASS Documentation](https://docs.nvidia.com/cutlass/) | CTA/warp/instruction tiling、pipeline、epilogue |
| Deep-learning library | [cuDNN Core Concepts](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/core-concepts.html) | Tensor Core、layout、operation graph |
| NCCL semantics | [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) | all-reduce/broadcast/reduce/all-gather/reduce-scatter 的正式语义与 rank 一致性 |
| NCCL implementation context | [Fast Multi-GPU collectives with NCCL](https://developer.nvidia.com/blog/?p=6598) | ring/chunk/monolithic kernel/topology 的实现背景；具体版本行为仍以当前文档为准 |
| NCCL double-tree | [Massively Scale Deep Learning Training with NCCL 2.4](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/) | 互补树、对数 startup depth 与流水化 bandwidth 设计；不把朴素 tree 式当 NCCL 精确模型 |
| NVSwitch/ring latency | [3x Faster AllReduce with NVSwitch and TensorRT-LLM MultiShot](https://developer.nvidia.com/blog/3x-faster-allreduce-with-nvswitch-and-tensorrt-llm-multishot/) | ring 的 $2p-2$ steps 与 latency 动机；产品宣传数字未用于普遍结论 |
| AMD architecture | [ROCm GPU architecture documentation](https://rocm.docs.amd.com/en/docs-6.3.0/conceptual/gpu-arch.html) | CU/wavefront/CDNA 对照，提醒常数与 CUDA 不可照搬 |
| AMD CDNA 2 example | [AMD Instinct MI250 microarchitecture](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi250.html) | wavefront/CU/Infinity Fabric/HBM 的官方术语；未把单 SKU 数字泛化 |
| Arm memory management | [Arm Memory Management Guide](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/Learn%20the%20Architecture/LearnTheArchitecture-MemoryManagement-101811_0100_00_en.pdf?revision=1fdc3375-d81c-4457-b786-04fb98557de0) | MMU、VA→PA、SMMU/IOMMU 定位 |
| Arm SMMU | [Arm SMMU Software Guide](https://developer.arm.com/documentation/109242/latest/) | SVA、PASID、PRI/fault、stage translation、P2P/ACS |
| GPUDirect RDMA | [GPUDirect RDMA Documentation](https://docs.nvidia.com/cuda/archive/12.3.0/gpudirect-rdma/index.html) | PCIe locality，以及该路径对 IOMMU pass-through/identity mapping 的约束 |

## 2. Primary papers

| 主题 | 来源 | 采用的结论/边界 |
|---|---|---|
| Roofline | Williams, Waterman, Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, CACM 2009, [DOI](https://doi.org/10.1145/1498765.1498785) | $P\le\min(P_{peak},BI)$；是上界模型而非 cycle simulator |
| FlashAttention | Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022, [arXiv](https://arxiv.org/abs/2205.14135) | 通过 tiling/online softmax 减少 HBM I/O，不将其误述为近似 attention |
| Ring all-reduce | Patarasuk & Yuan, *Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations*, JPDC 2009, [DOI](https://doi.org/10.1016/j.jpdc.2009.05.002) | ring reduce-scatter + all-gather 的带宽模型；真实 NCCL 还会按拓扑/协议选择算法 |
| Systolic arrays | Kung & Leiserson, *Systolic Arrays for VLSI*, 1979, [CMU record](https://www.cs.cmu.edu/~christos/courses/826-resources/PAPERS+BOOK/systolic.pdf) | 数据随规则阵列传播与局部复用的经典模型；不据此断言未公开 Tensor Core 物理实现 |

## 3. 用户提供的中文/对话材料：可访问性与使用方式

### 可见页面标题/摘要，未取得完整转录

- [Bilibili：掌握 CUDA GPU 并行编程（硬件与软件）中文语音](https://www.bilibili.com/video/BV19gBUBWEh1/?spm_id_from=333.1387.favlist.content.click)：静态页面可见课程标题和相关推荐，没有可靠完整转录；仅作为 CUDA 软硬件协同选题线索。
- [Bilibili：Tilelang/Triton/cuTile 的异同与选择](https://www.bilibili.com/video/BV1PFmdBmEdN/?spm_id_from=333.1387.favlist.content.click&vd_source=1fecee762931e992c96e5e166be13b76)：静态页面可见标题、日期和相关主题，没有完整转录；未归因任何具体技术判断。
- [ChatGPT 对话](https://chatgpt.com/c/69a76e70-87ac-83e8-a363-fe586256755e)：只返回 ChatGPT shell/history 标题，无共享正文；未使用其内容。

### 本次环境未能取得正文，保留为后续阅读入口

以下页面均**没有作为事实证据**；章节中的相应结论已回到上面的官方文档/原论文：

- [GPU 架构演进之 Gemini 问答](https://zhuanlan.zhihu.com/p/1984294397777568939)
- [Tensor Core 的软硬件协同设计](https://zhuanlan.zhihu.com/p/1952778893720262526)
- [如何理解 NCCL？](https://www.zhihu.com/question/63219175/answer/1997433472390813296)
- [一文看懂 ARM SMMU 内存管理](https://zhuanlan.zhihu.com/p/2019679464053822560)
- [线程束基本函数与协作组](https://zhuanlan.zhihu.com/p/2018433599020507875)
- [CUDA 标准库的使用](https://zhuanlan.zhihu.com/p/2019532918247179254)
- [Blackwell 矩阵乘法：基础介绍](https://zhuanlan.zhihu.com/p/2006030247439638741)
- [CUDA 流](https://zhuanlan.zhihu.com/p/2019443617530422289)
- [如何系统学习 GPU 架构？—进击的 Bruce](https://www.zhihu.com/question/319355296/answer/3374307130)
- [TensorCore 演进：寄存器的自我救赎之路](https://zhuanlan.zhihu.com/p/1929640607300719712)
- [如何系统学习 GPU 架构？—nicholaswilde](https://www.zhihu.com/question/319355296/answer/1931398398445060845)
- [GPU 基础知识 & 模型性能优化](https://zhuanlan.zhihu.com/p/2015887800717828407)
- [深入理解 GPU 硬件架构及运行机制](https://zhuanlan.zhihu.com/p/678001378)
- [GPU 编程模型与性能调优策略](https://zhuanlan.zhihu.com/p/1994438759010305490)
- [GEMM 的脉动阵列、内积、外积硬件架构比较](https://zhuanlan.zhihu.com/p/1997608528660153669)

## 4. Claim discipline

1. **硬件规格**必须同时写产品形态、dtype、dense/sparse 与理论/实测口径。
2. **性能数字**若来自 vendor blog，要保留 workload/config，不外推为普遍倍数。
3. **教学模型**（32-byte segments、occupancy、$\alpha$–$\beta$、Roofline）明确标为近似。
4. **未公开微架构**不从示意图、反汇编或二手文章强推。
5. **可运行代码**是 CPU analytical model；未声称在 CUDA GPU 上测量。
