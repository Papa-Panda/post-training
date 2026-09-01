# 07 — Multi-GPU 互连、NVLink/NVSwitch 与 NCCL

## 1. “有多张 GPU”不代表全连接同带宽

数据路径可能经过：GPU local HBM、NVLink、NVSwitch、PCIe switch/root complex、host memory、NIC、InfiniBand/Ethernet。拓扑决定：

- peer access 是否可用；
- link 数/带宽是否对称；
- 是否跨 CPU socket/NUMA；
- GPU 到 NIC 是否在同一 rail；
- 多个 collective 是否争用同一链路。

NVLink 是 point-to-point 高速链路；NVSwitch 提供基于 NVLink 的交换结构；NCCL 是选择通信路径和算法的软件库。三者不能互换使用。

## 2. $\alpha$–$\beta$ 模型

传输 $n$ bytes 的简化成本：

$$T(n)=\alpha+n\beta,$$

其中 $\alpha$ 是每步 latency，$\beta=1/B$ 是每 byte 时间。真实系统还有协议、chunk、拓扑、并发和 reduction compute。

### Ring all-reduce

reduce-scatter + all-gather，各有 $p-1$ steps，每步约传 $n/p$：

$$T_{\mathrm{ring}}\approx2(p-1)\alpha+2\frac{p-1}{p}n\beta.$$

优点：大消息时每 rank 的有效发送量趋近 $2n$，带宽利用率高。缺点：steps 随 $p$ 线性增长，小消息 latency 较差。

### Tree all-reduce

简化二叉 reduce + broadcast：

$$T_{\mathrm{tree}}\approx2\lceil\log_2p\rceil\alpha+2\lceil\log_2p\rceil n\beta.$$

它突出低 step count，但这个 bandwidth term 是**未流水化、每层传完整 tensor 的朴素上界**，不能拿来估算 NCCL double-tree。NCCL 的 double-tree 使用互补树、分片和 pipeline，目标是在 $O(\log p)$ startup depth 下保持接近带宽最优的每-rank volume。更合适的包络是：

$$T_{\mathrm{double\ tree}}\approx2\lceil\log_2p\rceil\alpha+C_{\mathrm{tree}}n\beta+T_{\mathrm{reduce}},$$

其中 $C_{\mathrm{tree}}$ 依赖 topology、channels、chunk 和 protocol，不能假装是固定常数。NCCL 会按消息、拓扑和协议选择算法，不应把这些教学式当实际 selector。

## 3. Collective 语义

- all-reduce：每 rank 输入同 shape，归约后每 rank 得完整结果；
- reduce-scatter：归约后每 rank 得一片；
- all-gather：各 rank 的片段拼成完整结果；
- all-to-all：每 rank 向所有 ranks 发送不同片段，MoE 路由常见；
- broadcast/reduce：一对多/多对一。

Tensor parallel 的 all-reduce 常可改写为 reduce-scatter + 后续 all-gather，与算子切分联动；不是单纯替换 API。

## 4. NCCL 的职责

NCCL 提供 GPU collective，针对 PCIe、NVLink 和跨节点网络进行 topology-aware 优化。经典 ring 实现将大消息切 chunk，用单个/少数持久 CUDA kernels 执行 copy/reduce/sync，避免为每个 chunk 启动独立 kernel。

关键观测：

- algorithm bandwidth 与 bus bandwidth 口径；`nccl-tests` 对 point-to-point all-reduce 使用 $\mathrm{busbw}=\mathrm{algbw}\times2(p-1)/p$ 归一化，但它不必等于带 in-network reduction 的真实 wire traffic；
- channels/protocol；
- topology graph；
- transport（P2P、shared memory、network）；
- collective 与 compute 是否真实 overlap。

## 5. 计算通信重叠

把梯度/activation 分 bucket，某 bucket ready 后立即启动 collective：

$$T_{\mathrm{step}}\approx T_{\mathrm{compute}}+T_{\mathrm{comm}}-T_{\mathrm{overlap}}.$$

`overlap` 不是免费：communication kernels 消耗 SM、copy/network resources 和 HBM bandwidth；过小 bucket 增加 $\alpha$ 与 launch，过大 bucket 延迟开始时间。

对于 ring all-reduce，粗略 crossover 可由比较 latency 与 bandwidth 项得到：

$$n^*\approx\frac{p\alpha}{\beta}.$$

这只是调参方向：小于该量更 latency-sensitive，大于该量更 bandwidth-sensitive。

## 6. LLM 并行策略的通信形状

| 并行 | 典型 collective | 代价特征 |
|---|---|---|
| Data parallel | gradient all-reduce / reduce-scatter | 参数/梯度规模，易与 backward bucket overlap |
| Tensor parallel | 每层 all-reduce/all-gather/reduce-scatter | 高频、延迟敏感；decode 小 batch 尤其明显 |
| Pipeline parallel | stage 间 point-to-point activation | bubble 与 microbatch schedule |
| Expert parallel | all-to-all | token routing 不均与网络 contention |
| Sequence/context parallel | activation all-gather/reduce-scatter | 随 sequence 与切分变化 |

## 7. 常见误区

- `nvidia-smi topo` 显示 NVLink 不等于 collective 自动满带宽；rank placement/channels/shape 仍重要。
- NCCL timeout 不一定是网络故障，也可能有 rank 控制流不一致、前序 CUDA error 或某 rank OOM。
- 通信时间下降不一定 step time 同比下降，若它原本已被 compute 覆盖。
- 把单向 link peak 与 aggregate bidirectional/bus bandwidth 混写会得到错误效率。

## 8. 可运行模型

`collective_cost()` 提供 ring/tree 的 $\alpha$–$\beta$ 教学模型，并在 tests 中验证 8-rank ring all-reduce 每 rank 单向 wire volume 为 $2(7/8)n$。它不模拟 topology/contention/protocol。

## 导航

- 上一篇：[06 库与 DSL](06_libraries_and_kernel_dsl.md)
- 下一篇：[08 VM/IOMMU/SMMU/UVM](08_virtual_memory_iommu_smmu_uvm.md)
- 相关：[10 LLM 系统映射](10_llm_training_inference_mapping.md) · [`vllm-rollout/`](../vllm-rollout/)
