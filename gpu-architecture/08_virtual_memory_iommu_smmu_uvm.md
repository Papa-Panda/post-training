# 08 — GPU Virtual Memory、IOMMU/SMMU 与 UVM

## 1. 三类地址不要混

- **CPU virtual address (VA)**：CPU process 页表解释；
- **device virtual address / I/O virtual address (IOVA)**：GPU/device 发起 DMA 时使用；
- **physical address (PA)**：最终 DRAM/HBM/system memory 位置。

IOMMU（Arm 系统称 SMMU）为非 CPU masters 做地址翻译和权限检查。统一虚拟地址空间不意味着物理内存统一，也不意味着访问成本相同。

## 2. 翻译链

简化 device DMA：

```text
(device, stream/PASID, VA/IOVA)
        -> context descriptor
        -> page-table walk / TLB
        -> PA + permissions
        -> memory/interconnect request
```

地址翻译开销可粗略拆成：

$$T_{\mathrm{access}}=T_{\mathrm{data}}+P_{\mathrm{TLB\ miss}}T_{\mathrm{walk}}+P_{\mathrm{fault}}T_{\mathrm{fault}}.$$

page fault/migration 可能比普通 HBM access 慢很多；随机超大工作集会同时伤害 locality、TLB 和迁移。

## 3. SMMU 的隔离职责

SMMU 不只是“让设备看到虚拟地址”，还要：

- 限制 DMA 可访问范围；
- 隔离 VM/process/device；
- 支持 stage-1/stage-2 translation；
- 处理 translation fault；
- 在支持 Shared Virtual Addressing 时，把 process address space 与设备请求关联。

Arm SMMU 软件指南指出，SVA 需要设备/总线/SMMU 支持多个 address spaces（PCIe 常用 PASID）和 I/O page faults（PCIe PRI 等）。这是一组系统条件，不是打开一个 runtime flag 就完成。

## 4. UVA、UVM 与 VMM 不是同一层

- **UVA (Unified Virtual Addressing)**：在支持的平台上统一 host/device virtual-address 范围和 pointer identity；不自动提供 page migration、物理内存统一、统一带宽或 peer reachability。
- **UVM / Managed Memory**：管理 allocation、mapping、placement、migration 与可访问性，典型入口是 `cudaMallocManaged`。
- **CUDA VMM**：显式分离 VA reservation、physical allocation、mapping 和 access permission，服务于 allocator、稀疏 remapping 与跨设备分享。

### UVM / Managed Memory

CUDA Managed Memory 给 CPU/GPU 暴露统一指针，runtime/driver 管理 placement、migration 和 coherence。它改善可编程性，但性能取决于访问模式。

典型路径：

1. GPU 访问不 resident page；
2. fault/访问计数触发 migration 或 remote mapping；
3. 更新 page tables/TLB；
4. kernel 继续。

若 CPU/GPU 交替细粒度写同一 pages，可能产生 thrashing。可用 prefetch/advice、批量 phase 和合适 page locality 改善；最终应检查 migration bytes/faults。

## 5. CUDA Virtual Memory Management

显式 VMM API 将 address reservation、physical allocation 和 mapping 分开：

```text
reserve VA -> create physical allocation -> map -> set access
```

用途包括：可扩展 allocator、连续 VA、跨 device/shareable allocation、按需 mapping。它不自动提供数据一致性；mapping lifetime、访问权限和 stream synchronization 仍由程序管理。

## 6. P2P、ACS 与隔离

PCIe peer-to-peer 可让设备间直接传输，避免 host-memory staging。但路径是否经过 IOMMU/SMMU 与系统拓扑有关。Arm 文档说明，如果 P2P 绕过 SMMU，SMMU 无法强制隔离；PCIe ACS/redirected validation 可让请求经受访问验证。

因此“P2P 更快”与“P2P 安全可用”是两个问题：

- 性能：路径、带宽、peer mapping；
- 正确性：地址可见、ordering、lifetime；
- 安全：IOMMU/SMMU/ACS、tenant/device isolation。

NVIDIA GPUDirect RDMA 文档还给出一个重要的平台约束：依赖参与 PCIe devices 看到一致 physical addresses 的路径，要求 IOMMU 关闭或使用 pass-through/identity mapping。不能把它泛化成“IOMMU 总是有害”——翻译提供 DMA isolation/virtualization，ATS/PASID/PRI 等能力和具体 GPU/NIC/driver 支持矩阵会改变可用方案。

## 7. 与 GPU kernel 性能的联系

- TLB reach 不足：大且随机的 embedding/KV 工作集；
- migration：oversubscription 或 UVM placement 不当；
- pinned host memory：支持 DMA/异步 copy，但过量 pin 会影响系统；
- GPUDirect/RDMA：减少 CPU staging，不绕过所有 registration、ordering 和安全约束；
- allocator fragmentation：有 free bytes 仍可能无法满足大连续 physical/mapping 需求。

## 8. 常见误区

- unified address ≠ uniform latency/bandwidth；
- virtual contiguous ≠ physical contiguous；
- IOMMU 一定很慢：TLB 命中与 batching 下开销不同，应测；
- page fault 只影响首次访问：反复 eviction/thrashing 会持续发生；
- UVM 是“无限显存”：超出 HBM 后工作集和访问模式决定性能，不能隐藏带宽物理限制。

## 9. 验证指标

- page faults 与 migration bytes；
- GPU/CPU residence；
- TLB/page-walk 指标（若工具暴露）；
- H2D/D2H/P2P 实际路径；
- NUMA placement；
- allocation/mapping/fault 时间线。

## 导航

- 上一篇：[07 Multi-GPU 与 NCCL](07_multi_gpu_interconnect_collectives.md)
- 下一篇：[09 Roofline 与 Profiling](09_roofline_profiling_tuning.md)
- 官方锚点：[Arm SMMU Software Guide](https://developer.arm.com/documentation/109242/latest/) · [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
