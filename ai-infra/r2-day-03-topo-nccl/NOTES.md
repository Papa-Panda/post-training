# NOTES - r2-Day03 通信拓扑

CPU proxy，待H100 NCCL验证

- NVLink BW: 900GB/s per GPU (H100 SXM5 18 links 900GB/s bi-dir)
- PCIe BW: 64GB/s Gen5 x16 双向
- IB 400Gbps = 50GB/s per link，8链 400GB/s aggregate
- Ring AllReduce 8卡 1GB: NVLink 1.94ms vs PCIe 27.3ms 差14倍 CPU proxy
- AllGather 0.97ms vs ReduceScatter 0.97ms 各占一半
- topo -m 预期：8卡H100 SXM5 NV12全互联，CPU 0-1 NUMA SYS跨，跨机NODE

待H100：
- nvidia-smi topo -m 真机输出拍照
- nccl-tests all_reduce_perf 1GB 8卡 真BW
- torch.distributed.all_reduce 1GB 真耗时 vs 理论 1.94ms
