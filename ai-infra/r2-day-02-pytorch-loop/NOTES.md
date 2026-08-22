# NOTES - r2-Day02 PyTorch 基础

CPU proxy，待H100 NCCL验证

- single-GPU time: 2.31s CPU proxy (mnist 1 epoch batch32)
- 2-GPU gloo time: 3.12s CPU proxy (all-reduce overhead)
- comm overhead: ~35% CPU proxy (true NCCL 7-15%预期)
- ckpt save: 12ms CPU 33KB
- memory proxy: 42.5GB bf16-mix峰值公式 (P-b)/G+b P=28GB fp32 b=0.22B*4 G=2 → 14.44GB → bf16-mix 42.5GB

待H100：
- torch.cuda.max_memory_allocated 真数
- torch.profiler CUDA trace AllGather 60% vs ReduceScatter 40%
- NCCL BW topo -m NVLink 900GB/s vs PCIe 64GB/s
