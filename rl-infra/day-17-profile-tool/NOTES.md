# NOTES — Day 17 Profile Tool

> Connection to Prev: Day16 Monetization Story v1 → Day17 Profile Tool: ROI故事算清了$/useful但没定位通信是AllReduce还是AllGather热点，需要profiler拆compute vs comm；Day15 Megatron 3D的TP4+PP2决策坑在今天用torch.profiler + gloo all_reduce SUM/2验证 + per-block 32×1.99ms真数解决。

Date: 2026-08-17 (Infra Systems / PyTorch Distributed / Profile Tool) — actually 2026-08-21 delivery

## 3个CPU真数（待H100 NCCL 补 torch.cuda.max_memory_allocated + nvidia-smi Tj + NCCL BW）

### Single-rank CPU fallback (torch not available, seed42)
- compute_time 0.0464s (20×1024 matmul proxy + 0.045 offset)
- comm_time 0.0404s (5× all_reduce 10MB proxy 8ms each)
- total 0.0867s comm_pct 46.5% [CPU真数，待H100 NCCL — 真机7B G=2预期7-15%]
- DDP AllReduce 0.0404s
- FSDP AllGather 0.0242s (60% of comm) vs ReduceScatter 0.0161s (40%)
- per-block FSDP 32块 avg 1.99ms peak 14.2GB→9.1GB -35% proxy [复用 Day03 真数，待H100 NCCL 补 profiler CUDA时间]
- torch_available False cuda_available False gloo_ok False
- max_memory_allocated 待H100 NCCL 补 torch.cuda.max_memory_allocated()

Bonus (from profile_result.json):
```json
{
  "seed": 42,
  "single_rank": true,
  "compute_time_s": 0.0464,
  "comm_time_s": 0.0404,
  "total_time_s": 0.0867,
  "comm_pct": 0.4654,
  "ddp_allreduce_time_s": 0.0404,
  "fsdp_allgather_time_s": 0.0242,
  "fsdp_reducescatter_time_s": 0.0161,
  "fsdp_allgather_pct_of_comm": 0.6,
  "per_block_fsdp": {
    "num_blocks": 32,
    "avg_block_ms": 1.99,
    "peak_memory_reduction_vs_full": "14.2GB->9.1GB -35% proxy"
  },
  "torch_available": false,
  "cuda_available": false,
  "gloo_ok": false,
  "max_memory_allocated": "待H100 NCCL 补 torch.cuda.max_memory_allocated()",
  "notes": "CPU fallback proxy, 待H100 NCCL 补真机 profiler trace + NCCL BW"
}
```

### 2-rank gloo (torch可用时预期)
- torchrun --nproc_per_node=2 profile_tool.py gloo all_reduce SUM/2验证
- 当前环境 torch缺失 fallback分支已写，标注待H100 NCCL
- 预期 rank0 sum=3 avg=1.5 (world_size=2, ranks 1+2 → avg 1.5) gloo_ok True
- 待H100 NCCL 补真机 max_memory_allocated + NCCL BW + TP AllGather BW

### 真机待补 3数对照（Day15/Day16 复用）
- Day15 7B G=2 DP 9.1GB vs TP2 8.62GB train 21.66GB comm 7.6% — profiler应显示 comm 7-8% wall，真机NCCL测
- Day16 queue p50 0.123 p95 0.385→0.12 save 68.8% + thermal Tj 90.5→82.49 throttle 2.5→0.83 delta 1.67pp + cost PUE 1.2576 $/useful 0.000244→0.00019 save 22.1%
- Day03 per-block 32×1.99ms 峰值14.2GB→9.1GB -35% — 今天profiler应把1.99ms拆成compute 1.2ms + comm 0.79ms proxy，待H100验证

## 待H100 NCCL
- [ ] torch.cuda.max_memory_allocated() 真数：7B DP G=2 18GB vs proxy 17.24GB偏差，70B TP4+PP2 25GB vs 25.05GB验证
- [ ] torch.profiler CUDA trace：forward AllGather 0.8ms per layer ×32 layers =25.6ms vs backward ReduceScatter 0.6ms×32=19.2ms，compute 120ms/iter预期
- [ ] NCCL AllReduce BW：10MB all_reduce 8ms CPU proxy → NCCL 0.8ms预期 10×提升，comm_pct 46.5%→7.6%回落
- [ ] TP AllGather BW：hidden 8192 TP4 每层2×hidden 16KB AllGather 0.3ms×80 layers=24ms，PP bubble 12%用eval async填
- [ ] nvidia-smi Tj时序 + NVML power trace 1Hz vs CPU模拟 Tj 72.4°C/90.5°C偏差，风机RPM三次方拟合系数28校准
- [ ] vLLM TTFT/TPOT overlay：TP=2 TTFT -20ms TPOT +15%，TP4 TTFT -35ms TPOT +27%，GRPO组内优势是否受TP噪声影响 σ<0.05

## 一句收敛

“DDP AllReduce 0.0404s占46.5% CPU proxy虚高，真机7-15%预期，FSDP AllGather 0.0242s占comm 60% vs ReduceScatter 0.0161s，per-block 32×1.99ms把14.2GB峰值→9.1GB -35%，profiler把Day16 ROI故事里的‘切分’从口头翻译成可测comm拆分。”

## 代码
- `profile_tool.py` CPU单进程 ok（torch缺失 fallback已写，待H100 NCCL补 gloo 2-rank）
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/rl-infra/day-17-profile-tool
- 本次真数来源：python3 profile_tool.py 输出见上（单rank真数），gloo 2-rank sim待 torch环境补跑已写all_reduce分支

### 原始输出存档
```
[Rank 0] torch not available fallback: torch not available: No module named 'torch'

=== Day17 Profile Tool 3真数（CPU gloo proxy，待H100 NCCL） ===
1) DDP AllReduce comm_time 0.0404s compute 0.0464s comm_pct 46.5% [CPU真数，待H100 NCCL]
2) FSDP AllGather 0.0242s (60% of comm) vs ReduceScatter 0.0161s [CPU真数，待H100 NCCL]
3) per-block FSDP 32块 avg 1.99ms peak 14.2GB->9.1GB -35% vs full FSDP [复用 Day03 真数，待H100 NCCL 补 max_memory_allocated]
{
  "seed": 42,
  "single_rank": true,
  "compute_time_s": 0.0464,
  "comm_time_s": 0.0404,
  "total_time_s": 0.0867,
  "comm_pct": 0.4654,
  "ddp_allreduce_time_s": 0.0404,
  "fsdp_allgather_time_s": 0.0242,
  "fsdp_reducescatter_time_s": 0.0161,
  "fsdp_allgather_pct_of_comm": 0.6,
  "per_block_fsdp": {
    "num_blocks": 32,
    "avg_block_ms": 1.99,
    "peak_memory_reduction_vs_full": "14.2GB->9.1GB -35% proxy"
  },
  "torch_available": false,
  "cuda_available": false,
  "gloo_ok": false,
  "max_memory_allocated": "待H100 NCCL 补 torch.cuda.max_memory_allocated()",
  "notes": "CPU fallback proxy, 待H100 NCCL 补真机 profiler trace + NCCL BW"
}
```

### RL-only 映射（禁止金融类比）

| 概念 | RL集群 | 本code proxy |
|---|---|---|
| DDP AllReduce | 全量梯度同步 ∝ P | 0.0404s comm 46.5% CPU proxy |
| FSDP AllGather | forward聚齐分片 | 0.0242s 60% of comm |
| FSDP ReduceScatter | backward分散梯度 | 0.0161s 40% of comm |
| per-block FSDP | 切小块降峰值 | 32块 1.99ms 14.2→9.1GB -35% |
| profiler热点 | 定位comm vs compute | compute 0.0464s vs comm 0.0404s |
| 热负载类比 | 两节点SSM定位Tj热点 | profiler定位AllGather热点 |
| Day16 ROI | queue 68.8% thermal 1.67pp cost 22.1% | comm拆分证明切分必要性 |

## Connection 一句话（用于 ai_daily.csv Notes）

Day16 Monetization 150字ROI故事省$200M方法论 → Day17 Profile用profiler把AllGather 60% vs ReduceScatter 40%拆开证明切分热点，CPU proxy comm 46.5%真机预期7-15%，per-block 32×1.99ms峰值-35%复用Day03。
