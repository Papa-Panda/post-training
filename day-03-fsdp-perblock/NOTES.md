# Day 3 - Done Date: 2026-08-04 22:10 PDT
Status: done (CPU verified, CUDA N/A 待H100) ## FSDP per-block 显存
- DDP 常驻 4P，FSDP 常驻 4P/G，峰值 (P-b)/G + b + (grad+opt)/G
- block 越小峰值越低，通信启动次数炸，per-block 是甜点
- all-gather 按 block 拼用完即丢，省大头是 optimizer 状态 2×参数一直分片
- reset_peak_memory_stats 看的是 block 非整模型 ## Profiler (2-rank gloo CPU)
- all_gather (0) 32 calls 2.635ms avg 14.55% CPU total
- all_gather (1) 32 calls 1.708ms avg 9.43%
- gloo:all_gather 64 calls 1.080ms avg (CPU memcpy)
- 判定: CPU 环境不能把 CPU time 误读成 NCCL 通信，需加 cuda 占位判断：`if torch.cuda.is_available:` 才量 comm，否则标待H100 ## 7B / 2×A100 快速估算
- P=28GB fp32 (bf16 14GB), b≈0.9GB (32 blocks)
- 常驻 56GB fp32 / 28GB bf16 mix
- 峰值约 42.5GB bf16 mix
- 结论: 2×80GB 够 ## infra note (已贴每日问题库格式)
1. 链路：产 X类 coding 数据 → 模型 Y 7B
2. 评测：A→B +5%
3. 成本：tokens/sec, GPU-hour, 失败率
4. 瓶颈：rollout约80%墙钟
5. 动作：FSDP per-block可把7B塞进2×A100可跑eval，省Z小时 Raw log: see profiler table in README, ckpt /tmp/fsdp_day3_ckpt.pt
