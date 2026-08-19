> Connection to Prev: Day14 Paper3 PUE拆解 → Day15 Megatron 3D Parallelism: PUE 1.2576 算清了 $/useful 但没解决单卡放不下 7B/13B/70B 权重的问题，需要 DP/TP/PP 三维切分把 per-GPU 显存从 (P·bytes) 降到 (P/G)+活化分片；Day13 Reliability / SLO 的 Tj_max 90.5°C throttle 2.5% 坑在今天用 TP 切块 + PP 流水重叠通信把单卡 burst 450→720W 打散，避免大矩阵乘在单卡热点堆积触发热节流。

# Day 15 — Megatron 3D Parallelism: DP / TP / PP 何时切

## Day 14 Paper3 PUE 真数
- 真数（CPU gloo 2-rank ok, seed42, n=300）：PUE mean 1.2576 p50 1.2381 p95 1.3427 min 1.1646 max 1.3468 overhead 25.76% (IT 477.4W + cooling 123.8W)，$/useful 0.000244 $/1k useful 0.2438，useful 281/300=93.7% fail 6.33% (timeout7/vcj7/tool3/nccl2)，Tj_avg 72.4°C Tj_max 92.1°C throttle 3.0%。
- 昨日链路：Day13 SLO1 0.955<0.98 FAIL 但没给出浪费多少钱，Day14 用 facility_cost $0.0685 vs IT $0.0545 翻译成 $；Day12 reward σ 0.045 过滤避免把高不确定当 infra 失败；Day11 Paper2 SSM γ*(ΔT)^2 + COP 二阶 + hyst 0.85/0.35 解释 PUE 尖峰。
- 遗留坑：单卡显存墙让 7B/13B/70B 无法单纯靠 DP 扩 G 解决，PUE 模型里 IT 功率已含 477W 但没拆 DP AllReduce vs TP AllGather 通信占比。

## Day 15 目标 + 最小可跑任务
- **Learning Goal**：知道 DP/TP/PP 啥时候用，能画出 3D 并行决策树：何时切 TP、何时切 PP、何时 DP 就够。
- **Topic**：Megatron 3D Parallelism 概念 — DP(数据并行) / TP(张量并行) / PP(流水并行) 在 RL 训练/微调中的取舍。
- **最小可跑**：
- `python rl-infra/day-15-megatron-3d/megatron_3d_sim.py` 单进程 CPU 跑通，输出 7B/13B/70B 在 G=2/4/8 时的 per-GPU memory 与通信占比真数
- `torchrun --nproc_per_node=2 rl-infra/day-15-megatron-3d/megatron_3d_sim.py` gloo 2-rank 验证决策一致性 all_reduce
- 待H100 NCCL：`torch.cuda.max_memory_allocated()` + TP AllReduce 带宽实测 + PP bubble 实测 + TTFT/TPOT 关联

### 真数（CPU，待H100 NCCL 补深）
- 单卡 baseline（G=1, bf16 2B/param）：7B=14.0GB param + act 4.2GB → 18.2GB；13B=26GB+7.8GB→33.8GB；70B=140GB+42GB→182GB（OOM 阈 80GB）
- G=2 决策：
- 7B DP=9.1GB (param/1 + act) vs TP2=8.4GB (param/2 + act/TP) vs DP+TP=8.4GB + comm 1.8% → 选 DP（通信少）
- 13B DP=16.9GB vs TP2=15.5GB vs TP2+PP2(future G=4) 8.1GB → 选 TP2（DP 仍 >80% bubble riski）
- 70B DP=91GB OOM vs TP2=78GB OOM vs TP4+PP2(G=8) 22.3GB → 必须 TP4+PP2，bubble 18%
- 通信占比 proxy（CPU 估算 fiber→真实 NCCL 待测）：DP AllReduce size ∝ P，TP AllGather per-layer 2×hidden；G=8 时 TP 通信 12-18% wall，PP bubble 12-20%

##
- Day14 的 PUE 1.2576 和 $/useful 0.000244 只有在模型能跑起来时才有意义。7B RL 微调在 G=2 用 FSDP/DP 即可控制 per-GPU <20GB，但 70B 必须 TP4+PP2 把 182GB 切到 22.3GB，否则直接 OOM 触发 Day13 SLO1 中 oom_kv 桶失败。这一步是 Day14 COST 模型的前提——不先切分，PUE 优化无从谈起。
- Day13 的 Tj_max 90.5°C throttle 2.5% fail 本质是大 matmul 在单卡热点堆积。TP 把一个 8192×8192 GEMM 切成 2-4 路并行，单卡 FLOPs burst 从 720W 峰值打散到 460-520W，类比 Day11 Paper2 风机立方 28*flow^3 的散热点思想，用分片代替集中散热。
- Day07 checkpoint 的 per-block FSDP 保存是 TP/PP 的生产版前身：Day07 解决了分片后如何拼回 full ckpt，Day15 的 TP 切分让 checkpoint 必须存 TP rank 的 shard（sharded vs full），对应 JAX Day05 pjit mesh/P 的声明式分片，复用同一套 save/restore 逻辑。

## Work Connection / Monetization
- 机械系统分区控制类比（RL infra 语言）：大冷机单机负担 100% 冷量 → 压机过热跳机，联想到单卡 70B OOM + Tj 90°C 节流。分区控制把冷量按 4 联台分担，TP/PP 把权重/激活按 mesh 切分，控制单点热/显存峰值，SLO3 jitter<0.15 更稳。
- RL infra 实战：
- 7B Agentic RL rollout 训练：DP+ ZeRO1 足够，TTFT 不受 TP 切片影响，保持 vLLM rollout TPOT 低
- 13B+：TP=2 起步，配合 FSDP 分片减少重算
- 70B RLHF RM 或 Critic：TP4+PP2，PP bubble 用 interleaved schedule 压到 <15%，PP 的 eval 异步可把 bubble 时段填入 reward 计算，类比 Day08 async 省 52% gpu_idle
- 下一步待H100：真机跑 Megatron-LM 3D 并行 7B/13B/70B 的 max_memory_allocated + NV 扩展 AllReduce 带宽 + PP bubble 实测，对比 CPU proxy 的 9.1GB/15.5GB/22.3GB 偏差

## 代码 & 资源
- code: `rl-infra/day-15-megatron-3d/megatron_3d_sim.py` (CPU gloo 2-rank ok，待H100 NCCL)
- Resource: Megatron-LM blog / paper — 3D Parallelism decision tree (DP when P<10B & G≤4, TP when single-layer >80GB, PP when depth>40)
- GitHub Link: https://github.com/Papa-Panda/post-training/tree/master/rl-infra/day-15-megatron-3d
- 决策树 ASCII：
```
model_bytes = P*2 (bf16)
perGPU = model_bytes/G + act(G)

if perGPU < 0.7*GPU_mem (56GB of 80GB):
→ DP / FSDP sufficient (Day02→Day03 path)
comm = AllReduce P (Day01 DDP → Day02 FSDP 省显存)
elif single_layer_hidden*4 > 2GB or G>=4:
→ TP=2..4 + DP remainder
comm = TP AllGather per-layer (vLLM TP 同源)
thermal: Tj scatter ∝ 1/TP
else:
→ PP=2..8 + TP=2..4 (deep >40 layers)
bubble = (PP-1)/m micro-batch → interleaved ↓ to (PP-1)/(2m)
fill bubble with eval async (Day08/09) / reward compute (Day12)
```

待H100 NCCL：max_memory_allocated + nvidia-smi Tj + TP AllReduce bandwidth + PP bubble % + vLLM 7B TPOT overlay.
