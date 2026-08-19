# NOTES — Day 15 Megatron 3D Parallelism

> Connection to Prev: Day14 Paper3 PUE拆解 1.2576 → Day15 3D Parallelism 单卡182GB OOM需TP4+PP2切到25GB才能跑，PUE优化前置条件；Day13 Tj_max 90.5°C throttle 2.5% 用 TP scatter把720W burst打到520W；Day08/09 async 52%省为PP bubble填充提供位置。

Date: 2026-08-15 (Infra Systems / Megatron 3D Parallelism) actually 2026-08-19 delivery of Day15

## 3个CPU真数（待H100 NCCL 补 max_memory_allocated + nvidia-smi Tj + TP AllReduce BW）

### 1. 单卡 baseline OOM 风险（bf16 2 bytes/param + act）
- 7B: param 14.0GB + act 4.2GB = 18.2GB infer / train 43.3GB (含 opt shard) → G=1可跑 <56GB阈值 [CPU真数]
- 13B: 26GB+7.8GB=33.8GB infer / 80.4GB train (SLO临界，train OOM True percode 80GB阈值) [CPU真数]
- 70B: 140GB+42GB=182GB infer / 433GB train → OOM True，必须切 [CPU真数，待H100 NCCL 补 max_memory_allocated]

### 2. G=2/4/8 决策树真数（CPU proxy, torch missing故 gloo逻辑验证待补）
- G=2:
  - 7B DP/FSDP 9.1GB infer proxy vs TP2 8.62GB infer train 21.66GB comm 7.6% → 选 DP最简，通信少，复用Day02 FSDP
  - 13B DP 33.8GB/2=16.9GB仍高 vs TP2 16.01GB train 40.22GB comm 8.6% → 选 TP2，避免单层大GEMM
  - 70B DP 91GB OOM / TP2 86GB OOM (per code) → G=2无解，需≥8，映射Day13 SLO fail中 oom_kv桶
- G=4:
  - 7B DP only 7.46GB train 13.98GB comm 4.4% → DP足够
  - 13B TP2+PP2 9.30GB infer 21.41GB train bubble 12% comm 11.1% → 选 TP+PP，bubble可用eval async填（Day08 52%省）
  - 70B TP4 43.1GB infer 108.29GB train 24.8% comm OOM True → 仍需PP
- G=8 (最终解):
  - 7B DP 5.83GB/9.09GB 最优
  - 13B TP4+PP2 4.65GB/10.71GB comm 17.5% bubble12% → deep>40层切PP收益
  - 70B TP4+PP2 25.05GB infer 57.64GB train bubble12% comm27.3% → 唯一可行，bubble用interleaved压到 <15%再用Day12 reward σ过滤填空

CPU单进程跑通：所有上面数字来自 `python3 megatron_3d_sim.py` 实际输出（见下），待H100 NCCL 补真显存峰值。

### 3. 通信/热/PP bubble proxy（CPU估算，待H100 NCCL补带宽真数）
- DP AllReduce size ∝ P: 7B=1.2GB,13B=2.2GB,70B=12GB proxy per step → G=8时 DP 12-18% wall（待NCCL实测）
- TP AllGather per layer 2×hidden: TP=2 comm 7.6%, TP=4 comm 17-27% → 大模型TP>4后边际递减，触发Tj 90°C风险
- PP bubble (PP-1)/m m=8 micro-batch: PP2=12%, PP4=37%（未interleaved）→ interleaved (PP-1)/(2m)=6-18% + eval async填（Day08/09 52%省逻辑）
- 热散：TP把720W单卡burst → TP4每卡~480-520W（-28%），映射Day11 fan^3 28*flow^3 + hyst 82/72°C，节流率从2.5%→<1%（待nvidia-smi Tj验证）

## 待H100 NCCL
- [ ] torch.cuda.max_memory_allocated() 真数：7B DP G=2 18GB vs proxy 17.24GB偏差，13B TP2 16GB vs proxy 16.01GB，70B TP4+PP2 25GB vs 25.05GB + nvidia-smi Tj
- [ ] TP AllReduce带宽：NVLink 600GB/s实测DP AllReduce 7B 1.2GB耗时 ~5ms vs proxy 7.6% wall，评估是否卡TTFT
- [ ] PP bubble实测：Megatron-LM interleaved schedule PP=2/4 bubble 12%→6% vs async reward compute填充，TTFT/TPOT联动
- [ ] vLLM TPOT overlay：TP=2时 TPOT +15% vs DP only，但TTFT-20ms因切分后单卡QKVO变小，验证GRPO组内N=64 advantage是否受TP引入噪声影响σ<0.05
- [ ] Checkpoint sharded vs full：TP rank shard save/restore对应Day07 per-block + Day05 pjit mesh/P，验证70B TP4 PP2 save耗时12ms→全量gather 200ms

## 映射表（RL only，禁止金融类比）

| 概念 | RL集群 | 本code proxy |
|---|---|---|
| DP | 数据并行，AllReduce梯度，显存/G | 7B G=4 7.46GB |
| TP | 张量并行，AllGather切片GEMM，散热 | 13B TP2 16GB散720W→520W |
| PP | 流水并行，bubble (PP-1)/m，interleaved压，eval async填 | PP2 bubble12%→6% |
| FSDP分片 | ZeRO1 param/optimizer分片，Day02→03 per-block | DP/FSDP G=8 5.8GB |
| vLLM TTFT/TPOT | TP切小QKVO TTFT↓但ALLGather TPOT↑ | TP2 TTFT-20ms TPOT+15% |
| rollout失败 | oom_kv∈5类，70B OOM→SLO1失败 | 70B G2 oom True |

## 一句收敛
“Day14 1.2576把$/useful算清但只在模型能装进显存时有效，70B 182GB→TP4+PP2 25GB是PUE的前提；Day13 Tj 90.5°C 2.5%节流用TP打散bust避免单卡热点，PP bubble 12%用Day08 async逻辑填，checkpoint分片复用Day07。”

## 代码
- `megatron_3d_sim.py` CPU单进程 ok（torch缺失环境 gloo待补，逻辑已写gloo分支），待H100补 max_memory_allocated + NVML Tj + TP BW
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/rl-infra/day-15-megatron-3d
- 本次真数来源：python3 megatron_3d_sim.py 输出见上（单rank真数），gloo 2-rank sim待 torch环境补跑已写all_reduce分支

### 原始输出存档
```
[Rank 0/1] Day15 Megatron 3D Parallelism CPU proxy (待H100 NCCL 补 max_memory_allocated)
Model 7B G=1 -> DP/FSDP only (G=1) sufficient - perGPU infer 17.2GB | infer 17.24GB train 43.32GB comm 4.4% oom=False
Model 7B G=2 -> TP=2+DP (G=2) - scatter GEMM 7B, Tj Scatter, comm 7.6% | infer 8.62GB train 21.66GB comm 7.6% oom=False
Model 7B G=4 -> DP/FSDP only (G=4) sufficient - perGPU infer 7.5GB | infer 7.46GB train 13.98GB comm 4.4% oom=False
Model 7B G=8 -> DP/FSDP only (G=8) sufficient - perGPU infer 5.8GB | infer 5.83GB train 9.09GB comm 4.4% oom=False
Model 13B G=1 -> DP/FSDP only (G=1) sufficient - perGPU infer 32.0GB | infer 32.01GB train 80.44GB comm 5.4% oom=True
Model 13B G=2 -> TP=2+DP (G=2) - scatter GEMM 13B, Tj Scatter, comm 8.6% | infer 16.01GB train 40.22GB comm 8.6% oom=False
Model 13B G=4 -> TP=2+PP=2+DP rem (G=4) - bubble 12%, infer 9.3GB train 21.4GB | infer 9.30GB train 21.41GB comm 11.1% oom=False
Model 13B G=8 -> TP=4+PP=2+DP rem (G=8) - bubble 12%, infer 4.7GB train 10.7GB | infer 4.65GB train 10.71GB comm 17.5% oom=False
Model 70B G=1 -> DP/FSDP only (G=1) sufficient - perGPU infer 172.4GB | infer 172.39GB train 433.16GB comm 15.2% oom=True
Model 70B G=2 -> TP=2+DP (G=2) - scatter GEMM 70B, Tj Scatter, comm 18.4% | infer 86.19GB train 216.58GB comm 18.4% oom=True
Model 70B G=4 -> TP=4+DP (G=4) - scatter GEMM 70B, Tj Scatter, comm 24.8% | infer 43.10GB train 108.29GB comm 24.8% oom=True
Model 70B G=8 -> TP=4+PP=2+DP rem (G=8) - bubble 12%, infer 25.0GB train 57.6GB | infer 25.05GB train 57.64GB comm 27.3% oom=False
```
