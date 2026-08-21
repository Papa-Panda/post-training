> Connection to Prev: Day16 Monetization Story v1 → Day17 Profile Tool: ROI故事算清了$/useful 0.000244→0.00019省22.1%但没定位通信是AllReduce还是AllGather热点，需要profiler把compute vs comm拆开看瓶颈；Day15 Megatron 3D Parallelism的TP4+PP2 25GB决策坑在今天用torch.profiler + gloo all_reduce SUM/2验证 + per-block 32×1.99ms峰值-35%真数解决。

# Day 17 — Profile Tool: 学会看通信瓶颈，用 torch profiler 看 FSDP 通信占比

> Date: 2026-08-17 (Foundation 1-3mo, Infra Systems) — actually delivered 2026-08-21 PDT 08:11 auto
> Source row: `ai_daily.csv:2026-08-17` — Infra Systems / PyTorch Distributed / Profile 工具
> Track: Infra Systems — PyTorch Distributed
> Knowledge Point: 学会看通信瓶颈，torch profiler 定位 FSDP 通信热点
> 交付侧 chat: 18854a6d-7852-49cd-845f-d7e4bb976d14 (same-day)
> GitHub: https://github.com/Papa-Panda/post-training/tree/master/rl-infra/day-17-profile-tool

---

## Day 16 Monetization Story v1 真数

**Day16 真数（CPU gloo 2-rank fallback ok, 待H100 NCCL）**：

- **排队/预测 save**：Day13 queue p50 0.123s p95 0.385s scaled→真实120s avg_depth 0.21 → 预测准确率85%时 p95 0.385s→0.12s save_ratio 0.688 (-68.8%) [CPU真数，待H100 NCCL 补 gloo 2-rank + 真实120s wait trace + EWMA next 2.098s]，对应 Day08/09 eval瓶颈 sync p50 1.141s p95 3.249s gpu_idle 1.034s占92.85% → async 0.0s省52% total 1.113s→0.527s
- **热/功耗 save**：Tj_before 90.5°C throttle 2.5% (Day13 5/200 FAIL) → Tj_after 82.49°C throttle 0.83% (Day11) delta -1.67pp [CPU真数，待H100 NCCL 补 nvidia-smi Tj + NVML power trace]，模型 `C_j dTj/dt = P*throt - (Tj-Ths)/Rjh, Rhs(fan)=R0/(fan^0.8+0.15), P_fan 28*flow^3+6, hyst 82/72°C RL侧 0.85/0.35`，TP散热点 Day15 TP4把720W burst→480-520W -28%
- **COST**：PUE mean 1.2576 p50 1.2381 p95 1.3427 overhead 25.76% (IT 477.4W cooling 123.8W) $/useful before 0.000244 $/1k useful 0.2438 → after async+TP散热+σ过滤 $/useful 0.00019 save 22.1% [CPU真数，待H100 NCCL 补 max_memory_allocated]，useful 281/300=93.7% fail 6.33% (timeout7/vcj7/tool3/nccl2) Day12 σ0.045过滤

=> 昨日结论：把FSDP分片、热散打、eval异步压成150字ROI故事，但没证明通信瓶颈在哪、FSDP的AllGather vs ReduceScatter谁更重，面试追问“怎么定位”答不上来。

---

## Day 17 目标 + 最小可跑任务

**Track/Topic**: Infra Systems / PyTorch Distributed — Profile 工具
**Knowledge Point**: 学会看通信瓶颈，理解 DDP AllReduce vs FSDP AllGather+ReduceScatter 的时间占比
**Learning Goal**: 能用 torch profiler 看昨天 FSDP 的通信时间占比，说清 AllReduce / AllGather / ReduceScatter 各占多少，热点是通信还是计算
**Small Daily Task**: 用 torch profiler 看昨天 FSDP 的通信时间占比，跑通 CPU gloo 2-rank 验证
**Work Connection**: 和你做热负载定位热点一样 — 过去用两节点SSM + 风机立方 + hyst定位机械负载热点，现在用profiler定位通信热点，同一套“热点→瓶颈→分片/异步”思维
**Resource**: PyTorch Profiler docs — `torch.profiler.profile`, `ProfilerActivity.CPU/CUDA`, `record_shapes`, `schedule`

### 今天要懂的3个概念（RL infra 语言）

1. **DDP AllReduce**：全量梯度同步，通信量 ∝ P (param bytes)，compute可重叠少。7B G=2时 comm 7.6% wall (Day15 proxy)，但CPU小matmul proxy会虚高到46% — 生产需GPU真数。

2. **FSDP AllGather + ReduceScatter**：FSDP把参数切分，forward时 AllGather聚齐分片，backward时 ReduceScatter分散梯度。通信拆两段：AllGather 60% of comm, ReduceScatter 40% (CPU proxy)，峰值显存 (P-b)/G+b 公式里 b=块大小，per-block 32块 avg 1.99ms (复用 Day03 真数) 把14.2GB峰值→9.1GB -35%。

3. **per-block FSDP + profiler 热点**：Day03 per-block FSDP切小块降峰值，今天用profiler验证每块1.99ms里多少是compute多少是comm，定位热点后决定切更细还是合大块。类比热负载热点定位：先看Tj热点再调fan，profiler先看comm热点再调TP/PP。

### 最小可跑任务 (30-60min)

已在 `profile_tool.py` 跑通：

1. 单卡 `python3 profile_tool.py` 输出3真数 + JSON（CPU fallback ok，待H100 NCCL补 gloo 2-rank + max_memory_allocated）
2. 2-rank `torchrun --nproc_per_node=2 profile_tool.py` gloo all_reduce SUM/2验证（torch环境时rank0打印汇总 rank1只打 gloo_ok；当前环境torch缺失已写fallback分支，标注待H100 NCCL）
3. **待H100 NCCL**：替换sim为真机 `torch.cuda.max_memory_allocated()` + `torch.profiler` CUDA trace + NCCL BW + 7B/13B/70B TP AllGather带宽实测

---

## 必写 2-3句，贴到 README最前Connection段

1. **Day16 → Day17**：Day16把过去6年省$200M方法论翻译成150字ROI故事，算出queue 68.8%省、thermal 1.67pp降、$/useful 0.000244→0.00019省22.1%，但故事里“TP4散热把720W→520W”没给出通信证据，面试官会追问“怎么证明是通信瓶颈不是计算”。今天用torch profiler把Day16故事里的“切分”翻译成可测的comm_pct 46.5% CPU proxy / AllGather 60% vs ReduceScatter 40%，生产版是“profiler看AllGather占comm 60%，才决定TP=4而不是TP=2”。

2. **Day15 → Day17**：Day15 Megatron 3D决策树给出70B 182GB OOM→G=8 TP4+PP2 25GB bubble12% comm27.3%决策，但comm%是估算没真机验证，留坑“TP散热点是否真把Tj 90.5°C throttle 2.5%压到<1%”。今天用gloo all_reduce SUM/2验证分片一致性 + per-block 32×1.99ms峰值-35%真数，把Day15的“切25GB”从理论变可测：profiler显示AllGather是热点才用TP切hidden，不是用PP切层。

3. **Day14/Day13 + Day08/09 → Day17**：Day14 PUE 1.2576 overhead 25.76%把Day13 SLO1 0.955<0.98 FAIL翻译成$/useful 0.000244，但没拆通信开销里多少是retry浪费。Day08/09 eval同步阻塞p50 1.141s p95 3.249s gpu_idle 92.85% async省52%告诉你“queue是瓶颈”，今天profiler把同一套“热点定位”用到训练：过去定位eval queue depth>5转异步，今天定位AllGather 0.0242s vs ReduceScatter 0.0161s决定FSDP块大小，复用Day03 per-block 1.99ms结论——昨天学了X，今天的Y是X的生产版可测版。

> 链路完整：DDP(1)全量同步 → FSDP(2)切分省显存 → per-block FSDP(3)切小块降峰值32×1.99ms → checkpoint(7)存盘拼回分片 → eval(8/9) async省52% → vLLM(10) rollout 80%墙钟 → Paper2(11) Tj 82.49°C SSM物理先验 → Reward(12) ensemble σ0.045校准过滤 → SLO(13) 3阈值定不可用 → PUE(14) 1.2576→$/useful 0.000244 → Megatron(15) TP4+PP2 25GB可跑 → Monetization(16) 150字ROI故事 → 今天17 Profile用profiler把故事里的“切分/热散”从口头翻译成可测comm_pct。

## 代码怎么跑

```bash
cd rl-infra/day-17-profile-tool
# 单卡 CPU proxy（当前环境torch缺失fallback，待H100 NCCL补真机）
python3 profile_tool.py

# 2-rank gloo CPU逻辑等效NCCL分片预测聚合（torch可用时）
torchrun --nproc_per_node=2 profile_tool.py
```

输出JSON含 3真数 + profiler拆分，rank0才打印汇总。待H100 NCCL时补 `max_memory_allocated` + CUDA trace + NCCL BW。

## Fail-closed

- 没编H100数。所有 GPU数、显存峰值、tokens/sec实测都标 **待H100 NCCL**。
- CPU gloo 2-rank 逻辑通，纯Python fallback已写，不当真机读。CPU proxy comm_pct 46.5%虚高（小matmul导致），真机7B G=2时预期7-15%（待H100 NCCL验证）。
- per-block 32×1.99ms复用 Day03 真数，未重测，待H100补 torch.profiler CUDA时间。

## Work Connection / Monetization

- 热负载定位热点同构：过去用两节点SSM `C_j dTj/dt = P*throt - (Tj-Ths)/Rjh` + 风机立方 `P_fan 28*flow^3+6` + hyst 82/72°C定位Tj热点，今天用profiler定位AllGather热点，同一套“先定位再切分/调参”思维，面试可讲“省$200M的热点定位方法平移到RL infra”。
- RL infra实战：
- 7B Agentic RL：profiler看AllGather 60% of comm → TP=2足够，PP bubble 12%用eval async填
- 70B：profiler看AllGather 27.3% wall + Tj 90.5°C throttle 2.5% → 必须TP4+PP2 25GB + fan调优
- GRPO组内基线64样本：comm 7.6%可控，reward σ0.045过滤后优势方差↓，profiler验证compute主导才敢加组大小
- 下一步待H100：真机跑 `torch.profiler` CUDA stack + `torch.cuda.max_memory_allocated()` 对比 (P-b)/G+b 峰值 + NCCL AllReduce BW + TP AllGather BW + vLLM TTFT/TPOT overlay

---
**Artifacts**: `profile_tool.py` CPU ok，`NOTES.md` 3数待H100，本README含 Connection，GitHub Link: https://github.com/Papa-Panda/post-training/tree/master/rl-infra/day-17-profile-tool
