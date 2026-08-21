# Day 6 NOTES - 2026-08-07 08:20 PDT

Status: done (离线笔记，H100前定版)

## infra note 5行版 (待H100验证)

- Eval: 基线 coding data 0.42 → 加小规模 GRPO 后 0.47 (+5%)，待真机小 eval 验证 (coding data, 200 samples)
- Throughput: 目标 tokens/sec 1.2k (vLLM rollout 集群, 7B bf16, 2×A100)，待H100 NCCL 验证 `torch.cuda.max_memory_allocated`
- Cost: GPU-hour $3.2 (训练 $ + vLLM $) / $/有用 rollout $0.018，失败重试占 12% 成本，类比 PUE Modeling
- Failure: rollout 失败率 12% (长 CoT 500→5000 tokens 超时/工具挂)，已加组内重试 3 次 + 冷却 10min
- Bottleneck: rollout 占 80% 墙钟，eval 占 15%，训练只占 5% (Agentic RL 长轨迹特性)，长 rollout 需拆 vLLM 集群独立调度

> FSDP per-block 7B 能进2×A100可跑小 eval，已 CPU gloo 验证逻辑，待 H100 NCCL + `max_memory_allocated`对比 DDP vs FSDP (常驻 4P/G vs 峰值 (P-b)/G+b)

## 追问准备
- TRPO → PPO → RLVR/GRPO 演进闭卷回想，留到 Day7 再默写，先保证 5 条桥顺。
- GPU 计时 / power smoothing 认知补齐，放 5 条后。

Code: 笔记为主，无需跑通。
