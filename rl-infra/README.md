# rl-infra — RL Infra Labs

原顶层的 `day-01` ~ `day-10` 全部收敛进这里，统一做 post-training / agentic RL infra 的 infra 轨道。

## 布局

```
rl-infra/
├── day-01-ddp-basics/           DDP 梯度同步、通信开销
├── day-02-fsdp/                 FSDP intro、显存对比
├── day-03-fsdp-perblock/        per-block FSDP + profiler
├── day-04-rlhf-vs-agentic-rl/   RLHF vs DPO vs GRPO 区分
├── day-05-jax-pjit/             JAX pjit/sharding matmul
├── day-06-paper1-rl-infra/      Paper1 autoscaling → RL bridge
├── day-07-checkpoint-recovery/  FSDP checkpoint & crash recovery
├── day-07-h100-beyond-7b/       H100 7B/13B/70B 外推、vLLM profiler
├── day-08-eval-infra/           Eval infra 瓶颈 — sync vs async / queue / nowcasting
├── day-10-vllm/                 vLLM rollout 基座
├── day-11-paper2-mech-load/     Paper2 机械负载非线性 SSM → GPU 热/功耗映射
├── day-12-reward-model/         Reward Model 不确定性 + OAS 校准 (金融→RL)
├── day-13-reliability-slo/      Cluster Reliability 3 SLOs — success/queue/power  (seed42 0.955/0.385s/0.146)
└── day-14-pue-cost/            Paper3 PUE → COST $/useful rollout (PUE 1.2576 $/1k useful 0.2438)
```

## 为什么要收敛

- 顶层原本 10+ 个 day-* 平铺，找东西慢；现在按轨道分，`rl-infra/` = infra 轨道，`ICL/` = 理论轨道，`ai-data/` = 数据轨道。
- `ai_daily.csv` 仍是事实源，里面 `Day` 对应的 `Topic` 指向这里的同名文件夹。

## 怎么跑

每个 `day-0x/` 里都有 `README.md` + `NOTES.md` + 单文件可跑脚本（CPU gloo 可跑，NCCL/H100 待真机）。

从根跑示例：

```bash
python rl-infra/day-01-ddp-basics/ddp_day1_mnist.py
python rl-infra/day-03-fsdp-perblock/fsdp_day3_profiler.py
```
