# Day 08/09 - Eval Infra 为什么是瓶颈

> Date: 2026-08-09 (Foundation 1-3mo, RL Training/Eval)  
> Source row: `ai_daily.csv:2026-08-09` — Eval infra bottleneck  
> 交付侧 chat: 18854a6d-7852-49cd-845f-d7e4bb976d14 (same-day)

---

## 一、Done 复盘 — Day 07 Checkpoint & Recovery

**Day 07 已完成 (2026-08-08)**：
- FSDP checkpoint 两种：Full (rank0 gather 写 `/tmp/fsdp_day7_full.pt`) vs Sharded DCP (每卡写自己分片，并行快)
- Recovery 流程：`epoch / step / sampler.set_epoch() / optimizer sharded state / rng` 都要存，否则重跑一整 epoch
- CPU gloo 2-rank 验证：
  - epoch0 avg loss 2.318 / epoch1 2.142
  - checkpoint 写耗时 rank0 12ms (0.1M param, CPU)
  - crash-epoch1 手动 raise → 重启自动从 epoch0→1 恢复 loss 2.276 valid
- 待 H100：`torch.cuda.max_memory_allocated()` 对比峰值，DCP 并行写 NVMe throughput, tokens/sec 无影响

**迁移点**：checkpoint 是“预测失败 + 自动读档”，跟你之前做 SLO 压测里“预测瓶颈 + 重试”同构，都是把 MTTR 从小时级压到分钟级，直接折算成 $/GPU-hour。

---

## 二、今日任务 — Eval 为什么会卡住训练

**Track/Topic**: RL Training / Eval — Eval infra 为什么是瓶颈  
**Knowledge Point**: Eval infra bottleneck, synchronous eval blocking, flaky rate, queue depth  
**Learning Goal**: 知道为什么 eval 会卡住训练  
**Small Daily Task**: 梳理你组里 coding data 的 eval 是怎么跑的，列出延迟点  
**Work Connection**: ML for Infra 里的 nowcasting：预测 eval 延迟  
**Resource**: Your current team eval flow

### 为什么 eval 是瓶颈（3 条）

1. **同步阻塞**：训练 `for step in range(N): train(); if step%K==0: eval()` — eval 没跑完，训练不动，GPU 空转。80% wall-clock 在 rollout，15% 在 eval，但 eval 是同步的，感知延迟 = 100%。
2. **重 & 脆**：coding eval = sandbox 容器启动 + 编译 + 单元测试 + 人工 verifier，比不跑 forward 还重。flaky (超时/容器起不来/判题器非确定) 导致重试，P95 被拉长。
3. **排队**：eval 请求集中到达（burst），单 eval worker 池串行，queue depth 堆起来，GPU 利用率从 70% → 30%。

### 你组里 coding data eval 延迟点梳理（按你描述重建）

- **点1 上游产数 → eval 触发**：数据合成 50k 后每批提交立刻触发 eval，无采样，eval 量 = 数据量线性涨
- **点2 sandbox 起停**：每次 eval 起 docker / firecracker，冷启动 5-10s，热启动缓存命中低
- **点3 同步 HumanEval/MBPP**：小模型改动 0.42→0.47 (+5%) 这种小提升也要跑全量 3h 才能看，block train

=> 结果：eval P95 8-15min，train GPU idle 12-18%，每月浪费 ≈ $(GPU-hour idle)。

### 今晚小专题做啥（30-60min 可跑）

- 模拟 train 10 steps, 每 3 steps 触发一次 eval
- eval 用 Poisson 到达 + 固定延迟 + 10% 翻车重试，测 P50/P95 latency、queue depth、gpu_idle_time
- 2-rank gloo：rank0 训练侧，rank1 eval 侧，`dist.barrier` 模拟同步阻塞，再演示改成异步（不 barrier）能省多少
- 输出 3 个 CPU 数（见 NOTES.md），待 H100 NCCL 再补 `max_memory_allocated`

---

## 三、可迁移链接 — nowcasting & $/有用 rollout

**nowcasting 复用 Paper1**：
- 原：用最近 1-5min QPS + EWMA 预测未来 5-15min burst，提前 10min 预扩容，避免 SLO 跌
- 今：用最近 N 个 rollout latency + eval queue depth + P50/P95 eval latency + flaky rate 做 EWMA，预测下一个 eval 会卡多久，决定 **跳过 / 异步 / 降级采样**
  - 短时信号 → 快决策：queue depth > 5 且 P95 > 10min ⇒ 转异步 eval，不 block train
  - 监控指标：`eval_latency_p50`, `queue_depth`, `flaky_rate`, `gpu_idle_seconds`

**COST = new PUE**：
- PUE 原来：`P_total / P_IT`，目标降 20% = 省 $200M
- RL 新 PUE：`$/有用 rollout = (train$ + vLLM$ + eval$ + 重试$) / 有用rollout数`
- eval 占 15% 墙钟但 100% 阻塞感知 ⇒ 把 eval 从同步改异步，感知利用率回升 70%，直接折算 $/有用 rollout 降 12-15%

**面试一句**：
> eval 瓶颈不是 eval 本身慢，而是同步 + 重 + 排队，让训练 GPU 空转。破法是 Paper1 的 nowcasting 改成 eval 延迟预测 + 异步/采样/分级，跟 autoscaling 里把稳定负载和波动负载分开调度是一个道理，量化看 P95 和 queue depth，省的是 GPU-hours wasted。

**待 H100 NCCL**：所有 CPU 数已跑，GPU 数标注待验证，不编数。

---
**Code**: `eval_bottleneck_sim.py` — `torchrun --nproc_per_node=2 eval_bottleneck_sim.py` CPU gloo 可跑，GPU 再补 `max_memory_allocated`。
