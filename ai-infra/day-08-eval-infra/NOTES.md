# 2026-08-09 Day 08 NOTES — Eval infra 为什么是瓶颈

Date: 2026-08-09 19:26 PDT (America/Los_Angeles)  
Status: done (CPU gloo 验证逻辑，待 H100 NCCL 真机验证 eval P95 / QPS / GPU 空转美元)  
Lab: `rl-infra/day-08-eval-infra/`  

## 核心问题

为什么 eval 会卡住训练？不是 eval 慢 0.5s，而是：

- 同步 eval = 训练 GPU 100% 被 block
- coding eval 重：sandbox 冷启动 5-10s + 编译 + 单元测试 + verifier
- flaky 重试把 P95 拉长 2-3x，queue depth 堆起来

=> rollout 占 80% wall-clock，eval 占 15% 但感知是 100% 阻塞，GPU 利用率 70% → 30%。

## 3 个 CPU 数 (gloo 2-rank, 待 H100 NCCL)

> CPU gloo 缩放版（真实 eval 5-10x），逻辑通，数字用来看比值，不当真机读。

### Sync 模式 (baseline, 2-rank gloo, train 10 steps, eval every 3)

- **eval_latency_p50**: **1.141s** (scaled, 映射真实 5-10s sandbox)
- **queue_depth_avg**: **1.00** / max 1 (单 eval worker 串行，无堆积是假象，真 burst 会到 5-7)
- **gpu_idle_time_accum**: **1.034s** / total_wall 1.113s → **bottleneck_ratio 92.85%**
- eval_count 3, flaky 1/3 → flaky_rate 33.3%, p95 3.249s (flaky 那次)
- 单 rank 对照：同 p50 1.141s, p95 3.249s, gpu_idle 1.034s, bottleneck 66.54%, total 1.554s

### Async 模式 (nowcasting 转异步后, 2-rank gloo)

- **eval_latency_p50**: 0.000s @rank0 (eval 在 rank1, rank0 不等)
- **gpu_idle_time_accum**: **0.000s** (rank0 idle 归零)
- total_wall 0.527s vs sync 1.113s → **省 52% wall-clock**
- rank1 侧真实 eval：0.63s / 0.72s / 1.14s, flaky_rate 0% 本轮

### 单 rank async (验证计算)

- p50 1.141s, p95 3.249s, gpu_idle 0.000s, bottleneck_ratio 192.93% (eval_time/total 单线程算术比 >100%，说明 sync 修 2x 开销)

### 待 H100 NCCL 真机补

- `torch.cuda.max_memory_allocated()` / `max_memory_reserved()` 对比 sync vs async 下常驻
- 真 coding eval：HumanEval 164 题全量 3h 的 P50 / P95 / queue depth 5-7 时的 gpu_hours_wasted
- ops：`P95 eval latency 8-15min 真值` / `flaky_rate 真值` / `GPU-hours wasted = (1-util)*wall-clock`
- $/有用 rollout：`$/useful = (train$ + vLLM$ + eval$ + retry$) / passed` 降幅 12-15% 需实测

## nowcasting → eval 预测 小实操

```python
alpha=0.3 EWMA over recent_latencies
pred_next = EWMA(recent)
if queue_depth>5 or pred_p95>10min: async/skip → 转后台，训练继续
else: sync eval 可接受
指标：queue_depth / P50 / P95 / flaky_rate / gpu_idle_seconds
```

已在 `eval_bottleneck_sim.py` 里跑通：step 3 后 pred 0.00s → step 6 pred 3.25s → step 9 pred 2.51s → EWMA next 2.098s。

## 你组里 coding eval 3 延迟点 + 解法

1. **sandbox 冷启动 5-10s** → 热池 + 预热容器 + 重用，同 image 复用 / 执行器池化
2. **HumanEval 全量 3h block** → 采样 20% fast-pass + 分级：fast (20%题) → full (100%题) 两级，fast fail fast
3. **verifier 非确定/超时** → 重试队列 + 超时 30s 熔断 + flaky 标记跳过 + 3 次重试带冷却 10min (复用 Day06 hysteresis)

## Code 怎么跑

```bash
# sync (模拟当前痛苦)
torchrun --nproc_per_node=2 eval_bottleneck_sim.py

# async (nowcasting 后优化)
torchrun --nproc_per_node=2 eval_bottleneck_sim.py --async-mode

# 单卡 fallback
python3 eval_bottleneck_sim.py
python3 eval_bottleneck_sim.py --async-mode
```

Output 每行都打印 step 级 queue_depth / pred / latency / idle，尾部汇总 3 numbers + 待H100 提示。

## 一句话可迁移

eval 瓶颈本质是同步 + 重 + 排队，把 autoscaling 里“稳定负载 vs 波动负载分开调度 + nowcasting 预测 burst” 翻译成“train vs eval 分开 + EWMA 预测 eval 延迟决定 async/skip”，量化看 P95 和 queue_depth，省的是 GPU-hours wasted，算进新的 PUE = $/有用 rollout。

## Fail-closed

- 没编 H100 数，所有 GPU 数字明确标 “待H100 NCCL”
- flaky_rate 33.3% 是本次 seed 42 的 CPU 模拟随机，不是真机实测，需大样本统计
- tokens/sec 1.2k 占位符不沿用，本 day 不提 tokens/sec，专注 eval latency / queue / idle
