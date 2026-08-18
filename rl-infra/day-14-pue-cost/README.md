> Connection to Prev: Day13 Reliability / SLO → Day14 Paper3 PUE拆解: SLO 定义了什么算失败但没算出失败浪费多少钱，需要 PUE→$/有用 rollout 把 GPU 空转/重试/制冷开销翻译成 dollars；Day12 Reward Model 的 reward σ 过滤坑在今天用 useful 分母扣除 + retry 成本单计解决，避免把高不确定 rollout 当成 infra 失败。

# Day 14 — Paper3 PUE拆解: 能效指标 → 训练成本 per token / per useful rollout

## 【昨日复盘】Day 13 Reliability / SLO 真数
- 真数（CPU gloo 2-rank ok, seed42, n=200）：success_rate 0.955 fail 4.5% 5类 timeout5 / vcj_parse3 / tool_retry1 / oom1 / nccl1，queue p50 0.123s p95 0.385s (scaled 1.2s≈真实120s) avg_depth 0.21，power_mean 459.2W σ67.0W jitter0.146 PASS，Tj_max 90.5°C avg73.4°C throttle2.5% 5/200 FAIL → slo_all FAIL。
- SLO 3 条：SLO1 job成功率≥0.98，SLO2 排队p95≤1.2s scaled (真实120s)，SLO3 功率抖动<0.15 + 节流<1%。
- 昨日链路：Day12 reward σ 0.045 + |cal-raw| 0.0539 是 SLO1 噪声源，Day11 Tj 82.49°C 热节流映射到 SLO3。

## 【今日主题】Day 14 目标 + 最小可跑任务
- **Learning Goal**：画出 PUE vs 训练成本 per token 的类比图，提炼能效指标建模可迁移到 RL infra。
- **Topic**：Paper3 拆解 — 数据中心 PUE 建模如何变成 RL 集群的 $/有用 rollout / $/k tokens。
- **最小可跑**：
  - `python rl-infra/day-14-pue-cost/pue_cost_model.py` 单进程 CPU 跑通 seed42 300 rollouts。
  - `torchrun --nproc_per_node=2 rl-infra/day-14-pue-cost/pue_cost_model.py` gloo 2-rank 验证通信 ok。
  - 输出 3 真数 + 待H100 NCCL 扩展：`torch.cuda.max_memory_allocated()`、`nvidia-smi` Tj、`nvml` power trace、真 vLLM 500→5000 tok 长尾 12-18%失败。

### 真数（CPU，seed42，待H100 NCCL 补深）
- PUE mean 1.2576 p50 1.2381 p95 1.3427 min 1.1646 max 1.3468 overhead 25.76%（IT 477.4W + cooling 123.8W）
- $/useful rollout 0.000244 $ /1k useful 0.2438 $ /1k tokens proxy 0.000116（$3.2/GPU-hr proxy，facility_cost $0.0685）
- useful 281/300=93.7% fail 6.33% (timeout7/vcj7/tool3/nccl2) filtered_high_uncert 0 retry 2.6s idle 9.1s Tj_avg 72.4°C Tj_max 92.1°C throttle 3.0%

### PUE vs COST 类比图（ASCII / Mermaid 思路）
```
DC视角:    P_IT ──┬──► P_cooling(COP, fan^3, hyst 0.85/0.35) ──┬─► P_total
                  └──► P_loss 3%                              └─► PUE = P_total/P_IT

RL视角:    GPU_IT_sec (rollout 80%墙钟 40-60k短/8-15k长)
               ├─ retry_sec (5类失败：timeout/tool/vcj/oom_kv/nccl)
               ├─ eval_idle_sec (同步→异步 nowcasting EWMA)
               └─ reward σ 过滤分母扣除 (Day12)
                     │
                $/hr 3.2 * PUE 1.2576
                     ▼
          $/useful = PUE * $IT / useful
                     ▼
          $/1k useful → $/1k tokens (proxy 2.1k tok/rollout)
                     ▼
               ROIC故事: 每1k有用rollout省 Z GPU-min
```

> Paper3 把二次换热 COP 退化 γ*(ΔT)^2 + 风机立方 + 冷机 hyst 建模成 PUE 曲线；RL 集群把同一套物理思维套到 $/有用 rollout，随负载因子动态看制冷 overhead，何时 PUE 1.16→1.35 吞掉 retry 省出来的钱。

## 【与之前内容的联系】
- Day13 定了 3 条 SLO（成功率/排队/功率抖动），但 SLO 只回答 pass/fail，没回答 waste 多少钱。今天用 PUE mean 1.2576 把 Day13 的 retry_sec 2.6s + eval_idle 9.1s 换算成 facility_cost $0.0685 vs IT_cost $0.0545，overhead 25.76% 是下一轮调度的硬约束：扩容还是降 Tj。
- Day12 的 reward model ensemble σ 0.045 与 |cal-raw| 0.0539 告诉我们一部分 rollout 被 filtered 不是 infra 失败。昨天 SLO 把所有非成功都算 fail，夸大了 fail_rate 6-15% vs 真实；今天把 filtered_high_uncert 单列，分母只计 useful 281，retry 成本单独计，避免误把不确定性当稳定性问题去扩机柜。
- Day11 Paper2 机械负载用 γ*(ΔT)^2 + COP 二阶 + 冷机 hyst 0.85/0.35 解决了 Day08/09 EWMA nowcasting 的滞后坑（RMSE 55.13kW）。今天直接复用该 SSM 算 cooling power：P_fan 28*flow^3+6 + P_chiller = P_IT/COP*1.15/0.35，解释为何 PUE p95 1.3427 出现在 burst 35 个 rollout 一次的长 CoT 时刻。

## Work Connection / Monetization
- 能效 → COST：把机房 PUE 1.x 思维翻译成训练集群每有用 rollout 花多少 GPU-min，压缩1天等待 = 省下一台 H100 小时费的预估，可嵌入 eval 异步调度优先级。
- 下一步：待H100 验证 max_memory_allocated + nvidia-smi Tj 真迹 + RAPL power，对比 CPU 模拟的 power_it_mean 477W / Tj_avg 72.4°C 偏差，补回真实 vLLM 7B tokens/sec 3.4-5k 预期。

## 代码 & 资源
- code: `rl-infra/day-14-pue-cost/pue_cost_model.py` (CPU gloo 2-rank ok)
- Paper3 draft: 数据中心 PUE modeling（SEER/COP 二次项、冷机 hyst、风机立方）
- GitHub Link: https://github.com/Papa-Panda/post-training/tree/master/rl-infra/day-14-pue-cost

待H100 NCCL：max_memory_allocated + nvidia-smi Tj + RAPL power trace + 真vLLM 500→5000 tok rollout长尾 12-18%失败替换模拟 + 70B外推 tokens/sec。
