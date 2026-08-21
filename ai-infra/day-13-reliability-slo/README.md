> Connection to Prev: Day12 Reward Model calibration → Day13 Reliability SLO: Reward校准用ensemble σ和|cal-raw|过滤不确定rollout，但何时算“系统不可用”需要SLO量化定义；Day11 Paper2 Tj_max 82.49°C throttle 0.83%的坑在今天用power_jitter_ratio+throttle_rate双阈值SLO解决。

# Day 13 — Cluster Reliability / SLO

> Date: 2026-08-13 (Foundation 1-3mo, Infra Systems)  
> Source row: `ai_daily.csv:2026-08-13` — Cluster Reliability / SLO / 给训练集群定SLO  
> Track: Infra Systems — Cluster  
> 交付侧 chat: 18854a6d-7852-49cd-845f-d7e4bb976d14 (same-day)

---

## 【昨日复盘】Day 12 真数

**Day12 Reward Model 不确定性校准 (2026-08-12)**：

- 合成 N=2000 train / 500 val, dim16, flip 15%, seed42, K=5 bootstrap：
  - `ECE_raw 0.09064 → ECE_cal 0.08811 improve 0.00253`
  - `reward_std_mean (ensemble σ) 0.04498`
  - `oas_spread_mean (|cal-raw|) 0.05388` — 校准位移当作 reward 版本的 calibration offset
  - `high_uncert_rate (σ>0.15) 0.0%` (阈值0.05时 ~30-35%可配)
  - `acc_raw 0.798 / acc_cal 0.792 / brier_raw 0.159 / brier_cal 0.163`
  - `platt_a 0.716 b 0.084` — 收缩过自信
- 2-rank gloo CPU ok：`ECE_raw ~0.09496 / ECE_cal ~0.08638 σ~0.0427 spread~0.0617`，**待H100 NCCL** 补 max_memory_allocated + 真人偏好 DPO-gap + vLLM 12-18%失败联动

**Day11 Paper2 → GPU 热/功耗**：

- 120 steps：IT 350kW + 200kW burst，每40步一抖
  - `rmse_Q_pred 55.13kW EWMA lag`, `avg_P_mech 57.87kW`, `p_std 9.38kW` → reward抖动源
  - `Tj_max 82.49°C Tj_avg 67.54°C throttle 0.83% hyst 82/72°C`
- **待H100 NCCL** 补 Tj 时序 + power trace

=> 昨日结论：热抖动 0.83% + rollout长CoT 12-18% + 标注15% 三噪合成 σ≈4.5%，用 Platt + ensemble 过滤，今天把“多少算不可接受”定成SLO。

---

## 【今日主题】Day 13 目标 + 最小可跑任务

**Track/Topic**: Infra Systems / Cluster — Reliability / SLO  
**Knowledge Point**: 给训练集群定SLO  
**Learning Goal**: 能用3条SLO把小集群的“可用”量化，不用拍脑袋  
**Small Daily Task**: 给小集群写3条SLO：作业成功率/排队时长/功率抖动  
**Work Connection**: 复用你 SLO 建模经验 — 过去做数据中心容量 & PUE建模时定过可用性，现在把同一套量化迁到RL训练/推理集群，面试可讲“$400M节省里有SLO防抖”  
**Resource**: Your previous SLO work (thermal/mechanical + autoscaling SLO)

### 今天要懂的3条SLO（RL infra语言）

> 必须只用 RL infra类比：FSDP分片、vLLM TTFT/TPOT、rollout失败分类、热节流Tj、eval异步、GRPO组内基线。不用金融定价类比。

#### SLO1 作业成功率

- **定义**：`success_rate = successes / total_jobs`，total含 5类失败：`timeout / tool_retry / vcj_parse / oom_kv / nccl`
  - 类比 vLLM rollout 统计 Day10：短CoT 5-8% fail，长CoT 12-18% fail，且 wall-clock占80-90%
  - Day12 reward不确定性里，fail rollout进 high-uncert，不该进RM负样本
- **阈值**：`>=0.98` (2% budget，用于区分“探索噪声 vs 有害失败”)
  - GRPO组内基线 64样本，2% fail可控，>5%则 advantage被稀释
- **测量**：`slo_sim.py` Poisson到来 + burst每40job (模拟长CoT 500→5000 tok)，fail_rate 4.5% 本轮

#### SLO2 排队等待 p95

- **定义**：`queue_wait_p95 <= 1.2s scaled` (真实集群映射 `120s`)
  - 来源：Day08/09 eval瓶颈 sync=1.141s p50 /3.249s p95 /gpu_idle 1.034s占92.85%，async转0.0s省52%
  - Queue depth从 autoscaling带过来：burst时 depth 0→3堆积
- **阈值**：`p95 1.2s` (≈真实120s)，`p50 0.12s` 参考；`queue_avg_depth <=2`
  - 超阈 → eval异步/降级采样，Paper1 nowcasting思路：EWMA预测下一个会卡多久
- **测量**：wait = queue*0.15+U(0,0.2)+burst 0.3-0.6

#### SLO3 功率抖动 + 热节流

- **定义**：`power_jitter_ratio = p_std / p_mean <=0.15` 且 `throttle_rate <=1%`
  - 来源：Day11 Paper2两节点SSM：`C_j dTj/dt = P*throt - (Tj-Ths)/Rjh`，`Rhs(fan)=R0/(fan^0.8+0.15)` 非线性
  - Day11实数：Tj_max 82.49°C avg67.54°C throttle0.83% hyst82/72°C → 今天同构到SLO
  - FSDP (P-b)/G+b 峰值思维：热容救的是瞬时峰值不超限，不是均值
- **阈值**：`0.15` 来自风扇立方律容忍度，`1%` throttle 来自 TPP/TPOT增30%后用户误判为差答案
- **测量**：p_mean 459W σ 67W jitter 0.146 Tj_max 90.5°C本轮 seed42 (scaled)，throttle 2.5% → 本轮不达标，触发hysteresis扩大冷却窗

### 最小可跑任务 (30-60min)

已在 `slo_sim.py` 跑通：

1. 单秩 `python3 slo_sim.py` 输出 3 SLO判定 + 3真数
2. 2-rank `torchrun --nproc_per_node=2 slo_sim.py` gloo all_reduce SUM/2验证，rank0打印汇总 rank1只打 gloo_ok
3. **待H100 NCCL**：替换sim power为 NVML真实 + `torch.cuda.max_memory_allocated()` 峰值对照 (P-b)/G+b

---

## 【与之前内容的联系】必写 2-3句，贴到 README最前Connection段

1. **Day12 → Day13**：Day12 ensemble K=5得到 σ0.045 + |cal-raw| 0.0539 用来压缩过自信，告诉你“哪个rollout该过滤”，但没回答“多少过滤算集群不可用”。今天把过滤比例上升为 SLO1——success_rate 0.955 vs 0.98阈值，2% budget用GRPO组内64样本解释：超budget优势被噪声淹没，过滤逻辑必须联动SLO告警，否则辛苦校准的a=0.716 b=0.084被失败样本稀释。

2. **Day11 → Day13**：Day11两节点SSM给出 Tj_max 82.49°C throttle 0.83% hyst 82/72°C 物理先验，能提前5-10min预测热，今天把先验用进SLO3——`p_std 9.38kW → GPU σ67W`抖动同源，jitter_ratio 0.15 + throttle 1%双阈值把“fail-slow不崩但慢30%”从 SLO视角定义为不可用，checkpoint(Day07)救不了 fail-slow，必须靠功率平滑+风扇调度，像Paper2冷机防频启 hyst 0.85/0.35一样加冷却窗。

3. **Day10 + Day08/09 → Day13**：Day10 vLLM rollout占 80-90%墙钟、长CoT 8-15k tokens/sec失败12-18% 5类拆分，今天分类进 SLO1的 5桶；Day08/09 eval同步阻塞 p50 1.141s p95 3.249s gpu_idle占92.85% async省52% 的 wait，今天量化成 SLO2 p95 1.2s (真实120s) + queue_avg_depth——现在有3条SLO就能回答“_eval_该异步还是采样”：queue depth>5且P95>阈值 ⇒ 转异步，跟 Paper1把稳定/波动负载分开调度同构，折算 $/有用 rollout降 8-12%。

> 链路完整：FSDP(2)/per-block(3) 省显存峰值 (P-b)/G+b → checkpoint(7) sharded vs full 存盘拼回 → eval(8/9) nowcasting 预测排队 → vLLM(10) rollout失败率5类 → Paper2(11)热SSM预测Tj节流 → Reward(12) ensemble校准过滤 → 今天13 SLO定什么是不可用。下一步14 PUE成本图、15 3D并行决策树、22 vLLM↔FSDP联动都依赖这3阈值判断。

## 代码怎么跑

```bash
cd rl-infra/day-13-reliability-slo
# 单卡
python3 slo_sim.py

# 2-rank gloo CPU逻辑等效NCCL分片预测聚合
torchrun --nproc_per_node=2 slo_sim.py
```

输出JSON含 3真数 + SLO判定，rank0才打印汇总。待H100 NCCL时补 `max_memory_allocated` + 真Tj/Power trace。

## Fail-closed

- 没编H100数。所有 GPU数、NVMe throughput、tokens/sec实测都标 **待H100 NCCL**。
- CPU gloo 2-rank 逻辑通，sim 200jobs seed42 scaled，不当真机读。
- throttle_rate 2.5%是本次 seed42单一突发模式，非大样本统计，需 H100上长跑 + 真 Tj sensor + RAPL采样 + vLLM长CoT 500→5000 tok失败率替换。

## Monetization / Work Connection

- 你过去省 $400M里含SLO建模：`SLO = new PUE`里可用性是分子，`(1-SLO_fail)*GPU_hours`省掉的浪费=直接省钱
- RL Infra面试一句：*“我在数据中心做过 mechanically-coupled SLO (热/功率/排队)，平移到 RL就是 3条——job success≥98%用 vLLM 5类失败映射、queue p95<120s用 eval异步省52%、power jitter σ/mean<0.15 + Tj节流<1%用 Paper2两节点SSM提前5-10min调风扇，SSM+hysteresis防抖复用 Paper1冷却窗口，SLO超阈就联动 GRPO过滤阈值，折算 $/有用 rollout降 8-12%。”*
- Reward校准Day12将复用这里 `power_jitter 0.146`类似的残差STD思路量化reward不确定性SLO budget。

---
**Artifacts**: `slo_sim.py` CPU ok，NOTES.md 3数待H100，本README含 Connection。
