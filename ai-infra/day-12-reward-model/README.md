> Connection to Prev: Day11 Paper2 mechanical → GPU thermal Tj_max 82.49°C throttle 0.83% is隐性失败源 → Day12 Reward 噪声是热的下一步：Tj高导致TPOT慢被误判为差答案，用 ensemble σ + |cal-raw| 补偿过滤；Day10 vLLM 长CoT 12-18% 5类失败进入不确定性集合，不进RM负样本；Day08/09 eval nowcasting 二阶γ*(ΔT)^2 同构到 reward 二次Brier/ECE。

# Day 12 — Reward Model 不确定性 + 校准

> Date: 2026-08-12 (Foundation 1-3mo, RL Training)
> Source: ai_daily.csv:2026-08-12 — Reward Model / 理解 reward 模型的不确定性
> Track: RL Training

---

## 【昨日复盘】Day 11 真数

**Day11 Paper2 → GPU 热/功耗**：
- 120 steps sim，IT 350kW + burst 200kW，每40步一抖：
  - `rmse_Q_pred` 55.13 kW（EWMA lag，二阶可压）
  - `avg_P_mech` 57.87 kW
  - `p_std` 9.38 kW → 同 reward 抖动源
  - `Tj_max` 82.49°C, `Tj_avg` 67.54°C, `throttle` 0.83%
- 代码 `paper2_mech_to_gpu_thermal.py` CPU gloo ok **待H100 NCCL**

**Day10 vLLM rollout**：7B G=2 峰值 ~42.5GB，短CoT 40-60k tps 失败5-8%，长CoT 8-15k tps 失败12-18%（超时40%/工具30%/VCJ15%/OOM10%/NCCL5%）占墙钟80%→90%

=> 昨日结论：rollout 占墙钟 80%→90%，热节流是隐性失败源，今天把热/失败映射为 reward 不确定性过滤。

---

## 【今日主题】Reward 模型为什么要校准

**Knowledge**: 3个噪声源
- 标注抖：人类偏好 15% 翻转（本code 0.15 flip）
- rollout抖：长CoT 12-18% 超时/工具失败回传 NaN
- 热抖：Tj>82°C 降频 TPOT +30% → 人误判差答案

=> `r_obs = r_true + ε_annot + ε_rollout + ε_thermal`

**Small Daily Task**: 训练一个简单分类 RM，用 ensemble + Platt 压校准

**Work Connection**: rollout/标注/热三噪 → reward 校准与过滤，省 $/有用 rollout 8-12%

### 校准两条线

1. **校准线**：Platt `logit_cal = a·logit + b`，本轮拟合 a=0.716 b=0.084，ECE 0.0906→0.0881
2. **不确定度线**：ensemble K=5 bootstrap → σ_mean 0.0450，|cal-raw| 0.0539。当 σ>0.15 为高不确定，本轮 0%，阈值0.05时约30%可过滤

本轮过滤逻辑：高 σ rollout 不进 RM 负样本，更新 GRPO 时降权，形成 rollout → σ → 惩罚闭环。

---

## 【与之前内容的联系】

1. **Day11 → Day12**：Day11 告诉你为啥会热节流，今天就是热的后果——Tj 82°C 不崩训练但让 reward 延迟被误判，必须加 offset 区分正常探索 vs 有害抖动，否则 GRPO 组内优势被噪声淹没。

2. **Day10 → Day12**：Day10 只拆了失败率，没说怎么进 reward。今天补：失败rollout算 σ，不管对错。高 σ 直接滤，阈值由 Day10 失败率反推，rollout 80% 墙钟里省 8-12% 有用样本。

3. **Day08/09 → Day12**：Eval nowcasting 用 EWMA 预测排队，Day11 二阶修正热 lag。今天同理，ECE 是一阶，Brier + ensemble 二阶，a=0.716 收缩过自信。FSDP (P-b)/G+b 省峰值，热容省节流，σ 过滤省无用 rollout，同一套。

---

## 代码怎么跑

```bash
cd rl-infra/day-12-reward-model
python3 reward_oas_calibration.py
torchrun --nproc_per_node=2 reward_oas_calibration.py
```

输出：ece_raw/cal, reward_std_mean, oas_spread_mean (|cal-raw|), high_uncert_rate, acc, brier, platt_a/b

2-rank gloo CPU 通，**待H100 NCCL** 补 max_memory_allocated + 真RM数据。

## Fail-closed

- 没编 H100 数，全部标待H100 NCCL验证
- 小合成 N=2000/500 dim16 flip15% seed42，ECE improve 0.0025 是真数，不夸大
- RM应在线校准：GRPO每500 steps 重拟合 a/b，EWMA 跟踪 σ 漂移，hysteresis 82/72°C 防震荡

**Artifacts**: `reward_oas_calibration.py` CPU ok，NOTES.md 3数待H100，本 README 含 Connection。
