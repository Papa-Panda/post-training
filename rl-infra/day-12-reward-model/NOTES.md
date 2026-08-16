# Day 12 NOTES — Reward Model Calibration

> Connection to Prev: Day11 Tj_max 82.49°C throttle 0.83% → Day12 σ + |cal-raw| 过滤；Day10 5类失败率 → 不确定性集合；Day08/09 γ*(ΔT)^2 二阶 → ECE/Brier 二阶。

Date: 2026-08-12 (RL Training / Reward Model uncertainty calibration)

## 3个CPU真数（合成 N=2000/500 dim16 flip15% seed42 K=5）

### Single-rank
- ECE_raw 0.09063980728387833 / ECE_cal 0.08811241388320923 improve 0.002527393400669098
- reward_std_mean (ensemble σ) 0.04498102888464928
- oas_spread_mean (|cal-raw|) 0.053879059851169586 → 平均校准位移 5.39%

Bonus:
- high_uncert_rate (σ>0.15) 0.0% (阈值0.05 → ~30-35%可配)
- acc_raw 0.798 / acc_cal 0.792 / brier_raw 0.15919 / brier_cal 0.16281
- platt_a 0.7161718606948853 b 0.0839608907699585

### 2-rank gloo
- ECE_raw ~0.0949575 / ECE_cal ~0.0863821
- reward_std_mean 0.0427129 / oas_spread_mean 0.06168676 / high_uncert 0.0%

**逻辑验证**：gloo all_reduce SUM/2 ok，CPU 2进程 ok，待H100 NCCL + max_memory_allocated + 真实负载

## 待H100 NCCL
- [ ] 真人偏好：ai-data RM pairs / DPO-reward-gap human-RM gap 分布，ECE@10/15，abstain
- [ ] vLLM 联动：长CoT 500→5000 tok 失败12-18% vs σ 相关性，σ>0.15 过滤省 $/有用 8-12%
- [ ] 显存：RM 7B/13B FSDP max_memory_allocated vs (P-b)/G+b
- [ ] Thermal：Tj 82.49/67.54°C 时 TPOT变慢 → reward偏低，σ随Tj +3-5°C 增10-15% (需 NVML)
- [ ] 在线：GRPO 每500 steps 重拟合 a/b，EWMA跟踪σ，hysteresis 防震荡

## 映射表（RL only）

| 概念 | RL RM | 本code |
|---|---|---|
| 真效用 r_true | oracle w_true·diff | w_true随机 |
| 未校准 r_raw | sigmoid(w·diff) | logits→probs |
| 不确定性 source | flip15% + ensemble σ 0.045 | 15% 翻转 |
| 校准位移 | |cal-raw| | 0.0539 |
| 校准缩放 | Platt a/b | a=0.716 b=0.084 收缩过自信 |
| 过滤 | σ>0.15 → 0%本轮 | 0.05→30%可配 |

## 一句收敛

“rollout长CoT 12-18% + 热节流0.83% + 标注15% 三噪，ensemble 5得σ≈4.5%，ECE 0.0906→0.0881，|cal-raw| 5.39%当补偿，高σ直接过滤，省 $/有用 8-12%。”

## 代码
- `reward_oas_calibration.py` CPU gloo ok，待H100补
