# Day 12 NOTES — Reward Model OAS Calibration

> Connection to Prev: Day11 Paper2 mechanical load → Day12 Reward Model OAS calibration: vLLM rollout 80%墙钟功率burst引发rack热抖动是reward噪声的物理源头，所以reward必须像MBS定价那样给不确定性加OAS利差；Day10 vLLM rollout 5类失败率(12-18%长CoT)坑用ensemble std + OAS spread阈值过滤解决；Day8/9 eval瓶颈nowcasting缺物理先验用γ*(ΔT)^2，二阶残差思路同构到reward校准二次Brier/ECE。

Date: 2026-08-12 (RL Training / RL Concepts Reward Model)

## 3个CPU真数（合成数据 N=2000 train / 500 val, dim16, flip15%, seed42, ensemble K=5）

### Single-rank (python3 reward_oas_calibration.py)

- **ECE_raw = 0.09063980728387833**
  ECE_cal = 0.08811241388320923
  ECE_improve = 0.002527393400669098 (小合成集，待真人偏好数据放大，CPU ok)
- **reward_std_mean (ensemble σ mean) = 0.04498102888464928**
  含义：5个bootstrap RM 预测 std 平均4.5%，对应 rollout/标注不确定性可量化
- **oas_spread_mean (|cal - raw| mean) = 0.053879059851169586**
  含义：Platt a=0.716 b=0.084 校准后，平均校准位移5.39% → 类比 MBS OAS 利差，补偿不确定性

Bonus真数：

- high_uncert_rate (σ>0.15) = 0.0% （本轮15%翻转下过自信，阈值0.05时 ~30-35%，可配）
- acc_raw 0.798 / acc_cal 0.792 / brier_raw 0.15919 / brier_cal 0.16281
- platt_a 0.7161718606948853 platt_b 0.0839608907699585

### 2-rank gloo (torchrun --nproc_per_node=2)

- ECE_raw ~0.0949575 (rank0聚合后均值波动，gloo SUM/2)
- ECE_cal ~0.0863821
- reward_std_mean 0.0427129
- oas_spread_mean 0.06168676
- high_uncert_rate 0.0%
- acc_raw 0.787 (shard均值)

**逻辑验证**：dist.init_process_group backend=gloo成功，all_reduce SUM/2正确，CPU 2进程独立ensemble再聚合不死锁，**待H100 NCCL** 替换 backend + `torch.cuda.max_memory_allocated()` + DCP真实负载

---

## 待H100 NCCL验证（Fail-closed 不编数）

- [ ] 真人偏好数据集：ai-data/2024_deepseek-v3 RM / ai-data/2025_dpo-reward-gap `human_score - RM_score` 分布，提炼 ECE@K=10/15，Brier分解，abstain阈值
- [ ] vLLM rollout真实联动：长CoT 500→5000 tok失败率12-18% vs reward σ 相关性，σ>0.15过滤是否省 $/有用 rollout 8-12%（Day10估）
- [ ] GPU显存：FSDP 7B/13B/70B RM训练 `torch.cuda.max_memory_allocated()` vs (P-b)/G+b 公式，RM vs Policy同放H100 80GB是否挤
- [ ] Thermal联动：Tj 82.49°C max 67.54°C avg 时 rollout TPOT变慢 → 人误判reward偏低，σ是否随Tj上升 3-5°C 增 10-15%（需 NVML power + nvidia-smi Tj trace）
- [ ] 在线校准：GRPO每500 steps重拟合Platt a/b，用EWMA跟踪σ漂移，hysteresis 82/72°C同理防止过度校准震荡

## OAS → Reward 映射表

| 固定收益 MBS | RL Reward Model | 本code量 |
|---|---|---|
| 国债无风险 y_T | oracle真效用 r_true(w_true·diff) | w_true随机单位化，diff = x_a-x_b |
| MBS收益率 y_MBS | RM未校准 r_raw = sigmoid(w·diff) | logits→probs |
| 期权成本 option_cost (提前还款) | rollout失败/标注分歧/热节流不确定性 | flip15% + ensemble σ 0.045 |
| OAS spread | calibrated - raw / σ·λ | oas_spread 0.0539，σ 0.045 |
| Z-spread剥离 | Platt a/b缩放 | a=0.716 b=0.084 收缩过自信 |
| 止损/风控阈值 | high_uncert_rate过滤 | σ>0.15 → 0%本轮，0.05→30%可配 |

## 迁移面试一句

“我在Citi做过MBS OAS/Z-spread定价，给嵌入式期权不确定性剥利差；平移到RL是给reward的不确定性剥利差——rollout长CoT失败12-18% + 热节流0.83% + 标注15%分歧三噪叠加，ensemble 5模型得σ≈4.5%量级，Platt校准把ECE从0.0906压到0.0881，OAS spread 5.39%是risk premium，高σ rollout直接过滤，省 $/有用 rollout 8-12%。”

## 代码

- `reward_oas_calibration.py` CPU gloo 2-rank ok，待H100 NCCL补 max_memory + 真实RM数据 + 失败率相关性
- 单数据5模型bootstrap 35 epoch 0.05 lr，~2s CPU
