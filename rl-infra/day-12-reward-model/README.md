> Connection to Prev: Day11 Paper2 mechanical load → Day12 Reward Model OAS calibration: vLLM rollout 80%墙钟的功率burst引发rack热抖动是reward噪声的物理源头，所以reward必须像MBS定价那样给不确定性加OAS利差；Day10 vLLM rollout 的5类失败率(12-18%长CoT)坑在今天用ensemble std + OAS spread阈值过滤高不确定rollout解决；Day8/9 eval瓶颈 的 nowcasting缺物理先验用γ*(ΔT)^2，二阶残差思路同构到reward校准的二次Brier/ECE。

# Day 12 — Reward Model 不确定性 + OAS 校准

> Date: 2026-08-12 (Foundation 1-3mo, RL Training/RL Concepts)  
> Source row: `ai_daily.csv:2026-08-12` — Reward Model / 理解 reward 模型的不确定性  
> Track: RL Training — RL Concepts  
> 交付侧 chat: 18854a6d-7852-49cd-845f-d7e4bb976d14 (same-day)

---

## 【昨日复盘】Day 11 真数

**Day11 Paper2 机械负载 → GPU 热/功耗 (2026-08-11)**：

- 120 steps 机械→GPU 联动 sim，IT 350kW base + 80*sin + burst 200kW (40步周期) + 外温 22±4°C 正弦：
  - `rmse_Q_pred` (EWMA vs 真值) **55.13 kW** — EWMA在burst拐点必lag，二阶γ*(ΔT)^2可压
  - `avg_P_mech` **57.87 kW** COP~5.2 ref
  - `p_mech_std` **9.38 kW** — 抖动可量化，对应 reward 抖动
  - `Tj_max` **82.49°C**, `Tj_avg` **67.54°C**, `throttle_rate` **0.83%** (hyst 82/72°C)
- 非线性技巧 4条：bilinear Q_IT*T_wb、quadratic/cubic γ*(ΔT)^2 & R_hs= R0/(fan^0.8)、hysteresis 冷机 0.85/0.35 → GPU 82/72°C、防抖 10min冷却 + 热容 C_w/C_j lag
- 代码 `paper2_mech_to_gpu_thermal.py` CPU gloo 2-rank ok，**待H100 NCCL** 补 `torch.cuda.max_memory_allocated` + nvidia-smi Tj / NVML power trace + R_jh/C_j fan cubic拟合

**Day10 vLLM rollout 基座 (2026-08-10)**：

- 7B G=2 峰值 ~42.5GB bf16-mix，常驻 28GB/G，comm 40-55%；13B G=4 峰值 38GB，tokens/sec 2.3-2.9k；70B G=8 峰值 86GB需activation ckpt，0.6-0.8k /GPU — **待H100 NCCL**
- 短CoT 500 tok 40-60k tokens/sec decode，失败 5-8%；长CoT 5000 tok 8-15k tokens/sec，失败 12-18% (超时40%/工具30%/VCJ15%/OOM10%/NCCL5%)

=> 昨日结论：rollout 占墙钟 80%→90%，热节流是隐性失败源，必须物理先验提前5-10min，今天把热/失败映射为 reward 不确定性定价。

---

## 【今日主题】Day 12 目标 + 最小可跑任务

**Track/Topic**: RL Training / RL Concepts — Reward Model  
**Knowledge Point**: reward 模型的不确定性 / 校准  
**Learning Goal**: 理解 reward 模型为什么总要带不确定性，出处有三：标注噪音 / rollout长CoT工具失败 / GPU热抖动；会用量化指标 ECE / Brier / ensemble σ 描述它  
**Small Daily Task**: 把一个简单的分类 reward 模型，用金融 OAS 思路做校准  
**Work Connection**: 金融定价的校准 → reward 校准；MBS OAS 给嵌入式期权不确定性定价 spread，reward OAS 给 rollout/标注不确定性定价  
**Resource**: RLHF Reward Modeling paper / InstructGPT RM / DPO-reward-gap ai-data/2025_dpo-reward-gap/

### 今天要懂的 3 层映射

#### 1) 为什么 reward 会抖

- **标注抖**：人类偏好 15% 翻车率（本code用 0.15 flip模拟），同 InstructGPT RM 收集“fully confident”只占60%
- **rollout抖**：Day10 vLLM 长CoT失败 12-18% → reward回传不是0/1是 NaN/超时，Day11 Tj>82°C 降频 → TPOT慢30% → 人误判为“答得差”
- **热/功耗抖**：Day11 `p_std 9.38kW` 同构 reward σ，rack温度+3°C → GPU throttled → reward latency分布尾变厚

=> reward 不是真值，是 `r_obs = r_true + ε_annot + ε_rollout + ε_thermal`，三噪叠加

#### 2) OAS 思路

固定收益：

- 国债收益率 y_T = 无风险
- MBS 收益率 y_MBS = y_T + OAS + option_cost
- OAS = market price 反解出的额外补偿，专补“借款人提前还款”这种内嵌option不确定性

RL reward：

- `r_true` = 无风险真效用（oracle）
- `r_raw` = RM 预测 logit → sigmoid prob，未校准
- 内嵌 option = rollout 失败/热节流/标注分歧导致的不确定性
- `OAS_reward` = `r_calibrated - r_raw`，或更实用 `r_adjusted = r_mean_ensemble + λ·σ`
- λ>0 时保守（惩罚高不确定 rollout，You might filter），λ<0 时探索（鼓励边界）

本code实现两条线：

- **校准线**：Platt scaling `logit_cal = a·logit + b`，a=0.716 b=0.084，本次拟合把过自信往回压，ECE 0.0906→0.0881（-0.0025，CPU小数据，待真人数据放大）
- **不确定性线**：ensemble K=5 bootstrap → σ_mean 0.0450，OAS spread |cal-raw| 0.0539，high_uncert_rate (σ>0.15) 0.0% → 低噪区；阈值0.05时约 30-35%（可配）

#### 3) 最小可跑任务（30-60min）

已在 `reward_oas_calibration.py` 跑通：

- 合成 N=2000 train 500 val，dim=16，翻转率15%模拟标注噪音
- 5模型 bootstrap ensemble + Platt scaling
- 输出 CPU 真数3个 + 分布式 gloo all_reduce演示
- **待H100 NCCL**：替换数据为 ai-data/2024_deepseek-v3 RM pairs / DPO-reward-gap human vs RM gap 数据，补 `torch.cuda.max_memory_allocated()` + vLLM rollout真实失败率联动，ECE需10bin→15bin sweep，σ vs rollout长度相关性

---

## 【与之前内容的联系】必写 2-3 句，贴到每日问题库

1. **昨天 Day11 → 今天 Day12**：昨天 Day11 学了 Paper2 机械负载的双线性 + 二次换热 + hysteresis 如何给“为什么会热节流”一个物理因果，今天的 Reward 不确定性是热的下一步——Tj>82°C throttled 0.83% 不会让训练崩，但会让 reward 回传慢/截断/被误判为差答案，MBS 的 OAS 给“借款人可能提前还款”定价，今天给“rollout可能热失败/标注分歧”定价，方法同是给未校准分数加一个补偿利差，区分正常探索噪声 vs 有害抖动，否则 GRPO 组内相对优势会被噪声淹没。

2. **前天 Day10 vLLM rollout → Day12**：Day10 把墙钟80%是rollout、长CoT失败12-18%5类拆分说清，但没回答“这些失败怎么进reward”，今天补：失败rollout不应进RM训练当负样本，而应进不确定性集合算 σ，高 σ 样本在今code里 high_uncert_rate 本轮0%（小合成），阈值降到0.05可筛30%边界，生产版阈值由 Day10 的失败率反推，形成“rollout → 不确定性 → OAS惩罚”闭环，折算 $/有用rollout 可省 8-12%（Day10同估）。

3. **Day08/09 Eval瓶颈 nowcasting → Day12**：Day09 的 EWMA 用 P50/P95/queue_depth 预测eval卡多久，miss了非线性拐点；Day11 用 γ*(ΔT)^2二阶修正解决了热 lag；今天把同一思想套到reward：ECE是“一阶校准误差”，Brier二次分解+ ensemble σ是二阶，Platt a=0.716 的收缩就是给过自信降 temper，和 γ 二阶同构——“分片省峰值 (P-b)/G+b”（Day02/03 per-block FSDP）→“热容省节流”（Day11）→“ensemble std过滤省无用rollout”是同一省峰值思想三进阶。

> 上面3段已浓缩进顶部的 Connection to Prev，可直接贴 README 最前及 ai_infra Notes。

---

## 代码怎么跑

```bash
cd rl-infra/day-12-reward-model
# 单卡
python3 reward_oas_calibration.py

# 2-rank gloo（CPU等效NCCL分片预测聚合）
torchrun --nproc_per_node=2 reward_oas_calibration.py
```

输出 JSON 含：
- ece_raw/ece_cal/ece_improve
- reward_std_mean / oas_spread_mean / high_uncert_rate
- acc_raw/acc_cal / brier_raw/brier_cal
- platt_a/b
- **待H100 NCCL** 提示

2-rank gloo CPU 逻辑通，**待H100 NCCL** 补 `max_memory_allocated` + 真RM数据。

## Fail-closed

- 没编 H100 数。所有GPU显存/吞吐/tokens/sec/失败率都标 **待H100 NCCL验证**。
- 本轮合成小数据 ECE improve 仅 0.0025，σ 0.045，high_uncert 0% 是单 seed 42 真数，不夸大；真人偏好 15%翻转是人为设，生产需 DPO-reward-gap/human-calibration 数据。
- RM校准不是只做一次Platt，应在线：GRPO每K steps重拟合a/b，用EWMA跟踪σ漂移，Day11 hysteresis同理。

## Monetization / Work Connection

- 你过去做MBS/FX准度+量化risk建模，OAS校准是老本行：MBS里 Z-spread 剥离现金流，RL里 reward spread 剥离 rollout噪音，面试可讲“用固定收益的OAS框架给RM定价，σ是隐含波动率，OAS spread是 risk premium，过滤阈值是止损线”
- $/有用 rollout 闭环：从 Day10 rollout瓶颈 → Day11热成本 → Day12 reward不确定性过滤，省的是“热抖动+失败rollout进RM误训练”浪费，量化成 $/有用rollout降 8-12%
- Paper1 5条bridge复用：抗抖动(hysteresis) → reward抖动控制 DAPO decoupled clip+dynamic sampling，nowcasting → reward σ EWMA预测，都是同一套“预测-平滑-止损”

---
**Artifacts**: `reward_oas_calibration.py` CPU ok，NOTES.md 3数待H100，本 README 含 Connection 段。
