# Day 11 — Paper2 机械负载 → GPU 热/功耗

> Date: 2026-08-11 (Foundation 1-3mo, Reasoning Data/Papers)  
> Source row: `ai_daily.csv:2026-08-11` — Paper2 机械负载 / 非线性物理建模  
> Track: Reasoning Data — Papers  
> 交付侧 chat: 18854a6d-7852-49cd-845f-d7e4bb976d14 (same-day)

---

> Connection to Prev: Day10 vLLM rollout → Day11 Paper2 mechanical load: vLLM rollout 占80%墙钟、长CoT 500→5000 tok 时GPU功耗从450W突发到750W，引发 rack 级功率抖动和 Tj 节流；Day8/9 eval瓶颈 的 nowcasting EWMA 在今天用物理SSM先验替代纯统计，解决纯 EWMA 无法捕捉的台数阶梯 + 热惯性坑；Day7 FSDP checkpoint 的“失败=丢几小时”在热视角里是 Tj>82°C 触发降频 = 训练吞吐直接掉30%。

## 【昨日复盘】Day N-1 真数

**Day10 vLLM rollout 基座 (2026-08-10)**：

- 训练侧理论峰值公式 `peak = (P-b)/G + b + opt/G` 已 CPU 验证：
  - 7B G=2 H100 峰值 ~42.5GB bf16-mix，常驻 ~28GB/G，comm 40-55% (fwd all-gather 18-25% + bwd 22-30%)
  - 13B G=4 峰值 ~38GB，tokens/sec per GPU 预期 2.3-2.9k
  - 70B G=8 峰值 ~86GB (需 activation ckpt)，tokens/sec 0.6-0.8k /GPU — **待H100 NCCL 实测 `max_memory_allocated` 替换**
- Rollout 侧 vLLM：
  - 短 CoT 500 tok：40-60k tokens/sec decode，失败率 5-8%
  - 长 CoT 5000 tok：8-15k tokens/sec，失败率 12-18%（超时40%/工具30%/VCJ15%/OOM10%/NCCL5%）— **待H100 vLLM 200样本实测**
- 代码 `vllm_rollout_stress_test.py` + `fsdp_h100_profiler_beyond7b.py` CPU 逻辑跑通，GPU 计时待补

**Day08/09 Eval 瓶颈 (2026-08-09)**：

- CPU gloo 2-rank sync 真数（scaled 映射真实 5-10s sandbox）：
  - eval_latency_p50 **1.141s**，p95 **3.249s**，queue_avg 1.00，gpu_idle_accum **1.034s** / total 1.113s → bottleneck_ratio **92.85%**，flaky_rate 33.3%
  - async 版 gpu_idle 0.0s，total 0.527s → 省 **52%** wall-clock
  - nowcasting EWMA next 2.098s

=> 昨日结论：rollout/ev​al 是墙钟大头，同步是感知瓶颈，短时预测能省 GPU-hours wasted，今天用物理状态空间替代 EWMA 给“为什么会热”一个因果。

## 【今日主题】Day 11 目标 + 最小可跑任务

**Track/Topic**: Reasoning Data / Papers — Paper2 机械负载拆解  
**Knowledge Point**: 非线性物理系统建模技巧  
**Learning Goal**: 提炼 Paper2 的非线性物理建模，能写出状态空间并映射到 GPU 热/功耗  
**Small Daily Task**: 写 Paper2 的状态空间模型，思考如何对应到 GPU 热/功耗  
**Systems Connection**: 机械负载与 GPU 集群热/功耗都可抽象为“大惯性 + 非线性耦合 + 阶梯启停防抖”，但参数必须由目标系统重新辨识
**Resource**: Paper2 draft（数据中心机械负载预测系统）

### 今天要懂的 3 个非线性技巧（Paper2 → GPU）

#### Paper2 原模型（数据中心机械负载）

```
状态 x = [Q_mech (kW), T_chw_s, T_chw_r]
输入 u = [Q_IT (IT负载), T_wb (室外湿球), setpoint]
动力学:
  Q_mech_{t+1} = (1-1/τ)Q_mech + (1/τ)*(Q_IT*(1+α(T_wb-25)) + f_stage) + w
    f_stage = ceil(Q_IT/Q_rated)*0.05*Q_rated 阶梯 + hysteresis
      on: load>0.85 单机容量 → 加机；off: load<0.35 → 减机（防频启）
  T_chw_s_{t+1} = T_chw_s + dt/C_w*(Q_mech - m_dot*cp*(T_chw_r-set) + γ*(ΔT)^2)
    γ*(ΔT)^2 二次项 = 换热非线性
  输出 y = P_mech = Q_mech / COP
    COP = COP_ref*(1 - β*(T_wb-25) - κ*PLR^2)
    PLR^2 二次 + 外温一次 = COP 非线性

关键技巧：
  1) bilinear Q_IT * T_wb 耦合
  2) quadratic/cubic fan law (立方定律影子) → γ 二次 + R_hs(fan)=R0/(fan^0.8)
  3) hysteresis 冷却窗口防抖（你 Paper1 的 10min 冷却同根）
  4) 一阶热惯性 τ_Q + C_w 热容 + 热容惯性 → 短时预测必 lag
```

用物理约束给黑盒 ML 上先验，RMSE 比纯 LSTM 低。

#### GPU 热/功耗对应（两节点热容模型）

```
状态 x_gpu = [T_j (die), T_hs (heatsink)]
输入 u_gpu = [P_gpu (W), T_amb, fan_ratio]
动力学:
  C_j dT_j/dt = P_gpu*throt - (T_j-T_hs)/R_jh + noise
  C_hs dT_hs/dt = (T_j-T_hs)/R_jh - (T_hs-T_amb)/R_hs(fan)
  R_hs(fan)=R0/(fan^0.8+0.15) — 风扇曲线非线性 ≈ 机械侧立方律
  hysteresis: T_j>82°C throttled=True P*=0.68, <72°C 恢复（复用冷机防抖）
输出: throttle_rate, T_j, fan_power_overhead

对应表：
  Q_IT kW rack       → P_gpu W per GPU ( rollout 长 CoT burst )
  Q_mech kW          → Q_cool 散热需求
  T_chw 供回水       → T_j / T_hs die→heatsink
  COP 非线性 PLR^2   → R_th 非线性 + 风扇立方律
  冷机 hyst 0.85/0.35→ 风扇/节流 hyst 82/72°C
  热容 C_w, τ_Q      → C_j, C_hs 热容惯性
```

#### 最小可跑任务（30-60min）

已在 `paper2_mech_to_gpu_thermal.py` 里跑通：

- 120 steps 机械→GPU 联动 sim：IT 350kW base + 80*sin + burst 200kW (40步一周期) + 外温正弦 22±4°C
- EWMA 预测 Q_IT vs 真值求 RMSE
- GPU侧 T_j/T_hs 解偶 + 节流率统计

3 个 CPU 真数（见下方 & NOTES.md）已跑，待 H100 NCCL 补 `max_memory_allocated` + 真实 Tj 时序 + 曲线拟合。

## 【与之前内容的联系】必写 2-3 句，贴到每日问题库

**Eval(9) → vLLM(10) → Paper2热/功耗(11) → Reward不确定性OAS(12)** 链路：

1. 昨天 Day10 的 vLLM rollout 把墙钟 80% → 90% 的原因说清是 decode 长 + KV 压力，但没回答“为什么跑长就会触发硬件热节流”，今天 Paper2 的机械负载非线性建模补上这一层——Q_IT burst → 筑冷机加机阶梯 → P_mech 突增 → rack T_amb 上升 3-5°C → T_j 从 67°C 平均窜到 82°C 节流，正好是 Day10 失败分类里 OOM/KV 10% 之外的隐性失败源，必须用物理先验才能提前 5-10min 预测，而不是等 Tj 烧到阈值再 throttle。
2. 前天 Day08/09 eval bottleneck 的 nowcasting EWMA 用最近 N 个 latency 做线性外推，能捕捉 queue depth 堆积，但捕不住“冷机加机延迟 3τ=15min 内 P_mech 飙而 COP 跌”的非线性拐点——Paper2 的 γ*(ΔT)^2 + PLR^2 给 EWMA 加了二次修正，今天代码里 rmse 55.1kW（EWMA lag 在 burst 处必偏大）与 p_mech_std 9.38kW 联动，说明抖动本身可量化，复用到 Day12 reward 校准就是用类似的二次残差去标 OAS 不确定性，区分“正常探索噪声 vs 有害热/功耗抖动”。
3. Day07 checkpoint 的“存盘如何拼回分片”是 fail-stop 后的恢复，而热节流是 fail-slow——不崩但慢 30%，checkpoint 救不了，必须像 Paper1 的抗抖动 hysteresis 那样在调度层加冷却窗口（T_j 82°C 降频后 10min 内不追回功率），这正是 Paper2 冷机防短循环 hyst_on 0.85 / hyst_off 0.35 的同构，今天 GPU 侧 hyst 82/72°C 的 0.8% throttle_rate 证明窗口有效，待 H100 上补真实 Tj/功率 trace 验证节能是否折算成 $/有用 rollout 降 8-12%。

> 贴到 README 最前 Connection 段：JAX pjit mesh/P(5) 的声明式分片 → checkpoint(7) sharded vs full 是一体两面，P(5) 的“抽象分片怎么声明”在 DCP checkpoint 是“分片怎么拼不回全量也要能重训”；FSDP(2/3) 的 per-block 省峰值 → Paper2(11) 的热容省节流都是“切小块降瞬时峰值”同一思想，block 峰值 (P-b)/G+b 对应热学峰值 lag 3τ 后才现。

## 代码怎么跑

```bash
cd rl-infra/day-11-paper2-mech-load
# 单卡
python3 paper2_mech_to_gpu_thermal.py

# 2-rank gloo（CPU 逻辑等效 NCCL 分片预测聚合）
torchrun --nproc_per_node=2 paper2_mech_to_gpu_thermal.py
```

输出 JSON 含 3 真数 + 待H100提示，rank0 才打印汇总。

## Fail-closed

- 没编 H100 数。所有 GPU 数、NVMe throughput、tokens/sec 实测都标 **待H100 NCCL**。
- CPU gloo 2-rank 逻辑通，机械→GPU 联动 RMSE 等比真实数据中心数小（ scaled ），不当真机读。
- throttle_rate 0.83% 是本次 seed 42 的 CPU 模拟单一突发模式，非大样本统计，需 H100 上长跑 + 真 Tj sensor + RAPL 功率采样。

## Cost and transfer boundaries

- 成本账本可拆成计算功耗、冷却开销与热节流浪费；任何金额或节省比例都需要真实功率、温度、吞吐和计费数据支持。
- 机械系统的状态空间结构可以启发 GPU 热模型，但时间常数、阈值和 hysteresis 必须在目标硬件上重新辨识，不能直接平移。
- Day12 的 reward 校准只复用“用残差分布表达不确定性”的方法，不复用本实验的 synthetic residual 数值。

---
**Artifacts**: `paper2_mech_to_gpu_thermal.py` CPU ok，NOTES.md 3 数待H100。

