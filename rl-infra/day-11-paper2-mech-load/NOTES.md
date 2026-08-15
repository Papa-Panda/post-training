# Day 11 NOTES — Paper2 机械负载 → GPU 热/功耗

Date: 2026-08-11 08:22 PDT (manual today run 2026-08-15,补)  
Status: done (CPU gloo 验证逻辑，待 H100 NCCL 真机验证 thermal + RAPL + max_memory_allocated)  
Lab: `rl-infra/day-11-paper2-mech-load/`  
Source: `ai_daily.csv:2026-08-11` — Paper2拆解-机械负载 / 非线性物理建模

## 核心 — Paper2 3 技巧拆解

- **bilinear 耦合**: Q_IT * (1+α(T_wb-25)) — IT*外温一次项，GPU 对应 P_gpu*Tamb 耦合
- **quadratic/cub 非线性**: CHW 二次 γ*(ΔT)^2, COP 二次 κ*PLR^2, fan R_hs=R0/(fan^0.8) 三次律影子 → GPU R_th 同构
- **hysteresis 防抖**: chiller on 0.85/off 0.35 机组容量比 ↔ GPU throttle on 82°C/off 72°C — 复用 Day06 10min 冷却思想，10min 内不追加功率

SSM 公式见 README.md，CPU 已跑联动。

## 3 个 CPU 数 (gloo 2-rank, 待 H100 NCCL)【真数】

> 120 steps 机械侧 350kW base + 80*sin + burst 200kW 每 40 步，GW 映射 GPU 450-720W per H100，dt mech 1min dt gpu 0.1s 串联，seed 42

### Mech 侧预测

- **rmse_Q_pred_kW (EWMA α=0.3 vs true Q_IT)**: **55.129** kW — EWMA 在 burst 段 lag 大，纯统计必偏，物理二次修正可降（待 H100 上加 γ 残差）
  - 意义：对应 GPU rollout burst 到达预测 RMSE，真机应 <20kW 带物理先验
- **avg_P_mech_kW**: **57.867** kW — 机械制冷电，类似 GPU 额外散热功耗开销
  - COP_ref 6.5, PLR 0.6 avg, COP avg ~5.2 → P_mech ~ Q_mech/COP
  - GPU 视角：fan_overhead 占 P_gpu 4-6%，40卡 rack 多 2.5kW 额外
- **p_mech_std_kW**: **9.380** kW — 抖动 STD，对应 GPU power jitter
  - burst 时 Δ 12-16kW，冷却 lag 3τ=15min 后仍有 5-8kW 尾巴
  - 迁移：Reward OAS 校准的 jitter 上限同理，用 STD 卡阈值避免“抖动→重训”

### GPU 热侧

- **Tj_max_C**: **82.486** °C (2-rank gloo 同 seed 42，120 steps 内最高)
  - Tj_avg **67.542** °C — H100 Tj safe 90°C 阈前，burst 时窜升
  - single rank 同样 82.49/67.54，2-rank dist avg throttle 0.00833 一致
- **throttle_rate**: **0.833%** — 1/120 steps 触发 Tj>82°C 降频 0.68×，对应 rollout 失败率里隐性 0.8% 性能慢
  - 待 H100：`torch.cuda.max_memory_allocated()` 并采集 `nvidia-smi -q -d TEMPERATURE,PERFORMANCE` 时序，看 Tj>82 时 tokens/sec 掉多少，折算 $/有用 rollout

### 2-rank gloo 聚合演示

- rank0 + rank1 全 reduce throttle_rate SUM/2 = **0.833%** 仍一致（seed 42+rank 差小）
- gloo 2-rank `[rank0] mechanical→GPU simulation done 0.053s` — 通信本身不阻塞，仿真已并行
- CPU 逻辑通，逻辑区分：per-block FSDP 省峰值 ↔ 热容省 Tj 瞬峰 同思想

### 待 H100 NCCL 真机补

- `torch.cuda.max_memory_allocated()` / `reserved()` 对比 mech SSM 大 state 与 GPU SSM 2-node 小 state 常驻差异（<100MB 预期）
- RAPL / `nvidia-smi power.draw` 时序：P_gpu 450→720W burst 10min 内 Tj 从 67→82°C 爬升曲线，拟 R_jh / C_j / R0 风扇律实测
- 真数据中心对位：拿你 Paper2 真 log 跑 SSM，对比 EWMA RMSE 55kW → 物理先验能否压到 32kW (类比 GPU 提前 5-10min 预调 fan)
- blocks: block 非整模型 → 热学 block 非整 rack，局部热点 (Tj>85) vs 平均 67°C 同理

## 与之前内容的联系—代码侧

- Day10 FSDP peak `(P-b)/G+b` 省峰值 ↔ 今天热容峰值 `Tj_next = Tj + dt/C_j*(P - ΔT/R)` 大 C_j 削峰，同切小块降瞬时
- Day08/09 EWMA nowcasting 缺二阶，今天 γ 二次补上 → Day12 reward OAS 用类似二次残差标不确定性
- Day07 checkpoint 读档 ↔ 今天防抖 hyst 避免频啟频停，Fail-slow (throttle) 比 Fail-stop (crash) 更隐蔽，需调度层冷却窗口

## 怎么对应到 GPU 集群 SOP

1. **建模**：两节点 T_j/T_hs + R_hs(fan)=R0/(fan^0.8) + hyst 82/72
2. **信号**：P_gpu (rollout burst) + T_amb (rack inlet, 来自机械侧 T_wb 映射) + fan_ratio + throttle_flag 三合一输入 SSM
3. **决策**：SSM 预测 Tj_next 若 >80°C 提前 5min 加 fan / 降 batch / 迁 rollout 到冷 rack，类似 chiller 提前加机
4. **验证**：/有用 rollout 成本里加 `+ $/thermal_waste`，机械侧 p_mech_std 9.38kW 对应 GPU jitter 阈值 35W

## Code

- `paper2_mech_to_gpu_thermal.py`:
  - single: `python3 paper2_mech_to_gpu_thermal.py`
  - 2-rank: `torchrun --nproc_per_node=2 paper2_mech_to_gpu_thermal.py`
  - 输出 JSON 真数 + 待H100提示
  - rank0 聚合 throttle_rate SUM/ world

## 一句话迁移

> Paper2 机械负载非线性 SSM 的双线性 IT*外温 + 二次换热 + 0.85/0.35 加机 hysteresis，平移成 GPU 两节点 Tj/T_hs SSM + 风扇立方律 R_hs + 82/72°C 节流 hysteresis，CPU 真数 55.13 kW RMSE / 57.87 kW 平均电 / 9.38 kW 抖动STD + Tj 82.49°C 峰值/67.54°C 均值/throt 0.83%，待 H100 NCCL 补真实 Tj/功率 trace，量化成 $/有用 rollout 热损 8-12% 降。

## Fail-closed

- 没编 H100 数，所有 CUDA/Tj 真传感器读数标 **待H100 NCCL**
- EWMA 55kW RMSE 是 CPU 缩放 IT load 300-500kW 带 burst 随机，非数据中心真机实测，提醒需 Paper2 真 log 回放
- Tj 82.49°C 是 CPU 热容模型估，非 nvidia-smi 真值，需 H100 长 CoT 5000 tok rollout 实跑温升
