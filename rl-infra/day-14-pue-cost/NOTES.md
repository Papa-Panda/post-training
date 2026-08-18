# NOTES — Day 14 Paper3 PUE → $/useful rollout

## 3 个 CPU 真数（待H100 NCCL 补 max_memory_allocated + nvidia-smi Tj + RAPL）

1. PUE 建模
- seed42 n=300，CPU single & gloo 2-rank ok
- PUE mean 1.25757439732819 p50 1.2380727943349583 p95 1.3427335637251325 min 1.1645596872997601 max 1.346807816525648 overhead 25.757% [CPU真数，待H100 NCCL补 max_memory_allocated]
- power_IT_mean 477.43446461875186W cooling_mean 123.84662162513972W Tj_avg 72.38419315951872°C Tj_max 92.06252530823865°C throttle_rate 3.00%
- 模型：COP = 5.2 - 0.04*ΔT - 0.002*ΔT^2，P_fan=28*flow^3+6，P_chiller=(P_IT/COP)*1.15 hyst on /0.35 off (0.85/0.35)，P_loss 3% → PUE=(IT+cool)/IT

2. $/useful rollout 经济性
- $IT proxy $3.2/GPU-hr：gpu_sec 49.5596s retry 2.6210s eval_idle 9.1131s total_hour 0.0170260 hr gpu_cost_it $0.05448335608808062 facility_cost $0.06851687369688517 (PUE*IT $)
- useful 281/300=0.9367 fail 19/300=6.333% (timeout7/vcj_parse7/tool_retry3/nccl2) filtered_high_uncert 0 (σ阈0.15, mean σ 0.045 连接Day12)
- cost_per_useful 0.00024383229073624615 $ /1k useful 0.24383229073624615 $ /1k tokens proxy 0.00011611061463630768 $ (2.1k tok/rollout)
- [CPU真数，待H100 NCCL 补 tokens/sec 真数 + $/GPU-hr 计费表 + vLLM 长尾 12-18%]

3. SLO → COST 联动（Day13连接）
- Day13 SLO1 0.955<0.98 FAIL vs 今天 useful_ratio 0.9367 fail_rate 0.0633；把 Day12 σ 0.045 过滤分母扣除后 useful 计数更准，避免把高不确定 rollout 误算 infra fail
- Day13 SLO2 queue p50 0.123s p95 0.385s scaled→真实120s ↔ 今天 eval_idle_sec 9.113s /300=30ms per rollout，异步 nowcasting 可省 52% (Day08实测 gpu_idle 1.034s→0.0s)
- Day13 SLO3 jitter 0.146 PASS Tj_max 90.5°C FAIL throttle 2.5% ↔ 今天 Tj_max 92.06°C throttle 3.0% flow^3 风机法则解释，hyst 82/72°C 解决
- [CPU真数，待H100 NCCL 补 nvidia-smi Tj + fan RPM + 冷机表]

### to H100 checklist
- [ ] torch.cuda.max_memory_allocated() vs CPU模拟 memory
- [ ] nvidia-smi Tj + NVML power trace 采样 1Hz 对比 power_it_mean 477W
- [ ] RAPL + fan power 真实立方拟合系数校准 28*flow^3
- [ ] 真vLLM 7B rollout 500→5000 tok tokens/sec 3.4-5k，失败率 5类分布替换模拟 6.33%
- [ ] 70B外推：IT 800W+ burst 功率墙，PUE overhead 30%+ 时 $/1k useful 翻倍阈值

### GitHub
- https://github.com/Papa-Panda/post-training/tree/master/rl-infra/day-14-pue-cost
- 文件：pue_cost_model.py CPU gloo 2-rank ok，README Connection to Prev 已写，NOTES 3 真数已填待H100。

### Connection 一句版
Day13 SLO 没算钱导致无法决策是否扩容，Day14 用 PUE 1.2576 把 retry+idle+cooling 转成 $/useful 0.000244，区分噪声 vs 真失败。
