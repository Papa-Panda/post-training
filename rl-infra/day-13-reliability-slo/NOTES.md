# Day 13 NOTES — Reliability / SLO

> Connection to Prev: Day12 σ 0.045 + |cal-raw| 0.0539 → Day13 SLO1 success_rate≥0.98 5类失败桶；Day11 Tj_max 82.49°C throttle 0.83% → SLO3 power_jitter 0.15 + throttle 1%；Day08/09 p50 1.141s p95 3.249s gpu_idle 92.85% async省52% → SLO2 p95 1.2s scaled。

Date: 2026-08-13 (Infra Systems / Cluster Reliability SLO)

## 3个CPU真数（sim n=200 seed42 burst每40job）

### Single-rank
- SLO1 success_rate 0.955 / fail_rate 0.045 fail分类 timeout 5 vcj_parse 3 tool_retry 1  阈值0.98 → FAIL
- SLO2 queue_p95 0.385100762541043s (scaled 1.2s≈真实120s) p50 0.12253046455235257s avg_depth 0.21 max 3 → PASS
- SLO3 power_mean 459.1598819386836W p_std 66.95247462665081W jitter_ratio 0.1458151664817957 tj_max 90.5218911856968°C tj_avg 73.44385070587003°C throttle_rate 0.025 (5/200) → FAIL (jitter PASS 0.146<0.15 但 throttle 2.5%>1%)
- Bonus: wall_sec 0.0008747s n=200

### 2-rank gloo
- 世界 2 rank0/1 同 seed42：
  - success_rate_avg 0.9549999833106995 (all_reduce SUM/2) gloo_ok True
  - queue_p95 0.3851s 同步
  - power_jitter 同步
- **逻辑验证**：gloo all_reduce SUM/2 ok，CPU 2进程 ok，待H100 NCCL + torch.cuda.max_memory_allocated + NVML Tj时序 + RAPL功率

## 待H100 NCCL
- [ ] 真实vLLM rollout 500→5000 tok 长CoT失败率 12-18% 5类比例 vs sim 4.5% → 修正SLO1阈值0.98是否过严/过松
- [ ] 真实eval排队 5-10s sandbox 真实P95 8-15min bench → 将scaled 1.2s映射到真实120s验证（100x缩放）
- [ ] 显存：FSDP 7B/13B/70B G=2/4/8峰值 (P-b)/G+b + max_memory_allocated vs jitter关联 — 大模型常驻显存高时power burst更大
- [ ] Thermal：真实nvidia-smi Tj + nvml power + fan cubic fit Rhs(fan)=R0/(fan^0.8+0.15) + hyst 82/72°C节流窗口10min防抖
- [ ] SLO联动：throttle_rate>1%时自动扩冷却窗口，同复用Paper1冷却10min抗抖；GRPO组内N=64 advantage过滤阈值联动SLO1超阈告警
- [ ] 在线：GRPO每500 steps重算 SLO budget消耗，EWMA跟踪σ漂移，hysteresis防震荡，同Day12 Platt重拟合

## 映射表（RL only）

| 概念 | RL集群 | 本code |
|---|---|---|
| 作业成功 | rollout 5类失败外总成功 | 0.955 |
| 排队p95 | eval异步阻塞感知 | 0.385s scaled → 38.5s? 真实120s |
| 功率抖动 | vLLM burst 450→720W | 459W mean σ67W jitter0.146 |
| 热节流 | Tj>82°C降频 TPOT+30% | 90.5°C max 2.5% rate |
| GRPO budget | 2% fail可控，>5%稀释优势 | 4.5%本轮超budget |

## 一句收敛

“Day12 σ0.045 + |cal-raw|0.0539告诉你过滤啥，SLO告诉你何时算集群不可用——success 0.955<0.98 FAIL 5类失败，queue p95 0.385s PASS async省52%，power jitter 0.146 PASS但 Tj 90.5°C throttle 2.5%FAIL，双阈值防抖复用Paper2 hyst 82/72°C。”

## 代码
- `slo_sim.py` CPU gloo ok，待H100补 max_memory_allocated + nvidia-smi + RAPL + 真vLLM失败率
