# NOTES — Day 16 Monetization Story v1

> Connection to Prev: Day15 Megatron 3D 182GB→TP4+PP2 25GB → Day16 Monetization v1: PUE 1.2576算清$/useful但缺可讲的跨界ROI故事，需要把分片/热散/eval异步压缩成150字；Day14 PUE 1.2576 overhead 25.76% + Day13 Tj 90.5°C throttle 2.5%坑用 $/useful+ jitter/throttle双阈值+TP散热解决。

Date: 2026-08-16 (RL Training / Monetization) — actually 2026-08-20 delivery of Day16

## 3个CPU真数（待H100 NCCL 补 torch.cuda.max_memory_allocated + nvidia-smi Tj + vLLM tokens/sec）

### 1. 排队/预测 save：nowcasting burst → eval异步省等待
- Day13 queue p50 0.123s p95 0.385s scaled→真实120s avg_depth 0.21
- 预测准确率85%时 p95 0.385s→0.12s save_ratio 0.688 (-68.8%) [CPU真数，待H100 NCCL 补 gloo 2-rank 验证 + 真实120s wait trace + EWMA next预测2.098s]
- 对应 Day08/09 eval瓶颈 sync p50 1.141s p95 3.249s gpu_idle 1.034s占92.85% → async 0.0s省52% total 1.113s→0.527s
- RL讲法：burst预测每准10% → SLO2 p95降0.05s，扩容决策延迟-1天 ≈ 1×GPU-hr proxy，待H100补 tokens/sec

### 2. 热/功耗 save：SSM+fan^3+hyst把节流打下来
- Tj_before 90.5°C throttle 2.5% (Day13实测 5/200 FAIL) → Tj_after 82.49°C throttle 0.83% (Day11实测) delta -1.67pp [CPU真数，待H100 NCCL 补 nvidia-smi Tj + NVML power trace + R_jh/C_j fan立方拟合]
- 模型：C_j dTj/dt = P*throt - (Tj-Ths)/Rjh, Rhs(fan)=R0/(fan^0.8+0.15), P_fan 28*flow^3+6, hyst on/off 82/72°C RL侧 0.85/0.35
- TP散热点：Day15 TP4把单卡720W burst→480-520W -28%，节流率从2.5%→<1% proxy
- RL讲法：throttle 1%阈值来源于TPOT +30%后用户误判为差答案，fail-slow也算 SLO FAIL，checkpoint(Day07)救不了 must功率平滑

### 3. COST：PUE→$/useful翻译成可讲ROI
- PUE mean 1.2576 p50 1.2381 p95 1.3427 min 1.1646 max 1.3468 overhead 25.76% (IT 477.4W cooling 123.8W Tj_avg 72.4°C Tj_max 92.1°C throttle 3.0%) [CPU真数，待H100 NCCL 补 max_memory_allocated]
- $/useful before 0.000244 $/1k useful 0.2438 $/1k tokens proxy 0.000116 useful 281/300=93.7% fail 6.33% —> after async+TP散热点+σ过滤 $/useful 0.00019 save 22.1% [CPU真数，待H100 NCCL 补 $3.2/GPU-hr计费表 + 真vLLM 3.4-5k tokens/sec]
- Day12 σ 0.045 ensemble K=5 + |cal-raw| 0.0539 OAS校准位移过滤高不确定rollout不进 useful分母，避免把RM噪声当infra失败扩机柜
- 映射：每1k有用rollout省 0.0538 $ proxy → 周3000 rollout省 0.16 $ proxy (CPU小样本，待H100放大) + GRPO组内N=64优势方差↓ ∝ sqrt(N)*σ

## 待H100 NCCL
- [ ] torch.cuda.max_memory_allocated() 真数：7B DP G=2 18GB vs proxy 17.24GB偏差，70B TP4+PP2 25GB vs 25.05GB验证
- [ ] nvidia-smi Tj时序 + NVML power trace 1Hz vs CPU模拟的 Tj 72.4°C / 90.5°C偏差，风机RPM三次方拟合系数28校准
- [ ] vLLM TTFT/TPOT overlay：TP=2 TTFT -20ms TPOT +15%，TP4 TTFT -35ms TPOT +27%，GRPO组内优势是否受TP噪声影响 σ<0.05
- [ ] 真rollout 5类失败分布替换模拟6.33%：timeout/tool/vcj/oom_kv/nccl → SLO1 budget 2%分配，useful分母扣除逻辑跑真人偏好DPO-gap联动
- [ ] PP bubble interleaved实测 + eval async填充奖励计算 filler，省gpu_idle 52%再压

## 一句收敛
“Day15把70B 182GB切到25GB才让PUE有意义，Day14 1.2576把$/useful算到0.000244但没讲成ROI story，今天把queue 68.8%+thermal 1.67pp+cost 22.1%三真数压成150字跨界叙事，SLO×COST→面试可讲$。”

## 代码
- `monetization_v1.py` CPU单进程 ok（torch缺失 fallback已写，待H100 NCCL补 gloo 2-rank）
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/rl-infra/day-16-monetization-v1
- 本次真数来源：python3 monetization_v1.py 输出见上（单rank真数），gloo 2-rank sim待 torch环境补跑已写all_reduce分支

### 原始输出存档
```
[Rank 0/1] Day16 Monetization Story v1 CPU proxy (待H100 NCCL 补 max_memory_allocated)
gloo_check: torch_not_installed CPU fallback ok (待H100 NCCL 补 gloo 2-rank)
1) queue p50 0.123 p95 0.385 -> pred_improved 0.12 save_ratio 0.688 [CPU真数，待H100 NCCL]
2) thermal Tj 90.5C throttle 2.5% -> 82.49C throttle 0.83% delta 1.67% [CPU真数，待H100 NCCL]
3) cost PUE 1.2576 overhead 25.76% $/useful 0.000244 -> 0.00019 save 22.1% [CPU真数，待H100 NCCL]

Story v1 (330字):
过去6年我做预测与SLO：用nowcasting预测burst把排队p95从真实120s压到阈值内，用两节点SSM+风机立方+hyst 0.85/0.35把Tj 90.5°C节流2.5%压到<1%，用FSDP分片把70B 182GB峰值切到TP4+PP2 25GB，把PUE 1.2576 overhead 25.76%翻译成$/useful 0.000244决策扩容。迁移到RL：把rollout5类失败(timeout/tool/vcj/oom_kv/nccl)+eval异步省52% gpu_idle+GRPO组内64基线抗抖合成SLO1≥98%/SLO2 p95<SLO3 jitter<0.15，让小规模后训练稳定、可复现、省$/1k useful。

{"day": 16, ...}
```

### RL-only 映射（禁止金融类比）

| 概念 | RL集群 | 本code proxy |
|---|---|---|
| nowcasting burst | rollout泊松到来 + 队列深度 | p95 0.385→0.12s -68.8% |
| FSDP分片 | ZeRO1 (P-b)/G+b 峰值切分 | 70B 182→25GB TP4+PP2 |
| vLLM TTFT/TPOT | TP切分TTFT↓但ALLGather TPOT↑ | TP2 TTFT -20ms TPOT +15% |
| rollout失败 | 5类 timeout/tool/vcj/oom/nccl | fail 6.33%→2% via过滤 |
| 热节流Tj | 两节点SSM + fan^3 + hyst | Tj 90.5→82.49 throt 2.5→0.83% |
| eval异步 | sync gpu_idle 92.85%→0% | save 52% gpu_idle |
| GRPO baseline | 组内64样本相对优势抗抖 | σ 0.045过滤后方差↓ |
| PUE→COST | cooling overhead → $/useful | 1.2576→0.000244→0.00019 -22% |
