# RL Infra 小规模后训练可见 scope — H100 >7B 扩展验证版 (2026-08-10)

基于 2026-08-07 TBD 版扩展，补齐 7B 之外的压力位，标明 CPU 理论 vs H100 实测区间。

---
1. **链路**: X类 coding 数据 50k → 训 Y 7B/13B (家用口径: 从ML-for-Infra带入的预测方法迁移到post-training)
   - 7B 已 CPU gloo 验证逻辑，2×A100/H100 可跑 eval
   - 13B 需 4×H100 80GB，70B 需 8×H100 + activation ckpt（理论估算，待 H100 `max_memory_allocated` 验证）

2. **评测**: eval A 73% → B 78% (+5% abs, coding eval / verifier 可验证，单测集 TBD，基线待锁定)
   - Agentic RL 长 CoT 500→5000 tok 时 verifier 稳定性 ↑，工具链失败占比 ↑

3. **成本**: 
   - tokens/sec = **CPU理论 1.2k 占位符 → H100预期**：7B G=2 训练 3.4-5k /GPU，rollout vLLM 短CoT 40-60k / 长CoT 8-15k (decode bound)
   - GPU-hour = $3.2 占位 (待 H100 单价) 
   - $/有用 rollout = 新 PUE = GPU_cost * P95_wall / (1-fail_rate) ，失败重试占 12% 成本类比 PUE overhead
   - 13B G=4 tokens/sec ~2.3-2.9k/GPU，70B G=8 ~0.6-0.8k/GPU (seq 2k + ckpt)

4. **失败率**: 12% 占位符 → 拆分布（H100 跑 `vllm_rollout_stress_test.py` 200样本统计）：
   - 7B 短CoT 500 tok：总失败 5-8% 分布 超时25%/工具25%/VCJ20%/OOM15%/NCCL15%
   - 7B 长CoT 5000 tok：总失败 12-18% 分布 **超时40%/工具30%/VCJ15%/OOM10%/NCCL5%**（~80%墙钟经验延续）
   - 13B/70B 长CoT：15-25%，工具链占比上升
   - 重试/抗抖动: hysteresis + DAPO decoupled clip + 3次重试 + 冷却10min

5. **瓶颈**: rollout 占 80% wall-clock (Placeholder) → 长CoT 5000 tok 时 → 90% wall-clock，待NCCL计时确认 → 解法: train / vLLM rollout 集群分离调度 + nowcasting预测 eval 延迟 → 异步 eval
   - P50/P95 arrival：短CoT ~0.8s/1.2s，长CoT ~2.6s/3.2s (CPU sim，待H100 vLLM实测)

**FSDP per-block**:
- 7B bf16-mix 常驻估 ~28GB/G 峰值 ~42.5GB，可进 2×A100 80GB / 2×H100 80GB，可跑小 eval — CPU理论验证过，待H100 NCCL计时确认 fwd all-gather 18-25% / bwd reduce-scatter 22-30% + peak_mem
- 13B：26GB bf16参数，G=2 峰值~68-75GB (>80%风险)，G=4 峰值~38GB 更稳，4×H100底线
- 30B：60GB bf16，G=4 ~78GB，G=8 ~40GB，8×推荐
- 70B：140GB bf16，G=8 ~86GB 临界需act ckpt，16×舒适

**可信度控制**:
- 不把 1.2k / $3.2 / 12% 写成实测值，本文 H100 数值为预期区间，标 "待H100验证"
- 代码已备：`day-07-h100-beyond-7b/fsdp_h100_profiler_beyond7b.py` (torchrun 2/4/8) + `vllm_rollout_stress_test.py`
- 后续补：H100跑一次FSDP profiler → 补tokens/sec, GPU-hour, 失败率真实分布，补rollout P95 arrival图

**一句话scope**: "把数据中心省钱方法论翻译为RL训练的 $/useful rollout：7B已CPU验证可进2×H100跑小eval，13B/70B需4×/8×H100 + per-block FSDP + 训练/vLLM分离调度，长CoT 500→5000tok时wall-clock 80%→90%，失败率12%拆为超时40%/工具30%/VCJ15%/OOM10%/NCCL5%，待H100 NCCL计时替换1.2k/tps占位"

---
*Parent: infra_note_2026-08-07_tbd_h100.md*  
*Task: bigger-gpu-task-reminder 2026-08-10 21:50 PDT*  
*Status: CPU logic验证，H100实测待执行*
