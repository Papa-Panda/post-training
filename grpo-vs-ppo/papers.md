# Papers — GRPO vs PPO 关键引用

## 核心

- **PPO — Schulman et al. 2017**: Proximal Policy Optimization Algorithms. https://arxiv.org/abs/1707.06347 — clip surrogate, GAE, RLHF 标准。
- **DeepSeekMath — GRPO 2024**: DeepSeekMath: Pushing Limits of Mathematical Reasoning in Open LMs. https://arxiv.org/abs/2402.03300 — 首次提 GRPO，组内均值/方差当 baseline，去 critic。
- **DeepSeek-R1 2025**: Incentivizing Reasoning Capability in LLMs via RL. https://arxiv.org/abs/2501.12948 — 大规模 GRPO + rule reward + KL，验证 671B 训练可行。
- **DAPO 2025**: DAPO: An Open-Source LLM RL System at Scale. clip-higher, dynamic grouping, token-level loss. https://arxiv.org/abs/2503.14476
- **Dr.GRPO 2025**: Understanding R1-Zero-Like Training. 去掉 len/σ bias. https://arxiv.org/abs/2503.19626

## 相关 Variants

- **ReMax** — Li et al. 2023: Greedy baseline for REINFORCE, 用贪心采样当 baseline 省 value。https://arxiv.org/abs/2310.10505 — 比 GRPO 更早的无 critic 尝试。
- **RLOO** — Kool et al. 2019: REINFORCE leave-one-out baseline，group mean 思想源头。https://arxiv.org/abs/1910.12855
- **DPO** — Rafailov 2023: Direct Preference Optimization。https://arxiv.org/abs/2305.18290 — RLHF 无 RL 版，跟 GRPO 同为省 critic 路线但用偏好对。
- **VAPO** — Value Augmented? Epic推理 RL 扩展，PPO 长链价值视角。

## 按需深读

- **GAE** — Schulman 2015: High-Dimensional Continuous Control via GAE. https://arxiv.org/abs/1506.02438
- **OpenRLHF / verl** implementation: 官方代码对 PPO/GRPO loss 写法最清晰。

> 注：无雇主标识，纯开源论文。建议阅读顺序 PPO→GAE→RLOO→ReMax→GRPO→R1→DAPO/Dr.GRPO。
