# Day 4 - Done (补)

Date: 2026-08-05 08:49 PDT (补 08-06)
Status: done (离线笔记，待真机验证不影响 infra 链路)

## 核心
- RLHF/PPO: 训 RM → PPO 优化，重、稳难
- DPO: 偏好对直接推策略，稳、便宜，cold-start 友好
- GRPO: 组内 rollout 相对打分，省 critic，适合 Agentic 长 CoT + 工具

## infra 问题提炼
- 博客: Dec 2024 OpenAI RLHF infra
- Q: 长 rollout 500→5000 tokens 为啥要拆 vLLM 集群和训练集群？
- A: rollout 成瓶颈 80%墙钟，失败率指数涨，需独立 autoscaling / 重试 / 机型，类比 nowcasting 预测波动负载

## 金融映射
- RM = bond price 有噪
- GRPO group = comparable basket 去噪
- 相对打分 = OAS 校准压方差

Code: 笔记为主，无需跑通，概念对齐即可。
