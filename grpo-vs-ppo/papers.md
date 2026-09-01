# Papers & primary sources

只列本专题实际依赖的来源。论文中的结果只证明其公开实验设置，不自动外推到任意模型、reward 或集群。

## Foundations

1. **PPO** — Schulman et al., *Proximal Policy Optimization Algorithms* (2017). [arXiv](https://arxiv.org/abs/1707.06347)
   用于：clipped surrogate、multiple minibatch epochs、PPO-Clip 与 adaptive-KL 的区分。

2. **GAE** — Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2015). [arXiv](https://arxiv.org/abs/1506.02438)
   用于：TD residual 的 exponentially weighted sum、$\gamma/\lambda$ bias-variance trade-off。

3. **RLOO for RLHF** — Ahmadian et al., *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs* (2024). [arXiv](https://arxiv.org/abs/2402.14740)
   用于：leave-one-out baseline 作为 critic-free alternative。不要与 2019 combinatorial-optimization 文献混引。

## GRPO and reasoning-RL recipes

4. **DeepSeekMath** — Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (2024). [arXiv](https://arxiv.org/abs/2402.03300)
   用于：GRPO 原始 token-level clipped objective、group score baseline、outcome/process supervision、$k_3$ KL estimator。

5. **DeepSeek-R1** — Guo et al., *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025). [arXiv](https://arxiv.org/abs/2501.12948)
   用于：大规模 reasoning RL 中使用 GRPO 与 rule-based reward 的公开描述。正文 sequence shorthand 不应覆盖 DeepSeekMath 的 token-level定义。

6. **Dr.GRPO** — Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective* (2025). [arXiv](https://arxiv.org/abs/2503.20783)
   用于：response-level length bias、question-level difficulty bias，以及去 response-length/per-group-std weighting 的修正。

7. **DAPO** — Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (2025). [arXiv](https://arxiv.org/abs/2503.14476)
   用于：Clip-Higher、dynamic sampling、token-level policy-gradient loss、overlong reward shaping。其论文报告的具体 benchmark/step 数不外推为通用超参数。

8. **GSPO** — Zheng et al., *Group Sequence Policy Optimization* (2025). [arXiv](https://arxiv.org/abs/2507.18071)
   用于：length-normalized sequence ratio 与 sequence-level clipping；geometric-mean ratio 不是 exact full-sequence importance ratio。

## KL estimation

9. **KL pitfalls** — *On a few pitfalls in KL divergence gradient estimation for RL* (2025). [arXiv](https://arxiv.org/abs/2506.09477)
   用于：区分 KL **value estimator** 与 KL **gradient estimator**；样本来自当前 policy 的条件不能省略。

## Public long-horizon case

10. **GLM-5.2 official release blog** — *GLM-5.2: Built for Long-Horizon Tasks* (2026). [Official post](https://huggingface.co/blog/zai-org/glm-52-blog)
    用于：compacted sub-traces 数量/长度不均、转向 critic-based PPO、individual-rollout/token-level advantage 的公开说明。未据此保留“行业首创”、固定显存比例、榜单或价格结论。

## Reading order

PPO → GAE → RLOO → DeepSeekMath → DeepSeek-R1 → Dr.GRPO / DAPO → GSPO → KL pitfalls → long-horizon case study。

读每个实现时，先填五列表：**behavior policy、reference policy、advantage unit、ratio unit、loss aggregation**。标签相同而这五项不同，实际算法就不同。
