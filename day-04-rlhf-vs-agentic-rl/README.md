# Day 4 - 2026-08-05 - Agentic RL vs RLHF

**Learning Goal**: 能用一句话区分 RLHF / DPO / GRPO

**工作关联**: 金融里 reward 不确定性定价思路

## 一句话区分
- **RLHF (PPO)** = 学一个 reward 模型，再用 PPO 去优化它。经典但重，贵在要训 RM 而且 PPO 不稳。
- **DPO** = 跳过 RM，直接用偏好对去推最优策略的闭式解。把 RLHF 的两步压成一步 SFT，稳、便宜，适合小规模 cold-start。
- **GRPO** = Group 相 对 策略 优化，不要 value 模型了，一组 rollout 里互相打分当 baseline。省显存，适合 Agentic RL 里 response 长、工具多的场景。

一句话：
> RLHF 是学 RM 再 PPO，DPO 是用偏好直接 SFT，GRPO 是组内相对省掉 critic，专给长链条 agent 用。

## Dec 2024 OpenAI RLHF infra blog提炼的1个问题
来源: OpenAI Dec 2024 post-training infra (o1 发布前后)

> **问题**: 当 rollout 长度从 500 tokens 翻到 5000 tokens (长思维链 + 工具)，为什么他们要把 vLLM rollout 集群和训练集群分开，甚至不同机型/调度？

拆解:
- RM / verifier 成了瓶颈，不是 train
- rollout 失败率随长度指数涨，80% 墙钟耗在 rollout，重试策略决定 SLO
- 和你做 nowcasting autoscaling 一样：rollout 是波动负载，训练是稳定负载，放一起会互相踩

这就是 Agentic RL vs RLHF 的 infra 本质区别：RLHF 是短 response，train 重；Agentic RL 是长 trajectory + 工具，rollout 重。

## 金融类比
Reward 不确定性定价 = OAS 思路
- RM 输出一个标量 reward，就像 bond 价格，里面有噪音
- GRPO 用 group baseline，就像用一组 comparable bonds 来估 OAS，去掉市场噪音
- 多 rollout 取相对值，相当于对 reward 做校准，压低方差
