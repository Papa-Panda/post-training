# 青稞Talk — 固定追更区

> B站技术 talk 系列「青稞Talk」，RL / post-training / infra 选题密度极高。
> 从 2026-09-20 起固定追，每期一条 log + 一篇要点纪要。
> 服务于从 ML for Infra → Post-training / Agentic RL Infra 转型。

## 怎么用

1. 看到值得追的一期，在 `watching-log.csv` 加一行（状态 `todo`）。
2. 看完（或提炼完要点）后，用 `EPISODE_TEMPLATE.md` 在 `episodes/` 下写纪要，命名 `EP{期号}-{slug}.md`。
3. 回到 `watching-log.csv` 把状态改成 `done`，填上纪要路径。

## 追更日志（摘要）

完整日志见 `watching-log.csv`。

| 期号 | 标题 | 状态 |
|---:|---|---|
| 154 | Rethinking On-Policy Distillation of Large Language Models：现象学、机制与 Recipe | ✅ 要点已提炼 |
| 155 | SkyRL：模块化 RL 后训练框架设计，与 397B Office Work Agent 的 RL 训练实战 | ✅ 要点已提炼 |

## 建议补看（RL infra 主线相关，往期）

- 124期《大模型强化学习的Scaling Law》
- 83期《统一 SFT & RL：迈向大语言模型后训练的统一视角》
- 78期《从 LLM-RL 到 Agentic RL：如何让语言模型成为自主智能体》
- 75期《FlashRL：探讨现代 RL 框架中推理与训练的错位问题及解决方案》
- 68期《slime：专为 RL Scaling 设计的大规模 RL 训练框架及实践》
- 49期《verl 源码解读与 HybridFlow 编程范式讲解》
- 106期《GDPO：解决 GRPO 在多奖励 RL 训练中的"优势崩溃"问题》
- 101期《MiniMax M2.1：Agent 后训练经验与认知》

## 相关资源

- 系列主页：B站搜索「青稞Talk」
- SkyRL 代码：https://github.com/NovaSky-AI/SkyRL
- SkyRL-Agent 论文：https://arxiv.org/abs/2511.16108
