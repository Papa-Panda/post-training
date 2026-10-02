# 青稞Talk — 固定追更区

> B站技术 talk 系列「青稞Talk」，RL / post-training / infra 选题密度极高。
> 从 2026-09-20 起固定追，每期一条 log + 一篇要点纪要。
> 服务于从 ML for Infra → Post-training / Agentic RL Infra 转型。

## 怎么用

1. 看到值得追的一期，在 `watching-log.csv` 加一行（状态 `todo`）。
2. 看完（或提炼完要点）后，用 `EPISODE_TEMPLATE.md` 在 `episodes/` 下写纪要，命名 `EP{期号}-{slug}.md`。
3. 回到 `watching-log.csv` 把状态改成 `done`，填上纪要路径。

## 追更日志（摘要）

完整日志见 `watching-log.csv`（全量 208 行：官网 156 期 + B站合集独有 52 条），全量索引见 `episode-index.csv`。每期纪要在 `episodes/` 下，同名 `.html` 为阅读版（Pages 链接写在各纪要文首）。原始字幕存 `transcripts/`。

| 期号 | 标题 | 状态 |
|---:|---|---|
| 49 | verl 源码解读与 HybridFlow 编程范式讲解 | ✅ 要点已提炼 |
| 68 | slime：专为 RL Scaling 设计的大规模 RL 训练框架及实践 | ✅ 要点已提炼 |
| 75 | FlashRL：探讨现代 RL 框架中推理与训练的错位问题及解决方案 | ✅ 要点已提炼 |
| 78 | 从 LLM-RL 到 Agentic RL：如何让语言模型成为自主智能体 | ✅ 要点已提炼 |
| 83 | 统一 SFT & RL：迈向大语言模型后训练的统一视角 | ✅ 要点已提炼 |
| 101 | MiniMax M2.1：Agent 后训练经验与认知 | ✅ 要点已提炼 |
| 125 | 重探 On-Policy Distillation（OPD）：三类典型失败以及修复路径 | ✅ 要点已提炼 |
| 126 | STream3R & 4RC：面向几何与运动理解的流式前馈 3D/4D 重建 | ✅ 要点已提炼 |
| 154 | Rethinking On-Policy Distillation of Large Language Models：现象学、机制与 Recipe | ✅ 要点已提炼 |
| 155 | SkyRL：模块化 RL 后训练框架设计，与 397B Office Work Agent 的 RL 训练实战 | ✅ 要点已提炼 |
| B106 | Intern-S1：科学多模态基础模型（B站合集期号） | ✅ 要点已提炼 |
| B124 | 拒绝阈值化：迈向共生关系的新型人机信任（B站合集期号） | ✅ 要点已提炼 |

## 编号体系说明（重要）

官网预告编号与 B站合集编号是两套体系，约 EP93 之后开始分叉：同一期号在两边可能是不同 talk，也有同一 talk 改标题的情况。全量裁决结果见 `episode-index.csv` 的备注列：

- 同一 talk 100 条视频对应 99 个官网期号（EP78 为重复上传，双 BV 并存）；
- 56 期官网有预告但合集无视频，记为 official-only；
- 52 条视频为合集独有，期号记为 `B<n>`。
- 例：官网 106 期 GDPO、124 期 Scaling Law 均无合集视频；合集的 106 是 Intern-S1、124 是人机信任，两场另有纪要（B106/B124）。

## 建议补看（RL infra 主线相关，往期）

- 124期《大模型强化学习的Scaling Law》（官网期号；合集无此视频，纪要待补）
- 83期《统一 SFT & RL：迈向大语言模型后训练的统一视角》✅
- 78期《从 LLM-RL 到 Agentic RL：如何让语言模型成为自主智能体》✅
- 75期《FlashRL：探讨现代 RL 框架中推理与训练的错位问题及解决方案》✅
- 68期《slime：专为 RL Scaling 设计的大规模 RL 训练框架及实践》✅
- 49期《verl 源码解读与 HybridFlow 编程范式讲解》✅
- 106期《GDPO：解决 GRPO 在多奖励 RL 训练中的"优势崩溃"问题》（官网期号；合集无此视频，纪要待补）
- 101期《MiniMax M2.1：Agent 后训练经验与认知》✅

## 相关资源

- 系列主页：B站搜索「青稞Talk」
- SkyRL 代码：https://github.com/NovaSky-AI/SkyRL
- SkyRL-Agent 论文：https://arxiv.org/abs/2511.16108
