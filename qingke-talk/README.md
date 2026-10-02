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
| 59 | [大模型推理强化学习中的熵机制](episodes/EP59-entropy-mechanism-rl.md) | ✅ 要点已提炼 |
| 62 | [ProRL：延长强化学习训练框架，拓展大语言模型的推理边界](episodes/EP62-prorl.md) | ✅ 要点已提炼 |
| 67 | [大模型训练流水线并行四部曲：吞吐、内存、负载均衡与线性扩展](episodes/EP67-pipeline-parallelism.md) | ✅ 要点已提炼 |
| 69 | [GSPO：大规模强化学习训练算法，迈向持续拓展的语言模型强化学习](episodes/EP69-gspo.md) | ✅ 要点已提炼 |
| 71 | [RLPR：基于参考概率奖励的强化学习，推广 RLVR 到通用领域推理问题](episodes/EP71-rlpr.md) | ✅ 要点已提炼 |
| 74 | [ROLL：面向 Agentic 场景的生产级大规模强化学习训练框架](episodes/EP74-roll.md) | ✅ 要点已提炼 |
| 76 | [NeMo RL：让大规模 MoE 模型权重 Refit 加速 10 倍](episodes/EP76-nemo-rl.md) | ✅ 要点已提炼 |
| 80 | [RL for LRMs：探讨面向推理模型的 RL 最新研究](episodes/EP80-rl-for-lrms.md) | ✅ 要点已提炼 |
| 92 | [RLinf：面向具身智能的"渲训推一体化"开源强化训练框架](episodes/EP92-rlinf.md) | ✅ 要点已提炼 |
| 101 | MiniMax M2.1：Agent 后训练经验与认知 | ✅ 要点已提炼 |
| 102 | 从 TRPO 到 SAPO：大模型 RL 算法演进 | ✅ 要点已提炼 |
| 111 | RLinf-USER：面向现实世界机器人在线策略学习的统一且可扩展系统 | ✅ 要点已提炼 |
| 123 | DeepSeek V4 模型在 SGLang 中的系统级优化与全栈适配 | ✅ 要点已提炼 |
| 125 | 重探 On-Policy Distillation（OPD）：三类典型失败以及修复路径 | ✅ 要点已提炼 |
| 126 | STream3R & 4RC：面向几何与运动理解的流式前馈 3D/4D 重建 | ✅ 要点已提炼 |
| 128 | MinT：面向百万级 LoRA 策略的训练与推理基础设施 | ✅ 要点已提炼 |
| 133 | VeRL-Omni：基于VeRL及vLLM-Omni构建的面向多模态生成模型的开源 RL 后训练框架 | ✅ 要点已提炼 |
| 140 | DRIFT：在线自进化后训练框架 | ✅ 要点已提炼 |
| 145 | JitRL——无需梯度更新的即时强化学习 | ✅ 要点已提炼 |
| 129 | [从 ARPO，到 AEPO，再到 Agent-World：探索通用智能体训练的可行路径](episodes/EP129-arpo-aepo-agent-world.md) | ✅ 要点已提炼 |
| 152 | 如何在真实 Coding Agent Harness 中接入 RL？ | ✅ 要点已提炼 |
| 153 | [EnvHarness，Agent 和环境如何"左脚踩右脚"实现自进化](episodes/EP153-envharness.md) | ✅ 要点已提炼 |
| 154 | Rethinking On-Policy Distillation of Large Language Models：现象学、机制与 Recipe | ✅ 要点已提炼 |
| 155 | SkyRL：模块化 RL 后训练框架设计，与 397B Office Work Agent 的 RL 训练实战 | ✅ 要点已提炼 |
| 156 | 聊聊自我改进、递归自我改进（RSI），以及与自博弈的关系 | ✅ 要点已提炼 |
| B106 | Intern-S1：科学多模态基础模型（B站合集期号） | ✅ 要点已提炼 |
| B124 | 拒绝阈值化：迈向共生关系的新型人机信任（B站合集期号） | ✅ 要点已提炼 |

## 编号体系说明（重要）

官网预告编号与 B站合集编号是两套体系，约 EP93 之后开始分叉：同一期号在两边可能是不同 talk，也有同一 talk 改标题的情况。全量裁决结果见 `episode-index.csv` 的备注列：

- 同一 talk 100 条视频对应 99 个官网期号（EP78 为重复上传，双 BV 并存）；
- 56 期官网有预告但合集无视频，记为 official-only；
- 52 条视频为合集独有，期号记为 `B<n>`。
- 例：官网 106 期 GDPO、124 期 Scaling Law 均无合集视频；合集的 106 是 Intern-S1、124 是人机信任，两场另有纪要（B106/B124）。
- 2026-10-02 复核发现：RollArt、重要性采样与熵调控、OPD 反直觉、MinT-OPD、Harness Engineering 五条 B站独有期此前登记的 BV 实际属于官网对应期视频，其真实 BV 待重新枚举合集确认（索引备注列已标注）。

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
