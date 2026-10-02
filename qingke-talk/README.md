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
| 77 | [Theory of Agent: From Definition, to Behavior and Objective](episodes/EP77-theory-of-agent.md) | ✅ 要点已提炼 |
| 78 | 从 LLM-RL 到 Agentic RL：如何让语言模型成为自主智能体 | ✅ 要点已提炼 |
| 83 | 统一 SFT & RL：迈向大语言模型后训练的统一视角 | ✅ 要点已提炼 |
| 1 | [SceneTex：高质量三维室内场景纹理图生成](episodes/EP01-scenetex.md) | ✅ 要点已提炼 |
| 2 | [ChatDev：大语言模型驱动的多智能体协作与演化](episodes/EP02-chatdev.md) | ✅ 要点已提炼 |
| 3 | [从 3D LLM 到 MultiPLY，3D 具身基础模型的构建](episodes/EP03-3d-llm-multiply.md) | ✅ 要点已提炼 |
| 4 | [Mini-Gemini：挖掘多模态视觉语言大模型的潜力](episodes/EP04-mini-gemini.md) | ✅ 要点已提炼 |
| 5 | [3D-VLA：构建生成式三维具身世界模型](episodes/EP05-3d-vla.md) | ✅ 要点已提炼 |
| 6 | [实时渲染 3DGS 中的反走样及逆渲染应用](episodes/EP06-3dgs.md) | ✅ 要点已提炼 |
| 7 | [MixEval：混合评测数据集来拟合大语言模型的人类评估](episodes/EP07-mixeval.md) | ✅ 要点已提炼 |
| 8 | [VideoBooth：文本和图像提示共同驱动的视频生成](episodes/EP08-videobooth.md) | ✅ 要点已提炼 |
| 9 | [具身多模态大模型的视觉表征预训练研究](episodes/EP09-embodied-multimodal-pretraining.md) | ✅ 要点已提炼 |
| 11 | [LLaMA Pro：扩展 Transformer 块优化的大型语言模型继续预训练](episodes/EP11-llama-pro.md) | ✅ 要点已提炼 |
| 12 | [VillagerAgent：减少幻觉、提高任务分解效率的多智能协作体框架](episodes/EP12-villager-agent.md) | ✅ 要点已提炼 |
| 10 | [PiSSA：收敛快、误差小的大模型参数高效微调方法](episodes/EP10-pissa.md) | ✅ 要点已提炼 |
| 13 | [LLaMA Factory：从预训练到RLHF，大模型高效训练框架](episodes/EP13-llama-factory.md) | ✅ 要点已提炼 |
| 33 | [XGrammar：高效实现 LLM 灵活且可移植的结构化生成](episodes/EP33-xgrammar.md) | ✅ 要点已提炼 |
| 42 | [COAT：显存高效的 FP8 训练，实现高效深度学习](episodes/EP42-coat.md) | ✅ 要点已提炼 |
| 41 | [PC-Agent：面向复杂 PC 任务的多模态智能体框架](episodes/EP41-pc-agent.md) | ✅ 要点已提炼 |
| 44 | [InferCept、Preble&Cognify：面向下一代 AI Agent 工作流系统的构建](episodes/EP44-infercept-preble-cognify.md) | ✅ 要点已提炼 |
| 38 | [Satori：通过训练 LLM 做自回归搜索来增强推理能力](episodes/EP38-satori.md) | ✅ 要点已提炼 |
| 39 | [PRIME: 结合隐式过程奖励的强化学习](episodes/EP39-prime.md) | ✅ 要点已提炼 |
| 45 | [B-STaR & SimpleRL-Zoo：通过强化学习自我提升推理性能和效率](episodes/EP45-bstar-simplerl-zoo.md) | ✅ 要点已提炼 |
| 46 | [从 TinyZero 到 APR：语言模型推理能力的探索与自适应并行化](episodes/EP46-tinyzero-apr.md) | ✅ 要点已提炼 |
| 48 | [从 TTS 到 TTRL：无标签数据强化学习探索与展望](episodes/EP48-tts-ttrl.md) | ✅ 要点已提炼 |
| 53 | [Chain-of-Model（模型链）：引入因果建模的大模型 Scaling 结构](episodes/EP53-chain-of-model.md) | ✅ 要点已提炼 |
| 57 | [Fast-dLLM：无需重训的扩散大语言模型推理加速](episodes/EP57-fast-dllm.md) | ✅ 要点已提炼 |
| 58 | [Virtual Community 虚拟社区：面向人、机器人与社会的开放世界模拟平台](episodes/EP58-virtual-community.md) | ✅ 要点已提炼 |
| 61 | [GUI-Reflection：让多模态 GUI 智能体获得反思纠错能力的训练框架](episodes/EP61-gui-reflection.md) | ✅ 要点已提炼 |
| 59 | [大模型推理强化学习中的熵机制](episodes/EP59-entropy-mechanism-rl.md) | ✅ 要点已提炼 |
| 60 | [Satori-SWE：用 Evolutionary Test-Time Scaling 让小语言模型做 SWE](episodes/EP60-satori-swe.md) | ✅ 要点已提炼 |
| 62 | [ProRL：延长强化学习训练框架，拓展大语言模型的推理边界](episodes/EP62-prorl.md) | ✅ 要点已提炼 |
| 66 | [SIMoE：稀疏插值混合专家，大模型升级再造的自动化专家发现框架](episodes/EP66-simoe.md) | ✅ 要点已提炼 |
| 67 | [大模型训练流水线并行四部曲：吞吐、内存、负载均衡与线性扩展](episodes/EP67-pipeline-parallelism.md) | ✅ 要点已提炼 |
| 69 | [GSPO：大规模强化学习训练算法，迈向持续拓展的语言模型强化学习](episodes/EP69-gspo.md) | ✅ 要点已提炼 |
| 71 | [RLPR：基于参考概率奖励的强化学习，推广 RLVR 到通用领域推理问题](episodes/EP71-rlpr.md) | ✅ 要点已提炼 |
| 74 | [ROLL：面向 Agentic 场景的生产级大规模强化学习训练框架](episodes/EP74-roll.md) | ✅ 要点已提炼 |
| 76 | [NeMo RL：让大规模 MoE 模型权重 Refit 加速 10 倍](episodes/EP76-nemo-rl.md) | ✅ 要点已提炼 |
| 79 | [UserRL & UserBench「知人者智」：以用户为中心的智能体交互与训练](episodes/EP79-userrl.md) | ✅ 要点已提炼 |
| 81 | [MemGen：生成式隐式记忆，Agent Memory 的第三种可能](episodes/EP81-memgen.md) | ✅ 要点已提炼 |
| 82 | [OpenCUA：用于构建 Computer-Use Agent 的开源框架](episodes/EP82-opencua.md) | ✅ 要点已提炼 |
| 80 | [RL for LRMs：探讨面向推理模型的 RL 最新研究](episodes/EP80-rl-for-lrms.md) | ✅ 要点已提炼 |
| 84 | [SimpleVLA-RL：简单可拓展的VLA强化学习训练](episodes/EP84-simplevla-rl.md) | ✅ 要点已提炼 |
| 87 | [QeRL：量化技术增强强化学习 Reasoning 探索](episodes/EP87-qerl.md) | ✅ 要点已提炼 |
| 86 | [从 DeepSeek-OCR 到 Glyph：深入理解图像-文本压缩技术](episodes/EP86-deepseek-ocr-glyph.md) | ✅ 要点已提炼 |
| 89 | [Generative RLHF-V：面向多模态 RLHF 的人类意图对齐框架](episodes/EP89-generative-rlhf-v.md) | ✅ 要点已提炼 |
| 91 | [KTransformers，在大模型微调与推理中的系统化实践](episodes/EP91-ktransformers.md) | ✅ 要点已提炼 |
| 95 | [从 π_0 到 π_RL：面向流匹配 VLA 的强化学习后训练框架](episodes/EP95-pi-rl.md) | ✅ 要点已提炼 |
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
