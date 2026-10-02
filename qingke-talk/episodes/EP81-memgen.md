# EP81 — MemGen：生成式隐式记忆，Agent Memory 的第三种可能

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP81-memgen.html

> 「我们想构造一个 agent memory system，它是一个动态的过程，能够在 agent 本身的 reasoning 过程中无缝地交织 memorization 和 memory 的过程。」——讲者对研究目标的表述

## 元信息

- 期号：81
- 标题：MemGen：生成式隐式记忆，Agent Memory 的第三种可能
- BV：BV1wMWEzPEUD
- 时长：00:59:31（讲授约 48 分钟 + Q&A 约 11 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：张贵斌（新加坡国立大学博士生；字幕中自述姓名，字幕末处主持人致谢作「张桂斌」系识别异写，以讲授开场自述为准）
- 相关论文：MemGen（讲者团队工作；英文标题口播近似为 Generative Latent Memory for Self-Evolving Agents，arXiv 号未在字幕中给出，待补录）
- 相关代码：视频中未给出仓库地址
- B站链接：https://www.bilibili.com/video/BV1wMWEzPEUD/
- 官网期号：81（预告链接不确定，未附）
- 字幕原文存档：本地 `transcripts/EP81.txt`（1434 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP81，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如 MemGen 在字幕中作「MEIJEN/MEMORGEN」，memory weaver 作「river/VIVO」，GRPO 作「GRPU」，均按上下文与领域常识校正）。讲者在实验部分给出的数字多为指图口述、未逐项报数，凡此类数字下文标注（字幕口径），精度以论文原文为准。

## 一句话总结

MemGen 把 agent 的经验不写进 prompt、也不训进主模型参数，而是外挂两个只训 LoRA 的模块：一个 trigger 在解码的语义间隔处判断「此刻需不需要记忆」，一个 weaver 据当前上下文现场生成若干 latent token 插入 KV cache；主模型全程冻结，却能在多类任务上超过对主模型做全参数 GRPO 的效果，且学到的记忆可跨域迁移、抗持续学习遗忘。

## 核心

### 记忆的分类学：MemGen 站在第三格

讲者先把 agent memory 的版图切开。他区分的不是技术而是**用途**：

- **Chatbot / personalized memory**：记用户偏好并随时间增删改（如「100 天前说喜欢吃苹果、50 天后说不喜欢」），代表工作是 MemGPT、Mem0 一类。本文不研究这一格。
- **Self-improving memory**（本文的 scope）：让 agent 利用过往任务轨迹/经验提升新任务（含域外任务）的解决能力。这一格内部再分三条实现路线：
  - **Retrieval-based memory**：经验以文本/结构化形式存外部库（向量库、图库），新问题来了做相似度检索再塞进 prompt。讲者团队前作 G-Memory 即属此类（三层图：query 图、interaction 图、insight 图）。判词：本质是 context engineering，性能有上限。
  - **Parametric memory**：把经验直接训进主模型参数（SFT、GRPO 都算），或训一个独立 critic 模型把反思能力参数化（讲者举 ICLR 2024 的早期工作为例）。判词：动主模型参数，灾难性遗忘难以避免。
  - **Latent memory**（第三种可能）：以 latent embedding / hidden states 作记忆载体，是更「machine-native」的表示。先行工作 MemoryLLM、Memory³、M+ 维护 latent 记忆池并按相似度检索回填 KV cache——但仍是「新问题来了先检索一次」的静态形态。

讲者的不满在于：人脑记忆是与推理交织的连续流（continuous flow of thought），边说边联想；而现有三类都是在推理之外另设一道检索工序。研究问题因此定为：**能否让记忆的召唤与编织发生在解码过程内部、与推理无缝交织。**

### 架构：冻结的 reasoner + trigger + weaver

系统三件套，主模型（reasoner）全程冻结：

1. **Reasoner**：负责 thinking 与 action 的 agent 本体，参数不动。
2. **Memory trigger**：输入当前已解码的全部 hidden states（即一个 $J \times D$ 的 embedding 矩阵， $J$ 为已生成 token 数、 $D$ 为 embedding 维度），输出当前时刻的记忆调用概率，经 Bernoulli 采样得到 0/1 决策。
3. **Memory weaver**：被唤醒后读同一份上下文，生成一段长度为 $K$ 的 latent token 序列（实践中 $K$ 约为 4 或 8），直接写入 reasoner 的 KV cache。这段记忆不解码成自然语言、不打断正常解码，reasoner 在后续生成中「读到」它。

两个关键的工程取舍：

- **只在语义间隔处判断**。逐 token 调 trigger 代价太高且干扰句内解码；MemGen 只在逗号、句号、换行符处做调用判断（讲者引 Anthropic 可解释性工作的类似结论背书：句中干预更有效且不扰动句内生成）。
- **实现上是两个 LoRA**。论文实践中 trigger 与 weaver 都是附着在冻结主模型上的 LoRA adapter；讲者明确说这不是唯一形态——也可以是独立小模型（如 reasoner 为 7B 时用 1.5B 小模型全量微调）、其他 PEFT 方法，代价最高的选项是复制一份主模型专存记忆（太贵，违背初衷）。记忆的形态是一个参数化网络，具体用 LoRA 只是当前实现。

### 训练：weaver 学「记什么」，trigger 学「何时记」

两个模块分开训，顺序是**先训 weaver、固定后、再训 trigger**：

- **Weaver 的 loss 几乎任意**（SFT / GRPO / DPO 皆可）：核心约束只有一条——不更新 reasoner 参数，经验全部注入 weaver。实验对比了 MemGen-SFT 与 MemGen-GRPO 两个版本，后者更好（reward 直接对齐答对与否）。
- **Trigger 用 GRPO 训**：固定 weaver 后，trigger 对每个问题采样 4 或 8 条轨迹，奖励 = 轨迹的稀疏 0/1 对错信号 **减去调用次数惩罚**。第二个分量是刻意的：每个标点都插记忆会拖慢解码、甚至起负作用，trigger 必须学会只在关键处召唤。讲者强调这套训练保证了生成出的 latent embedding「不干扰原解码、且能提点」——不是靠结构先验，而是 loss 直接朝答案正确率优化出来的。

### 实验：四个研究问题

设置（字幕口径）：基座模型 1.5B / 3B / 8B；约 8 个数据集，一半是 ReAct 式多轮 agent 任务（TriviaQA、PopQA、ALFWorld），一半是单次生成任务（GPQA、MATH、代码等）；基线覆盖原模型与 CoT、参数化路线（SFT、GRPO、REINFORCE++ 等）、检索路线（MemoryBank、ExpeL、Agent Workflow Memory）与隐式推理路线（Soft CoT 等）。

- **RQ1 能否超过另两类记忆**：参数化方法整体强于检索方法，但全参数 GRPO 的代价是动主模型。MemGen-GRPO 只微调两份 LoRA，在 ALFWorld 上超过全参数 GRPO（字幕口径：检索基线 Agent Workflow Memory 约 32–40、GRPO 约 55，MemGen-GRPO 在其上再高出不止一两个点）。讲者的定性结论是：这是一种「更温和、但性能不输全量微调」的干预方式。
- **RQ2 跨域泛化**：在 ALFWorld 上攒的经验迁到 TriviaQA、ScienceWorld、Fever 上仍持续涨点（ScienceWorld 约 2 倍提升，字幕口径）；而 SFT 与 MemoryBank 在迁移时掉点。机制证据来自 trigger 的激活计数：在训练域 GSM8K 上平均激活约 86 次，到 GPQA 约减半，到完全异域的代码任务约 20 次（约为训练域的 1/4）——trigger 学会了「别域经验没用时少召唤甚至不召唤」，这是不掉点的原因。
- **RQ3 持续学习**：四个数据集轮流训练、每轮后测 GPQA。SFT 路线先涨后崩（11 → 16 → 17 → 13 → 2.53，字幕口径，最终原能力基本丧失）；检索路线聊胜于无；MemGen-SFT 维持在约 20（11 → 19 → 20 → 21 → 约 20）。讲者同时坦白：近来多篇工作表明 RL 本身抗遗忘就远强于 SFT，本实验只比了 SFT 版本，RL 版本未及做。
- **RQ4 这些 latent token 到底是什么**：把 latent token 强行按词表最近邻解码出来，人类完全不可读（混着德语词、keyword 一类碎片）。但三层分析显示其内部有结构：
  - 不同数据集产生的 latent token 经 t-SNE 降维后按域聚类（数学类彼此交织、代码类彼此靠近）；
  - 同一数据集内部再聚类，各簇有共享「后缀」模式的倾向（粗粒度统计约 50%–60% 同簇共享同一结尾模式）；
  - **事后干预实验**（借鉴 failure taxonomy，把错误分为 planning failure、tool response/parsing error、demand misunderstanding、answer 格式错误等类）：消融掉 cluster 2，planning 类失败显著上升（约 6 例 → 17–18 例）；消融 cluster 3，工具响应/解析/格式类错误上升；cluster 1、4 与 think-action 一致性、需求理解类错误相关。即无外部监督下，记忆自发分化出近似 planning memory / procedural memory / working memory 的功能分工。讲者强调这不是泾渭分明的划分，只是偏好倾向。

### 边界与未决（多为 Q&A 中的坦白）

- **跨基座不兼容**：latent embedding 绑定基座模型的表示空间，换 reasoner 必须重训一套 trigger/weaver；想通用需加 projection head，效果无保证（讲者称用 SLM 训练 + 投影头已有跑通的实验）。
- **离线记忆**：当前 weaver 不能实时写入新经验（来一条轨迹更新一次做不到），online 注入是其团队在探索的方向。
- **推理开销**：最坏情况每个 token 需 reasoner、trigger、weaver 三次推理；但因 trigger 只在标点处判断，实测额外 inference delay 不超过 5%（论文附录口径，字幕转述）。
- **讲者的定位**：trigger/weaver 本身当然也是 parametric memory，只是记忆的承载与传递形式是 latent 的；trigger 与 weaver 理论上也可合并为一个网络训练。至于为什么不用 special token 让 reasoner 自己发记忆请求——那需要动 reasoner，违背「不碰主模型推理能力」的设计出发点。

## 关键数字

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 基座模型规模 | — | 1.5B / 3B / 8B | 字幕 |
| 每次插入的 latent token 数 $K$ | — | 约 4 或 8 | 字幕 |
| Trigger 训练轨迹采样数 | — | 每题 4 或 8 条 | 字幕 |
| ALFWorld 成功率 | Agent Workflow Memory 约 32–40；全参数 GRPO 约 55 | MemGen-GRPO 超过 GRPO（高出不止一两个点） | 字幕（指图口述） |
| 跨域迁移（ScienceWorld） | 在 ALFWorld 训练 | 约 2 倍提升 | 字幕 |
| Trigger 平均激活次数 | 训练域 GSM8K 约 86 次 | GPQA 约减半；代码任务约 20 次（约 1/4） | 字幕 |
| 持续学习 GPQA 得分轨迹 | SFT 路线：11 → 16 → 17 → 13 → 2.53 | MemGen-SFT：11 → 19 → 20 → 21 → 约 20 | 字幕（指图口述） |
| 记忆簇消融 | 原模型 planning 类失败约 6 例 | 移除 cluster 2 后约 17–18 例 | 字幕 |
| 额外推理延迟 | 无 MemGen | 不超过 5% | 字幕（转述论文附录） |

## 可迁移

- **记忆即外挂 LoRA、主干冻结**：这是一个把「经验沉淀」与「基座能力」解耦的工程范式——对 RL infra 的意味是：经验库可以独立迭代、独立回滚、按任务域热插拔，而不需要为攒经验重训/微调交付模型本身；同一基座挂不同 weaver 即可分化行为。
- **「何时调用」的决策要显式训练并计费**：trigger 的 GRPO 奖励里直接扣调用次数，与推理成本同账。做 agent 记忆/工具调用门控时，可借鉴这条：门控不是规则阈值，而是一个被任务奖励和调用成本联合训练的小策略。
- **语义边界插入**：只在标点/换行处干预解码，是低成本且低干扰的注入点选择；对需要在生成中途注入信息（检索结果、校验反馈、隐式记忆）的系统都可复用。
- **可解释性方法**：对不可读的 latent 表示，用「按簇消融 × 人工 failure taxonomy 计数」验证功能分工，比强行解码最近邻 token 有信息量得多。

## 疑问 / 下一步

- Weaver 的 online 写入未实现：经验只能离线批量训入。若要让 agent 在部署中持续攒经验，增量更新 weaver 时如何避免它自身遗忘、如何与 trigger 的调用分布漂移共适应，讲者未给方案。
- 跨基座迁移只有 projection head 一条无保证的路线；不同规模/家族基座间 latent 记忆能否蒸馏复用，尚无证据。
- RQ3 只比了 SFT 版本的遗忘；MemGen-GRPO 版本与「RL 本身抗遗忘」的结论叠加后增益还剩多少，未验证。
- 记忆簇的功能分工是事后归因（消融 + 错误计数），因果强度有限；且簇内约一半成员并不遵循共享后缀模式，其余结构未解释。

## 原文金句（1-2句）

> 「人脑的记忆其实是 reasoning 和 memory 彼此交织的，在 reason 的过程中可能会同步地联想起 memory。」——讲者以此立论现有记忆机制与目标形态的差距

> 「我们把之前的经验性知识全部注入到 weaver 里面去，而当一个 context 来了之后，它可能就能够回忆起之前相关的经验。」——讲者解释 generative 的含义：记忆按当前上下文现场定制生成

> 「在没有任何外部监督的情况下，MemGen 进化出了这样不同的 memory 层次和 memory 功能，我们认为这是一个很有意思的现象。」——讲者对 RQ4 簇消融结果的判词
