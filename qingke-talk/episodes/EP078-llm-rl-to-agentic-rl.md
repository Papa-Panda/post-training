# EP078 — 从 LLM-RL 到 Agentic RL：如何让语言模型成为自主智能体

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP078-llm-rl-to-agentic-rl.html

> 「…agentic environment、agentic training，还有 trustworthiness，是我们认为在未来的 Agentic RL 里面值得关注的三个事情。」——讲者总结（字幕 56:41–56:52）

## 元信息

- 期号：78
- 标题：从 LLM-RL 到 Agentic RL：如何让语言模型成为自主智能体
- BV：BV1sxWgzUEZo（https://www.bilibili.com/video/BV1sxWgzUEZo/；另有同题重复上传 BV1sBsFz9EVw）
- 时长：01:10:35（4235 秒；讲授约 57 分钟 + Q&A 约 14 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：张桂彬（Guibin Zhang，新加坡国立大学计算学院博士生，导师颜水成 Shuicheng Yan；G-Designer、G-Memory、MaAS 等工作作者；字幕中主持人介绍作「张贵斌」，以论文作者页为准）
- 相关论文：The Landscape of Agentic Reinforcement Learning for LLMs: A Survey（Guibin Zhang、Hejia Geng 等，arXiv:2509.02547，2025-09；后发表于 TMLR），https://arxiv.org/abs/2509.02547
- 相关代码：综述配套汇编（开源环境 / benchmark / 框架清单）见论文项目页
- 字幕原文存档：本地 `transcripts/EP78.txt`（1808 条，带时间戳）

> 📝 提炼方式说明：本纪要原为论文还原版，现据 B站 AI 字幕原文（已存档）升级为字幕实录版：正文以讲授口径为准，个别专名有识别误差（如 G-Memory 字幕作「g memory」、 BrowseComp 作「bross Comp」），均以论文与公开资料为准。旧版中仅见于综述、talk 未展开的内容（如 DeepCoder-14B 的 LiveCodeBench 口径）保留并标注（论文补充）。

> 关联：讲者是本仓库多期议题的"上游综述"——EP80 的记忆/过程奖励讨论、EP129 的 ARPO/AEPO、EP153 的环境自进化，都可在这张版图里定位。

## 一句话总结

这期是 Agentic RL 的"概念奠基课"：传统 LLM-RL 本质是**单轮、无环境、完全可观测的退化决策问题**（给 prompt、出答案、拿奖励），而 agent 要在动态环境里多步行动、只能看到局部观测，必须升级为 **POMDP** 形式化。讲者沿六个维度（奖励、转移、动作空间、目标、算法、环境）拆清两者的形式化差异，再用"能力轴 × 应用域轴"双轴分类法把 RL 如何增强 planning、tool use、memory、self-improvement、reasoning、perception 六类能力组织成一张版图，并给出三个未来判断：agentic environment、agentic training、trustworthiness。

## 核心

### 引入：先划清"Agentic RL"到底指什么

讲者开场先处理了命名争议：传统 RL 社区会觉得"RL 本来就是训练 agent 的"，但 2025 年的语境下，Agentic RL 特指"帮助 LLM 获得 agent 特质的 RL 算法"。综述的 focus 只有一条——**RL 如何在动态环境中增强 LLM agent 的能力**；明确排除三类：价值对齐型 RLHF、传统 RL 算法本身、在静态 benchmark（如 GSM8K、MATH500）上刷分。判据就是两个词：dynamic environment + agentic 特质。

### 方法一：六个维度的形式化对比

讲者逐项对比偏好式 RL 微调（PBRFT）与 Agentic RL（字幕口径）：

1. **轮次结构**：PBRFT 的轮次 $T = 1$ ，给指令、生成、由外部 reward model 打分、更新策略，一次结束；Agentic RL 训练的是 ReAct 式 agent，多轮调用工具、自主决定继续或终止。
2. **环境**：PBRFT 没有"环境"概念（解一道数学题就是输出一次文本）；Agentic RL 的核心一环是动态环境——每次 action 都改变环境本身（ALFWorld、ScienceWorld 这类文本世界，WebShop、WebArena 这类网页环境，GAIA、deep research 这类真实世界环境）。这正是它被称为 **POMDP**（部分可观测马尔可夫决策过程）的原因。
3. **动作空间**：不只是自然语言输出，还包括工具调用、API/MCP、bash terminal、代码编译器、memory 管理等结构化动作。
4. **转移不确定性**：PBRFT 的下一步状态在行动确定后即确定；Agentic RL 中环境返回带不确定性——同一个 web search 查询在 2024 年与 2025 年返回完全不同的结果。
5. **学习目标**：Agentic RL 是长程多轮过程，需要显式设计折扣因子 $\gamma$ ，近期收益与远期收益区别定价。
6. **算法谱系**：从 PPO/DPO 家族转向 rule-based RL（GRPO 及其变体、REINFORCE++、GSPO 等）；讲者提醒变体多到"三四页列不完"，综述只收相对代表性的。

在此之上，讲者借 Lilian Weng 2023 年的 agent 定义并加以润色，给出能力分解：agent = LLM 本体 + reasoning/planning + memory + perception + tool use + self-improvement——后续讲授就沿这六个能力逐个展开"RL 怎么增强它"。

### 方法二：RL 如何增强六类能力（讲授主线）

**Planning** 分两路：RL as external guidance（参数不动，用 MCTS、遗传算法等在推理期引导搜索，如 Tree-of-Thoughts 式分叉选择）与 RL as internal driver（直接用 RL 训练模型自身的规划能力，讲者称这一方向"相对较新、工作还不多"，举了腾讯近期用 RL 训练 agent planning 的工作为例）。

**Tool use** 分三个阶段（讲者的历史叙事）：① ReAct 式 prompt 工程 → SFT 内化（AgentTuning、FireAct、Toolformer 一系；轨迹数据从 AgentTuning 的 2–3 千条长到 AgentBank 的 5 万条）；② tool-integrated RL（2025 年 3–4 月涌现三四十篇：在 GRPO 训练中让模型自由决定何时唤醒工具、填什么参数；已内化为 QwQ、Kimi K2、GLM-4.5、LongCat 等模型的基础能力，多模态侧如 DeepEyes 的图像工具调用）；③ multi-turn tool RL（讲者特别澄清：一次 rollout 内多次工具调用不算 multi-turn，要反复"think→answer→再 think→answer"才算；此阶段主流仍是 outcome-based reward，细粒度动作的信用分配很难，SPARL、GiGPO 等已在尝试，讲者判断未来半年仍是核心探索方向）。

**Memory** 分三类：RAG-style 外部记忆（MemoryBank、HippoRAG；RL 的参与可以很轻，如用 REINFORCE 训一个 reranker 从 top-10/20 检索结果里挑有用的；Memory-R1 则训练 memory 管理模型自主决定搜索/添加/更新/删除）、token-level 记忆（MemAgent 分块维护自然语言记忆池解决长文本；通义前一天开源的 ReSum 每轮用 summary tool 压缩对话再喂下一轮，讲者称这是"多轮真实世界 agent 训练中非常关键的技术"）、结构化记忆（图数据库；讲者团队的 G-Memory 本身无 RL 元素，用 RL 训练结构化记忆管理 agent 是其展望；团队也在做 latent token-level memory）。

**Self-improvement** 分三路：verbal self-correction（reflection、self-debug，不更新参数，"口头强化学习"）、internalizing self-correction（更新参数：ACCA 一类 actor-critic 结构，actor 答题、critic 批评，critic 把错改对也得奖励，用 DPO 训练；讲者团队的 **AgentTracer** 针对 deep research 长轨迹"错了之后无法定位哪一步错"的痛点，专门训练一个 8B 模型做 step-level 错误定位——现有大模型在这件事上很差： GPT-4o 只有 3.44% 、 Claude 4 Sonnet 18.97% ，AgentTracer-8B 可达约 20% 乃至 57.63% （字幕口径），而且只用了 2.5K–3K 条自动收集的数据；把诊断结果喂回第二轮求解，效果优于 Self-Refine/Critic 式 prompt 方法）、iterative self-training / self-play（Absolute Zero 的零数据自提问题自训练；MAPoRL 的多智能体 RL，agent 自行决定何时与同伴通信合作，效果优于单 agent RL）。

**Reasoning 与 perception** 讲者从简：reasoning 已有很好的综述，只做 fast/slow reasoning 的简单辨析；perception 指让纯文本训练的模型获得图像、视频、音频等模态能力（并推荐麦炜老师组 RL for vision LLM 的专门综述）。

### 方法三：应用域轴与环境/框架清单

垂直领域按 search & deep research（R1-Searcher、Search-R1 一系；开源工作多为 7B/8B、最大 Qwen2.5-32B；新趋势是 ZeroSearch、SSRL 这类"单模型自己搜索自己"；讲者另推荐华为的 deep research RL 综述）、coding（三层：code generation、iterative code refinement（SWE-bench 一系）、automated software engineering）、数学（informal 自然语言推理 vs formal 定理证明，两块重合很小，提醒读者区分）、GUI（RL-free 的 prompt 式 → SFT 内化 → RL-based；关键区别是环境 static 还是 interactive）、vision、embodied（VLN 导航与 VLA 操作两块）、multi-agent（RL-free 的 prompt 分工 → 训外部组件而不训 agent 本身（如 GPTSwarm 训连接矩阵、MaAS 训 supernet）→ 真正训 agent 参数（MALT、MAPoRL、FlowReasoner；训全部 agent 还是只训 meta agent 是差异点）；讲者坦承大规模多 agent 训练仍不成熟）。

环境节的主张：**dynamic/interactive 环境（状态会因 action 改变）与静态 benchmark 的分野，是 RL 环境与评测集的本质区别**。框架分三类：Agentic RL 专用框架（AReaL、AgentFly、Agent Lightning 等）、RLHF tuning 框架（OpenRLHF）、通用 RL 框架（verl、TRL 等）；讲者现场承认 v1 清单漏了 slime，欢迎联系补录。

### 结论/观点（区分事实与判断）

- 事实：六维形式化差异、双轴分类、环境/框架清单是综述可核查内容。
- 判断（讲者立场）：三个最值得关注的开放方向——① **trustworthiness**（security：agent 权限更大、可被投毒的接口更多，如记忆库投毒、工具返回有害信息、图像中的隐形有害 token；hallucination：工具幻觉即幻觉参数/函数名；sycophancy 谄媚）；② **agentic training**（把 agent 训练前移到预训练：通义前一天发布的 continual agentic pretraining 把 BrowseComp 提到约 40 分（字幕口径），讲者称"很震撼"，但也直言这"不是一般 university lab 能做的事"）；③ **agentic environment scaling**（现有环境太老太简单、GAIA 这类又随时间失效；出路是自动 reward 设计与自动课程生成——环境生成器像世界模型一样随 agent 行动生成后续任务与场景，讲者认为"非常 promise"）。

### Q&A 要点

- **概念包含关系**：Agentic RL ⊃ Tool-integrated RL ⊃ Search-R1（一类实现）；deep research 与 Agentic RL 是交集关系。
- **multi-turn 的严格定义**：一次 rollout 里想多久、调多少工具都只算 single-turn；多轮指多轮 think-answer 循环。难点在稀疏奖励下每一步的信用分配。
- **过程奖励怎么设计**：指向综述中 GiGPO、SPARL 的专门设计。
- **采样率太低怎么办**（base model 直接接 MCP 效果差）：GMPO、GSPO 一系的共识是做数据筛选、剔除失败率过高的轨迹；数据效率与质量是 RL 的核心问题。
- **self-evolve 中最重要的能力**：讲者选 memory——分 chatbot/personalized memory 与 self-improving memory 两类，后者（从已解的 1000 条轨迹里挑经验帮当前问题）"更 promise"，与 self-improvement 直接相关。
- **企业场景（5–15 个工具）需要多大模型**：与预训练/后训练见过的数据强相关；熟悉的 pandas 数据分析场景 14B 够用，陌生场景 32B 也未必行。
- **上手建议**：verl 等框架已较成熟，可直接上手；入门者多看前沿 paper、多跑代码。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 综述综合工作量 | — | 500+ 篇 | 论文补充（综述口径，talk 未展开） |
| Tool-use SFT 轨迹规模 | AgentTuning 约 2–3 千条 | AgentBank 约 5 万条 | 字幕（tool use 节） |
| Tool-integrated RL 涌现量 | — | 2025 年 3–4 月约三四十篇 | 字幕（tool use 节） |
| 开源 search agent 模型规模 | — | 多为 7B/8B，最大 Qwen2.5-32B | 字幕（search 节） |
| Step-level 错误定位准确率 | GPT-4o 3.44% 、 Claude 4 Sonnet 18.97% | AgentTracer-8B 约 20%–57.63% | 字幕（self-improvement 节；末项数字识别存疑） |
| AgentTracer 训练数据量 | — | 仅 2.5K–3K 条（自动收集 pipeline） | 字幕（self-improvement 节） |
| Continual agentic pretraining | BrowseComp 此前低位 | 约 40 分 | 字幕（future 节；通义工作，讲者转述） |
| 代表实证（综述所引） | DeepCoder-14B 训练前 | LiveCodeBench Pass@1 约 +8pt（outcome reward） | 论文补充（综述口径，talk 未展开） |

## 可迁移

- 对 coding data / RL infra 工作的 1-2 个直接可试的点：
  1. **给 agent 训练任务先写 POMDP 五元组**：状态是什么、观测是什么（上下文窗口里到底有什么）、行动空间、奖励何时到账、折扣怎么取。写不出来的任务，多半会在 rollout 阶段以不可调试的形式爆出来。
  2. **ReSum 式轮次压缩值得直接借鉴**：多轮 agent 的 context 膨胀用"每轮 summary 后丢弃原文"治理，讲者判断这会成为未来 agent 的标准范式——与本仓库 harness 的 token 记账问题同构。
- Infra 视角（扩展性 / 成本 / 评测自动化）的启发：POMDP 化之后，infra 的核心矛盾从"生成吞吐"变成"环境吞吐与状态管理"：环境实例的并发、状态快照/回滚、观测组装（token 记账）成为 rollout 成本的主导项——这与本仓库 skyrl-agent / harness-engineering 主题直接接续。讲者对"环境生成器"的展望，正是 EP153 EnvHarness 在做的事。

## 疑问 / 下一步

- AgentTracer-8B 的准确率字幕口径（约 20% 与 57.63% 两个数）对应什么评测切分，字幕未说清，需查论文原文核对。
- POMDP 框架下长程信用分配目前主要靠 outcome reward + 组内相对优势（如 GRPO 系），何时需要真正的过程奖励或价值模型？讲者在 Q&A 只指向 GiGPO/SPARL，未给判据；可对照 EP155 的实战口径再看。
- 讲者预告综述 v2 会补 multi-turn RL 新工作（Seed Agent-GM-RL、DeepDive 等），后续可跟踪 v2 的分类变化。

## 原文金句（1-2句）

> 「我们认为一个 dynamic 的、interactive 环境是很重要的事情。」——讲者论"环境为何是 RL 与静态 benchmark 的分野"（字幕 51:32–51:40）

> 「主流不主流我不知道，但这肯定不是一般的 university lab 能做的事情。」——被问 agentic pretraining 会不会成为主流时讲者的回答（字幕 69:36–69:49）
