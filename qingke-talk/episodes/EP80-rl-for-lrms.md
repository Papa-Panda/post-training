# EP80 — RL for LRMs：探讨面向推理模型的 RL 最新研究

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP80-rl-for-lrms.html

> 「我们会发现其实这些方法起效的一个原因，本质上还是在于它是去将模型的这种不确定性给它转化为了这种性能。」——张开源总结无监督（内在）奖励方法为何有效的机制

## 元信息

- 期号：80
- 标题：RL for LRMs：探讨面向推理模型的 RL 最新研究
- BV：BV1b9sGzwEAc
- 时长：01:09:10（4150 秒）
- 提炼日期：2026-10-02
- 分享嘉宾：张开源（博士；代表其团队介绍推理模型 RL 系列工作，团队工作含 PRIME、TTRL、self-search 等；讲者机构以其 GitHub 论文列表页为准）。主持：王过（青稞社区主理人；字幕作「金科社区」）
- 相关论文：本期为综述式分享，对应团队的 RL for LRMs 论文列表（paper list）与 PPT 均已开源至 GitHub（讲者称约 1800~1900 star）；文中点名 PRIME、TTRL、GPM、HP、Absolute Zero、DAPO、GSPO、Kimi K2、DFT/IFT 等工作
- 相关代码：同上 GitHub 仓库（地址见其论文列表页）
- B站链接：https://www.bilibili.com/video/BV1b9sGzwEAc/
- 官网期号：EP80（预告链接未确认，略）
- 字幕原文存档：本地 `transcripts/EP80.txt`（1728 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP80，已存档）清洗提炼。AI 字幕对专名识别误差较多，纪要按公开资料校正：LRM/RLVR 在字幕中多作讹字，GRPO 作「GRPU/GPU」，TTRL 作「TTLL」，self-search 工作作「SSLL」，PRIME 作「PREM」，reward hacking 作「river hiking」等，均以原工作名称为准。讲者团队自述的工作归属（如 PRIME、TTRL、GPM、HP、self-search）按字幕口径记录。

## 一句话总结

这是一场把「推理模型 RL」拆成奖励设计、策略优化、采样策略三大模块的综述式分享：讲者系统梳理了从可验证奖励到无监督内在奖励、从 PPO/GRPO 到 off-policy 修正、从动态采样到结构化采样的技术版图，并明确给出三个基础问题的判断——RL 目前更多是锐化（sharpening）而非发现（discovery）、SFT 才负责引入新能力、以及真正有效的训练 trick 只有少数几个（截断重要性采样、序列级 loss 聚合、batch 级 advantage 归一化、在线过滤）。

## 核心

### 背景：从 RLHF 到 RLVR 的范式转变

讲者的历史坐标：2022–2023 年的 RLHF/DPO 只解决偏好对齐（更有用、更无害），没有本质提升解题与推理能力；2024 年底起 RLVR（基于可验证奖励的 RL）才显著提升模型解决任务的能力。能力侧的标志是任务时间跨度：按讲者口径，模型以 50% 成功率完成任务的时长，GPT-5 已达 2~3 小时、Claude 4.5 在特定任务上可连续工作 30 多小时——AI 正从 L2 推理迈向 L3 agent。

形式化上，LLM 的 RL 没有传统 MDP 里清晰的 action/state 区分：next-token 预测即 action，输出拼回上文即 state 变化，奖励一般只给 response 级。这决定了后面所有模块的设计约束：奖励设计（如何给出稳定可靠的奖励）、policy 优化（有了奖励如何更新）、采样策略（词表级动作空间下如何采样）。

### 奖励设计（一）：可验证奖励与生成式奖励

- **规则型可验证奖励**：主流范式，用 tag 约束 thinking/answer 的输出格式，直接算正确性 + 格式奖励。讲者引「可验证者定律」（verifier's law，字幕作「杰森为提出了非常有名的…very fid lad 定额」，按公开出处为 Jason Wei 的 verifier's law）：只要任务可验证、不断 scale 数据与模型，基本都能做到专家级。
- **不可验证任务走生成式奖励**：从 LLM-as-judge 给离散分数，到先写 critique 再打分，再到直接用 reasoning model 做奖励模型——打分时能展开分析、反思、重估，分数更准，但推理慢、大规模 RL 里会成为瓶颈。
- **model-based verifier**：介于两者之间，对等式变换、单位换算等「规则难对齐、语义可判定」的任务，用 prompt 驱动一个模型做验证（Kimi K1.5 thinking 中有应用，字幕作「C的1.5thinking」）。
- **checklist 奖励**：把任务拆成一系列子问题逐项核对，简单模型即可打分，在 agent 与复杂任务里常用。
- **reward model 与 policy 共同进化**：独立 RM 必然面临 reward hacking，两条共训路线——self-rewarding（模型自生成自打分，Kimi K2 的 judge 用法是其规模化版本）与外加 verifier 模型共训（字幕作「l tango」的工作：generator 出题多解，outcome reward 同时监督 policy 与一个过程 verifier，过程奖励聚合回结果奖励，两个模型迭代共训）。

### 奖励设计（二）：稠密奖励三粒度与无监督奖励

结果奖励之外，稠密（过程）奖励按粒度分三类：

- **token 级**：团队 2025 年 1 月的 PRIME——把同一问题的正确/错误答案做成 pair 训 DPO 模型，利用 DPO 蕴含的隐式 PRM 逐 token 取分作为过程奖励。讲者口径：比纯结果奖励收敛更快、信号利用率更高、上限可能更高。
- **step 级**：两条路——额外训一个过程奖励模型（团队的 GPM：用 DeepSeek 系 7B 模型逐过程打分，准确率高、用于后续 TTS 更高效）；或蒙特卡洛采样估计步值（o1 之后大量工作如此），但采样昂贵。团队 9 月的工作改在「不确定性高的位置」才采样，并定义了基于 attention 影响力的分数来定位这些位置。
- **turn 级**：agent 场景每轮有明确边界，可分别核对工具名（function name）与参数（arguments）的正确性直接给分；或由最终结果反推每轮得分（与蒙特卡洛同理，团队 ARPO 类工作用了相关处理；字幕作「AARPU」）。

无监督/无标注奖励是 scale 奖励的关键一节，分两类：

- **model-specific（模型内在）**：TTRL（团队今年 4 月的工作）用模型对自身输出多数投票的一致性当奖励即可稳定训练；进一步可直接用 token 级 logits 的熵/置信度判对错当奖励；或模型自任 judge。Absolute Zero 是巧妙极端：同一模型同时扮出题者与解题者，出题用程序验证，全程自给奖励。
- **model-agnostic（模型外）**：启发式规则（如回复越长正确率越高、格式对即大概率对）能训出效果，但讲者提醒陷阱——有些提升其实来自评测管线里默认的格式提取，方法本身未必起效；另一路是强化预训练，直接用 next-token 预测从数据里取内在奖励。

讲者对这批方法的统一判断（近两周将放出的新工作）：内在奖励方法可用一个统一框架收纳，其起效本质是「把模型的不确定性转化为性能」；但它们对超参敏感，更适合小规模与 test-time 学习场景。

奖励的组合用法：rule-based 与生成式奖励加权相加（Qwen2.5 技术报告起的做法，团队 EMNLP 工作验证有效）；以及 GRPO 式 group 内归一化降方差，8 月起 group 级归一化被拓展到更多场景。

### 策略优化：PPO 骨架、off-policy 修正与正则之争

主流仍是 PPO 风格策略梯度：advantage 由 reward 计算、clip 防漂移、各类归一化。分化在两点：

- **critic 路线 vs critic-free**：critic 与 policy 同量级、训练成本高，故主流是 GRPO、RLOO、REINFORCE++ 等去 critic 方法；混合 group 与 batch 两级统计量算 advantage 可能更有效。
- **off-policy 是核心矛盾**：训练/推理引擎的精度差异、异步采样、replay、混合外部数据都会造成分布 mismatch。阶段性（截断）重要性采样是目前各场景都有效的标准解（讲者提到 Meta 近期 scale-up RL 工作也用了它）；GSPO 的序列级 loss 聚合、NFT 等工作提示可能存在超越 on-policy 梯度的新形式。off-policy 的数据级版本：DFT/IFT 把 SFT 目标改造成与 RL objective 统一，只用有监督数据即可超过 SFT、逼近 RL；团队的 HP 工作把 SFT、GRPO 与 mismatch 修正统一进一个框架，让模型按训练状态自选做 SFT 还是 GRPO，稳定性与上限都更好。

正则三类，讲者给了明确的当下判断：

1. **KL 约束**：对 reasoning model 可能无用甚至有害，DAPO 等工作去掉 KL 反而更好——推理能力是训出来的，拿 reference model 锚住反而限制探索；
2. **熵**：熵塌缩是训练崩溃的前兆；对高方差 token 施更严格的更新约束 + 熵惩罚（Kimi K2 用法），曲线更稳；
3. **长度惩罚**：可做，但 GPT-OSS 那种高压缩比未必是训出来的，与数据关系更大。

### 采样策略：动态过滤与结构化探索

- **动态采样**：PRIME 早期就用在线过滤——只保留 group 内准确率在 0.2~0.8 的 prompt；DAPO 的动态采样本质相同，但需要补新 prompt、整体更慢，两种做法可按速度/效果取舍。
- **结构化采样**：与蒙特卡洛同源，团队的 attention-guided 工作只在高不确定位置做结构化采样，得到的过程分数可直接当过程奖励用，探索性与训练上限都更好。
- **超参调度**：训练中逐步升温增不确定性、逐步放长生成上限避免截断、overlong masking 等，都是论文里反复出现的有效操作。

### 三个基础问题：讲者的立场

1. **RL 是 sharpening 还是 discovery？** 既有工作用 pass@k 证明 RL 后模型很难超过 base model 上限（即只是降不确定性）。团队在纯净合成设定（复合函数操作字符串）里复现了这一点：一、二阶复合函数上 RL 上限与 base 一致；但到三、四阶复合函数，pass@k 上限确实被抬高——discovery 在更高难度、严格防污染的设定下成立。结论：不要脱离具体任务与难度谈 RL 的本质，科学发现场景里的数据污染评估本身是未解难题。
2. **SFT 与 RL 是记忆还是泛化？** 年初工作说 RL 泛化好、SFT 易过拟合；但新证据显示在 SFT 模型上做 RL，其 OOD 分布准确率未必超过 SFT 本身。讲者立场：SFT 仍是引入新能力的更好方式（比如从 80 分的分布里把新能力装进来），RL 负责在此基础上从 80 冲 90、100；两者不互斥，混合/统一（HP 路线）是方向。
3. **模型先验与 trick 审计**：base model 常比 instruct model 更好训（SFT 会伤探索性）；Qwen 系比 Llama 系涨得多，但这未必能移植——要放在 pretrain→RL 的全 pipeline 里看数据多样性与混合。trick 方面，讲者引阿里淘宝的工作把 trick 全列了一遍，结论是核心有效的只有几个；团队综述与实验共同验证的有效集是：截断重要性采样（TIS/CIS）、GSPO 式序列级 loss 聚合、batch 级 advantage 归一化、动态采样/在线过滤。Meta 近期发布的 scale-up RL 工作被当作正面参照（小模型上即可验证 scaling 趋势）。

outcome vs process 奖励之争讲者明确不下结论：只给结果奖励模型也能长推理，但中间会胡编、最后蒙对；引入过程奖励则必然诱发新的 reward hacking——引 OpenAI 年初分析：对过程施加强压后错误会减少，但模型「一定会发现新的捷径」。这仍是开放问题。

### 资源、框架与应用（略讲部分）

讲者整理了 math/coding 等训练资源与开源框架，但判断静态榜单（math、coding leaderboard）已刷到过拟合、价值递减；更有价值的是在真实/动态环境（dream 或真实世界）里训练。框架观察：现有 RL 框架本质是对训练与推理引擎的 workflow 封装，创新多由算法驱动，共同痛点都是 off-policy/训推不一致（LM Sys 博客近期也在讨论）。应用覆盖 coding agent、多模态、机器人、科学、医疗，细节见论文列表。

### 未来挑战

- **新范式**：持续/continual 学习（甚至 in-context RL）、基于 memory 的 RL（超越 RAG 式记忆）；
- **环境构建**：model-based RL 是当红方向——团队 8 月的 self-search 工作让模型用自身参数知识当搜索引擎训练多轮搜索，训完换接真实 Google 搜索后存在 sim-to-real 式泛化，讲者称之为「world model 的雏形」；Meta 针对 agent/coding 的 world model 工作同路；
- **激发内在能力**：更高效的训练与新架构、latent space 推理（信息比特密度高于语言）、RL 与 pretraining 结合（强化预训练，目前仅两三个工作）；
- **新载体**：diffusion 语言模型上做 RL（结构化、物理化学等任务可能更高效，但做法未定）、架构与算法协同优化、把 RL 用于闭环极长的科学创新与自我进化。

### Q&A 要点

- **推理模型 RL 需不需要过程监督？** 没有定论。过程监督的好处是效率与稳定；GRPO 训练崩溃可能正是过程监督缺位——PPO 的 critic 本身就是一种过程监督，GRPO 不够稳时值得回头试 PPO。GSPO 对 MoE 的有效性尚无细致分析，训练崩溃时可一试。
- **隐式（DPO/PRIME 式）奖励的价值**：主要在训练效率与稳定性；OpenAI 至今用 PPO，侧面说明 critic 式过程监督仍有其稳的优势。
- **通用场景奖励系统**：rule-based 与 model-based 混合可尝试；更稳的做法是在合成问题时就埋好 checklist 子问题，用它做规则奖励——数据合成阶段就把可验证性设计进去。
- **step-level 奖励还有没有意义**：数学任务里意义不大；agent 场景链路长、反馈稀疏，仍有价值，本质是换效率与抗崩溃。
- **continual RL 与传统 continual learning 的区别**：问题本质相近，LLM 的特异性在于模块更多（policy、reward、采样轨迹），且更可能靠 test-time/in-context 学习实现持续知识积累并涌现新能力；小规模 toy 上价值不大，agent 场景结合 judge/understanding 模型才可能体现价值。

## 关键数字总表

| 指标 | 数值 | 来源 |
|---|---|---|
| 任务时间跨度（50% 成功率，讲者口径） | GPT-5 约 2~3 小时；Claude 4.5 特定任务 30+ 小时 | 字幕（背景部分） |
| 在线过滤保留的 prompt 准确率区间 | 0.2~0.8（group 内） | 字幕（动态采样部分） |
| PRIME 提出时间 | 2025 年 1 月（字幕口径） | 字幕（token 级奖励部分） |
| TTRL 提出时间 | 2025 年 4 月（字幕口径） | 字幕（无监督奖励部分） |
| self-search 工作时间 | 2025 年 8 月（字幕口径） | 字幕（未来挑战部分） |
| 团队 paper list GitHub star | 约 1800~1900 | 字幕（结尾部分） |
| 复合函数干净设定结论 | 1~2 阶：RL 上限 = base；3~4 阶：pass@k 上限被抬高 | 字幕（基础问题部分） |
| 被验证有效的核心 trick 集 | 截断重要性采样、序列级 loss 聚合、batch 级 advantage 归一化、动态采样/在线过滤 | 字幕（基础问题部分） |

## 可迁移

- 对 coding data / RL infra 工作的直接可试点：给训练数据在合成阶段预埋 checklist 子问题，把「可验证性」做进数据生产管线，而不是训完再补 verifier；采样侧默认开在线过滤（保留 group 准确率 0.2~0.8 的 prompt），这是讲者两处独立提到、且 PRIME 与 DAPO 双向验证过的最便宜收益。
- Infra 视角：off-policy/训推不一致被讲者定为当前 RL 框架的共同头号痛点，截断重要性采样是现成的标准解；选型与自研时应优先看框架对异步采样下 off-policy 修正的支持，而不是算法数量。

## 疑问 / 下一步

- 讲者预告「近两周放出」的无监督奖励统一框架工作值得追踪：它声称内在奖励起效本质是把不确定性转化为性能，若其统一框架成立，TTRL/熵奖励/Absolute Zero 可被同一套超参逻辑管理。另：讲者对 process reward 的保留态度（必然诱发新 hacking）与 EP125/EP154 的 OPD 线互为对照，后续可交叉验证。

## 原文金句（1-2句）

> 「我觉得 SFT 还是引入新能力的更好的一种方式……我们如果从 80 分达到 90 分，甚至 100 分之后的这种能力，还是要靠 RL 能够去提升的。」——张开源谈 SFT 与 RL 的分工

> 「当我们对过程给予了非常大的压力或者非常强的一个奖励之后……他同时一定会去发现新的这种捷径。」——张开源引 OpenAI 分析谈过程奖励的代价
