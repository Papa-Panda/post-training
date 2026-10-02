# EP156 — 聊聊自我改进、递归自我改进（RSI），以及与自博弈的关系

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP156-spiral-spice-spade-rsi.html

> 「第二代模型造出第三代模型，recursive 的意思就是这个 improvement 算子本身也在 improve：新一代的 $\theta_2$ 写出的训练代码 $F_2$ ，要比上一代 $\theta_1$ 写出的 $F_1$ 更高效。」——讲者对 RSI 的定义拆解（字幕 03:26–04:38，按干净口径转写）

## 元信息

- 期号：156
- 标题：聊聊自我改进、递归自我改进（RSI），以及与自博弈的关系（B站标题：《从 SPIRAL、SPICE 到 SPADE：大模型 RL 后训练从自博弈走向递归自我改进》）
- BV：BV1PZhs6bEsy
- 时长：01:01:52（讲授约 46 分钟 + Q&A 约 15 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：刘博（华盛顿大学，与 Natasha Jaques 合作；即将赴 Stanford 攻读 CS 博士——字幕自述，导师与正式头衔以个人主页为准；字幕中 Stanford 作「散福」，仅标注口径）
- 相关论文：SPIRAL、SPICE（Self-Play In Corpus Environments）、SPADE 三篇系列工作（arXiv 编号待从论文页补录）
- 相关代码：讲者 Q&A 中提到有生成环境的可交互 demo 网站（链接待补录）
- B站链接：https://www.bilibili.com/video/BV1PZhs6bEsy/
- 官网预告：https://qingkeai.online/blog/Exploring-Self-Play-Towards-Recursive-Self-Improvement
- 字幕原文存档：本地 `transcripts/EP156.txt`（1199 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP156，已存档）清洗提炼；个别专名有识别误差，如 SPICE 在字幕中作「space」、SPADE 作「sped」、Natasha Jaques 作「娜塔莎JX」、Jason Weston 作「jason western」，均以公开资料为准。凡字幕口径无法确证处，下文标注（字幕口径，存疑）。

## 一句话总结

讲者用自己三篇系列工作回答一个问题：RLVR 之后，如何把人类逐步移出「造数据」的流水线。三篇工作的共同骨架是 self-play——让同一个模型分饰两个角色互相出题、做题：SPIRAL 在人为定义的零和文字游戏里自博弈，筛选出可泛化的 CoT pattern 并迁移到数学推理；SPICE 把出题者放回预训练语料这个「真实世界的缩影」，Challenger 从语料挖题加答案、用 Reasoner 答题的方差做 reward；SPADE 再进一步，让 designer 直接生成可执行的 Gym 风格环境代码，用「带 hint 与不带 hint 的表现差」定义 regret 做自动课程。讲者把这条线定位为 RSI（递归自我改进）中 improvement operator 的「数据/环境生成」那一部分：环境代码只是算子的一小块，真正的 RSI 还要求模型改进 RL 算法本身。

## 核心

### 开场：RSI 三个字母逐个拆

讲者先给定义。Improvement 是人类研究员 curate 数据、造 eval、训出更好的模型这个过程；Self 是让模型参与这个过程——用 coding agent 写代码生成训练数据、设计 reward、提升训练代码效率，在 benchmark 上迭代自己；Recursive 是关键的一跳：第一代模型造第二代、第二代造第三代，而且 improvement 算子本身在变强。用他的形式化：权重 $\theta_1$ 写出训练代码 $F_1$ 得到新权重 $\theta_2$ ， $\theta_2$ 又写出更好的训练代码 $F_2$ ，在定义的指标下 $F_2$ 优于 $F_1$ ，这才叫 recursive self-improvement。

这个 $F$ （improvement operator）的成分包括 backbone 代码、训练数据生成代码、评测数据生成代码、loss 与优化算法代码，乃至与芯片通信的 CUDA 代码。今天三篇工作只覆盖其中「训练数据/环境生成」这一块，讲者反复强调这一点：让 AI 自己写出算子中数据部分的代码，是他全部工作与 RSI 的关系。

动机的时间线：2024 年底 o1 与 DeepSeek-R1 preview 刚出，讲者在想 RLVR 之后是什么训练方式。由 AlphaGo/AlphaZero 很自然地想到 self-play——模型与自己交互产生数据、用于提升自己。他的核心假设是：thinking model 之所以强，是因为它的 CoT pattern 能泛化，而 RL 的作用正是自动筛选出这些可泛化的 thinking pattern；那么在 reward 完全确定（谁输谁赢人定得死死的）的人为文字游戏里做 self-play，对手持续变强会自动给出越来越难的题，应该能把可泛化的 pattern 筛出来并迁移到通用 reasoning。同时他立了两条边界：self-play 必须在大规模上 work，且目标是通用 reasoning 能力，而不是在某个文字游戏上做 superhuman。

### SPIRAL：零和文字游戏里的自博弈

第一篇 SPIRAL（2024 年底，与 Simon Yu、刘子辰等合作——字幕口径，机构未确证）：同一个模型塞入不同 prompt，分饰 player 0 与 player 1，玩人为定义的双人零和文字游戏（围棋式对弈、井字棋、扑克类）。因为输赢 reward 确定、不存在 reward 对错问题，模型只需在规则内自由探索。优化用多轮 REINFORCE 加 baseline 降方差（字幕称该 baseline 为「RAE」，存疑），不用 GRPO；并做了 environment skating（字幕口径，疑指游戏多样化采样）避免过拟合单一游戏——灰色线对照显示不用降方差技巧会直接训崩。

结果侧：文字游戏上的 self-play 能泛化到通用数学与 general reasoning benchmark；做了 4B 到 8B 的 scaling，也在 Llama、Qwen 以及 Distill-Qwen-7B 一类较强 reasoning model 上验证可继续提升。讲者举的 pattern 例子是期望估计与分类讨论：游戏决策与数学解题共享同类 thinking pattern，这正是他想要的迁移。

局限也很直白：游戏仍是人定义的，只是把「人类出题」的 effort 转移成「人类出游戏规则」的 effort。好在游戏简单、模拟快、人类积淀多，拿来即用——但这引出下一篇的问题：能不能让模型真正自己出题，还要自己给答案。

### SPICE：从预训练语料里挖题

第二篇 SPICE（Self-Play In Corpus Environments，2025 年 5 月，讲者 Meta 实习期间与 Jason Weston 合作）：角色换成 Challenger 与 Reasoner。Challenger 从预训练语料中挖掘有价值的问题及答案，Reasoner 负责作答。与同时期的 Absolute Zero / R-Zero 相比，讲者刻意区分了 motivation：Absolute Zero 强调「无中生有」的 zero data，但模型内部知识固定时无中生有出难题在他看来「有点不太对劲」；他认为 self-improving system 必须与真实世界交互、从中找 verification signal——物理世界由物理定律保证 ground truth，预训练语料则是人类观察世界的记录，是「真实世界的一个缩影」，出题者与语料交互就等于间接与世界交互。

SPICE 的一个关键设计是 Challenger 的 reward：用 Reasoner 回答这道题的 outcome 方差衡量题目价值——比如 8 道题 4 对 4 错时方差最高、题目最有价值。这本质是把 DAPO 一类 dynamic sampling 里「不太简单也不太难」的启发式，变成训练出题者的 reward。实验在 Qwen 与 Llama 上做，对比 R-Zero 与 Absolute Zero 在数学与 general reasoning benchmark 上更好，并验证了有预训练语料明显优于无语料，正好回扣 motivation。

定位上，SPICE 仍是 single-turn 的 prompt-answer 范式，对应 RLVR 时代。从 2025 年起 reasoning model 变 reasoning agent、single-turn 变 multi-turn，环境要有状态转移——这是第三篇的出发点。

### SPADE：生成可执行环境，用 regret 做自动课程

第三篇 SPADE（与 Simon Yu、Natasha Jaques 等合作，字幕口径）：先看一个反例 toy——400 个人写的环境、或用很强的模型（字幕作「GPT5.5」，存疑）生成的环境，拿去训 30B 模型，提升比较有限、甚至出现退化迹象。讲者的判断是：不光要生成有价值的环境，还要有针对当前 agent 水平的自动课程设计，这正是 self-play 能派上用场的地方。

架构仍是双角色，这次叫 environment designer 与 reasoning agent：

- **Designer**：从预训练语料筛 passage，结合 memory 机制（储存过去生成的全部环境代码，以及衡量这些环境价值的 regret 指标；检索时从 high-regret 样例中随机抽两个塞进 prompt 做 few-shot），生成一段完整的、可直接用 Python 解释器执行的 Gym 风格环境代码（reset/step 接口，有状态转移、每步有 reward），并同时生成关于这段代码的 privileged information，即 hint。
- **Reasoning agent**：在同一个环境上跑两条轨迹，一条能看到 hint、一条看不到。带 hint 的表现近似「看到环境代码后的 optimal policy」，两条轨迹的表现差定义为 regret：差越大，说明这个环境在「能做出来」的前提下学习空间越大，越适合当前 agent。反过来，看了 hint 也做不出来的环境会被筛掉。Regret 因此同时做了可行性筛选与课程排序。
- **自动课程**：同一 input 在不同训练阶段会被反复筛到，随着 agent 变强，designer 能针对同一 input 提出更难的环境；agent 若训崩变弱，也能选出没那么难的环境供它重新学。讲者认为这是环境必须动态调整的原因——对比 frontier lab 常见的「生成一堆、人为筛选、人定课程」，self-play 把课程自动化了。多样性则由预训练语料本身保证，这是他坚持要与真实世界（的缩影）交互的第二个理由。

实验在 4B、8B、30B 上做，8 个 benchmark 上持续提升，且模型越大提升越多；除文字游戏类环境外，也生成了 tool-use 类环境并在相应 benchmark 上验证有效。Q&A 补充的环境有效性保障：先确保代码能跑，再用 probe 类启发式检查（执行不同 action 应有不同 reward），再加 LM self-judge；hint 的生成 prompt 专门调过，防止把环境代码的答案直接说出来造成 reward hacking。

讲者对 SPADE 的定位很克制：环境是 improvement operator 的一部分，代码可执行意味着这一部分不需要任何人类中间参与；算子的其他部分（更 sample-efficient 的算法、更快的代码乃至硬件）都还要逐个攻克。

### Q&A 要点

- **Self-play 与 RLVR 正交吗？** 不正交。Self-play 本身也是 RL，只不过同时优化两个带对抗性质（接近零和）的 reward。
- **Hint 怎么防泄题？** Hint 有专门的 generation prompt 约束，不能把生成代码的答案直说，否则就是 reward hack；环境 make sense 与否另有可运行性检查、probe 启发式与 LM self-judge 三层保障。
- **策略循环（A 胜 B、B 胜 C、C 胜 A）怎么办？** Self-play 能学到均衡，但均衡未必是好策略；游戏 AI 领域用 population-based 的方法（如 PSRO）在平均意义上打败更多对手。
- **Self-play 下一步是什么？** 目前 self-play 只用来生成数据（当前形态是环境）；有了环境与 reward 之后「怎么更新自己」的 RL 算法仍是人定的（GRPO、REINFORCE 等）。讲者点名 David Silver 等人已在尝试搜索更好的 RL 算法——让模型改进 RL 算法本身，才是通往真正 RSI 的必经一步。
- **博弈类型能诱导特定推理能力吗？** 可以：在 social deduction、debate 类游戏里做 self-play，可能筛出能说善辩的 thinking pattern。
- **结构化环境一定比非结构化好吗？** 方向是 automate，所以非结构化语料最终都要转成结构化形态：SPICE 转成 prompt-answer，SPADE 转成代码。
- **Memory 机制如何实现？** 储存全部历史环境代码及其 regret 值，检索时从 high-regret 样例中随机采样塞进 prompt；讲者坦承这块没有做很深刻的设计，是 follow-up work。
- **与 collaborative multi-agent 怎么看？** 两者并行；讲者转述与 Noam Brown 的讨论：OpenAI 内部更偏 collaborative multi-agent，理由是它对 test-time compute 的利用更有效。
- **迁移机制的解释？** 没有形式化解释；猜想是 RL 筛选出可泛化的 CoT pattern 乃至内部 circuit——「wait」「but」等高频词出现得越多、思考越久，正确答案越容易出现，这与 test-time scaling 的现象一致。
- **零和假设限制多大？** 讲者明确说零和「很有必要」，因为对抗性正是 self-play 拥有自动课程的关键。
- **当前进步瓶颈？** 两条：一是 starting point——模型规模变大后，仅靠注入预训练语料可能不足以生成足够难的环境，需要更复杂的生成机制打底，否则人出的题仍更靠谱（而人出的题尚未穷尽）；二是 self-play 的训练不稳定性——二值化策略会坍缩到单一模式，讲者不认为这是 bug：「RL 就是要坍缩到 reward 高的模式上，想要多模式就得多个策略。」他也希望有更多资源验证方法在更大模型上是否 work。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| SPIRAL 模型规模 | — | 4B–8B scaling；另在 Distill-Qwen-7B 级模型验证 | 字幕 |
| SPICE Challenger 价值信号 | Reasoner 对同一题的答对分布 | 8 题 4 对 4 错时方差最高、价值最大 | 字幕 |
| SPICE 对比 | R-Zero / Absolute Zero | 数学与 general reasoning benchmark 上更好；有语料优于无语料 | 字幕 |
| SPADE 模型规模 | — | 4B / 8B / 30B | 字幕 |
| SPADE 评测范围 | — | 8 个 benchmark，规模越大提升越多 | 字幕 |
| 反例对照 | 400 个人写环境、强模型生成环境训 30B | 提升有限、个别出现退化 | 字幕 |
| SPADE 自动课程信号 | 带 hint vs 不带 hint 两条轨迹 | 表现差 = regret，差越大越适合当前 agent | 字幕 |

## 可迁移

- 「自动课程」是这场分享最可迁移的抽象：不管数据是人出的、模型出的还是环境生成的，都需要一个随当前策略水平动态调整难度的机制。SPICE 的方差 reward 与 SPADE 的 regret 本质都是同一个量——挑「现在做对一半」的样本，前者用答对率方差、后者用信息差，都可以直接借到 RL 数据筛选与 curriculum 设计里（与 EP152 里「rollout 4 次保留答对 1–3 次」是同一思想的不同实现）。
- 环境可执行化是一个值得抄的工程判断：把环境写成 Gym 风格的代码（reset/step）而不是自然语言描述，环境本身就自带可验证性（能跑、action 有区分度），且天然无人类中间环节；做 agentic RL 环境合成时，「生成代码 + 探针校验 + LM judge」三层保障比让模型直接吐环境描述更可靠。
- RSI 的 $F$ 算子拆解提供了一个评估「自我改进」工作的坐标系：任何自称 self-improving 的系统，先问它自动化了算子的哪一块（数据？评测？算法？硬件代码？），以及算子本身是否在跨代变强——只把数据生成自动化的工作，离 recursive 还差算法那一块。
- 「预训练语料是真实世界的缩影」是对合成数据路线的一个有用提醒：无中生有地让模型给自己出题，价值上限受限于模型内部已有知识；接入外部语料/真实交互拿到 verification signal，才是生成质量的天花板所在。

## 疑问 / 下一步

- Regret 用「带 hint 表现」近似 optimal policy，这个近似有多紧？如果 hint 太强，regret 主要测的是信息量而非环境难度；论文里应该有 regret 与真实学习增益的相关性分析，值得核对。
- 三篇工作的 scaling 都只到 30B，讲者自己把 starting point 列为瓶颈：基座越强，语料直挖的环境越不够难——frontier 级别上 designer 需要怎样的生成机制（更复杂的 harness？多步合成？），是这条线最关键的开放问题。
- SPIRAL 用 REINFORCE 加 baseline 而非 GRPO，在多轮 setting 下这个选择在今天是否仍成立？字幕称 baseline 为「RAE」（存疑），具体形式需查论文。
- 讲者提到生成环境有可交互 demo 网站、SPICE 中文名是 Self-Play In Corpus Environments，这些公开材料连同三篇 arXiv 编号都待补录进元信息。

## 原文金句（1-2句）

> 「我可以把预训练语料模拟成真实世界的一个缩影。」——SPICE 的 motivation（字幕 23:40–23:46）

> 「零和假设是很有必要的，因为对抗性是 self-play 拥有自动化课程的关键。」——Q&A 回答（字幕 59:35–59:43）
