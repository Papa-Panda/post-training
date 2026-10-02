# EP77 — Theory of Agent: From Definition, to Behavior and Objective

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP77-theory-of-agent.html

> 「我们把 agent 看作是一个 goal-oriented tool-use decision maker。」——讲者给出的 agent 新定义

## 元信息

- 期号：77
- 标题：Theory of Agent: From Definition, to Behavior and Objective
- BV：BV1dHWEzWETN
- 时长：01:21:03（讲授约 59 分钟 + Q&A 约 22 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：王鸿儒（香港中文大学博士生；字幕中主持人介绍作「港中文博士生王鸿儒博士」，另有「王洪文」等识别变体，正式姓名以讲者主页为准）
- 相关论文：position paper *The ... of Agent as Tool-Use Decision Maker*（讲者原话标题在字幕中作「The power also save agent as to use decision maker」，确切题名以 arXiv/讲者主页为准）；配套工作包括 SMART 、 OTCPPO/OTCGRPO（"teach model to act efficiently"）、 Alita
- B站链接：https://www.bilibili.com/video/BV1dHWEzWETN/
- 官网预告：官网第 77 期（链接未确认，仅记期号）
- 字幕原文存档：本地 `transcripts/EP77.txt`（1955 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP77，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如 LeCun 在字幕中作「乐坤/洛坤」、OTCPPO 在字幕中作「OSCPU/OSCPU」，均以论文与公开资料为准）。本期是 position paper 讲解，观点性内容多于实验报告；凡属讲者立场而非实验结论之处，以下按字幕口径如实标注。

## 一句话总结

讲者提出一套 agent 的理论框架：把 agent 定义为「目标导向的工具使用决策者」，把工具按来源二分为内部工具（从模型内部世界模型抽取知识）与外部工具（从真实环境抽取知识），论证好的 agent 行为不是最大化 accuracy ，而是让「知识边界」与「工具使用决策边界」精确对齐（知行合一）；在这个目标下，SFT/prompting 都受限于「最优行为由 agent 与任务共同决定」这一性质，RL（以及他们具体的 outcome×tool-use 乘性奖励 OTCPPO/OTCGRPO）才是能持续把两个边界对齐的训练方式。

## 核心

### 出发点：agent 的研究正在从刷榜走向科学化

讲者开场回顾了 agent 应用的火爆（OpenAI Deep Research 、 Manus 、操作电脑/手机/browser 的系统、他们与普林斯顿合作的 Alita 在 GAIA 上拿到过 top-1），随即提出问题：把 benchmark 的 accuracy/success rate 刷到最高，就是好 agent 吗？他主张 agent 这个学科正在从「争论什么是 agent、需要哪些模块」的阶段，走向理论化、科学化，并给出定义、行为、目标三层框架（与 Princeton 、 UIUC 、西北大学等机构的研究者合作）。

### 定义：tool 是什么， agent 是什么

两条主流观点对比：ReAct 式的 reasoning+acting，以及「agent = tools + memory + planning + action」式的模块拼盘。讲者主张核心行为只有 reasoning 与 acting 两件，其余（ memory 等）都是工具的实例。

工具的定义：一个东西是 tool ，当且仅当它 (i) 有用（useful：能完成一个或多个任务，有输入输出）、(ii) 按需调用（on-demand：只在需要时被调用，而非常驻激活）。按这个标准，他分出两类：

- **内部工具（internal tools）**：从 agent 的内部世界模型（ internal world model ，即模型参数所编码的知识）里抽取知识。包括 CoT 、反思（ reflection ）、 decomposition 等认知技巧，也包括人造的概念（军事、经济、法律、对话策略等）。以内部工具为主的典型产品是 OpenAI o1 。
- **外部工具（external tools）**：需要与外部环境交互、从真实世界抽取知识，如 API 、搜索、 computer use ，甚至「人」本身（需要澄清意图时才调用帮助）。典型产品是 Apple Intelligence 这类要调各种 app/API 的系统。

Memory 被单独讨论：它横跨两类——做成 RAG 检索的长期记忆更像外部工具，只指解题上下文的短期记忆则属内部工具。

新定义：**agent 是一个 goal-oriented tool-use decision maker**。任何任务的完成过程都可以写成统一的轨迹形式：第 $n$ 步调用工具 $T_n$ 、返回知识 $K_n$ ，工具既可以是内部的也可以是外部的。这带来三个好处（讲者自列）：unified format（与 ReAct 兼容——把内部部分打包成 $R$ 后就退化回 ReAct 形式）、flexible and robust（只靠 outcome-based reward 就能激发内部工具与外部工具两类行为，与已有实验现象对齐）、以及一个潜在的新 scaling law——继 next-token prediction 之后的 "next-tool prediction"：学会每一步该调什么工具，本质是 procedure knowledge scaling ，从交互与经验中学习、不断自我进化。

自主 agent 的判据也随之改写（承 LeCun 2022 年「on the path to autonomous machine intelligence」）：真正自主的 agent 应在完成任务的前提下，尽可能少地在外部环境里采取 action ——因为这意味着它把尽可能多的知识学进了内部世界模型。等价表述：minimize external tool use 的同时就是在 maximize internal reasoning ；当内部世界模型与外部世界模型完全一致时，不需要任何外部工具即可解题，「general agent is a world model」。

讲者还区分了 self-agent 与 theory of mind ： theory of mind 建模的是他人的 mental state（ beliefs 、 desires 、 intentions 、 emotions ），而 self-agent 除了建模外部环境，还要建模自己的内部世界模型来做决策——这是他认定的核心区别。

### 行为：两个边界必须对齐

框架的核心概念对：

- **知识边界（knowledge boundary）**：这个模型知道什么、不知道什么。内部知识是它知道的，外部知识是它不知道的。
- **工具使用决策边界（tool-use decision boundary）**：每一步决定用内部工具还是外部工具。

理想 agent 的定义就是让这两个边界 exactly matched——用讲者的话说就是「知行合一」：知道的就用内部工具解决，不知道的才去调外部工具。训练要做的事因此明确：当知识在边界外而模型错用内部工具（ internal tool overuse / external tool underuse ）时要把决策边界缩小；当知识在边界内而模型错去调外部工具（ internal underuse / external overuse ）时要把决策边界扩充。推理期则相反方向自然发生：外部工具取回的知识会扩充知识边界，边界一直扩张直到触及尚无人知晓的新知识，那就是 new knowledge discovery 。

讲者给出三条形式化原则（在 talk 中以「拉马（lemma）」形式口述，细节在论文 appendix）：

1. **基础原则**。给定模型在给定时刻的知识边界是固定的；随时间推移能力进化、知识边界向外扩展；且知识边界可以被重新分布（同一底座做 SFT/RL 会在某些任务上增强、另一些上削弱）。由此可推出一个 AGI/ASI 的简洁判据：当 agent 内部知识扩张的速度等于人类世界整体获取新知识的速度时，它就是 AGI ；远超之时即 ASI 。
2. **唯一性与多样性**。每个模型因训练语料、配方、架构不同，都有自己独特的知识边界与决策边界；在所有模型之上存在最小知识边界（人类最核心的共识，如真善美，应被所有模型的语料包含）与最大知识边界（其外是所有模型都不知道的，如未解决的数学/物理猜想）。
3. **知识的动态守恒**。任一给定时刻世界知识 $W_t$ 对所有模型都相同；对任一可解任务，存在一个最少且固定的所需知识量 $N_q$ ，它由内部与外部知识组合而成，且这个数值同时由任务本身的复杂性与该 agent 的能力共同决定（同一个问题给不同模型，所需资源不同）。由此推出「能力守恒定律」：只要允许动态 offload （把问题外包给更强的模型/工具），不同规模的模型在 agent 视角下可以等价——8B 模型把难题全外包给 70B ，与 70B 自己解题，在这个账上没有区别。这正是 routing/大小模型分工类工作的理论表述。

### 目标：为什么训练方法最终收敛到 RL

讲者论证的逻辑链：

1. **轨迹格式统一了，难点转移到数据**。所有交互都可以格式化成 tool-interaction 形式让模型学习「知识获取」而非只做知识压缩（字节的 all-in-one 式预训练在往这走）；但交互数据难以爬取，要么模拟、要么靠真实人机交互数据——所以 agent 反而更适合有数据壁垒的大公司。
2. **SFT 低效**。最优行为由 agent 与任务共同决定，按同一份 SFT 数据训所有模型会扰乱各自的知识边界与决策边界的对齐。补救思路是 SMART 那篇的做法：不知道每一步具体该调什么，但可以判断一个子任务是「已知还是未知」，据此构造数据去逼近所有模型共享的 maximum knowledge boundary（如复杂数学运算、用户私有信息、快速变化的事实等天然属于外部知识的类别）。讲者报告这种造数训出的模型输出更少、 accuracy 更高、决策 confidence 也更高。
3. **RL 是能动态对齐两个边界的方式**。核心挑战是 reward function 设计（算法与基建重要但属于优化目标之外的因素）。他们的具体解法是把工具调用次数写进奖励：定义「工具生产力」= 答对题数 / 所用工具总数；工具奖励满足单调性即可（答对前提下工具用得越少、奖励越大，他们用 sin/cos 形式因为更平缓易训）。最终统一奖励是乘性形式而非加性：

$$R = \alpha \cdot R_{tool} \cdot R_{outcome} + R_{format}$$

讲者强调乘法形式的三个好处：(i) 与任务定义对齐—— outcome 为 0 时整体为 0 ，杜绝「答错但不用工具也拿分」的 reward hacking ，且数学上最大程度保住 outcome accuracy ；(ii) 可泛化到各种 tool 形式与 outcome/format 奖励的线性组合。他们称此为 OTCPPO/OTCGRPO 一系的工作（细节与理论证明在论文中）。

实验口径（字幕原话，讲者明确说快速过）：严格沿用 Search-R1 与 ToRL 的 setting ，工具调用次数显著降低、工具生产力提升 2–3 倍；小模型上 accuracy 略降，模型越大则持平甚至提高——因为规模越大内部世界模型越准，被激发出的内部推理能力越强。

两个 case study 是讲者自认最有价值的发现：

- **cognitive offloading（认知卸载）**。一个「A 、 B 是否都是京剧作曲家」的问题：纯 outcome reward 训出的 agent 三次搜索（搜 A 、搜 B 、搜 A 与 B ）才答对；而 OTCPPO 只搜一次（同时搜 A 跟 B ）就答对。只优化 outcome 会让模型倾向于依赖外部工具、无法约束自身行为，就像人长期依赖外部工具会「提笔忘字」。
- **最小外部工具 = 最大内部推理**。更极端的例子：用更准的估计发现同一个问题内部知识其实就够（工具调用 0 次也答对），说明逼模型少用外部工具，等价于在迫使它的内部世界模型更精确地模拟外部世界——这与框架的立论正好闭环，也解释了方法为何随模型规模变大而更有效。

四个行为模式的未来之辨：在「任务已完成」的前提下，行为可分 (1) 内外部工具都拉满（不在乎成本，过度优化、不 efficient）；(2) 内外部都最小（理想最优，但非常难训练、易过保守）；(3) 最大化内部、最小化外部（与 (2) 互补， OTCPPO 正是同时做 (2) 与 (3)）；(4) 最小化内部、最大化外部（浪费了 LLM 已学好的内部世界模型，讲者个人判断长期不 promising ，只在内部世界模型明显不够准的场景可能有增益）。他判断 (2) 与 (3) 最值得做，共同点都是要 minimize external tool cost 。

### 讲者立场小结（区分事实与观点）

以下是讲者在 talk 与 Q&A 中明确表述的个人判断，非实验结论：与工业界在 GAIA/SWE-bench 上拼精度没有优势（字节已把战场烧到 pretraining 阶段），学术界应转向行为模式与目标设计；能力不足时 dense reward 信号能帮助训练稳定，但有足够数据与基建时 outcome reward 本身就够用、只是行为会有不足；工具描述默认放 system prompt 便于注入新工具，工具稳定不变时直接学进参数也无妨； RL 训练中工具调用次数会下降，但 OTCPPO 的平均值约 1 次而非全面归零，且这与 prompt 关系不大、主要由 RL 奖励决定；记忆问题上他自称研究不多，倾向长期记忆用 RAG 、短期记忆是 long-context 问题。

Q&A 其他要点：工具调用轨迹需不需要人类标注——他认为问题本身的严重性（ task-agent 共决最优性被忽略时对后续 scaling 的影响）尚未被探索扎实，理论上每个工具调用决策都值得 revisit ，长期则应由 agent 自己 revisit 自己的决策；Web GUI 成功率低是否源于知识空间重合低——他认为这只是因素之一， next-token prediction 确实缺少 GUI 交互知识，解法涉及 pretraining/SFT 阶段的新探索；对「高效 RL 框架」的推荐问题，他澄清自己说的高效指行为效率而非同步/异步等基建效率，点名了 AReaL 、 slime 等框架。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| 工具生产力提升 | Search-R1/ToRL 严格同 setting | 提升 2–3 倍 | 字幕（实验部分） |
| 工具调用次数（case 1） | outcome-only agent 3 次搜索 | OTCPPO 1 次搜索答对 | 字幕（case study） |
| 工具调用次数（case 2） | 需外部搜索的先验判断 | 0 次（纯内部知识）答对 | 字幕（case study） |
| RL 后平均工具调用次数 | OTCPPO | 约 1 次，未全面归零 | 字幕（Q&A） |
| 实验模型规模 | 资源所限 | 仅 3B 与 7B | 字幕（Q&A） |
| Alita 在 GAIA 的成绩 | 当时 test/val 榜单 | top-1（讲者团队与普林斯顿合作） | 字幕（开场背景） |

## 可迁移

- 给任何 agent 训练/评测加一条「成本轴」：在 success rate 之外记录工具生产力（成功数 / 工具调用数），并用乘性而非加性方式把成本项并入奖励—— `答错即 0 分` 的乘法结构既保 accuracy 又天然防 reward hacking ，这是可以直接搬到 tool-use RL 配方里的一行改动。
- 「知识边界 vs 决策边界」的二维诊断可以做成现成的 eval 分类法：把失败样本按（知识在内/在外）×（用了内部/外部工具）四格归类，能直接区分出 over-thinking/乱搜与该搜不搜两类问题，比只看成功率更能指导数据构造（例如按 SMART 的思路只标注「已知/未知」来造 SFT 数据）。

## 疑问 / 下一步

- 框架的关键假设被讲者明确放在 appendix 且 talk 中未展开：内部知识与外部知识被假设完全正确、互不重叠、无冲突；真实场景里的知识冲突（knowledge conflict）、不确定性与置信度如何进入两个边界的形式化，是他承认「暂时没做更复杂讨论」的部分，也是这套理论离工程落地最近的一个缺口。
- 「最小化外部工具」与「难任务需要更多探索」之间的张力如何量化：讲者用 case study 说明少搜反而对，但对何时该主动多调外部工具（例如内部世界模型明显不够准的任务），框架只给了定性判断（路线 (4) 在这类场景可能有价值），没有给出可操作的判据。

## 原文金句（1-2句）

> 「如果这个语言模型觉得当前这一步需要支持的是 internal（知识），它就用 internal tools；如果需要的是 external，它就调用 external tools——这就是知行合一：你的知识跟你的行为完全 match、完全 aligned。」——讲者解释两个边界对齐

> 「你不断去 minimize 它的 external tool 调用，某种意义上就是在迫使这个语言模型的 internal world model 尽可能精确地模拟外部的 world model。」——讲者解读 OTCPPO 的核心发现
