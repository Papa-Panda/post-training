# EP45 — B-STaR & SimpleRL-Zoo：通过强化学习自我提升推理性能和效率

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP45-bstar-simplerl-zoo.html

> 「我们只有有了不断会有更高质量的合成数据、更 high quality 的 response，我们才能去推进我们的模型不断的去变强。」——讲者在讲 self-training 瓶颈时的判断

## 元信息

- 期号：45（B站标题为「青稞Talk 45期」）
- 标题：B-STaR & SimpleRL-Zoo：通过强化学习自我提升推理性能和效率
- BV：BV1d7pYzbE1C
- 时长：01:11:59（讲授约 54 分钟 + Q&A 约 17 分钟；字幕末条 4311 秒）
- 提炼日期：2026-10-02
- 分享嘉宾：曾伟豪（香港科技大学，博士在读；研究方向为大模型 post-training 与 reasoning；字幕中自述姓名；B站字幕主持人介绍时将姓名识作「邓伟豪」，以讲者自述为准）
- 相关论文：*B-STaR: Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners*（ICLR 2025，arXiv:2412.17256）；*SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild*（arXiv:2503.18892）
- 相关代码：github.com/hkust-nlp/simpleRL-reason（讲者称 GitHub 约 4000 star）；B-STaR 代码见 github.com/mistobaan/b-star
- B站链接：https://www.bilibili.com/video/BV1d7pYzbE1C/
- 字幕原文存档：本地 `transcripts/EP45.txt`（1736 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文清洗提炼，来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如主持人介绍讲者时作「邓伟豪」、SimpleRL-Zoo 在字幕中作「simple21zoom/simple IO zoom」，均以讲者自述与论文为准）。论文（arXiv:2412.17256、arXiv:2503.18892）仅作元信息交叉核对；凡数值未在字幕口播中给出明确数值的，以下写明（字幕口径）不硬凑数字。

> 关联：本期是青稞Talk 早期一期把「传统 self-improving」和「zero RL」并到同一讲的分享；讲者本人的判断框架（自动监测并平衡 exploration / exploitation）是 B-STaR 和 SimpleRL-Zoo 的共线。

## 一句话总结

曾伟豪把 self-training 的成败归结到两个随训练漂移的能力——模型的探索能力（自己生成多样、高质量 response）与外部 reward 的利用能力（从候选里挑出对的）——并用 B-STaR 在每轮迭代自动调温度、采样数与 reward 阈值去最大化一个「平衡分」（balance score）保住这两个能力不要饱和；再往前推到更 online 的 zero RL，他给出 SimpleRL/SimpleRL-Zoo：在 7B base 上只用 8000 条开源数学数据 + 规则奖励即可涌现 long CoT 的三段式学习曲线，并在跨模型动物园里给出一套（含 format reward 是毒药、数据难度要匹配基座）能跨模型复用的经验配方。

## 核心

### 先分清：RL 在这里是数据合成策略，不是 teacher 的替代品

讲者开头先给了一个分类。post-training 合成数据只有三条来源：更强的 teacher（你总得先有更强的模型——对最前沿的实验室不可用）、更弱的 teacher（需要 student 本身是极强的 base）、以及模型用自身生成再过滤的 self-training。他把 RL 定位为「一种形式的数据合成策略」：相比 SFT 的模仿学习（通常难以超越 teacher），RL 鼓励自由探索、上限更高，但真正的前提条件是能持续产出高质量数据——这也是他整场反复追问的：「self training 中哪一步是最关键的？」

答案落在 filter 这一步：合成数据跑多快，取决于监督信号能不能持续区分出「更好」的 response。讲者并给了一条从 offline 到 online 的演化谱系：self-improving 的迭代越快（每个 iteration 生成更少、更新更频繁），它就越像 online RL。

### 前置发现：传统 self-improving 很快饱和，而且不是靠更强 PRM 能救的

讲者做了一组小型前置实验：在 GSM8k 与 MATH 上用两种监督信号——只看 final answer 的 rule-based reward，以及 final answer + 额外训的 process-based reward model——按训练步数跟踪 benchmark 表现。结论是（字幕口径）：无论哪种监督信号，性能都很快饱和，看不出随 compute 线性增长的 scaling 趋势。他提醒这是 DeepSeek-R1 之前那套比较 offline 的自改进设定；但它把讲者推去追问：到底是哪些因素卡住了 scaling？

### 两个能力：exploration 与 exploitation 的动态失衡

讲者把 self-improving 的动力学拆成两个量。**探索能力（exploration）**是模型自己经多次 sampling 生成多样、高质量 response 的能力；**利用能力（exploitation）**是外部 reward 从 K 个候选里至少筛中 S 个正确 response 的能力（他用统计指标 reward 的利用率去量）。两者都会随迭代动态变化——模型分布变、探索路径随之变；reward 的利用率也会变。

他给出两条训练曲线观察（字幕口径，为图表描述，未给出精确数值）：

1. 探索能力：初期快速上升，随后饱和、甚至下滑。讲者直言「模型自身的探索能力，有时候已经决定了算法的上限」。
2. 利用能力：整体也趋向饱和，且水平长期低于探索能力——也就是说真正卡住训练的不是生成，而是筛。

只有两者平衡，self-improving 才 work；一般设定下很难自动达成平衡，这是讲者为 B-STaR 准备的问题背景。

### B-STaR：用 Query Effect / Balance Score 自动调参，而非手动 schedule

问「什么样的训练数据算好？」讲者给了两条原则：筛出的正确 response 绝对数量要够多（否则训练数据不够），且它们在所选数据中的占比要够高（否则 noisy 数据会污染训练）。他据此定义了 query 级的指标（论文称 balance score，讲者口播称 Query Effect）：左边约束每道 query 被选中的正确 response 数 $N'_i$ 的绝对规模，右边约束它在候选中的占比；两头都高，这道 query 才是对当前 policy 有贡献的数据。

有了这个可量化的目标，B-STaR 的做法就是在每次迭代里把目标最大化：随当前 policy 与 reward 的状态，动态调整 rollout 温度、sampling 数与 reward 阈值（具体阈值在论文里是每轮用小量子集先搜再推广到全量）。讲者强调这与旧 self-improving 把超参写死（静态黑盒）有本质区别。

效果（字幕口径，图表展示未口播三位有效数字）：在 MATH 等数学推理与 APPS 代码推理上，相对包括 online FT、STaR/ReST-EM、Iterative RFT 在内的经典 self-improving 基线有大幅提升，且训练曲线在演示的步数范围内保持上行、没有过早饱和。他还展示了探索能力指标 pass@32（32 次 rollout 中至少一次正确）在 B-STaR 下同样保持良好、没有走弱。

一个有趣的自发现象：被自动搜出的温度不是常数——训练早期自动选到较低的约 $T \approx 0.5$ 左右，后期逐渐拉高。讲者把这解释为算法在后期更愿意扩大探索空间。sampling 数因当时算力没动，但他说消融显示更大的 sampling 数带来更好的 balance score 与最终性能。

### SimpleRL：8000 条数据 + rule-based reward 即可在 7B 上涌现推理

第二部讲 zero RL。讲者先展示用 DeepSeek-R1 案例：不经任何 SFT、从 base 直接做 RL，long CoT 与 self-reflection（如「wait, let me make sure」）是训练中涌现出来的，不需要 tree search 或 reward model。

SimpleRL 是他们抢在 R1 开源前（约 2024 年 11 月已跑通）的复现实验。设置（字幕口径）：

- 基座：Qwen2.5-Math-7B base，直接从 base 训，零 SFT；
- 数据：仅 8000 条开源数学数据，不做生成与精筛；
- 奖励：rule-based 三段式——答案对 +1，格式对但答案错 -0.5，格式错 -1；
- 算法：早期用 PPO 并保留 value / reference 组件；规则奖励替换掉了 reward model。

作为对照讲者点名了两条同期路线的资源开销（字幕口径）：RStar-Math 用了约 7M 条 SFT 数据 + RL 阶段 300 万数据 × 16 次 rollout；PRIME 用了 23 万 SFT + 15 万 RL 数据。SimpleRL 合计仅 8000 条 × 每次 rollout 8 次。讲者称结果比 base 约 +20 个绝对点（字幕口径，展示图表）、与 Qwen 官方 instruct 模型相当、与 PRIME 相当，并且在 AIME 上 SFT 反而会损害成绩，而 RL 能实现从易到难的泛化。

训练动态被讲者拆成三段：length 先快速下跌（格式对齐阶段）、随后 length 与 accuracy 同步回升、在第三阶段 length 还在涨但 accuracy 不动——他事后把这段解读为无有效探索、只长冗余词。这张曲线日后被许多复现研究复述为对 RL 的工作机制画像；讲者给自己的处方是后期应加大温度或最大 rollout 数以维持真实探索。

### SimpleRL-Zoo：把配方搬到动物园，用新指标判断「真变长」

讲者随后指出在 Qwen 上做的实验有特殊性：Qwen 预训练可能混入过 post-training 数据与反思行为。为回答「能否在野生 base 上复现」，SimpleRL-Zoo 扩展到 Qwen2.5 0.5B–32B、Mistral-7B、Llama-3.1-8B、DeepSeek-Math-7B、Mistral-Small-24B 等（字幕口径）。他称在这一系列上都观测到 accuracy 与 response length 同步提升，而 Stanford 团队在 Llama-3.1 上没看到（引讲者口径，未做交叉核对）。

讲者提出两个比 length 更真实的指标：截断比例（clip ratio，过长被 budget 截掉的 response 比例）与未截断的平均长度。在他举的例子里 Mistral-7B 的「变长」是不健康的——clip ratio 极高、绝大多数 response 被截断；而其它模型 clip ratio 保持低位。

他借 Stanford 的认知行为分类法统计 backtracking、verification、enumeration（穷举）与 subgoal setting 四种行为：Mistral-7B 四种基本无变化，而 Llama-3.1、DeepSeek-Math-7B、Mistral-Small-24B 会在 RL 里逐渐发育；Mistral-Small-24B 的 backtracking 从近零提升到约 +50% 左右（字幕口径）。对 Qwen2.5 的 7B/32B 这些行为较平稳，讲者的解释是 MATH 数据对它们已经太简单，不需要复杂或新策略。

pass@k 争论上他正面回应：DeepSeek 团队论文里 RL 模型在 pass@k 上并未超过 instruct；而 SimpleRL-Zoo 在 AMP/AIME/AMC/MATH500 上测得 pass@k 随训练步数持续抬升（展示了 K 从 1 到 128 的曲线），他借此主张 RL 是在系统性提升 reasoning 能力而非只移动概率质量。

### 动物园的三条经验法则

1. **format reward 会毒害弱基座**。对比 format reward 的有无实验里，Qwen2.5-7B 与 Llama-3-8B 都出现负面影响（字幕口径）；Llama-3.1-8B 一度直接把 accuracy 打到零。讲者的解释是早期用格式约束硬掰输出方式会约束探索、对弱 base 伤害最大。
2. **数据难度要匹配基座能力**。Mistral-7B 在 GSM8K 级简单数据上最稳健、面对 MATH Level 3–5 难度即退化弃答；Qwen2.5-7B 需要 Level 3–5 才激发长链与性能提升。讲者建议用 pass@k 落到合适区间去为不同模型定量划分难度带。
3. **pre-SFT 冷启动可能会反噬**。在 RL 前插入 short-CoT SFT（0、100、500 步对比）里，讲者称没有 SFT 的 base 上限最高、长度增长最积极，SFT 越深探索被限制得越狠（字幕口径）。Q&A 里他把缓解办法指向长 CoT 冷启动：R1-Zero 的长 CoT 数据冷启动反而可能小幅强化探索。

## 关键数字总表

| 指标 / 观察 | 字幕口径 | 来源 |
|---|---|---|
| SimpleRL 训练数据 | 8000 条开源数学数据，零 SFT | 字幕（方法部分） |
| SimpleRL 基座 | Qwen2.5-Math-7B base | 字幕（方法/实验） |
| SimpleRL 规则奖励 | 对 +1 / 格式对答案错 -0.5 / 格式错 -1 | 字幕（方法部分） |
| Single rollout 数 | 每条数据 rollout 8 次 | 字幕（对比表） |
| RStar-Math 资源（对比） | SFT 7M、RL 300 万 × 16 rollout | 字幕（对比表） |
| PRIME 资源（对比） | SFT 23 万、RL 15 万 | 字幕（对比表） |
| SimpleRL 相对 base 的幅度 | 约 +20 个绝对点（图表口播） | 字幕（实验结果） |
| B-STaR 自动温度 | 训练早期约 $T \approx 0.5$ ，后期自动拉高 | 字幕（配置可视化） |
| SimpleRL 课程资源（Q&A） | 早期 PPO 约 2 天、4 节点；GRPO 后小模型单卡数小时 | 字幕（Q&A） |
| SimpleRL-Zoo 动物园规模 | Qwen2.5 0.5B–32B、Mistral-7B、Llama-3.1-8B、DeepSeek-Math-7B、Mistral-Small-24B | 字幕（实验部分） |
| pass@k 跨度 | K 从 1 到 128 的变化曲线 | 字幕（pass@k 部分） |
| simpleRL-reason GitHub | 约 4000 star（讲者口述） | 字幕（开场介绍） |

## 可迁移

- 把 self-training 看作「两个能力都要监控」的事情：只追 accuracy 会把探索能力饱和误读为算法瓶颈；至少同时给出 pass@k 与 clip ratio 两条轨迹，再下结论。
- B-STaR 的自动调参思路直接就是一条工程处方：在 zero RL 里用「当前 policy 与 reward 的实际配对状态」实时调温度与阈值，比拍一个静态 schedule 更划算，且讲者的观察是这条调参本身就能再换出更好的性能。
- 对做 RL 数据的人最实用的启示：数据难度要相对于 policy 而言。讲者的说法是看 pass@k 在什么区间——太难的题对他来说等同没信号、太简单等同只给捷径。难度评定应随 policy 能力动态标注，而非一次性打标。
- format reward 与短 CoT 冷启动都要警惕：看似无害的「对齐格式」与「热身一下」，对 RL 里对弱基座可能是把探索空间收紧的隐性约束。默认关闭、先跑 baseline，再以实验说服自己加回来。
- 长度不能当第一信号：讲者的 length 与 clip ratio 两报一的办法很便宜——被截断的样本比例如果开始上升，length 上涨大概率是垃圾增长、不是更长的思维链。

## 疑问 / 下一步

- B-STaR 的 balance score 精确定义（对 $N'_i$ 的惩罚项形式与温度/阈值的自动搜索细节）讲者说要看论文；纪要按口播只给出概念框架，具体数学形式未从字幕确立。
- pass@k「大幅抬升」是讲者用曲线主张系统性提升，但讲者没有在正片里给出基座对比表的三位有效数字（GRAM8K / MATH500 / AIME24 的 percent），做成笔记时适合再打开论文核对。
- SimpleRL-Zoo 关于「Stanford 在 Llama-3.1 上未看到正向结果、我们让它 work 了」的差异来自讲者单方叙述，后续若要引用需另行核对 Stanford 文章的具体设置差在哪里。
- pre-SFT 冷启动伤害探索是讲者较强的判断，但他给出的对比是 short-CoT SFT；long-CoT 冷启动缓解只是「当前在跑实验」的预告语气，未定案。

## 原文金句（1-2句）

> 「我们只有有了不断会有更高质量的合成数据、更 high quality 的 response，我们才能去推进我们的模型不断的去变强。」——讲者在界定 self-training 真正的瓶颈（字幕 07:18–07:28）

> 「RL 它其实可以被当成一种形式的数据合成策略，它可以帮助我们去克服人类数据的限制。」——讲者对 RL 与数据工程之间关系的总结（字幕 06:55–07:07）

> 「response length 它仅仅是一个很表层的 metric，我们应该去关注更多更真实的、更能反映模型 reasoning 变化的 metric，让我们能让整个过程变得更加白盒。」——SimpleRL-Zoo 新指标提出时讲者的话（字幕 47:21–47:38）
