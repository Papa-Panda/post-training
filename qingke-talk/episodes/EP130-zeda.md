# EP130 — ZEDA：将 Post-Trained MoE 迁移为高效动态 MoE

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP130-zeda.html

> 「对于模型来讲，它所认为的困难可能并不是人类所认为的困难，反而是和 teacher 的差异、和输出的不确定度（相关）。」——讲者在分析实验中的判词

## 元信息

- 期号：130
- 标题：ZEDA：将 Post-Trained MoE 迁移为高效动态 MoE
- BV：BV1cJEd6kE1t
- 时长：01:03:31（讲授约 44 分钟 + Q&A 约 19 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：吕兴泰（清华大学直博二年级；导师周伯文教授——字幕自述音近「周博文」，以论文作者页 Bowen Zhou 为准）
- 相关论文：Xingtai Lv, Li Sheng, Kaiyan Zhang, et al., *Post-Trained MoE Can Skip Half Experts via Self-Distillation*，https://arxiv.org/abs/2605.18643 （v2 2026-06-08）
- 相关代码：https://github.com/TsinghuaC3I/ZEDA
- B站链接：https://www.bilibili.com/video/BV1cJEd6kE1t/
- 字幕原文存档：本地 `transcripts/EP130.txt`（1300 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP130，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如 ZEDA 在字幕中作「Z大」、AdaMoE 作「挨打MOE」、OPD 作「OBD」、MATH-500 作「max500」，均以论文与公开资料为准）。论文（arXiv:2605.18643）仅作交叉引用；凡讲授口径与论文版不同，以下标注（字幕口径）。

## 一句话总结

ZEDA 回答的是一个此前没人系统做过的迁移问题：部署时真正用到的 MoE 模型都已经走完预训练 + SFT + RL/OPD 全流程，能不能低成本地把这种 post-trained 静态 MoE 改造成按 token 分配计算量的动态 MoE？讲者的做法是两步：先往每层注入无参数的 zero expert（输出恒为零），让 router 可以在 top-k 槽位里「选空气」从而跳过真实 expert；再用原模型作 frozen teacher 做两阶段自蒸馏（先 SFT 后 OPD）恢复能力，并加一个只管「组间比例」不管「组内分布」的 Group Auxiliary Loss 把 zero expert 激活率钉在可控范围。结果是跳过一半以上 expert 计算、精度基本无损、端到端推理提速约 20%。

## 核心

### 引入：动态 MoE 的迁移版图里缺了 post-trained 这一块

静态 MoE 对每个 token 激活固定数量的 expert，计算量恒定；动态 MoE 则按 token 难度分配计算量，简单 token 少激活、难 token 多激活，已有大量工作表明简单 token 激活少量 expert 就够了。讲者把已有动态 MoE 工作分成两类：一类从零预训练（如美团 LongCat、MoE++，字幕作「MV加加」），一类从预训练好的 base 模型出发、针对单个任务做迁移（如 AdaMoE）。唯独「已经完整后训练过的模型如何迁移动态 MoE」是空白——而这恰恰是推理部署的实际场景：线上跑的全是 post-trained 模型，迁移成功就能直接摊薄线上推理成本。

动态 MoE 的实现方式讲者盘点了三种：expert-chooses-token（专家挑 token）、额外训练一个 allocator 预测每个 token 该激活几个 expert、以及插入零计算量 expert（zero / copy / constant 三种）。ZEDA 选第三种，理由直接：它对原模型参数的改动最小，最利于在不破坏已训分布的前提下做迁移。

### 第一步：zero expert injection，以及为什么不能用 copy expert

注入方式：在每层原有的 $N$ 个 normal expert 之外再加 $N_z$ 个 zero expert，激活总数（top-k）不变，但候选池扩大，token 选到 zero expert 的槽位就不产生计算。router 需要扩维新增参数，讲者用高斯初始化并对齐原 router 参数的均值与方差。

zero expert 的输出恒为零；另一种候选 copy expert 的输出等于输入。讲者做了一个预实验对比：简单 SFT 后在 5 个数学基准上，copy expert 让模型平均分从原模型的 80 多掉到 20 多（字幕口径）。失败机理分析是这场 talk 里很有意思的一段：

- **scale mismatch**：copy expert 引入的输出与原静态模型输出的相对 L2 范数差异全程保持在 80% 以上；zero expert 对应差异最高也就 20% 多。
- **direction mismatch**：把输出展平算余弦相似度，copy expert 的输出随层数加深与原模型输出趋于正交；zero expert 方向差异小得多。
- 进一步拆解 copy expert 的两部分（normal expert 输出 + 输入直通项）后发现，直通项是 scale mismatch 的主因，并且它会反过来把 normal expert 的输出也逐步「拉偏」。

结论：zero expert 之所以 work，不是因为它有什么魔力，而是因为它对原模型的扰动最小——「能不让网络算的东西就别动原分布」，这条原则后面还会反复出现。

### 第二步：两阶段自蒸馏 + Group Auxiliary Loss

注入后的模型只是个初始化，有时连完整的 thinking 过程都输出不了。恢复能力用两阶段自蒸馏，teacher 始终是原 post-trained 模型：

1. **SFT 阶段**：用原模型对 SFT prompt 做 rollout 得到数据，训动态模型；
2. **OPD 阶段**：标准的 reverse KL on-policy distillation。

Q&A 中讲者给了具体配置（字幕口径）：SFT 学习率 $2 \times 10^{-5}$ 、1 个 epoch、batch size 256；OPD 学习率 Qwen3 用 $5 \times 10^{-6}$ 、GLM 用 $1 \times 10^{-6}$ ，batch size $16 \times 2$ ，且只训 320 步——OPD 训太久会过拟合且开销极大。

两个阶段的 loss 都外加一项 **Group Auxiliary Loss**。它的动机：经典的 aux load-balancing loss（Switch Transformer 时期提出）会逼所有 expert 的被选概率和路由权重趋于一致，这对从头训没问题，但对一个已经训出不均衡路由分布的 post-trained 模型是破坏性的。讲者的改法是让 loss「看不见」单个 expert：只把 expert 分成 normal / zero 两组、只在组粒度上算均衡，并用系数 $\omega$ 控制目标比例，最终 zero expert 的激活概率收敛到 $p_z = \frac{\omega N_z}{N + \omega N_z}$ 附近（字幕口径）。这样既钉住了计算量预算，又不碰 normal expert 之间已经学好的分布。消融里 $\alpha$ （loss 权重）取 0.1 最好（字幕口径）；直接用经典 aux loss 性能「特别糟糕」。

Q&A 中两个相关追问值得记：一是为什么不用 LongCat 的 loss-free bias 调节——讲者说在迁移场景下 bias 控制波动大、钉不住目标比例，group aux loss 的均衡点可解析、更稳；二是只做路由熵最大化行不行——理论上能达到同样均衡点，但按 aux loss 的经验会非常不稳定。

### 实验：两个模型、十一个基准、一半计算

设置（字幕口径）：

- **Qwen3-30B-A3B**：128 个 normal expert、top-8；注入 64 个 zero expert。
- **GLM-4.7-Flash**：64 个、top-4；注入 32 个 zero expert。（注入数量 follow LongCat 设置，正好减半。）
- 11 个基准、3 个领域：数学（AIME 24/25/26、MATH-500 等）、代码（LiveCodeBench v5/v6、HumanEval+、MBPP+）、指令跟随（IFEval、IFBench）。

主结果（字幕口径，论文口径见关键数字表）：ZEDA 后两模型 11 任务平均分几乎无损（Qwen3 由 47.9 到 47.2；GLM 由 72.5 到 71.8），同时 zero expert 激活率达到 51.2%（Qwen3）与 53%（GLM）——即跳过一半以上 expert 计算。论文摘要口径：超过最强动态 MoE baseline 6.1 分（Qwen）与 4.0 分（GLM）。

对照实验讲了三层意思：

1. **胜过两个外部 baseline**：AdaMoE 与训练后剪枝类 Dynamic Capping 都在某个领域上大幅掉分，只有 ZEDA 三个领域都稳。
2. **胜过自己的静态变体 NET**（强行把 top-k 减半）：说明收益来自「动态地让不同 token 减不同量」，不是单纯少算。
3. **胜过 ZEDA-SFT-only**：OPD 阶段不可省；且消融显示只做 OPD（跳过 SFT）是所有变体里最差的——起始模型能力没对齐时强行 OPD，路由初始化太差。SFT 先行本质上是给 router 一个像样的起点。

成本与落地数字：整个迁移在单机 8 张 H200 上，Qwen3 不到 31–32 小时（论文口径 31 小时内、字幕 Q&A 口径 32 小时）、GLM 62 小时，含 teacher rollout 采 SFT 数据的时间；实测推理（输入输出各约 8K）prefill 与 decode 吞吐都提速约 20%（论文口径约 1.20×）。数据量上讲者给的经验值是 Qwen 级别模型准备 40K–60K 条 SFT 数据比较合适，且数据领域要覆盖目标部署领域——只在数学上训对代码有一定泛化，但不如直接备齐对应领域数据。

### 分析：模型到底把计算量花在哪

这是讲者自己说「比较有意思」的一部分。把每个 token 的 zero expert 激活率与训练时可观测量作图，得到三条规律：

1. **与 teacher–student 分歧正相关**：OPD 阶段的 teacher–student log-prob 差越大、student 熵越大的 token，zero expert 激活率越低——模型给自己不确定、与 teacher 分歧大的 token 分配更多计算。这符合直觉：不确定 = 难 = 多算。
2. **与 response 模式强相关，且反直觉**：可视化三条轨迹发现，数学公式与代码 token 反而被分配更少计算（zero 率更高），被分配更多计算的是句首的自然语言脚手架 token（如 "OK, I need to solve this problem"）。
3. **与人类标注的任务难度无关**：MATH-500 五个难度等级加 AIME 24，zero 率都在 51%–52% 之间，没有随人类难度单调变化；与层号也没有干净的规律。靠后 token zero 率略升，是因为回答尾部公式/代码密度更高，回到第 2 条。

讲者对第 3 条的解读很克制：不是模型「不懂难度」，而是当前训练信号（SFT/OPD）里没有人类难度这个变量，模型学到的是与 teacher 分歧和自身不确定性挂钩的难度观——如果在训练中额外喂人类难度信号，它大概率也能学会对人类难度的感知。

### 局限（讲者自列）

- 提速随序列变长衰减：输入输出从 2K 涨到 8K，speedup 逐渐变小，KV/attention 等固定开销占比上升后 MoE 计算节省被稀释。
- 只验证到 30B-A3B 量级，更大模型未验。
- 只省计算不省通信：实验里 all-to-all dispatch/combine 没做优化，讲者说通信更快时 ZEDA 的优势会更明显；同理若 attention 换成线性注意力，MoE 侧节省的占比会更大。

## Q&A 要点

- **频繁迭代权重的场景能不能用？** 看迭代周期与迁移成本（约 32 小时）的比值：迭代周期远大于 32 小时可用；若一小时就能迭一版，等收敛到确定版本再做 ZEDA。
- **和 LoRA / 全量微调比开销？** 讲者坦白没做对照实验；ZEDA 的 SFT/OPD 本身就是全量微调，同数据量下训练开销和全量微调相当、不比 LoRA 省——它的收益在推理侧（约 −20%），这是 LoRA 给不了的。
- **教师冻结、学生会不会过拟合教师的噪声路由？** 理论上蒸馏够久会把教师优缺点都学到，但实验中未观察到明显过拟合。
- **zero expert 与动态 K 的区别？** 结果上等价（都是部分 token 只用少数真实 expert），但 zero expert 方案对原模型参数扰动最小，迁移场景优先选它。
- **超参稳定性**：SFT 阶段太短会让路由初始化变差，这是 OPD-only 变体垫底的原因；两阶段各自训到指标涨幅趋平即停，不必追求训满。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| Qwen3-30B-A3B 11 任务平均分 | 原模型 47.9 | ZEDA 后 47.2 | 字幕（实验部分） |
| GLM-4.7-Flash 11 任务平均分 | 原模型 72.5 | ZEDA 后 71.8 | 字幕（实验部分） |
| zero expert 激活率（= 跳过的 expert 计算占比） | — | Qwen3 51.2% / GLM 53%（均过半） | 字幕（实验部分） |
| 相对最强动态 MoE baseline 的平均分优势 | — | Qwen +6.1 分 / GLM +4.0 分 | 论文 arXiv:2605.18643 摘要 |
| copy expert 预实验（5 个数学基准平均） | 原模型 80 多 | 约 20 多 | 字幕（方法部分） |
| 迁移总耗时（单机 8×H200，含采数据） | — | Qwen3 <31–32 小时 / GLM 62 小时 | 论文口径 31 小时；字幕 Q&A 口径 32 小时 |
| 实测推理提速（输入输出各约 8K，prefill 与 decode） | 原静态模型 | 约 +20%（约 1.20×） | 字幕（实验部分）；论文摘要同口径 |
| Group Aux Loss 权重 $\alpha$ | — | 0.1 较优 | 字幕（消融） |
| SFT 数据量经验值 | — | 40K–60K 条（Qwen 级别） | 字幕（Q&A） |
| OPD 训练步数 | — | 320 步 | 字幕（Q&A） |

## 可迁移

- 「先把可精确求解/零扰动的部分摘出去，剩下的再交给学习」这条选型逻辑在 infra 里通用：zero expert 的价值不在结构巧妙，而在它对已训分布零扰动。做模型压缩/路由改造时，先问哪种改造对现有分布扰动最小，往往比问哪种最强更重要。
- Group Auxiliary Loss 的「只管组间比例、不管组内分布」是可迁移的控制模式：任何想给已训练系统加预算约束又不想重训分布的场景（专家并行负载、KV 预算、工具调用次数），都可以把约束做在粗粒度聚合量上而不是逐元素均衡上。
- 这期给的难度观值得注意：模型自发的计算分配跟的是「与教师的分歧 + 自身熵」，不是人类难度。做 adaptive compute / 路由策略时，用 teacher–student 分歧当难度信号可能比人工难度标签更对齐模型实际需求。

## 疑问 / 下一步

- 更大规模 MoE（百 B 级、带 shared expert 的结构）上 zero expert 的最优注入比例与收益曲线未验证；讲者自己也把规模列为局限。
- 迁移后的模型如果继续 RL 训练，zero 率会如何漂移、Group Aux Loss 要不要全程挂着，没有实验。
- 通信侧的账没算清：all-to-all 未优化时是 +20%，换高效 EP 通信后实际能到多少，决定了这条路线在真实 serving 栈里的上限。
- 讲者对「为什么公式/代码 token 反而少算」没有给出机制解释，只是现象报告——这是理解动态计算分配的一个未决点。

## 原文金句（1-2句）

> 「对于模型来讲，它所认为的难（的 token），可能并不是人类所认为的困难，反而是（和）teacher 的差异，或者是和输出的不确定度（相关）。」——分析实验结论（字幕约 53:53–54:07，按干净口径转写）

> 「（copy expert 引入的）这部分会一点一点地把原本这些 normal expert 的输出，也拉得和最开始原始的静态 MoE 的输出变得不相像。」——讲者拆解 copy expert 失败机理时的总结（字幕约 21:02–21:13）
