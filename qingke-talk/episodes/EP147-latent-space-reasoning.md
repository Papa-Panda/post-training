# EP147 — LatentSeek & GradCuit：从价值引导的隐空间优化看大语言模型推理

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP147-latent-latent-space-reasoning.html

> 「A great latent space is one where following the gradient of value is enough to solve complex problems.」——讲者反复回到的判词：推理的核心不在生成更多 token，而在找到一个只需沿 value 梯度下降就能解题的隐空间

## 元信息

- 期号：147
- 标题：大语言模型在 latent space（隐空间）里的优化和推理（B站标题：LatentSeek & GradCuit：从价值引导的隐空间优化看大语言模型推理）
- BV：BV1Bw8m6GEd7
- 时长：01:08:07（讲授约 48 分钟 + Q&A 约 20 分钟）
- 提炼日期：2026-10-02
- 分享嘉宾：李恒利（北京大学；字幕中自述姓名，英文名 Hengli Li，以论文作者页为准；GradCuit 论文的 project lead）
- 相关论文：*Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space*（LatentSeek），https://arxiv.org/abs/2505.13308 ；*GradCuit: Credit-Assigned Gradient Flow Enables Robust and Interpretable Test-Time Latent Reasoning*，https://arxiv.org/abs/2608.02585
- 相关代码：https://github.com/bigai-nlco/LatentSeek ；https://github.com/Yuzhaoxin946/GradCuit
- B站链接：https://www.bilibili.com/video/BV1Bw8m6GEd7/
- 官网预告：https://qingkeai.online/blog/LatentSeek%26GradCuit
- 字幕原文存档：本地 `transcripts/EP147.txt`（1586 条，带时间戳）

> 📝 提炼方式说明：本纪要基于 B站 AI 字幕原文（青稞Talk EP147，已存档）清洗提炼，信息来源为 AI 字幕原文（经人工清洗；个别专名可能有识别误差，如 GradCuit 在字幕中作「GRACKET/GRIT」、LatentSeek 作「latent thick」、Coconut 作「coconuts」等，均以论文为准）。论文（arXiv:2505.13308 与 arXiv:2608.02585）仅作交叉引用；凡讲授口径与论文版不同，以下标注（字幕口径）。官网对本期的定性是「LatentSeek & GradCuit」，讲授亦以此为序；两项工作同属 BIGAI/UCLA 一线（论文通讯 Zilong Zheng、Ying Nian Wu），李恒利为共同作者与 GradCuit 的 project lead。

## 一句话总结

讲者主张把推理从「在 token 空间里 likelihood 最大化」改写为「在隐空间里沿 value 梯度优化」：给定 prompt $c$ ，不要问什么样的 latent $z$ 最能生成答案 $x$ ，而要问什么样的 $z$ 能生成 reward 最高的 $x$ ——因为验证一个答案比求解它便宜。两篇工作沿这一 formulation 各推进一步：LatentSeek 把 LM head 前最后一层的 hidden states 当作一串可优化的 latent 向量，用 self-reward（模型给自己的答案打分）驱动实例级 policy gradient，在 CoT 基线之上平均涨约 10 个点（字幕口径）；GradCuit 则发现 LatentSeek 的死穴在 credit assignment——latent 与后续推理之间隔了一层不可导的 decoding，于是把 latent 前移到 Transformer 中间层，让每个后续 token 的 log-prob 都经由 self-attention 对每个 latent 可导，reward-weighted 梯度可以直接、稠密地回流，学习率鲁棒性与可解释性同时变好。两篇的共同判词是：**推理的瓶颈可能不在采样策略，而在有没有一个「沿 value 梯度走就能解题」的隐空间**。

## 核心

### 引入：为什么要在 latent 里推理，以及隐空间从哪来

讲者从统计学史切入：从 David Mumford 一代的统计学家起，「感知」就被理解为反演——看到观测 $i$ ，找其背后的 latent structure / hidden causes（柏拉图洞穴的现代形式）。贝叶斯写法是经典的

$$p_\theta(z \mid i) \propto p_\theta(i \mid z) \, p_\theta(z)$$

——什么样的隐表示能生成所见之物；AIGC 与 diffusion 的思想根源都在这里。语言模型如今的主流是自回归

$$p_\theta(x \mid c) = \prod_t p_\theta(x_t \mid x_{<t}, c)$$

而推理（o1/R1 之后）默认走 CoT：把思考外化为长 token 序列。讲者列出 CoT 的三重限制：只能落在人类可读语言里（真思考有 incubation period 式的潜意识过程，未必可言说）；序列太长、慢且难控制；训练侧还要为它收集昂贵的长标注。问题于是变成：**能不能让模型用人类看不懂的方式思考？**

当下 latent reasoning 有两条来路：Coconut 一派让连续 hidden state 自回归地迭代（像 loop transformer 的 for 循环）；贝叶斯/EM 一派仍按老路找 $z^* = \arg\max_z \log p_\theta(x \mid z, c)$ 再 ELBO 训练。讲者指出两派在 inference 侧并无本质区别——都是先 infer 隐结构、再生成答案，区别只在怎么把参数 $\theta$ 训出来。

### 核心 formulation：从 likelihood-driven 到 value-driven

讲者的关键一步是把 inference 侧的目标换掉。旧框架找 likelihood 最好的 $z$ ；新框架找能导出 reward 最高答案的 $z$ ：

$$z^* = \arg\max_z \; R(x, c), \quad x \sim \pi_\theta(\cdot \mid z, c)$$

（附带一个可丢弃的先验项）。直觉是「验证比求解简单」：求解要从生成分布里采样出推理路径，验证只需 reward 给个对错判断；这套 formulation 等把验证能力转移进求解过程。 $z$ 的诱导分布由此是一个由 value 定义的 energy-based 分布。两个边界要划清：其一， $\pi_\theta$ 全程冻结，这是纯 inference 框架，叫 test-time training 并不准确，更像 test-time search/reasoning；其二，它与 RL 后训练的区别正在于此——RL 也用 reward，但更新的是参数本身，这里参数不动、只动当前实例的 latent。讲者的一句凝练（talk 中英文原话）是：

> A great latent space is one where following the gradient of value is enough to solve complex problems.

即推理的核心问题被重新表述为：**找到这样一个隐空间，在其中只做最简单的（value 引导的）梯度下降，就能解开复杂问题**。后面两篇工作都是在不同位置上找这个空间。

### LatentSeek：在最后一层 hidden states 上做 policy gradient

第一篇（*Seek in the Dark*）的空间选址：LM head 之前、最后一层 Transformer 的输出。给定 prompt $c$ ，latent 是一串 $N$ 个向量 $z = (z_1, \dots, z_N)$ ，先经 LM head 解码成 token，再把这些 token 与 prompt 一起送回模型继续生成答案，最后拿 reward。流程是闭环：生成→打分→按 policy gradient 更新 latent→再生成，直到 reward 超过阈值或超时。

三个实现决定值得单独记：

1. **独立性假设与 policy gradient 形式**。因为解码这一步不可导，无法沿梯度直接上升，改用 policy gradient；为使更新可分解，假设各 $z_t$ 相互独立，每位的梯度是

$$[\nabla_z J(z)]_t = \mathbb{E}_{x \sim \pi(x \mid z, c)}\left[R(x, c) \, \nabla_{z_t} \log \pi(x_t \mid z_t)\right]$$

讲者坦承这个假设「肯定不完全对」，并给出理论补丁（论文 Appendix C）：他们定义了一个借自多证明者交互证明的复杂度类 MIP-Bounded（每个 prover 只许说一个 token，对应每个 latent 独立），并证明只要能找到多项式时间的 verifier，这个受限类与原 MIP 在表达能力上等价（乃至等价于指数时间图灵机）——即独立性的代价可被一个好 verifier 补偿。这个类并非首创，ITCS 2013 的 *On the Power of Many One-Bit Provers* 已有相近定义。

2. **初始化代替先验**。formulation 里先验被删掉，补救办法是拿第一次推理产生的前 $\rho$ 比例 token 的 hidden states 做出发点（ $\rho$ 经验取约 20%），等价于把先验经由初始化带回来；讲者说完整做法（假设高斯先验、加 $L_2$ 正则）当然可行，但为了简单直接删掉。

3. **Self-reward**。reward 不用训好的 reward model，也不用更强的外部大模型：直接问模型本身「这个答案对的程度是多少，给个百分比」。理由有二：训好的 reward model 在 outcome reward 上反而不如 self-reward（过程奖励上 PRM 稍好，判最终答案它会错得更多，论文附录有实验）；而若可用 GPT 级闭源模型打分，何不直接让它答题。reward 的定义由此完全内生。

实验（字幕口径）：与 CoT 基线比平均涨 10.75 分，MATH-500 约 +3.04 分、AIME2024 约 +4.07 分（AIME 只有 30 题，该涨幅「还是有点说法的」）；骨干是 Qwen2.5-7B、LLaMA3.1-8B 这一代模型（讲者承认偏旧）。几个分析结论：

- **新的 test-time scaling 轴**：旧 scaling 是生成更长或采更多（BoN、self-consistency），这里是增加 latent 优化的迭代轮数，accuracy 随轮数上升；简单题平均一轮收敛，难模型/难设置轮数更多、涨幅也更大。
- **梯度方向不可替代**：消融把 policy gradient 换成纯 random search（高斯噪声加在 $z$ 上），保留 reward 做选择，性能显著更差——如 LLaMA3.1-8B 上高约 30 个点、Qwen2.5-7B 上高约 20 个点（字幕口径）； $\nabla \log \pi$ 本身就含有指向正确方向的信息。
- **长度不是作弊来的**：最终答案长度与普通 CoT 的比值在 1 附近，即不是靠让模型说更长取胜。
- **代价说清**：它是 sequential 算法，单题延迟天然高于可并行的 BoN/self-consistency；且当前推理 infra（vLLM、SGLang）很难把 latent 取出来，这个框架在真实部署上吃亏，data-parallel 摊薄后理论上可比，但未做工程验证（字幕口径）。
- **定性发现**：优化后的 latent 解码出来是「Let's find this, , let you more understand it down step to and let」一类人类不可读的 token 串，却能导出正确答案——与 prompt tuning 的区别是，这里的 latent 是从模型自身推理中 infer 出来的、逐实例优化的，而非在整个数据集上训出来的一段前缀。PCA 下看各 latent 的优化轨迹则形状各异、看不出规律——这正是下一篇的引子。

### GradCuit：把 latent 前移，让 reward 梯度经 attention 直接回流

第二篇（*GradCuit: Credit-Assigned Gradient Flow …*，名字取 gradient through circuit）从 LatentSeek 的两个病入手：**credit assignment 不可知**（一串 latent 都在更新，不知道哪个起了作用）与 **latent dynamics 不稳定**（轨迹什么形状都有，稍改就崩，对调参极其敏感）。解法是一个位置变化带来的机制变化：

- **空间前移**：latent 不再放在 LM head 前，而插入 Transformer 中间层（经验上 25%–50% 深度，如 32 层模型取第 8 或 16 层；讲者借 Twitter 评论的说法，叫在更深的「记忆宫殿 mind palace」里推理）。
- **机制随之改变**：中间层的 latent 不必先解码成 token，可直接经 causal self-attention 影响后续所有位置的生成；于是后续每个 token 的 log-prob 都对每个 latent 可导。用论文摘要的话说，self-attention 为每个 continuation token 的 log-prob 提供了一条通往每个前置 latent 的可微通路，**reward-weighted 梯度从整条续写直接、稠密地分配给每个 latent**。相对 LatentSeek 少掉一层不可导的解码，梯度估计的第一层噪声源消失；优化仍用同一套 policy-gradient 式更新，且全程同一套参数（这一点讲者在 Q&A 中被反复确认）。
- **可解释性第一次有了抓手**：把每个后续 token 对各 latent 的梯度模长打出来，发现影响力集中在 because / therefore / however 这类 **reasoning connectors（推理转折词）** 上——latent 主要在调控「逻辑往哪转」，而不是盯着「 $1+1=2$ 算没算对」或答案格式；这与讲者们起初的猜测（以为会关注答案内容或 boxed 格式）相反。因为不再解码，中间 token 也不再出现 LatentSeek 那种乱码串，生成文本基本可读。

实验对照（字幕口径）：GradCuit 在多个骨干（含 Qwen3-4B、LLaMA3.2-3B）与 GPQA（物理/化学/生物）等新基准上取得同组最优；论文口径为五骨干三基准平均 64.5%，比 CoT 高 6.6 个百分点、比最强竞品高 2.4 个百分点。鲁棒性是更硬的卖点：

- 学习率在原值 ±60% 范围内扫，LatentSeek 性能大幅波动，GradCuit 波动很小；准确率标准差 0.82 对 1.53 （论文口径一致），「腰斩」。
- 方向鲁棒性（消融，讲者自己都称意外）：在 GradCuit 的中间层空间里做纯 random search，竟能与 LatentSeek 的梯度法打得有来有回、个别设置还更好——说明这个空间里「对的方向太多，随便走都容易踩到对的」；当然有 reward 引导的梯度法仍明显更强。

讲者收尾的自我定位：可解释性离「完全可解释」还差得远，这只是在 credit 如何分配上迈了一小步；这条线上可对照的工作还包括 LTPO（中科院/牛津）、他们自己把同一套方法迁到图像生成的 Mirage 一篇（ICLR），以及 CVPR 的 Demirage 一类后续。

### Q&A 要点（含讲者最坦白的边界）

- **稀疏奖励下单样本梯度为什么够？** 他们实际只采样一次来估梯度，「单单（这件事）不可理解」但一次样本已足够好；多次采样取平均会更好，这也是 policy gradient 的性质使然。
- **Coherence 无法保证**。独立性假设把 latent 之间的关联完全抹掉，这是 LatentSeek 乱码 token 的根源；GradCuit 因为不解码而绕开了这个问题，但讲者补了一句反问：可解释性也许只是人类的需求，模型未必需要。
- **早期事实/计算错误会被回溯修正，还是会扭曲后续逻辑去迎合错误前提？** 讲者称这个问题「值得单独做一篇 paper」：目前不可解释，只能说 GradCuit 在这方面有一点小进步，latent 在中间层之后的 dynamics 仍是黑盒。
- **与 Coconut 的区别**：Coconut 的 latent 是自回归生成的、优化的是 log-prob；这里的 latent 彼此独立、用 value 引导的 policy gradient 优化。与 iCoT 的区别：iCoT 把长 CoT 的各 token 表示逐层塞回去，在一次 forward 内完成 CoT；这里是按 value 做反向传播引导。
- **「latent」与「hidden representation」**：实现上无区别；可认为 latent 服从高斯分布、infer 出来的是其均值，先验在实践中可删。
- **理论上限**：讲者直说不知道，得等后续工作；infra 支持（vLLM/SGLang 取 latent、做 gradient-based inference）若跟上，简单题上甚至可能加速推理，但没做过实验，不给肯定答复。

## 关键数字总表

| 指标 | 基线/口径 | 结果/数值 | 来源 |
|---|---|---|---|
| LatentSeek 平均分 | CoT 基线 | 约 +10.75 分 | 字幕 |
| LatentSeek MATH-500 | CoT 基线 | 约 +3.04 分 | 字幕 |
| LatentSeek AIME2024 | CoT 基线（30 题） | 约 +4.07 分 | 字幕 |
| LatentSeek GSM8K | BoN | +15.23 分 | 论文 arXiv:2505.13308 |
| 梯度 vs 随机搜索（LatentSeek 空间） | random search 消融 | LLaMA3.1-8B 约 +30 点、Qwen2.5-7B 约 +20 点 | 字幕 |
| latent 初始化比例 $\rho$ | 首次推理前段 token | 约 20% | 字幕 |
| GradCuit 平均准确率 | 五骨干三基准 | 64.5%（CoT +6.6 pp；最强竞品 +2.4 pp） | 论文 arXiv:2608.02585 |
| 学习率鲁棒性（准确率标准差） | LatentSeek 1.53 @ LR ±60% 扫描 | GradCuit 0.82 | 字幕/论文一致 |
| GradCuit latent 插入深度 | 25%–50% 层深（如 32 层取第 8/16 层） | 早期至中间层最优 | 字幕/论文 |
| LatentSeek 平均更新轮数 | 简单题 | 约 1 轮收敛（平均口径） | 字幕 |

## 可迁移

- 对 RL infra 的直接启发：这条工作线本质是把 policy gradient 从参数空间搬到激活空间——同一套「采样→reward→ $\nabla \log \pi$ 加权」的更新律，优化对象换成每实例的 hidden states。若推理引擎（vLLM/SGLang 级别）将来支持把中间层激活取出并回注，test-time 就会多出一个与 BoN/self-consistency 正交的 scaling 轴：按题难度分配的优化轮数，而不是一律堆采样数。
- Credit assignment 的定位法可借鉴：LatentSeek 的失败不是「隐空间不行」，而是监督信号隔了一层不可导解码；GradCuit 的全部收益都来自把信号通路改直。做 latent/agent 侧的梯度或反馈设计时，先查「反馈到被优化量之间有几层不可导」，比先调学习率更优先。
- Self-reward 的取舍判据：训好的 ORM 在「判最终答案」上不如让模型自评百分比——在 verifier 选型时，outcome 级判定与 process 级判定要用不同口径分别验，不要默认 PRM 全面占优。
- 「Reasoning connectors 梯度最大」这一归因方法（逐 token 打梯度模长）是一个轻量的可解释性探针：可用于检查任何 latent/prefix 干预到底在调控逻辑转折还是在拟合格式。

## 疑问 / 下一步

- 讲者自陈未解的核心问题：隐空间优化能否回溯修正早期事实错误、还是只会扭曲后续逻辑迎合错误前提——当前两篇都不可解释，这决定了该方法在长链推理上的可信边界。
- 基础设施缺口是落地前置条件：主流推理引擎取不出、也回注不了中间层 latent，GradCuit 式「梯度穿过 attention」在生产推理栈里暂无对应实现；data-parallel 摊薄延迟的说法停留在口头估计。
- LatentSeek 的实验骨干（Qwen2.5/LLaMA3.1 代）偏旧，强推理模型（R1 式长 CoT 模型）上这套实例级优化还剩多少增益，talk 未覆盖；GradCuit 的 GPQA 细分数值亦未在讲授中给出。
- 理论侧只证明了「独立性可被好 verifier 补偿」的表达能力等价性，对最优 verifier 不存在、reward 有偏时的收敛与偏差控制，讲者在 Q&A 中明确说尚无答案。

## 原文金句（1-2句）

> 「A great latent space is one where following the gradient of value is enough to solve complex problems.」——讲者的核心命题：好的隐空间里，最简单的梯度下降就足以解题

> 「（可解释性）这件事情有可能……单纯对人类而言需要可解释，但有可能对模型而言，它并不需要。」——Q&A 谈 coherence 与乱码 token 时讲者的回应
