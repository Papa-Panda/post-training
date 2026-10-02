# EP53 — Chain-of-Model（模型链）：引入因果建模，全新的大模型 Scaling 结构

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP53-chain-of-model.html

> 「我们既需要训练一个非常大的模型，但我们并不总是需要它。」——讲者对现行 scaling 范式的质疑原话

## 元信息

- 期号：EP53
- 标题：Chain-of-Model（模型链）：引入因果建模，全新的大模型 Scaling 结构
- BV：BV18K4szzEzT
- 时长：01:12:18（字幕末条 4319 秒 ≈ 01:11:59，含主持人开场与 Q&A）
- 提炼日期：2026-10-02
- 分享嘉宾：宋恺涛（微软亚洲研究院高级研究员；主持人介绍：发表论文 41 篇以上、Google Scholar 引用 12K 以上、开源社区 3 万多 star）
- 提炼方式：B站 AI 字幕原文（已存档 `transcripts/EP53.txt`，2114 条，带时间戳）清洗提炼；字幕把 Chain 普遍识别为「china / channel」、CoLM 作「column / colon」，ESP 作「ESP / SKYLA」等，均已按语境校正
- 相关工作：Chain-of-Model 范式及其语言模型实例 CoLM（Chain-of-Language-Model）

> 提炼方式说明：本纪要基于 AI 字幕原文（经清洗，专名可能有识别误差）。讲授中大量数字是在读幻灯片表格，字幕转写多处含混（如参数量、加速比的个别读数），下文只记讲者明确说清的数字，含混处标注存疑，不臆测补齐。

## 一句话总结

Chain-of-Model 把「因果」从序列维度搬到表征维度：一个表征被切成若干个依次嵌套的子表征（chain），第 $i$ 个子表征只能使用前 $i$ 个子表征的信息，于是同一个模型天然内嵌一串由弱到强的子模型——可以只激活前几条 chain 做弹性推理、可以拿现成模型当第一条 chain 再向外扩展维度继续训练、还可以在扩展后把微调出的专家接回大模型。讲者把它定位为 dense 与 MoE 之外的一种新的 scale 架构（scaling architecture），并在 Transformer 每一层（linear、attention、FFN、normalization、embedding、分类头）上给出了保持链式性质的实现，实例名为 CoLM。

## 核心

### 1. 背景：对现行 scaling 范式的三个质疑

讲者从人脑的动态性出发，对当下范式连发三问：

1. **固定激活**。无论 dense（训完 2T 就永远跑 2T）还是 MoE（DeepSeek 670B 每次固定激活约 27B），部署后任何 instruction 都要走同一套全量激活；但人对「1+1 等于几」并不需要绞尽脑汁。Existing 范式只有 fixed-size activation，缺 flexibility 与 extensibility。
2. **扩容必须从头训**。模型从 30B 做到 670B，每一代都要 train from scratch，因为各尺寸之间维度不对齐（dimension mismatch）；而人的学习是持续的、容量是渐进扩充的，不会把旧知识推倒重学。为什么不能复用已经训好的 30B、175B 再往上扩？
3. **理解与生成共用同一份能力**。Transformer 里 understanding（prefilling，处理 input 的 KV）与 generation（decoding）绑在同一个 dense 连接的模型上，但读一本书显然比写一本书简单——理解需要的 capability 不该和生成一样大。这也解释了 BERT 类只做理解的模型 scale 不上去的现象。

对应想要的性质：复用已有模型能力并渐进增强（progressive）；同一个模型内提供多档能力做弹性推理（快慢系统，简单问题走经验/memory 直接答，难问题才深想）；更快的 prefilling；以及大小模型之间动态切换时 KV 不必重算（现状是大小模型 KV 不能复用，切换有成本）。

### 2. 范式定义：链式表征、链式层、链式模型

讲者先把 foundation architecture 拆成两层：base architecture（如 Transformer、RNN）与 scaling architecture（如 dense、MoE）。Chain-of-Model 属于后者，其理念分三级递推：

**Chain representation（链式表征）**：一个 hidden 维度为 $D$ 的表征，不再当作不可分的整体，而是显式定义为 $N$ 个子表征的拼接，各子表征维度之和仍是 $D$ 。设想 hidden size 为 8：只激活前 2 维就是 scale 1（宽度 2 的尺度），激活前 4 维是 scale 2，激活全部是 scale 3。每个更大的 scale 复用更小 scale 的全部维度，于是能力随 scale 单调递增、且天然可复用。

难点在连接：若 scale 2 用到了 scale 3 的信息，嵌套性质就被破坏。于是定义：

**Chain layer（链式层）**：输入 $X$ 与输出 $Y$ 同为链式表征时，若每个 $Y_i$ 只条件依赖于 $X_{\le i}$ （即只用下标不超过 $i$ 的子表征），这一层就叫 chain layer。这正是把序列里的因果性（每个 token 只看自己和之前）搬到了表征维度上。

**Chain model（链式模型）**： $L$ 层全部满足 chain layer 性质的模型整体。它继承三个性质：

- **通用性（generality）**：任何网络都是 $N = 1$ 的 chain layer，所以任何现有模型都是 Chain-of-Model 的特例；任何网络都能被扩展成 chain 结构，不需要对网络有特殊要求。
- **因果性（causality）带来的弹性激活**：只要 $Y_i$ 就只需激活 $X_{\le i}$ 及相关权重，后面的权重完全不必算——这是 dense / MoE 都做不到的（MoE 也得部署全量 670B 且每次固定激活约 27B / 30B）。
- **组合性（compositionality）**：两层 chain layer 复合成的网络仍是 chain layer（由条件依赖的传递性直接可证），于是性质可以从一层无痛推广到整个模型。

与 MoE 的区别讲者讲得很细：MoE 的 experts 是等价能力的（128 个 expert 每个参数量一样、top-k 选取），Chain-of-Model 的「expert」是从弱到强的——第一条 chain 是个 1B 级 expert，加上第二条变 3B、再加第三条变 7B（字幕举例口径），像小学老师到高中老师逐级增强而非同级并列；MoE 只用在 FFN，Chain-of-Model 必须贯穿到模型层面；激活方式 MoE 是稀疏选取、这里是嵌套式激活。且两者正交：第一条 chain 可以是 dense、第二条是 MoE、第三条再 dense。讲者还强调这是一个全新的维度，与深度扩展、宽度扩展、NAS 都正交（后面讨论部分逐项对比）。

### 3. 在 Transformer 每一层上的实现（CoLM）

讲者逐模块给了保持链式性质的做法，核心在 linear 与 attention 两处：

- **Chain linear**：权重矩阵是块下三角结构——每个 $Y_i$ 只由 $W$ 中对应的块乘 $X_{\le i}$ 得到。极端情形（每维一条 chain）就是标准下三角，但实践中每维一条 chain 既不高效也学不好表征，要用 block-wise 分块（如 1024 维切成 256 / 256 / 512）。超参 $C$ 决定各 chain 维度： $d_{x_i}$ 按 $c_i$ 占总和的比例从 $D_x$ 中切分（此处原文公式字幕转写含混，仅记其定义作用）。naive 实现（for 循环逐块算）会带来多次 data access 与 all-reduce，input 方向的耗时甚至超过标准 linear（字幕口径：权重只有 104K 量级时 naive 版 input 处理仍更慢）；解法是 block-wise sparse kernel 把条状 block 张量并行分发到不同核上，优化后 input 方向可加速到一倍以上。讲者坦言这只是找到的一条路，imbalance tensor 的基础设施未来可能有更好的支持。
- **Chain attention**：QKVO 四个 linear 换成 chain linear 还不够，点积本身也必须满足因果性——若某个 head 混入了两条 chain 的信息，输出就会污染嵌套性质（边界不对齐时最容易发生，如 250 / 262 这样切分）。理想是每个 head 只含一条 chain 的信息，实现上令各 $c_i$ 之和等于 head 总数 $H$ ，于是 $c_i$ 的物理意义就是分配给 chain $i$ 的 head 数。每个 chain 生成自己的 K、V，query 只与同 chain 的 K、V 做 attention，chain 之间不碰撞。
- **KV sharing（扩展设计）**：更进一步，让所有 chain 的 K、V 都由第一条 chain 计算。这同样满足链式性质（ $Y_1$ 只依赖 $X_1$ ，而 $X_1$ 是 $X_{\le i}$ 的子集），换来两个独特性质：prefilling 大幅加速（KV 只算第一条 chain 那份）；不同 scale 之间无缝切换（KV 同源，放大缩小都不用重算）。代价是精度约掉 1 个点左右，可通过增加参数量补回。讲者把第一条 chain 解释成被单独拎出的 understanding 模块，但与 encoder-decoder 不同，它是嵌套在大模型内部的子模型而非独立模型。与 GQA / MQA 结合时复制顺序要换：按 $K_1 K_2 K_3 K_4$ 的顺序复制而非交错，因为这里的 query 是有顺序含义的。
- **FFN**：由 linear 加激活组成，直接把 linear 换成 chain linear 即可，组合性保证成立。
- **Normalization**：用最简单的 trick——对每条 chain 各自在其维度上做 normalization，不用全局统计量；少了 all-reduce、稍提效率，讲者称对效果基本没影响，normalization 主要起稳定作用，这样已足够。
- **Embedding**：训练时无特殊设计，激活时按选定的 chain 数只取前面相应维度。
- **分类头与目标函数**：借鉴 Matryoshka（俄罗斯套娃）式的多头 loss，每个 scale 用共享权重的 head 各自做预测；实际训练为省算力，先完整训好最大 scale 的 head，再用多头 loss 做 post-training 单独训各 scale 的分类头。

### 4. 实验（32 张 40G A100 量级，仅验证可行性）

讲者先声明实验规模有限（32 张 40G 的 A100，主要做同步对比），结论按其口播：

- **主性能对比**：不用 KV sharing 时，CoLM 本质是标准模型的一种新的参数分布，关键在找最优 setting。同为 2048 维、16 层时 chain 版参数量约为原模型的 0.86（块下三角所致），性能低约 0.1 个点；加大维度比加层数更有效（讲者明确 increase dimension 比 increase layer 稍好），如扩到 2560 维即可与原模型相当，3072 维的 setting 做到 44.51（所指 benchmark 字幕未点名，存疑不补）。chain 切到 8 份（8888）时因第一条 chain 只有 100 多维，性能下降更明显；16/16 切分时第一条 chain 仍占约 400M 参数（1.1B 模型口径），下降较小。
- **模型扩展（model expansion）**：拿现成模型当第一条 chain、向外扩维度继续训——对 TinyLlama 与 Llama-3.2-1B 各扩训约 2 万步，TinyLlama 很快提升约 1 个点；Llama-3.2-1B 本身更强，提升约 0.15 个点但稳定。做法上保留原模型的 logic（原维度完整保留），新维度在其外做加法。
- **Prefilling 加速**：KV 全由第一条 chain 算时，64K 长度下：第一条 chain 占 16 份中的 1 份（约 400M 参数）可加速 30% 多；第一条 chain 只占 8 分之 1（约 140M 参数）时从约 2500 降到不足 1000、加速 2.5 倍以上；且模型越大加速越好（此处时间单位字幕未明说，只记倍数）。再叠加 sparse attention 类 post-hoc 加速可到 27 倍——因为第一条 chain 本身就是个标准模型，与这类方法完全兼容。
- **Chain tuning（讲者用 claim 的口吻提出、尚缺合适 benchmark 验证）**：冻结第一条 chain、只微调后面的 chain，由于后面 chain 基本只含 query 相关参数，微调后模型生成的 KV 仍可被原始模型无缝接收——即专家微调完还能接回大模型用。

### 5. 讨论：与相邻范式的关系与局限

- **深度扩展 vs 宽度扩展**：深度扩展擅长 feature 能力且能复用权重续训，但不能保留原模型的 logits（最终概率预测）——Google 的做法是在不同层另接 classifier 才能维持 scale 能力；宽度扩展（含本方法）扩展性更强，能把从第一层到最后一层乃至分类头完整保留。本方法与二者正交。
- **NAS / 弹性推理**：NAS 是训一个 supernet 再用 sampling policy 抽子网；讲者反问——训完 175B 为什么不能直接选其中 30B 来用、为什么小模型总要靠 sample 才得到？Chain-of-Model 把各 scale 联合训在一起，不需要 sampling policy，弹性推理效率很高；代价是只支持同质架构，NAS 支持异质架构。
- **局限一（infra）**：chain linear 本质是 imbalance（不均衡）tensor 计算，与 DP、pipeline 并行、context 并行都兼容，与 tensor 并行不太合拍（naive 实现把每块单独做 TP 会成倍增加 all-reduce 次数）。讲者认为未来的分布式计算需要涵盖这种不均衡的链式分布，现有分布式体系尚未涵盖。
- **局限二（最优 setting 未知）**：各 chain 维度怎么分配是开放问题，尤其是第一条 chain 直接决定 understanding 能力、且影响所有后续 chain。实验里用等分（8888、16/16）主要是为 infra 效率；更细的 chain（8 份、16 份在 kernel 层测过）会伤性能上限；chain 维度取 128 / 256 的 block-wise 倍数效率更高。每个模型尺寸可能需要一套自己的 scaling law，这点讲者明确说还没探讨清楚。

Q&A 中讲者再确认：链式设计本身不带来额外计算量或内存开销（CPU 视角零开销），开销都来自实现层的多次 data access（推理）与 all-reduce 增多（训练）；子模型之间不存在反向干扰——后面的 chain 不影响前面的 chain，但前面的 chain 训得差，后面一定受影响（嵌套式单向影响）。后续或可扩展成 tree model（分叉结构），但需要更强的 infra 支持。代码还在 review 中、承诺最终开源，论文附录已放 linear / attention / FFN 的 demo code。

## 关键数字总表

| 指标 | 数值 | 来源 |
|---|---|---|
| 实验算力规模 | 32 张 40G A100 | 字幕（实验设置） |
| 同配置（2048 维、16 层）chain 版参数量 | 约为同尺寸标准模型的 0.86 | 字幕（性能对比） |
| 同配置下性能差距 | 低约 0.1 个点 | 字幕（性能对比） |
| KV sharing 的精度代价 | 约 1 个点（可加参数补回） | 字幕（KV sharing） |
| 现成模型扩展幅度 | TinyLlama / Llama-3.2-1B 向外扩维度续训约 2 万步 | 字幕（model expansion） |
| 扩展收益 | TinyLlama 约 +1 点；Llama-3.2-1B 约 +0.15 点 | 字幕（model expansion） |
| prefilling 加速（64K、第一条 chain 为 1/16 参数量） | 30% 多 | 字幕（prefilling） |
| prefilling 加速（64K、第一条 chain 为 1/8、约 140M） | 2.5 倍以上（约 2500 → 不足 1000，单位未明说） | 字幕（prefilling） |
| 叠加 sparse attention 后的 prefilling 加速 | 27 倍 | 字幕（prefilling） |
| 典型 chain 切分示例 | 1024 维切 256 / 256 / 512；实验常用等分 8888、16/16 | 字幕（实现与实验） |

## 可迁移

- **弹性推理的另一种实现路径**：不用训多个尺寸、不用 supernet sampling——把模型内部表征切成嵌套 chain，部署时按任务难度选激活档位（第一条 chain 即是一个完整可用的标准模型）。对 RL rollout / agent 场景的直接启发是：rollout 采样可以用小档位快速生成、难样本再切大档位，而 KV sharing 保证切换档位时已算好的 KV 不浪费。
- **模型扩展的工程姿势**：拿已训好的开源模型当第一条 chain、向外扩维度续训，只需约 2 万步就有可测收益，且原模型的 logits 与行为完整保留——这比从头训一个更大模型或做深度堆叠更贴近「在现成 checkpoint 上渐进加能力」的 infra 需求；chain tuning（冻第一条 chain、只训后面、KV 可接回原模型）同理可用于专家微调。
- **Infra 视角的提醒**：块下三角权重把参数量压到同尺寸的约 0.86，但 naive 实现会因多次 data access 与 all-reduce 反而更慢——新架构的真实成本在 kernel 与分布式层（imbalance tensor），评估这类「理论计算量更省」的设计时必须把 block-wise kernel 与并行策略一起算账。

## 疑问 / 下一步

- 各 chain 维度的最优分配（尤其第一条 chain 多大才够担 understanding）没有 scaling law 支撑，讲者只给了等分经验做法；这是该范式能否 scale 到大模型的第一道未答题，需等论文或后续工作。
- 44.51 一数出自 3072 维 setting，但对应哪个 benchmark 字幕没有点名，且全实验都在 1B 量级、32 张卡上完成；Chain-of-Model 在几十 B 以上是否还保持「同配置只低 0.1 点」完全没有证据。
- chain tuning「微调后 KV 可接回原模型」目前只是 claim，没有 benchmark 验证；且只微调后面 chain 对最终能力上限的影响（后面 chain 只能利用前面 chain 的信息）值得追问。

## 原文金句

> 「我们并不总是需要它。也就是说我们在实际……有的任务，比如说 1+1=2 这种很简单的题目，那它是不是还需要我们去激活那么大的一个值？」

> 「如果说我们要去增扩我们的一个大脑容量，它是一个逐渐扩充的一个范围……但在 deep learning 的这个范式里面，它并不是这样的，它总是需要我们 train from scratch。」

> 「我第一个 chain 本身来说是一个非常重要的一个 chain，因为它直接 determine 你的 understanding 的能力。」（Q&A 谈最优 setting 时）
